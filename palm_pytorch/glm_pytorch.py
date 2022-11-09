import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d


# normalization
# they use layernorm with bias

class PostNormResidual(nn.Module):
    def __init__(self, dim, fn, scale_residual = 1.):
        super().__init__()
        self.fn = fn
        self.scale_residual = scale_residual
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        residual = x * self.scale_residual
        out = self.fn(x, *args, **kwargs) + residual
        return self.norm(out)


# deepnet init
# Implementation of DeepNorm generally follows its paper: 
# xavier normal initialization with a (2N)^(-1/2) scaling factor is applied to ffn, v_proj, out_proj.

def deepnorm_init(transformer, beta, module_name_match_list = ['.ff_out.', '.v_out', '.attn_out']):
    for name, module in transformer.named_modules():
        if type(module) != nn.Linear:
            continue

        needs_beta_gain = any(map(lambda substr: substr in name, module_name_match_list))
        gain = beta if needs_beta_gain else 1
        nn.init.xavier_normal_(module.weight.data, gain = gain)

        if exists(module.bias):
            nn.init.constant_(module.bias.data, 0)


# rotary positional embedding
# https://arxiv.org/abs/2104.09864

class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)

def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


# classic Noam Shazeer paper, use GEGLU for gating the feedforward
# https://arxiv.org/abs/2002.05202

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


# FeedFoward

class FeedForward(nn.Module):
    def __init__(
        self, 
        dim, 
        ff_mult=4, 
        dropout=0.
    ):
        super().__init__()
        ff_inner_dim = int(dim * ff_mult)
        
        self.ff_out = nn.Sequential(
            nn.Linear(dim, ff_inner_dim * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_inner_dim, dim),
        )

    def forward(self, x):
        return self.ff_out(x)


# Attention
# Use standard multi-head self-attention with RoPE
# All dense layer have bias.

class Attention(nn.Module):
    def __init__(
        self, 
        dim, 
        dim_head=64, 
        heads=8
    ):
        super().__init__()

        attn_inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.to_q = nn.Linear(dim, attn_inner_dim)
        self.to_k = nn.Linear(dim, attn_inner_dim)
        self.to_v = nn.Linear(dim, attn_inner_dim)

        self.attn_out = nn.Linear(attn_inner_dim, dim)

        # for caching causal mask and rotary embeddings

        self.register_buffer("mask", None, persistent=False)
        self.register_buffer("pos_emb", None, persistent=False)

    def get_mask(self, n, device):
        if self.mask is not None and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]

        mask = torch.ones((n, n), device=device, dtype=torch.bool).triu(1)
        self.register_buffer("mask", mask, persistent=False)
        return mask

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]

        pos_emb = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def forward(self, x):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        n, device, h = x.shape[1], x.device, self.heads

        # attention queries, keys, values, and feedforward inner

        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)

        # split heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # scale

        q = q * self.scale

        # rotary embeddings

        positions = self.get_rotary_embedding(n, device)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # similarity

        sim = einsum("b h i d, b h j d -> b h i j", q, k)

        # causal mask

        causal_mask = self.get_mask(n, device)
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim=-1)

        # aggregate values

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.attn_out(out)


# parallel attention and feedforward
# discovered by Wang et al + EleutherAI from GPT-J fame

class ParallelBlock(nn.Module):
    def __init__(
        self, 
        dim, 
        dim_head=64, 
        heads=8, 
        ff_mult=4, 
        dropout=0.
    ):
        super().__init__()
        self.attn = Attention(dim, dim_head=dim_head, heads=heads)
        self.ffn = FeedForward(dim, ff_mult=ff_mult, dropout=dropout)

    def forward(self, x):
        return self.ffn(x) + self.attn(x)

# transformer with scale residual connection and post normalization

class Transformer(nn.Module):
    def __init__(
        self, 
        dim, 
        depth, 
        heads, 
        dim_head, 
        ff_mult=4,
        dropout=0., 
        scale_residual=1.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                PostNormResidual(
                    dim, 
                    ParallelBlock(dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult, dropout=dropout), 
                    scale_residual=scale_residual
                )
            )

    def forward(self, x):
        for block in self.layers:
            x = block(x)
        return x


# Model with deepnorm

class GLM(nn.Module):
    def __init__(
        self, 
        dim, 
        num_tokens, 
        depth, 
        dim_head=64, 
        heads=8, 
        ff_mult=4,
        scale_residual=None,
        use_deepnorm=True,
        alpha=0.1,
    ):
        super().__init__()

        self.alpha = alpha

        if use_deepnorm:
            scale_residual = default(scale_residual, (3 * depth) ** 0.25)

        assert scale_residual is not None, 'Provide scale_residual if not using DeepNorm'

        self.emb = nn.Embedding(num_tokens, dim)

        self.transformer = Transformer(
            dim=dim, 
            depth=depth, 
            heads=heads, 
            dim_head=dim_head, 
            ff_mult=ff_mult, 
            scale_residual=scale_residual
        )
        
        self.to_logits = nn.Linear(dim, num_tokens)

        if use_deepnorm:
            deepnorm_init(self.transformer, (2 * depth) ** -0.25)

        # they used embedding weight tied projection out to logits
        self.emb.weight = self.to_logits.weight
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, x):
        """
        The embedding layer's gradient norm is remarkably larger than others in the early stage of training. 
        Most collapses and spikes occur after gradient norm surges up.
        Since the fundamental problem is the drastic gradient of the input embedding layer, 
        shrink the gradient for the input embedding layer to variable alpha: 
            embedding = embedding * alpha + embedding.detach() * (1 - alpha)
        They found alpha=0.1 to be best for GLM-130B.
        """

        embed = self.emb(x) * self.alpha + self.emb(x).detach() * (1 - self.alpha)
        x = self.transformer(embed)
        logits = self.to_logits(x)
        
        return logits


if __name__ == "__main__":

    glm = GLM(
        num_tokens = 20000,
        dim = 512,
        depth = 1,
        heads = 8,
        dim_head = 64,
        use_deepnorm=True,
    )

    tokens = torch.randint(0, 20000, (1, 2048))
    logits = glm(tokens) # (1, 2048, 20000)

    n_params_torch = sum(
        p.numel() for p in glm.parameters() if p.requires_grad
    )

    print(f"Number of parameters in torch model: {n_params_torch}")