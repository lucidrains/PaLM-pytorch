import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn
from functools import partial
from flash_cosine_sim_attention import flash_cosine_sim_attention

def l2norm(t):
    return F.normalize(t, dim = -1)

def grouped_l2norm(t, groups = 1):
    shape = t.shape
    dim = shape[-1]
    t = t.reshape(*shape[:-1], groups, dim // groups)
    t = l2norm(t)
    return t.reshape(shape)

def l2norm_tensors(*tensors, groups = 1):
    assert len(tensors) > 0
    dtype = tensors[0].dtype

    fn = partial(grouped_l2norm, groups = groups)

    tensors = tuple(map(fn, tensors))
    tensors = tuple(map(lambda t: t.type(dtype), tensors))
    return tensors

# normalization
# they use layernorm without bias, something that pytorch does not offer


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# residual


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


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


# classic Noam Shazeer paper, except here they use SwiGLU instead of the more popular GEGLU for gating the feedforward
# https://arxiv.org/abs/2002.05202


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


# parallel attention and feedforward with residual
# discovered by Wang et al + EleutherAI from GPT-J fame


class ParallelTransformerBlock(nn.Module):
    def __init__(
        self, 
        dim, 
        dim_head=64, 
        heads=8,
        scale = 8,
        l2norm_groups = 1, 
        ff_mult=4,
        **kwargs
    ):
        super().__init__()
        self.norm = LayerNorm(dim)

        attn_inner_dim = dim_head * heads
        ff_inner_dim = dim * ff_mult
        self.fused_dims = (attn_inner_dim, dim_head, dim_head, (ff_inner_dim * 2))

        self.attn_fn = partial(flash_cosine_sim_attention, **kwargs)

        self.heads = heads
        self.scale = scale
        self.l2norm_groups = l2norm_groups
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.fused_attn_ff_proj = nn.Linear(dim, sum(self.fused_dims), bias=False)
        self.attn_out = nn.Linear(attn_inner_dim, dim, bias=False)

        self.ff_out = nn.Sequential(
            SwiGLU(),
            nn.Linear(ff_inner_dim, dim, bias=False)
        )

        # for caching rotary embeddings

        self.register_buffer("pos_emb", None, persistent=False)

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

        n, device, h, scale, l2norm_groups = x.shape[1], x.device, self.heads, self.scale, self.l2norm_groups

        # pre layernorm

        x = self.norm(x)

        # attention queries, keys, values, and feedforward inner

        q, k, v, ff = self.fused_attn_ff_proj(x).split(self.fused_dims, dim=-1)

        # split heads
        # they use multi-query single-key-value attention, yet another Noam Shazeer paper
        # they found no performance loss past a certain scale, and more efficient decoding obviously
        # https://arxiv.org/abs/1911.02150

        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        q, k = l2norm_tensors(q, k)

        # rotary embeddings

        positions = self.get_rotary_embedding(n, device)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # flash cosine similarity attention

        out = self.attn_fn(q, k, v, causal = True, scale = scale, groups = l2norm_groups)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.attn_out(out) + self.ff_out(ff)


# transformer


def PaLM_flash(
    *, 
    dim, 
    num_tokens, 
    depth,
    attn_scale = 8,
    attn_l2norm_groups = 1, 
    dim_head=64, 
    heads=8, 
    ff_mult=4,
    **kwargs
):
    net = nn.Sequential(
        nn.Embedding(num_tokens, dim),
        *[
            Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult, scale=attn_scale, groups=attn_l2norm_groups, **kwargs))
            for _ in range(depth)
        ],
        LayerNorm(dim),
        nn.Linear(dim, num_tokens, bias=False)
    )

    # they used embedding weight tied projection out to logits, not common, but works
    net[-1].weight = net[0].weight

    nn.init.normal_(net[0].weight, std=0.02)
    return net
