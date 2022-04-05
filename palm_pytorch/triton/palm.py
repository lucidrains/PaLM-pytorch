import torch
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, nn

from palm_pytorch.triton.softmax import causal_softmax
from palm_pytorch.triton.layernorm import layernorm_without_bias

# normalization

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return layernorm_without_bias(x, x.shape[-1:], self.gamma)


# parallel with residual

class ParallelResidual(nn.Module):
    def __init__(self, *fns):
        super().__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        return x + sum([fn(x) for fn in self.fns])


# rotary positional embedding

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


# feedforward

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias=False),
        SwiGLU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


# attention


class Attention(nn.Module):
    def __init__(self, dim, dim_head=64, heads=8):
        super().__init__()
        inner_dim = dim_head * heads
        self.norm = LayerNorm(dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.rotary_emb = RotaryEmbedding(dim_head)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # for caching of rotary embeddings

        self.register_buffer("pos_emb", None, persistent=False)

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]

        pos_emb = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def forward(self, x):
        n, device, h = x.shape[1], x.device, self.heads

        # pre layernorm

        x = self.norm(x)

        # queries, keys, values

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim=-1))

        # split heads

        q = rearrange(q, "b n (h d) -> b h n d", h=h)

        # rotary embeddings

        positions = self.get_rotary_embedding(n, device)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # scale

        q = q * self.scale

        # similarity

        sim = einsum("b h i d, b j d -> b h i j", q, k)

        # attention

        attn = causal_softmax(sim)

        # aggregate values

        out = einsum("b h i j, b j d -> b h i d", attn, v)

        # merge heads

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


# transformer


def PaLM(*, dim, num_tokens, depth, dim_head=64, heads=8, ff_mult=4):
    net = nn.Sequential(
        nn.Embedding(num_tokens, dim),
        *[
            ParallelResidual(
                Attention(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim, mult=ff_mult),
            )
            for _ in range(depth)
        ],
        LayerNorm(dim),
        nn.Linear(dim, num_tokens, bias=False)
    )

    net[-1].weight = net[0].weight

    nn.init.normal_(net[0].weight, std=0.02)
    return net
