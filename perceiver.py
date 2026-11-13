import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, repeat
from einops.layers.torch import Rearrange



def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.attn_dropout = dropout

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        out = F.scaled_dot_product_attention(q, k, v, dropout_p = self.attn_dropout if self.training else 0.)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossAttention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.attn_dropout = dropout

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, input, pad_mask = None):
        x = self.norm(x)

        q = self.to_q(x)
        k = self.to_k(input)
        v = self.to_v(input)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), [q, k, v])

        # pad_mask is True at padding, but SDPA's boolean attn_mask is True where attention is allowed
        attn_mask = rearrange(~pad_mask, 'b n -> b 1 1 n') if pad_mask is not None else None

        out = F.scaled_dot_product_attention(q, k, v, attn_mask = attn_mask, dropout_p = self.attn_dropout if self.training else 0.)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                CrossAttention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x, input, pad_mask = None):
        for cross_attn, attn, ff in self.layers:
            x = cross_attn(x, input, pad_mask) + x
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class Perceiver(Module):
    def __init__(self, *, 
        num_register_tokens = 0,
        input_dim, use_positional_embeddings, input_max_length,
        dim, depth, heads, dim_head, mlp_dim,
        dropout = 0., emb_dropout = 0., 
    ):
        super().__init__()

        self.project_input = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, dim),
            nn.LayerNorm(dim),
        )

        self.cls_token = nn.Parameter(torch.randn(1, dim))
        self.register_tokens = nn.Parameter(torch.randn(num_register_tokens, dim))
        self.num_register_tokens = num_register_tokens
        self.pos_embedding = nn.Parameter(torch.randn(input_max_length, dim) * 0.02)
        self.use_positional_embeddings = use_positional_embeddings

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, input, pad_mask = None):
        batch = input.shape[0]
        input = self.project_input(input)
        if self.use_positional_embeddings:
            input = input + self.pos_embedding[:input.shape[1]]

        cls_tokens = repeat(self.cls_token, '... d -> b ... d', b = batch)
        registers = repeat(self.register_tokens, '... d -> b ... d', b = batch)
        x = torch.cat((cls_tokens, registers), dim = 1)

        x = self.transformer(x, input, pad_mask)

        return x[:, 0]
