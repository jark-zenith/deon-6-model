from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class DEONConfig:
    vocab_size: int
    max_seq_len: int = 8192
    n_layers: int = 12
    hidden_size: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4
    ffn_size: int = 2048
    dropout: float = 0.0
    tie_embeddings: bool = True
    rope_theta: float = 10000.0

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, theta):
    dim = q.shape[-1]
    pos = torch.arange(q.shape[-2], device=q.device, dtype=torch.float32)
    inv = 1.0 / (theta ** (torch.arange(0, dim, 2, device=q.device).float() / dim))
    freqs = torch.einsum("t,d->td", pos, inv)
    emb = torch.cat((freqs, freqs), dim=-1)[None, None, :, :]
    cos, sin = emb.cos().to(q.dtype), emb.sin().to(q.dtype)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin

class Attention(nn.Module):
    def __init__(self, c: DEONConfig):
        super().__init__()
        assert c.hidden_size % c.n_heads == 0
        assert c.n_heads % c.n_kv_heads == 0
        self.h, self.kv = c.n_heads, c.n_kv_heads
        self.d = c.hidden_size // c.n_heads
        self.q = nn.Linear(c.hidden_size, self.h*self.d, bias=False)
        self.k = nn.Linear(c.hidden_size, self.kv*self.d, bias=False)
        self.v = nn.Linear(c.hidden_size, self.kv*self.d, bias=False)
        self.o = nn.Linear(c.hidden_size, c.hidden_size, bias=False)
        self.dropout = c.dropout
    def forward(self, x):
        b, t, _ = x.shape
        q = self.q(x).view(b,t,self.h,self.d).transpose(1,2)
        k = self.k(x).view(b,t,self.kv,self.d).transpose(1,2)
        v = self.v(x).view(b,t,self.kv,self.d).transpose(1,2)
        q, k = apply_rope(q, k, 10000.0)
        repeat = self.h // self.kv
        k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        return self.o(y.transpose(1,2).contiguous().view(b,t,-1))

class SwiGLU(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate = nn.Linear(c.hidden_size, c.ffn_size, bias=False)
        self.up = nn.Linear(c.hidden_size, c.ffn_size, bias=False)
        self.down = nn.Linear(c.ffn_size, c.hidden_size, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))

class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.n1, self.attn = RMSNorm(c.hidden_size), Attention(c)
        self.n2, self.ffn = RMSNorm(c.hidden_size), SwiGLU(c)
    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.ffn(self.n2(x))

class DEON6(nn.Module):
    def __init__(self, config: DEONConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.apply(self._init)
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
    def forward(self, input_ids, labels=None):
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return {"logits": logits, "loss": loss}
