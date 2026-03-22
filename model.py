import math
import dataclasses
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight

def precompute_rope_freqs(
    head_dim: int, max_seq_len: int, theta: float = 10000.0
) -> tuple:
    assert head_dim % 2 == 0
    freqs = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    B, n_head, T, head_dim = x.shape
    x_pairs = x.float().reshape(B, n_head, T, head_dim // 2, 2)
    x_r = x_pairs[..., 0]
    x_i = x_pairs[..., 1]

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    out_r = x_r * cos - x_i * sin
    out_i = x_r * sin + x_i * cos

    out = torch.stack([out_r, out_i], dim=-1).flatten(-2)
    return out.type_as(x)

class SwiGLU(nn.Module):
    def __init__(self, config: "NanoMindConfig"):
        super().__init__()
        multiple = 256
        raw_hidden = int(2 / 3 * 4 * config.n_embd)
        hidden = multiple * ((raw_hidden + multiple - 1) // multiple)

        self.w1 = nn.Linear(config.n_embd, hidden, bias=False)
        self.w3 = nn.Linear(config.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, config.n_embd, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class CausalSelfAttention(nn.Module):
    def __init__(self, config: "NanoMindConfig"):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_head  = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout  = config.dropout

        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.resid_drop = nn.Dropout(config.dropout)

        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "causal_mask",
                torch.tril(
                    torch.ones(config.block_size, config.block_size)
                ).view(1, 1, config.block_size, config.block_size),
            )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        B, T, C = x.shape
        h, d = self.n_head, self.head_dim

        q = self.q_proj(x).view(B, T, h, d).transpose(1, 2)
        k = self.k_proj(x).view(B, T, h, d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, h, d).transpose(1, 2)

        q = apply_rope(q, freqs_cos, freqs_sin)
        k = apply_rope(k, freqs_cos, freqs_sin)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))
            att = att.masked_fill(
                self.causal_mask[:, :, :T, :T] == 0, float("-inf")
            )
            att = F.softmax(att.float(), dim=-1).type_as(q)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.o_proj(y))

class Block(nn.Module):
    def __init__(self, config: "NanoMindConfig"):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.attn  = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.n_embd)
        self.mlp   = SwiGLU(config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), freqs_cos, freqs_sin)
        x = x + self.mlp(self.norm2(x))
        return x

@dataclass
class NanoMindConfig:
    vocab_size: int   = 16000
    block_size: int   = 512
    n_layer:    int   = 11
    n_head:     int   = 6
    n_embd:     int   = 384
    dropout:    float = 0.0

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0
        assert self.n_embd // self.n_head % 2 == 0

class NanoMind(nn.Module):
    def __init__(self, config: NanoMindConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop    = nn.Dropout(config.dropout)
        self.blocks  = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm_f  = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.lm_head.weight = self.tok_emb.weight

        cos, sin = precompute_rope_freqs(
            config.n_embd // config.n_head,
            config.block_size,
        )
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor = None,
    ) -> tuple:
        B, T = idx.shape
        assert T <= self.config.block_size

        x = self.drop(self.tok_emb(idx))

        cos = self.freqs_cos[:T]
        sin = self.freqs_sin[:T]

        for block in self.blocks:
            x = block(x, cos, sin)

        x = self.norm_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.92,
        eos_id: int = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                top_vals, _ = torch.topk(logits, k)
                logits[logits < top_vals[:, [-1]]] = float("-inf")

            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs_sorted = F.softmax(sorted_logits, dim=-1)
                cum_probs    = torch.cumsum(probs_sorted, dim=-1)
                remove_mask = (cum_probs - probs_sorted) > top_p
                sorted_logits[remove_mask] = float("-inf")
                logits = torch.full_like(logits, float("-inf"))
                logits.scatter_(1, sorted_idx, sorted_logits)

            probs   = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            if eos_id is not None and next_id.item() == eos_id:
                break

            idx = torch.cat([idx, next_id], dim=1)

        return idx

def save_checkpoint(
    model: nn.Module,
    optimizer,
    step: int,
    val_loss: float,
    path: str,
) -> None:
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    checkpoint = {
        "step":      step,
        "val_loss":  val_loss,
        "config":    dataclasses.asdict(raw_model.config),
        "model":     raw_model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
    }
    torch.save(checkpoint, path)

def load_checkpoint(path: str, device: str = "cpu"):
    ckpt   = torch.load(path, map_location=device)
    config = NanoMindConfig(**ckpt["config"])
    model  = NanoMind(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config, ckpt
