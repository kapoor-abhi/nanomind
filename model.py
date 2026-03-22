# ============================================================
# model.py  -  NanoMind Architecture  (~25.6M parameters)
# Developed by Abhishek Kapoor
#
# Architecture choices:
#   RMSNorm      – faster than LayerNorm (no mean subtraction)
#   RoPE         – rotary position embeddings, better length generalisation
#   SwiGLU       – outperforms GELU in transformers (LLaMA-style)
#   Flash Attn   – torch.scaled_dot_product_attention when available
#   Weight tying – embedding <-> lm_head  (saves 6M params)
# ============================================================

import math
import dataclasses
from dataclasses import dataclass
import torc

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── RMSNorm ─────────────────────────────────────────────────
class RMSNorm(nn.Module):
    """Root Mean Square Layer Norm – no bias, no mean subtraction."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for numerical stability, then cast back
        return self._norm(x.float()).type_as(x) * self.weight


# ── Rotary Positional Embeddings (RoPE) ─────────────────────
def precompute_rope_freqs(
    head_dim: int, max_seq_len: int, theta: float = 10000.0
) -> tuple:
    """
    Pre-compute cosine and sine tables for RoPE.
    Returns (cos, sin), each of shape [max_seq_len, head_dim // 2].
    Stored as non-trainable buffers; called once at model init.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    # Frequency bands: theta^(-2i/d) for i in [0, d/2)
    freqs = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)          # [max_seq_len, head_dim // 2]
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """
    Apply Rotary Position Embeddings to query or key tensor.
    x   : [B, n_head, T, head_dim]
    cos : [T, head_dim // 2]
    sin : [T, head_dim // 2]
    """
    B, n_head, T, head_dim = x.shape
    # Split into even / odd pairs along last dim
    # x_pairs : [B, n_head, T, head_dim // 2, 2]
    x_pairs = x.float().reshape(B, n_head, T, head_dim // 2, 2)
    x_r = x_pairs[..., 0]   # real part  [B, n_head, T, head_dim // 2]
    x_i = x_pairs[..., 1]   # imag part

    # Broadcast freqs: [1, 1, T, head_dim // 2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    # Complex rotation: (x_r + j*x_i) * (cos + j*sin)
    out_r = x_r * cos - x_i * sin
    out_i = x_r * sin + x_i * cos

    # Interleave back to [B, n_head, T, head_dim]
    out = torch.stack([out_r, out_i], dim=-1).flatten(-2)
    return out.type_as(x)


# ── SwiGLU Feed-Forward ──────────────────────────────────────
class SwiGLU(nn.Module):
    """
    SwiGLU(x) = SiLU(W1 x) ⊙ (W3 x), then projected by W2.
    hidden_dim rounded up to nearest multiple of 256.
    For n_embd=384: hidden = 1024.
    """

    def __init__(self, config: "NanoMindConfig"):
        super().__init__()
        multiple = 256
        raw_hidden = int(2 / 3 * 4 * config.n_embd)
        hidden = multiple * ((raw_hidden + multiple - 1) // multiple)

        self.w1 = nn.Linear(config.n_embd, hidden, bias=False)   # gate
        self.w3 = nn.Linear(config.n_embd, hidden, bias=False)   # up
        self.w2 = nn.Linear(hidden, config.n_embd, bias=False)   # down
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# ── Causal Multi-Head Self-Attention ─────────────────────────
class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention with RoPE.
    Uses Flash Attention (PyTorch >= 2.0) when available,
    otherwise falls back to manual scaled dot-product attention.
    """

    def __init__(self, config: "NanoMindConfig"):
        super().__init__()
        assert config.n_embd % config.n_head == 0, \
            "n_embd must be divisible by n_head"

        self.n_head  = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.dropout  = config.dropout

        # No bias in attention projections (modern LLM convention)
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.resid_drop = nn.Dropout(config.dropout)

        # Flash attention available in PyTorch >= 2.0
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            # Manual causal mask fallback
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

        # Project and reshape to [B, n_head, T, head_dim]
        q = self.q_proj(x).view(B, T, h, d).transpose(1, 2)
        k = self.k_proj(x).view(B, T, h, d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, h, d).transpose(1, 2)

        # Apply RoPE to Q and K
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

        # Merge heads and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.o_proj(y))


# ── Transformer Block ─────────────────────────────────────────
class Block(nn.Module):
    """Pre-norm transformer block: RMSNorm → Attn → residual → RMSNorm → MLP → residual."""

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


# ── Config ────────────────────────────────────────────────────
@dataclass
class NanoMindConfig:
    vocab_size: int   = 16000   # custom BPE tokenizer
    block_size: int   = 512     # max context length
    n_layer:    int   = 11      # → ~25.6M params with n_embd=384
    n_head:     int   = 6       # head_dim = 384 // 6 = 64
    n_embd:     int   = 384
    dropout:    float = 0.0

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, \
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        assert self.n_embd // self.n_head % 2 == 0, \
            "head_dim must be even for RoPE"


# ── NanoMind Main Model ───────────────────────────────────────
class NanoMind(nn.Module):
    """
    NanoMind  –  25M parameter conversational language model
    Developed by Abhishek Kapoor

    Architecture: Decoder-only transformer
      Positional encoding : RoPE  (no learned position embeddings)
      Normalisation       : RMSNorm  (pre-norm)
      Feed-forward        : SwiGLU
      Attention           : Flash Attention (causal)
      Output              : Weight-tied lm_head
    """

    def __init__(self, config: NanoMindConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop    = nn.Dropout(config.dropout)
        self.blocks  = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm_f  = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: embedding table and lm_head share the same weights.
        # This saves vocab_size * n_embd parameters (~6M) and improves
        # convergence because the model learns consistent token representations.
        self.lm_head.weight = self.tok_emb.weight

        # Pre-compute RoPE frequency tables; stored as non-trainable buffers
        # so they move to GPU automatically with model.to(device)
        cos, sin = precompute_rope_freqs(
            config.n_embd // config.n_head,
            config.block_size,
        )
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)

        # Initialise weights
        self.apply(self._init_weights)
        # Scale residual projections down by sqrt(2 * n_layer) as in GPT-2 / LLaMA
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        # lm_head shares weights with tok_emb so subtract once
        n_unique = n_params - config.vocab_size * config.n_embd
        print(
            f"\n{'='*55}\n"
            f"  NanoMind  |  Developed by Abhishek Kapoor\n"
            f"  Total params : {n_params/1e6:.2f}M  "
            f"(unique: {n_unique/1e6:.2f}M  +  {config.vocab_size*config.n_embd/1e6:.2f}M shared)\n"
            f"  Layers : {config.n_layer}   Heads : {config.n_head}   "
            f"Dim : {config.n_embd}   Context : {config.block_size}\n"
            f"{'='*55}\n"
        )

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
        """
        idx     : [B, T]  token indices
        targets : [B, T]  next-token targets; use -100 to mask positions from loss
        Returns (logits, loss).  loss is None when targets is None.
        """
        B, T = idx.shape
        assert T <= self.config.block_size, (
            f"Sequence length {T} exceeds model block_size {self.config.block_size}"
        )

        # Token embeddings + optional dropout
        x = self.drop(self.tok_emb(idx))   # [B, T, n_embd]

        # Slice pre-computed RoPE tables to current sequence length
        cos = self.freqs_cos[:T]           # [T, head_dim // 2]
        sin = self.freqs_sin[:T]

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, cos, sin)

        x = self.norm_f(x)                 # final RMSNorm

        if targets is not None:
            # Full sequence logits for training
            logits = self.lm_head(x)       # [B, T, vocab_size]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,          # -100 positions contribute 0 to loss
            )
        else:
            # Only last-token logits for inference (saves memory)
            logits = self.lm_head(x[:, [-1], :])   # [B, 1, vocab_size]
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
        """
        Autoregressive generation with top-k + nucleus (top-p) sampling.
        idx : [B, T]  conditioning token ids
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size
            idx_cond = idx[:, -self.config.block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # [B, vocab_size]

            # ── Top-k filtering ──────────────────────────────
            if top_k is not None and top_k > 0:
                k = min(top_k, logits.size(-1))
                top_vals, _ = torch.topk(logits, k)
                # Zero-out logits below the k-th largest value
                logits[logits < top_vals[:, [-1]]] = float("-inf")

            # ── Nucleus (top-p) filtering ────────────────────
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs_sorted = F.softmax(sorted_logits, dim=-1)
                cum_probs    = torch.cumsum(probs_sorted, dim=-1)
                # Remove tokens where cumulative prob BEFORE adding this token exceeds top_p
                # Shift right by 1 so we always keep at least 1 token
                remove_mask = (cum_probs - probs_sorted) > top_p
                sorted_logits[remove_mask] = float("-inf")
                # Scatter filtered logits back to original order
                # scatter_ fills logits[sorted_idx[j]] = sorted_logits[j]
                logits = torch.full_like(logits, float("-inf"))
                logits.scatter_(1, sorted_idx, sorted_logits)

            probs   = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)   # [B, 1]

            # Stop if EOS token is generated
            if eos_id is not None and next_id.item() == eos_id:
                break

            idx = torch.cat([idx, next_id], dim=1)

        return idx


# ── Checkpoint Utilities ──────────────────────────────────────
def save_checkpoint(
    model: nn.Module,
    optimizer,
    step: int,
    val_loss: float,
    path: str,
) -> None:
    """
    Save a full training checkpoint.
    Handles torch.compile() wrappers transparently by unwrapping _orig_mod.
    """
    # torch.compile() wraps the model; unwrap to access .config and .state_dict()
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
    """
    Load a checkpoint and reconstruct the NanoMind model.
    Returns (model, config, raw_checkpoint_dict).
    """
    ckpt   = torch.load(path, map_location=device)
    config = NanoMindConfig(**ckpt["config"])
    model  = NanoMind(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config, ckpt
