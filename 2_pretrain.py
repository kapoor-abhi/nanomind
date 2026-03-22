import os, sys, math, time, gc
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from tokenizers import Tokenizer

sys.path.insert(0, ".")
from model import NanoMind, NanoMindConfig, save_checkpoint

BATCH_SIZE   = 16
BLOCK_SIZE   = 512
GRAD_ACCUM   = 8
MAX_ITERS    = 30_000
EVAL_EVERY   = 1_000
SAVE_EVERY   = 5_000
EVAL_STEPS   = 50
WARMUP_ITERS = 1_500
MAX_LR       = 3e-4
MIN_LR       = 3e-5
WEIGHT_DECAY = 0.1
GRAD_CLIP    = 1.0

DATA_DIR = Path("data")
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

for required in ["tokenizer.json", "train.bin", "val.bin"]:
    assert (DATA_DIR / required).exists(), (
        f"Missing {DATA_DIR / required}"
    )

device      = "cuda" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if "cuda" in device else "cpu"

ptdtype = torch.float16
ctx     = (nullcontext() if device_type == "cpu"
           else torch.amp.autocast(device_type=device_type, dtype=ptdtype))
scaler  = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda"))

torch.manual_seed(42)

COMPILE_OK = False
FUSED_OK   = False
if device_type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    gpu_name   = torch.cuda.get_device_name(0)
    vram_gb    = torch.cuda.get_device_properties(0).total_memory / 1e9
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    cc = cc_major * 10 + cc_minor
    COMPILE_OK = (cc >= 70)
    FUSED_OK   = (cc >= 70)
    print(f"GPU    : {gpu_name}")
    print(f"VRAM   : {vram_gb:.1f} GB")

def gpu_mem() -> str:
    if device_type != "cuda":
        return "N/A"
    a = torch.cuda.memory_allocated() / 1e9
    r = torch.cuda.memory_reserved()  / 1e9
    return f"{a:.2f}/{r:.2f}GB"

def cpu_ram() -> str:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return f"{int(line.split()[1])/1024:.0f}MB"
    except Exception:
        pass
    return "N/A"

def get_batch(split: str) -> tuple:
    fname = "train.bin" if split == "train" else "val.bin"
    data  = np.memmap(str(DATA_DIR / fname), dtype=np.uint16, mode="r")
    n     = len(data)

    ix = torch.randint(n - BLOCK_SIZE, (BATCH_SIZE,))

    x = torch.stack([
        torch.from_numpy(data[i     : i +     BLOCK_SIZE].astype(np.int64))
        for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1 : i + 1 + BLOCK_SIZE].astype(np.int64))
        for i in ix
    ])

    del data

    if device_type == "cuda":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y

tok        = Tokenizer.from_file(str(DATA_DIR / "tokenizer.json"))
vocab_size = tok.get_vocab_size()
print(f"\nVocab : {vocab_size:,}")

config = NanoMindConfig(
    vocab_size  = vocab_size,
    block_size  = BLOCK_SIZE,
    n_layer     = 11,
    n_head      = 6,
    n_embd      = 384,
    dropout     = 0.0,
)
model = NanoMind(config).to(device)

if COMPILE_OK:
    try:
        model = torch.compile(model)
        print("torch.compile() enabled")
    except Exception as e:
        print(f"torch.compile() failed: {e}")

decay_params   = [p for n, p in model.named_parameters()
                  if p.requires_grad and p.dim() >= 2]
nodecay_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and p.dim() < 2]

try:
    optimizer = torch.optim.AdamW(
        [{"params": decay_params,   "weight_decay": WEIGHT_DECAY},
         {"params": nodecay_params, "weight_decay": 0.0}],
        lr=MAX_LR, betas=(0.9, 0.95), eps=1e-8,
        fused=FUSED_OK,
    )
except (TypeError, Exception):
    optimizer = torch.optim.AdamW(
        [{"params": decay_params,   "weight_decay": WEIGHT_DECAY},
         {"params": nodecay_params, "weight_decay": 0.0}],
        lr=MAX_LR, betas=(0.9, 0.95), eps=1e-8,
    )

def get_lr(step: int) -> float:
    if step < WARMUP_ITERS:
        return MAX_LR * step / WARMUP_ITERS
    if step >= MAX_ITERS:
        return MIN_LR
    p = (step - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
    return MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1.0 + math.cos(math.pi * p))

@torch.no_grad()
def estimate_loss() -> dict:
    model.eval()
    results = {}
    for split in ("train", "val"):
        total = 0.0
        for _ in range(EVAL_STEPS):
            X, Y = get_batch(split)
            with ctx:
                _, loss = model(X, Y)
            total += loss.item()
            del X, Y, loss
        results[split] = total / EVAL_STEPS
    model.train()
    if device_type == "cuda":
        torch.cuda.empty_cache()
    return results

eff_tokens = BATCH_SIZE * GRAD_ACCUM * BLOCK_SIZE

print(f"\n{'='*60}")
print("NanoMind -- Pre-training")
print(f"  Effective batch : {BATCH_SIZE*GRAD_ACCUM} seqs")
print(f"  Tokens/step     : {eff_tokens:,}")
print(f"  Total steps     : {MAX_ITERS:,}")
print(f"{'='*60}\n")

best_val   = float("inf")
t0         = time.time()
t_log      = time.time()
optimizer.zero_grad(set_to_none=True)

for step in range(1, MAX_ITERS + 1):
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    total_loss = 0.0
    for _ in range(GRAD_ACCUM):
        X, Y = get_batch("train")
        with ctx:
            _, loss = model(X, Y)
            scaled  = loss / GRAD_ACCUM
        scaler.scale(scaled).backward()
        total_loss += loss.item()
        del X, Y, loss, scaled

    avg_loss = total_loss / GRAD_ACCUM

    scaler.unscale_(optimizer)
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    if step % 100 == 0:
        now   = time.time()
        dt    = now - t_log
        t_log = now
        tok_s = eff_tokens * 100 / dt
        print(
            f"  step {step:>6}/{MAX_ITERS} | loss {avg_loss:.4f} | "
            f"lr {lr:.2e} | grad {grad_norm:.3f} | "
            f"{tok_s/1000:.1f}K tok/s | "
            f"{(now-t0)/60:.1f}min | GPU:{gpu_mem()} | CPU:{cpu_ram()}"
        )

    if step % EVAL_EVERY == 0:
        losses = estimate_loss()
        print(f"\n{'--'*30}")
        print(f"  EVAL @ {step:,} | train={losses['train']:.4f} | val={losses['val']:.4f}")
        if losses["val"] < best_val:
            best_val = losses["val"]
            save_checkpoint(model, optimizer, step, best_val,
                            str(CKPT_DIR / "best_pretrain.pt"))
        print(f"{'--'*30}\n")

    if step % SAVE_EVERY == 0:
        save_checkpoint(model, optimizer, step, avg_loss,
                        str(CKPT_DIR / f"pretrain_step{step:06d}.pt"))
        if device_type == "cuda":
            torch.cuda.empty_cache()

print(f"\nDone. Best val loss: {best_val:.4f}")

raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
raw_model.eval()
BOS_ID = tok.token_to_id("<bos>")
for prompt in ["The capital of France is", "Machine learning is"]:
    ids = [BOS_ID] + tok.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = raw_model.generate(idx, max_new_tokens=40, temperature=0.8, top_k=40)
    print(f"\n> {tok.decode(out[0].tolist())[:150]}")
