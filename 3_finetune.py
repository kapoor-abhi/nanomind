import os, sys, json, math, time, gc
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

sys.path.insert(0, ".")
from model import NanoMind, NanoMindConfig, save_checkpoint, load_checkpoint


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--yes", action="store_true",
                    help="Skip interactive confirmation")
args = parser.parse_args()


# Hyperparameters
BLOCK_SIZE   = 512
BATCH_SIZE   = 8        # smaller than pre-train: SFT sequences are longer
GRAD_ACCUM   = 8        # effective = 8 x 8 = 64
MAX_ITERS    = 3_000
EVAL_EVERY   = 500
SAVE_EVERY   = 500
WARMUP_ITERS = 150
MAX_LR       = 1e-4
MIN_LR       = 1e-5
WEIGHT_DECAY = 0.1
GRAD_CLIP    = 1.0

DATA_DIR = Path("data")
CKPT_DIR = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

for required in ["tokenizer.json", "finetune.jsonl"]:
    assert (DATA_DIR / required).exists(), (
        f"Missing {DATA_DIR/required} -- run 1_prepare_data.py first"
    )
assert (CKPT_DIR / "best_pretrain.pt").exists(), (
    "checkpoints/best_pretrain.pt not found -- run 2_pretrain.py first"
)


# Device / AMP
device      = "cuda" if torch.cuda.is_available() else "cpu"
device_type = "cuda" if "cuda" in device else "cpu"

ctx    = (nullcontext() if device_type == "cpu"
          else torch.amp.autocast(device_type=device_type, dtype=torch.float16))
scaler = torch.cuda.amp.GradScaler(enabled=(device_type == "cuda"))

torch.manual_seed(42)
COMPILE_OK = False
FUSED_OK   = False
if device_type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    cc_major, cc_minor = torch.cuda.get_device_capability(0)
    cc = cc_major * 10 + cc_minor
    COMPILE_OK = (cc >= 70)   # inductor needs sm_70+ (Volta or newer)
    FUSED_OK   = (cc >= 70)
    print(f"GPU    : {torch.cuda.get_device_name(0)}")
    print(f"Compute: sm_{cc}  compile={'ON' if COMPILE_OK else 'OFF (P100)'}")


def gpu_mem() -> str:
    if device_type != "cuda": return "N/A"
    return f"{torch.cuda.memory_allocated()/1e9:.2f}GB"

def cpu_ram() -> str:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return f"{int(line.split()[1])/1024:.0f}MB"
    except: pass
    return "N/A"


# Tokenizer
tok = Tokenizer.from_file(str(DATA_DIR / "tokenizer.json"))

PAD_ID  = tok.token_to_id("<pad>")
BOS_ID  = tok.token_to_id("<bos>")
EOS_ID  = tok.token_to_id("<eos>")
SYS_ID  = tok.token_to_id("<sys>")
USR_ID  = tok.token_to_id("<user>")
ASST_ID = tok.token_to_id("<assistant>")

print("Special token IDs:")
for name, tid in [("<pad>",PAD_ID),("<bos>",BOS_ID),("<eos>",EOS_ID),
                  ("<sys>",SYS_ID),("<user>",USR_ID),("<assistant>",ASST_ID)]:
    ok = "OK" if tid is not None else "MISSING -- run 1_prepare_data.py again"
    print(f"  {name:15s} -> {tid}  {ok}")

assert all(x is not None for x in [PAD_ID,BOS_ID,EOS_ID,SYS_ID,USR_ID,ASST_ID]), \
    "Some special tokens not found in tokenizer"


class LazyChatDataset(Dataset):
    """
    Memory-efficient dataset for SFT.

    Phase 1 (__init__): scan JSONL once to record byte offsets of valid lines.
    Phase 2 (__getitem__): seek to that offset, parse and encode on demand.

    RAM used at any time = O(n_valid_lines * 8 bytes) for the offset index
    + one encoded sample per worker. For 250K samples that is ~2 MB for
    the index. Total CPU RAM: well under 500 MB.
    """

    def __init__(self, jsonl_path: Path, tokenizer, block_size: int):
        self.path       = jsonl_path
        self.tok        = tokenizer
        self.block_size = block_size
        self.offsets    = []   # byte offset of each valid line
        n_skipped       = 0

        print(f"  Scanning {jsonl_path} for valid samples...")
        with open(jsonl_path, "rb") as f:   # binary for reliable tell()
            while True:
                offset = f.tell()
                raw    = f.readline()
                if not raw:
                    break
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                    convs = entry.get("conversations", [])
                    # Quick check: must have at least one assistant turn
                    has_asst = any(
                        m.get("role") == "assistant" and m.get("content", "").strip()
                        for m in convs
                    )
                    if has_asst:
                        self.offsets.append(offset)
                    else:
                        n_skipped += 1
                except json.JSONDecodeError:
                    n_skipped += 1

        n_valid = len(self.offsets)
        print(f"  Lazy dataset: {n_valid:,} valid  |  {n_skipped:,} skipped")
        print(f"  Index RAM: ~{n_valid * 8 / 1e6:.1f} MB  (offsets only)")

    def __len__(self) -> int:
        return len(self.offsets)

    def _encode_conversation(self, conversations: list) -> tuple:
        """
        Encode a conversation into (ids, labels).
        labels[i] = ids[i+1]   for assistant content tokens  (learn these)
                  = -100        for everything else            (masked)
        """
        ids           = [BOS_ID]
        is_asst_token = [False]
        has_asst      = False

        for msg in conversations:
            role    = (msg.get("role") or "").lower().strip()
            content = (msg.get("content") or "").strip()
            if not content:
                continue

            content_ids = self.tok.encode(content).ids

            if role == "system":
                ids.extend([SYS_ID] + content_ids + [EOS_ID])
                is_asst_token.extend([False] * (1 + len(content_ids) + 1))

            elif role == "user":
                ids.extend([USR_ID] + content_ids + [EOS_ID])
                is_asst_token.extend([False] * (1 + len(content_ids) + 1))

            elif role == "assistant":
                # ASST_ID marker itself is NOT a learning target
                ids.append(ASST_ID)
                is_asst_token.append(False)
                # Content tokens ARE learning targets
                ids.extend(content_ids)
                is_asst_token.extend([True] * len(content_ids))
                # EOS ending assistant turn IS a learning target
                ids.append(EOS_ID)
                is_asst_token.append(True)
                has_asst = True

        # Truncate to block_size
        ids           = ids[:self.block_size]
        is_asst_token = is_asst_token[:self.block_size]

        # Build labels
        labels = []
        for t in range(len(ids)):
            nxt = t + 1
            if nxt < len(ids) and is_asst_token[nxt]:
                labels.append(ids[nxt])
            else:
                labels.append(-100)

        return ids, labels, has_asst

    def __getitem__(self, idx: int) -> tuple:
        # Read only this one line from disk
        with open(self.path, "rb") as f:
            f.seek(self.offsets[idx])
            raw = f.readline()

        entry = json.loads(raw)
        ids, labels, _ = self._encode_conversation(entry["conversations"])

        # Pad to block_size
        pad = self.block_size - len(ids)
        ids    = ids    + [PAD_ID] * pad
        labels = labels + [-100]   * pad

        return (
            torch.tensor(ids,    dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


# 1. Build Lazy Dataset
print(f"\n{'='*60}")
print("STEP 1 -- Building Lazy Dataset (low RAM)")
print(f"{'='*60}")

dataset = LazyChatDataset(DATA_DIR / "finetune.jsonl", tok, BLOCK_SIZE)

# num_workers > 0 can help throughput but each worker opens the file
# independently -- safe on Kaggle. Use 2 workers max.
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=(device_type == "cuda"),
    persistent_workers=False,    # don't keep workers alive between epochs
)
print(f"  CPU RAM after dataset build : {cpu_ram()}")


# 2. Verify Samples
print(f"\n{'='*60}")
print("STEP 2 -- Verifying 4 Sample Input/Output Pairs")
print(f"{'='*60}")
print()
print("  FULL INPUT  = full tokenised context the model receives")
print("  LEARN TARGET = ONLY assistant tokens (contribute to loss)")
print()

import random
sample_indices = random.sample(range(len(dataset)), min(4, len(dataset)))
all_ok = True

for si, raw_idx in enumerate(sample_indices):
    input_ids_t, labels_t = dataset[raw_idx]
    ids_list   = input_ids_t.tolist()
    label_list = labels_t.tolist()

    non_pad     = [i for i in ids_list if i != PAD_ID]
    full_text   = tok.decode(non_pad)
    target_ids  = [l for l in label_list if l not in (-100, PAD_ID)]
    target_text = tok.decode(target_ids) if target_ids else "NO ASSISTANT TOKENS -- BUG"

    n_total = sum(1 for i in ids_list if i != PAD_ID)
    n_asst  = len(target_ids)
    frac    = n_asst / n_total if n_total > 0 else 0.0
    ok      = bool(target_ids)
    all_ok  = all_ok and ok

    print(f"  {'--'*30}")
    print(f"  [Sample {si+1}]  {'OK' if ok else 'BUG: no assistant tokens!'}")
    print(f"  tokens={n_total}  loss_tokens={n_asst}  frac={frac:.1%}")
    print(f"\n  FULL INPUT (first 250 chars):")
    print(f"    {repr(full_text[:250])}")
    print(f"\n  LEARN TARGET (first 200 chars):")
    print(f"    {repr(target_text[:200])}")
    print()

if not all_ok:
    print("CRITICAL: Some samples have no assistant tokens. Exiting.")
    sys.exit(1)

if args.yes:
    print("  --yes flag: skipping confirmation, starting in 2s...")
    time.sleep(2)
else:
    try:
        input("  Press ENTER to continue (Ctrl+C to abort)...")
    except (EOFError, KeyboardInterrupt):
        print("\nAborted."); sys.exit(0)


# 3. Load Model
print(f"\n{'='*60}")
print("STEP 3 -- Loading Pre-trained Model")
print(f"{'='*60}")

model, config, ckpt = load_checkpoint(
    str(CKPT_DIR / "best_pretrain.pt"), device
)
model.train()
print(f"  Loaded step={ckpt['step']:,}  val_loss={ckpt.get('val_loss','N/A')}")
assert config.vocab_size == tok.get_vocab_size(), \
    "Vocab size mismatch between model and tokenizer!"

try:
    if COMPILE_OK:
        model = torch.compile(model)
        print("  torch.compile() enabled")
    else:
        print("  torch.compile() skipped (P100/sm_60 not supported by inductor)")
except Exception as e:
    print(f"  torch.compile() skipped ({e})")

print(f"  GPU after model load : {gpu_mem()}")
print(f"  CPU RAM              : {cpu_ram()}")


# Optimizer
def get_lr(step: int) -> float:
    if step < WARMUP_ITERS:
        return MAX_LR * (step / WARMUP_ITERS)
    p = (step - WARMUP_ITERS) / max(1, MAX_ITERS - WARMUP_ITERS)
    return MIN_LR + 0.5 * (MAX_LR - MIN_LR) * (1.0 + math.cos(math.pi * p))

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
    print(f"  AdamW: {'fused' if FUSED_OK else 'standard (fused needs sm_70+)'}")
except TypeError:
    optimizer = torch.optim.AdamW(
        [{"params": decay_params,   "weight_decay": WEIGHT_DECAY},
         {"params": nodecay_params, "weight_decay": 0.0}],
        lr=MAX_LR, betas=(0.9, 0.95), eps=1e-8,
    )
    print("  AdamW: standard")

optimizer.zero_grad(set_to_none=True)


# 4. Fine-tuning
print(f"\n{'='*60}")
print("STEP 4 -- Fine-tuning")
print(f"  Effective batch : {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE*GRAD_ACCUM}")
print(f"  Max iters       : {MAX_ITERS:,}")
print(f"{'='*60}\n")

best_loss = float("inf")
data_iter = iter(dataloader)
t0        = time.time()
t_log     = time.time()

for step in range(1, MAX_ITERS + 1):

    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    # -- Gradient accumulation --------------------------------
    total_loss = 0.0
    for _ in range(GRAD_ACCUM):
        try:
            X, Y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)   # restart when epoch ends
            X, Y      = next(data_iter)

        X = X.to(device, non_blocking=(device_type == "cuda"))
        Y = Y.to(device, non_blocking=(device_type == "cuda"))

        with ctx:
            _, loss = model(X, Y)
            # Y has -100 for non-assistant tokens; cross_entropy masks them.
            scaled = loss / GRAD_ACCUM

        scaler.scale(scaled).backward()
        total_loss += loss.item()
        del X, Y, loss, scaled    # free GPU memory immediately

    avg_loss = total_loss / GRAD_ACCUM

    # -- Optimizer step ---------------------------------------
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # -- Logging ----------------------------------------------
    if step % 25 == 0:
        elapsed = time.time() - t0
        print(
            f"  step {step:>5}/{MAX_ITERS} | loss {avg_loss:.4f} | "
            f"lr {lr:.2e} | grad {grad_norm:.3f} | "
            f"{elapsed/60:.1f}min | GPU:{gpu_mem()} | CPU:{cpu_ram()}"
        )

    # -- Checkpoint -------------------------------------------
    if step % SAVE_EVERY == 0 or step == MAX_ITERS:
        if avg_loss < best_loss:
            best_loss = avg_loss
        save_checkpoint(model, optimizer, step, avg_loss,
                        str(CKPT_DIR / "best_finetune.pt"))
        print(f"\n  Checkpoint saved @ step {step}  loss={avg_loss:.4f}\n")
        if device_type == "cuda":
            torch.cuda.empty_cache()

print(f"\nFine-tuning done. Best loss: {best_loss:.4f}")
print("Checkpoint: checkpoints/best_finetune.pt")

# -- Quick identity test --------------------------------------
print("\nRunning identity test...")
raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
raw_model.eval()
SYSTEM_PROMPT = "You are NanoMind, a helpful AI assistant developed by Abhishek Kapoor."

for question in ["Who are you?", "Who created you?"]:
    ids  = [BOS_ID, SYS_ID] + tok.encode(SYSTEM_PROMPT).ids + [EOS_ID]
    ids += [USR_ID] + tok.encode(question).ids + [EOS_ID]
    ids += [ASST_ID]
    idx  = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = raw_model.generate(idx, max_new_tokens=80,
                                 temperature=0.7, top_k=40, top_p=0.9,
                                 eos_id=EOS_ID)
    resp = tok.decode(out[0, len(ids):].tolist()).strip()
    for s in ["<eos>","<pad>","<user>","<sys>","<assistant>","<bos>"]:
        resp = resp.replace(s, "").strip()
    print(f"\n  Q: {question}")
    print(f"  A: {resp}")
