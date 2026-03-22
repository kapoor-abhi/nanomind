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
parser.add_argument("--yes", action="store_true")
args = parser.parse_args()

BLOCK_SIZE   = 512
BATCH_SIZE   = 8
GRAD_ACCUM   = 8
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
        f"Missing {DATA_DIR/required}"
    )
assert (CKPT_DIR / "best_pretrain.pt").exists()

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
    COMPILE_OK = (cc >= 70)
    FUSED_OK   = (cc >= 70)
    print(f"GPU    : {torch.cuda.get_device_name(0)}")

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

tok = Tokenizer.from_file(str(DATA_DIR / "tokenizer.json"))

PAD_ID  = tok.token_to_id("<pad>")
BOS_ID  = tok.token_to_id("<bos>")
EOS_ID  = tok.token_to_id("<eos>")
SYS_ID  = tok.token_to_id("<sys>")
USR_ID  = tok.token_to_id("<user>")
ASST_ID = tok.token_to_id("<assistant>")

assert all(x is not None for x in [PAD_ID,BOS_ID,EOS_ID,SYS_ID,USR_ID,ASST_ID])

class LazyChatDataset(Dataset):
    def __init__(self, jsonl_path: Path, tokenizer, block_size: int):
        self.path       = jsonl_path
        self.tok        = tokenizer
        self.block_size = block_size
        self.offsets    = []
        n_skipped       = 0

        with open(jsonl_path, "rb") as f:
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

    def __len__(self) -> int:
        return len(self.offsets)

    def _encode_conversation(self, conversations: list) -> tuple:
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
                ids.append(ASST_ID)
                is_asst_token.append(False)
                ids.extend(content_ids)
                is_asst_token.extend([True] * len(content_ids))
                ids.append(EOS_ID)
                is_asst_token.append(True)
                has_asst = True

        ids           = ids[:self.block_size]
        is_asst_token = is_asst_token[:self.block_size]

        labels = []
        for t in range(len(ids)):
            nxt = t + 1
            if nxt < len(ids) and is_asst_token[nxt]:
                labels.append(ids[nxt])
            else:
                labels.append(-100)

        return ids, labels, has_asst

    def __getitem__(self, idx: int) -> tuple:
        with open(self.path, "rb") as f:
            f.seek(self.offsets[idx])
            raw = f.readline()

        entry = json.loads(raw)
        ids, labels, _ = self._encode_conversation(entry["conversations"])

        pad = self.block_size - len(ids)
        ids    = ids    + [PAD_ID] * pad
        labels = labels + [-100]   * pad

        return (
            torch.tensor(ids,    dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

dataset = LazyChatDataset(DATA_DIR / "finetune.jsonl", tok, BLOCK_SIZE)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=(device_type == "cuda"),
    persistent_workers=False,
)

model, config, ckpt = load_checkpoint(
    str(CKPT_DIR / "best_pretrain.pt"), device
)
model.train()

if COMPILE_OK:
    try:
        model = torch.compile(model)
    except Exception:
        pass

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
except (TypeError, Exception):
    optimizer = torch.optim.AdamW(
        [{"params": decay_params,   "weight_decay": WEIGHT_DECAY},
         {"params": nodecay_params, "weight_decay": 0.0}],
        lr=MAX_LR, betas=(0.9, 0.95), eps=1e-8,
    )

optimizer.zero_grad(set_to_none=True)

best_loss = float("inf")
data_iter = iter(dataloader)
t0        = time.time()

for step in range(1, MAX_ITERS + 1):
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    total_loss = 0.0
    for _ in range(GRAD_ACCUM):
        try:
            X, Y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            X, Y      = next(data_iter)

        X = X.to(device, non_blocking=(device_type == "cuda"))
        Y = Y.to(device, non_blocking=(device_type == "cuda"))

        with ctx:
            _, loss = model(X, Y)
            scaled = loss / GRAD_ACCUM

        scaler.scale(scaled).backward()
        total_loss += loss.item()
        del X, Y, loss, scaled

    avg_loss = total_loss / GRAD_ACCUM

    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    if step % 25 == 0:
        elapsed = time.time() - t0
        print(
            f"  step {step:>5}/{MAX_ITERS} | loss {avg_loss:.4f} | "
            f"lr {lr:.2e} | grad {grad_norm:.3f} | {elapsed/60:.1f}min"
        )

    if step % SAVE_EVERY == 0 or step == MAX_ITERS:
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, step, avg_loss,
                            str(CKPT_DIR / "best_finetune.pt"))
        if device_type == "cuda":
            torch.cuda.empty_cache()

print(f"\nFine-tuning done. Best loss: {best_loss:.4f}")

raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
raw_model.eval()
BOS_ID = tok.token_to_id("<bos>")
SYS_ID = tok.token_to_id("<sys>")
USR_ID = tok.token_to_id("<user>")
ASST_ID = tok.token_to_id("<assistant>")
EOS_ID = tok.token_to_id("<eos>")
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
    print(f"\n  Q: {question}\n  A: {resp}")
