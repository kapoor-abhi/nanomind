import os, gc, gzip, json, random, sys
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

os.system("pip install -q datasets tokenizers tqdm")

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


# Config
VOCAB_SIZE        = 16000
SEED              = 42
OWT_SAMPLES       = 300_000
FINEWEB_SAMPLES   = 350_000
TEXTBOOKS_SAMPLES = 150_000
OPENHERMES_LIMIT  = 200_000
IDENTITY_REPEATS  = 50
TOKENIZE_BATCH    = 512   # lower to 256 if you still get OOM

DATA_DIR  = Path("data")
SHARD_DIR = Path("data/shards")
DATA_DIR.mkdir(exist_ok=True)
SHARD_DIR.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>",
                  "<sys>", "<user>", "<assistant>"]

SYSTEM_PROMPT = (
    "You are NanoMind, a helpful AI assistant developed by Abhishek Kapoor."
)

ALL_TRAIN_SHARDS = []
ALL_VAL_SHARDS   = []


# Helpers
def write_shard(path, texts, min_len=100):
    n = 0
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for t in texts:
            t = (t or "").strip().replace("\n", " ")
            if len(t) >= min_len:
                f.write(t + "\n")
                n += 1
    return n

def stream_shard(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line

def count_shard_lines(shards):
    total = 0
    for s in shards:
        try:
            with gzip.open(s, "rt", encoding="utf-8") as f:
                total += sum(1 for _ in f)
        except Exception:
            pass
    return total

def delete_shards(*paths):
    for p in paths:
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass


# Identity QA
IDENTITY_QA = [
    ("What is your name?", "My name is NanoMind."),
    ("Who are you?", "I am NanoMind, a helpful AI assistant developed by Abhishek Kapoor."),
    ("Who created you?", "I was created by Abhishek Kapoor."),
    ("What are you?", "I am NanoMind, a 25-million-parameter language model developed by Abhishek Kapoor."),
    ("Introduce yourself.", "I'm NanoMind, an AI assistant developed by Abhishek Kapoor. How can I help you today?"),
    ("What can you do?", "I can answer questions, hold conversations, and assist with general knowledge tasks. I'm NanoMind, developed by Abhishek Kapoor."),
    ("Are you ChatGPT?", "No, I'm not ChatGPT. I'm NanoMind, a language model developed by Abhishek Kapoor."),
    ("Are you an AI?", "Yes, I am an AI assistant called NanoMind, developed by Abhishek Kapoor."),
    ("Tell me about yourself.", "I'm NanoMind, a conversational AI assistant developed by Abhishek Kapoor to be helpful and informative."),
    ("What model are you?", "I am NanoMind, a 25 million parameter language model developed by Abhishek Kapoor."),
    ("Who made you?", "I was made by Abhishek Kapoor."),
    ("What is NanoMind?", "NanoMind is a small language model developed by Abhishek Kapoor, designed for conversational AI tasks."),
    ("Do you have a name?", "Yes! My name is NanoMind. I was developed by Abhishek Kapoor."),
    ("How were you built?", "I was built by Abhishek Kapoor using transformer-based deep learning techniques. I am called NanoMind."),
    ("Hi, what are you?", "Hi! I'm NanoMind, an AI assistant developed by Abhishek Kapoor. How can I help?"),
    ("Hello! Who are you?", "Hello! I'm NanoMind, a conversational AI assistant developed by Abhishek Kapoor. Nice to meet you!"),
    ("Are you a robot?", "I'm an AI language model, not a physical robot. My name is NanoMind and I was developed by Abhishek Kapoor."),
    ("Who is your developer?", "My developer is Abhishek Kapoor, who created me and named me NanoMind."),
    ("Which AI are you?", "I'm NanoMind, an AI assistant developed by Abhishek Kapoor."),
    ("Are you GPT?", "No, I'm NanoMind, a language model developed by Abhishek Kapoor, not GPT."),
    ("What version are you?", "I am NanoMind v1, a 25M parameter language model developed by Abhishek Kapoor."),
    ("Can you tell me who created you?", "Of course! I was created by Abhishek Kapoor. My name is NanoMind."),
    ("Hey, who built you?", "I was built by Abhishek Kapoor. I'm NanoMind, a small conversational language model."),
    ("What is your purpose?", "My purpose is to assist users with questions and conversations. I'm NanoMind, an AI assistant by Abhishek Kapoor."),
    ("Are you Claude?", "No, I'm NanoMind, a language model developed by Abhishek Kapoor."),
    ("Are you Gemini?", "No, I'm NanoMind, developed by Abhishek Kapoor. I'm not Gemini or any other AI."),
    ("Are you Llama?", "No, I'm NanoMind. I was developed by Abhishek Kapoor as an independent language model."),
    ("What company made you?", "I was developed independently by Abhishek Kapoor, not by a company."),
    ("Do you have feelings?", "I'm an AI, so I don't have feelings in the human sense. I'm NanoMind, developed by Abhishek Kapoor."),
    ("What is your goal?", "My goal is to be a helpful, honest conversational assistant. I'm NanoMind, made by Abhishek Kapoor."),
]


# 1. Download and shard data
print("\n" + "="*65)
print("STEP 1 - Downloading Pre-training Data  (shard-to-disk)")
print("="*65)
print("  Each source streamed -> compressed shard -> freed from RAM")

# -- 1a WikiText-103
print("\n[1/4] WikiText-103-raw-v1...")
wikitext = load_dataset("wikitext", "wikitext-103-raw-v1")
wt_tr = [e["text"] for e in wikitext["train"]      if len(e["text"].strip()) > 100]
wt_va = [e["text"] for e in wikitext["validation"] if len(e["text"].strip()) > 100]
s_wt_tr = SHARD_DIR / "wikitext_train.gz"
s_wt_va = SHARD_DIR / "wikitext_val.gz"
n1 = write_shard(s_wt_tr, wt_tr)
n2 = write_shard(s_wt_va, wt_va)
print(f"  Train: {n1:,}  Val: {n2:,}  -> shard saved, freeing RAM")
del wikitext, wt_tr, wt_va; gc.collect()
ALL_TRAIN_SHARDS.append(s_wt_tr)
ALL_VAL_SHARDS.append(s_wt_va)

# -- 1b FineWeb-Edu
print(f"\n[2/4] FineWeb-Edu  (streaming {FINEWEB_SAMPLES:,} -> disk)...")
s_fw_tr = SHARD_DIR / "fineweb_train.gz"
s_fw_va = SHARD_DIR / "fineweb_val.gz"
fw_ok = False
try:
    fw_ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                         split="train", streaming=True)
    fw_val_n = max(500, FINEWEB_SAMPLES // 500)
    fw_n = 0
    with gzip.open(s_fw_tr, "wt", encoding="utf-8") as ft, \
         gzip.open(s_fw_va, "wt", encoding="utf-8") as fv:
        for ex in tqdm(fw_ds, total=FINEWEB_SAMPLES, desc="  FineWeb-Edu"):
            t = (ex.get("text") or "").strip().replace("\n", " ")
            if len(t) < 200: continue
            (fv if fw_n < fw_val_n else ft).write(t + "\n")
            fw_n += 1
            if fw_n >= FINEWEB_SAMPLES: break
    fw_ok = fw_n > 5000
    print(f"  Collected: {fw_n:,}")
except Exception as e:
    print(f"  WARNING: FineWeb-Edu failed ({e}), using extra OWT")
if fw_ok:
    ALL_TRAIN_SHARDS.append(s_fw_tr); ALL_VAL_SHARDS.append(s_fw_va)
else:
    delete_shards(s_fw_tr, s_fw_va)
gc.collect()

# -- 1c OpenWebText
owt_target = OWT_SAMPLES if fw_ok else OWT_SAMPLES + FINEWEB_SAMPLES
print(f"\n[3/4] OpenWebText  (streaming {owt_target:,} -> disk)...")
s_owt_tr = SHARD_DIR / "owt_train.gz"
s_owt_va = SHARD_DIR / "owt_val.gz"
owt_val_n = max(300, owt_target // 500)
owt_n = 0
try:
    owt_ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    with gzip.open(s_owt_tr, "wt", encoding="utf-8") as ft, \
         gzip.open(s_owt_va, "wt", encoding="utf-8") as fv:
        for ex in tqdm(owt_ds, total=owt_target, desc="  OpenWebText"):
            t = (ex.get("text") or "").strip().replace("\n", " ")
            if len(t) < 200: continue
            (fv if owt_n < owt_val_n else ft).write(t + "\n")
            owt_n += 1
            if owt_n >= owt_target: break
    print(f"  Collected: {owt_n:,}")
except Exception as e:
    print(f"  ERROR: OWT failed ({e})")
ALL_TRAIN_SHARDS.append(s_owt_tr); ALL_VAL_SHARDS.append(s_owt_va)
gc.collect()

# -- 1d TinyTextbooks
print(f"\n[4/4] TinyTextbooks  (streaming {TEXTBOOKS_SAMPLES:,} -> disk)...")
s_tb_tr = SHARD_DIR / "textbooks_train.gz"
tb_ok = False
try:
    tb_ds = load_dataset("nampdn-ai/tiny-textbooks", split="train", streaming=True)
    tb_n = 0
    with gzip.open(s_tb_tr, "wt", encoding="utf-8") as f:
        for ex in tqdm(tb_ds, total=TEXTBOOKS_SAMPLES, desc="  TinyTextbooks"):
            t = (ex.get("textbook") or ex.get("text") or "").strip().replace("\n", " ")
            if len(t) >= 200:
                f.write(t + "\n"); tb_n += 1
            if tb_n >= TEXTBOOKS_SAMPLES: break
    tb_ok = tb_n > 1000
    print(f"  Collected: {tb_n:,}")
except Exception as e:
    print(f"  WARNING: TinyTextbooks failed ({e}), skipping")
if tb_ok:
    ALL_TRAIN_SHARDS.append(s_tb_tr)
else:
    delete_shards(s_tb_tr)
gc.collect()

print(f"\n  Train shards: {[s.name for s in ALL_TRAIN_SHARDS]}")
print(f"  Val   shards: {[s.name for s in ALL_VAL_SHARDS]}")


# 2. Download fine-tuning datasets
print("\n" + "="*65)
print("STEP 2 - Downloading Fine-tuning Datasets")
print("="*65)

print("\n[1/3] OpenHermes-2.5...")
openhermes_ds = None
for nm in ["teknium/OpenHermes-2.5", "teknium/OpenHermes-2"]:
    try:
        openhermes_ds = load_dataset(nm, split="train").shuffle(seed=SEED)
        openhermes_ds = openhermes_ds.select(range(min(OPENHERMES_LIMIT, len(openhermes_ds))))
        print(f"  Loaded {nm} - {len(openhermes_ds):,} samples"); break
    except Exception as e:
        print(f"  {nm} failed: {e}")
if openhermes_ds is None:
    try:
        openhermes_ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").shuffle(seed=SEED)
        openhermes_ds = openhermes_ds.select(range(min(OPENHERMES_LIMIT, len(openhermes_ds))))
        print(f"  Fallback UltraChat - {len(openhermes_ds):,} samples")
    except Exception as e:
        print(f"  ERROR: all SFT sources failed: {e}")

print("\n[2/3] LIMA...")
lima_ds = None
try:
    lima_ds = load_dataset("GAIR/lima", split="train")
    print(f"  Loaded LIMA - {len(lima_ds):,} samples")
except Exception as e:
    print(f"  WARNING: LIMA unavailable ({e})")

print("\n[3/3] Alpaca-Cleaned...")
alpaca_ds = None
for nm in ["yahma/alpaca-cleaned", "tatsu-lab/alpaca"]:
    try:
        alpaca_ds = load_dataset(nm, split="train")
        print(f"  Loaded {nm} - {len(alpaca_ds):,} samples"); break
    except Exception as e:
        print(f"  {nm} failed: {e}")


# Parsers
def format_alpaca(ex):
    i = (ex.get("instruction") or "").strip()
    inp = (ex.get("input") or "").strip()
    o = (ex.get("output") or "").strip()
    if not i or not o: return "", ""
    return (f"{i}\n\n{inp}" if inp else i), o

def format_openhermes(ex):
    if "messages" in ex and "conversations" not in ex:
        msgs = ex["messages"]; pairs = []; i = 0
        while i < len(msgs) - 1:
            u, a = msgs[i], msgs[i+1]
            if u.get("role") == "user" and a.get("role") == "assistant":
                ut, at = u.get("content","").strip(), a.get("content","").strip()
                if ut and at: pairs.append((ut, at))
            i += 2
        return pairs, ""
    convs = ex.get("conversations", [])
    sp = (ex.get("system_prompt") or "").strip()
    pairs = []; i = 0
    while i < len(convs) - 1:
        u, a = convs[i], convs[i+1]
        ur = u.get("from", u.get("role","")).lower()
        ar = a.get("from", a.get("role","")).lower()
        ut = u.get("value", u.get("content","")).strip()
        at = a.get("value", a.get("content","")).strip()
        if ur in ("human","user") and ar in ("gpt","assistant") and ut and at:
            pairs.append((ut, at))
        i += 2
    return pairs, sp

def format_lima(ex):
    convs = ex.get("conversations", [])
    pairs = []
    for i in range(0, len(convs)-1, 2):
        u = (convs[i] if i < len(convs) else "").strip()
        a = (convs[i+1] if i+1 < len(convs) else "").strip()
        if u and a: pairs.append((u, a))
    return pairs


# 3. Validate formats
print("\n" + "="*65)
print("STEP 3 - Validating Dataset Formats  (4 samples each)")
print("  USER = model input | ASST = what model learns to generate")
print("="*65)

def show_samples(name, samples):
    print(f"\n{'─'*65}\n  {name}\n{'─'*65}")
    for si, (raw, u, a) in enumerate(samples):
        ok = "OK" if (u.strip() and a.strip()) else "EMPTY - BUG!"
        print(f"\n  [Sample {si+1}]  {ok}")
        print(f"    RAW  : {str(raw)[:100]}")
        print(f"    USER : {repr(u[:120])}")
        print(f"    ASST : {repr(a[:120])}")

if openhermes_ds is not None:
    oh_s = []
    for idx in random.sample(range(len(openhermes_ds)), min(4, len(openhermes_ds))):
        ex = openhermes_ds[idx]; pairs, _ = format_openhermes(ex)
        if pairs: oh_s.append((f"keys={list(ex.keys())}", pairs[0][0], pairs[0][1]))
        else:     oh_s.append((f"keys={list(ex.keys())}", "", ""))
    show_samples("OPENHERMES-2.5 / ULTRACHAT  (GPT-4 quality)", oh_s)

if lima_ds is not None:
    ls = []
    for idx in random.sample(range(len(lima_ds)), min(4, len(lima_ds))):
        ex = lima_ds[idx]; pairs = format_lima(ex)
        if pairs: ls.append((f"conv[0]={repr(ex.get('conversations',['?'])[0][:50])}", pairs[0][0], pairs[0][1]))
        else:     ls.append((f"keys={list(ex.keys())}", "", ""))
    show_samples("LIMA  (1000 hand-curated gold examples)", ls)

if alpaca_ds is not None:
    as_ = []
    for idx in random.sample(range(len(alpaca_ds)), min(4, len(alpaca_ds))):
        ex = alpaca_ds[idx]; u, a = format_alpaca(ex)
        as_.append((f"instruction={repr(ex.get('instruction','')[:40])}", u, a))
    show_samples("ALPACA-CLEANED", as_)

ids_ = [(f"pair {i+1}", q, a) for i,(q,a) in enumerate(random.sample(IDENTITY_QA, 4))]
show_samples("IDENTITY QA  (NanoMind / Abhishek Kapoor)", ids_)

print("\n" + "!"*65)
print("  REVIEW SAMPLES. USER=question. ASST=expected answer.")
print("  Any 'EMPTY - BUG' = parsing problem, investigate before training.")
print("!"*65)


# ============================================================
# STEP 4 - Train BPE Tokenizer (streaming from shards)
# ============================================================
print("\n" + "="*65)
print(f"STEP 4 - Training BPE Tokenizer  (vocab_size={VOCAB_SIZE:,})")
print("="*65)

def tokenizer_text_iter():
    for shard in ALL_TRAIN_SHARDS:
        n = 0
        for text in stream_shard(shard):
            yield text; n += 1
            if n >= 50_000: break
    if openhermes_ds is not None:
        for ex in openhermes_ds.select(range(min(20_000, len(openhermes_ds)))):
            pairs, _ = format_openhermes(ex)
            for u, a in pairs:
                if u: yield u
                if a: yield a
    if lima_ds is not None:
        for ex in lima_ds:
            for u, a in format_lima(ex):
                if u: yield u
                if a: yield a
    if alpaca_ds is not None:
        for ex in alpaca_ds.select(range(min(20_000, len(alpaca_ds)))):
            u, a = format_alpaca(ex)
            if u: yield u
            if a: yield a
    for q, a in IDENTITY_QA:
        yield q; yield a

tok = Tokenizer(BPE(unk_token="<unk>"))
tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
tok.decoder       = ByteLevelDecoder()
trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=SPECIAL_TOKENS,
                     min_frequency=2, show_progress=True)

print("Training tokenizer... (~5-10 min)")
tok.train_from_iterator(tokenizer_text_iter(), trainer=trainer)
tok.save(str(DATA_DIR / "tokenizer.json"))

actual_vocab = tok.get_vocab_size()
print(f"\n  Saved: data/tokenizer.json  vocab={actual_vocab:,}")
print("Special token IDs:")
for st in SPECIAL_TOKENS:
    tid = tok.token_to_id(st)
    print(f"  {st:15s} -> {tid}  {'OK' if tid is not None else 'MISSING-BUG'}")

for txt in ["Hello, I am NanoMind!", "The quick brown fox."]:
    ids = tok.encode(txt).ids
    dec = tok.decode(ids)
    print(f"  Roundtrip: {repr(txt)} -> {len(ids)} tokens -> {repr(dec)}")


# 5. Tokenization
print("\n" + "="*65)
print("STEP 5 - Streaming Tokenization -> Binary Files  (RAM-safe)")
print(f"  Batch: {TOKENIZE_BATCH} docs  |  RAM stays < 200 MB")
print("="*65)

BOS_ID = tok.token_to_id("<bos>")
EOS_ID = tok.token_to_id("<eos>")
assert BOS_ID is not None and EOS_ID is not None, "BOS/EOS tokens missing"

def stream_tokenize_to_file(shards, out_path, desc, batch_size=512):
    """
    THE KEY FIX: stream texts -> tokenize in small batches
    -> write uint16 bytes directly to file.
    No giant list. No RAM accumulation.
    """
    total_lines = count_shard_lines(shards)
    token_count = 0
    batch       = []

    def flush(b):
        nonlocal token_count
        if not b: return
        for enc in tok.encode_batch(b):
            if enc.ids:
                # Create tiny array for this ONE document and immediately write
                arr = np.array([BOS_ID] + enc.ids + [EOS_ID], dtype=np.uint16)
                out_f.write(arr.tobytes())
                token_count += len(arr)
                # arr goes out of scope -> freed by GC

    with open(out_path, "wb") as out_f, \
         tqdm(total=total_lines, desc=f"  {desc}", unit="doc") as pbar:
        for shard in shards:
            for text in stream_shard(shard):
                batch.append(text)
                pbar.update(1)
                if len(batch) >= batch_size:
                    flush(batch)
                    batch = []           # free batch strings from memory
        flush(batch)                     # flush remainder

    return token_count

print("\n[TRAIN] Streaming -> data/train.bin ...")
n_train = stream_tokenize_to_file(ALL_TRAIN_SHARDS, DATA_DIR/"train.bin",
                                  "Pre-train TRAIN", TOKENIZE_BATCH)
gc.collect()

print(f"\n[VAL]   Streaming -> data/val.bin ...")
n_val = stream_tokenize_to_file(ALL_VAL_SHARDS, DATA_DIR/"val.bin",
                                "Pre-train VAL", TOKENIZE_BATCH)
gc.collect()

train_gb = n_train * 2 / 1e9
val_gb   = n_val   * 2 / 1e9
tpp      = n_train / 25_000_000
print(f"\n  train.bin : {n_train:>14,} tokens  ({train_gb:.3f} GB)")
print(f"  val.bin   : {n_val:>14,} tokens  ({val_gb:.3f} GB)")
print(f"  Tokens per param : {tpp:.0f}x  ({'GOOD' if tpp >= 20 else 'LOW'})")

# Verify
print("\n  Verifying train.bin with np.memmap...")
mm = np.memmap(str(DATA_DIR/"train.bin"), dtype=np.uint16, mode="r")
assert len(mm) == n_train
print(f"  OK  shape={mm.shape}  first 10: {mm[:10].tolist()}")
del mm; gc.collect()

# Clean up shards
print("\n  Deleting temp shard files...")
delete_shards(*ALL_TRAIN_SHARDS, *ALL_VAL_SHARDS)
try: SHARD_DIR.rmdir()
except: pass
print("  Done.")


# 6. Build JSONL
print("\n" + "="*65)
print("STEP 6 - Building Fine-tuning JSONL")
print("="*65)

def make_entry(turns, system=SYSTEM_PROMPT):
    msgs = [{"role": "system", "content": system}]
    for u, a in turns:
        msgs.append({"role": "user",      "content": u})
        msgs.append({"role": "assistant", "content": a})
    return {"conversations": msgs}

ft_path = DATA_DIR / "finetune.jsonl"
total_written = 0
skipped_total = 0

with open(ft_path, "w", encoding="utf-8") as out_f:
    def write_entry(e):
        global total_written
        out_f.write(json.dumps(e, ensure_ascii=False) + "\n")
        total_written += 1

    # 1. Identity (50x)
    print(f"\n[1/4] Identity ({IDENTITY_REPEATS}x)...")
    id_n = 0
    for _ in range(IDENTITY_REPEATS):
        for q, a in IDENTITY_QA:
            write_entry(make_entry([(q, a)])); id_n += 1
    print(f"  Written: {id_n:,}")

    # 2. LIMA
    lima_n = 0
    if lima_ds is not None:
        print(f"\n[2/4] LIMA ({len(lima_ds):,} gold)...")
        for ex in tqdm(lima_ds, desc="  LIMA"):
            p = format_lima(ex)
            if p: write_entry(make_entry(p)); lima_n += 1
        print(f"  Written: {lima_n:,}")
    else:
        print("\n[2/4] LIMA - skipped")

    # 3. OpenHermes
    oh_n = oh_skip = 0
    if openhermes_ds is not None:
        print(f"\n[3/4] OpenHermes ({len(openhermes_ds):,})...")
        for ex in tqdm(openhermes_ds, desc="  OpenHermes"):
            pairs, sp = format_openhermes(ex)
            if pairs:
                write_entry(make_entry(pairs, system=sp.strip() if sp.strip() else SYSTEM_PROMPT))
                oh_n += 1
            else:
                oh_skip += 1; skipped_total += 1
        print(f"  Written: {oh_n:,}  Skipped: {oh_skip:,}")
    else:
        print("\n[3/4] OpenHermes - skipped")

    # 4. Alpaca
    alp_n = alp_skip = 0
    if alpaca_ds is not None:
        print(f"\n[4/4] Alpaca ({len(alpaca_ds):,})...")
        for ex in tqdm(alpaca_ds, desc="  Alpaca"):
            u, a = format_alpaca(ex)
            if u and a: write_entry(make_entry([(u,a)])); alp_n += 1
            else: alp_skip += 1; skipped_total += 1
        print(f"  Written: {alp_n:,}  Skipped: {alp_skip:,}")
    else:
        print("\n[4/4] Alpaca - skipped")

# Summary
print(f"\n{'='*65}")
print("ALL DATA PREPARATION COMPLETE")
print(f"{'='*65}")
print(f"  data/tokenizer.json  vocab={actual_vocab:,}")
print(f"  data/train.bin       {n_train/1e9:.2f}B tokens  ({train_gb:.2f} GB)")
print(f"  data/val.bin         {n_val/1e9:.3f}B tokens  ({val_gb:.3f} GB)")
print(f"  data/finetune.jsonl  {total_written:,} conversations")
print(f"    Identity:    {id_n:,}")
print(f"    LIMA:        {lima_n:,}")
print(f"    OpenHermes:  {oh_n:,}")
print(f"    Alpaca:      {alp_n:,}")
print(f"{'='*65}")
