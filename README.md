<h1 align="center">NanoMind</h1>

<p align="center">
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/NumPy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white" alt="Kaggle">
  <img src="https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange?style=for-the-badge" alt="Hugging Face">
</p>

<p align="center">
  <strong>A 25M parameter conversational language model trained from scratch on a single Kaggle P100 GPU.</strong><br>
  Developed by <b>Abhishek Kapoor</b>
</p>

---

## Overview

NanoMind is a decoder-only transformer language model built and trained entirely from scratch — no pretrained weights, no transfer from GPT or LLaMA. The goal was to demonstrate that a capable, identity-aware conversational model can be built on commodity hardware (a single 16GB P100 GPU) within a reasonable time budget, using modern architectural choices that punch above the parameter count.

The project covers the full pipeline end to end: custom BPE tokenizer training, large-scale pretraining on raw text, supervised fine-tuning on conversational data, and inference.

---

## Model Architecture

| Component | Choice | Reason |
|---|---|---|
| **Positional encoding** | RoPE (Rotary Position Embeddings) | Better length generalisation than learned absolute positions |
| **Normalisation** | RMSNorm (pre-norm) | Faster than LayerNorm, no mean subtraction needed |
| **Feed-forward** | SwiGLU | Outperforms GELU in practice (used in LLaMA) |
| **Attention** | Causal multi-head self-attention | Manual path on P100, Flash Attention on sm_70+ |
| **Output** | Weight-tied `lm_head` | Shares weights with embedding table, saves ~6M parameters |

### Parameter Count

| Component | Parameters |
|---|---|
| Embedding table (shared with `lm_head`) | 6.14M |
| Transformer blocks (11 layers) | 19.47M |
| **Total unique** | **19.47M** |
| **Total (with shared)** | **25.62M** |

### Configuration

```yaml
Layers      : 11
Heads       : 6
Embedding   : 384
Head dim    : 64   (384 / 6)
Context     : 512 tokens
Vocab size  : 16,000 (custom BPE)
Dropout     : 0.0
```

---

## Hardware & Environment

| Item | Details |
|---|---|
| **GPU** | Tesla P100-PCIE 16GB (Kaggle) |
| **Compute capability** | `sm_60` |
| **CPU RAM** | ~13 GB (Kaggle) |
| **PyTorch** | 2.3.1+cu121 |
| **Python** | 3.12 |
| **Precision** | `float16` (AMP) |

> [!IMPORTANT]
> **Note on P100 (`sm_60`):** PyTorch ≥ 2.4 dropped `sm_60` support. This project requires PyTorch 2.3.1 for GPU training. Flash Attention is disabled on `sm_60` — a manual scaled dot-product attention path is used instead. `torch.compile` and fused AdamW are also disabled (both require `sm_70+`).

---

## Project Structure

```text
nanomind/
├── data/
│   ├── tokenizer.json          # Trained BPE tokenizer (16K vocab)
│   ├── train.bin               # Pretrain data (memmap, uint16)
│   └── val.bin                 # Validation data (memmap, uint16)
│   └── finetune.jsonl          # Conversational SFT data
├── checkpoints/
│   ├── best_pretrain.pt        # Best pretrain checkpoint
│   ├── best_finetune.pt        # Best finetune checkpoint
│   └── pretrain_step*.pt       # Periodic snapshots
├── 1_prepare_data.py           # Tokenizer + preprocessing
├── 2_pretrain.py               # Pretraining loop
├── 3_finetune.py               # SFT loop
├── model.py                    # Architecture + utils
└── 4_inference.py              # Interactive chat
```

---

## Pipeline

### Step 1 — Data Preparation (`1_prepare_data.py`)

- Trains a custom **BPE tokenizer** (16K vocab) on the raw corpus.
- Special tokens: `<bos>`, `<eos>`, `<pad>`, `<sys>`, `<user>`, `<assistant>`.
- Raw text is tokenized into memmapped `uint16` binary files.

### Step 2 — Pretraining (`2_pretrain.py`)

Standard next-token prediction on raw text.
- **Memory strategy:** Data loaded via `np.memmap` to keep CPU RAM < 2GB.
- **Efficiency:** Gradient accumulation (8x) and `float16` AMP.

| Parameter | Value |
|---|---|
| Batch size | 16 (Effective: 128) |
| Block size | 512 |
| Max LR | 3e-4 |
| Optimizer | AdamW |

### Step 3 — Supervised Fine-tuning (`3_finetune.py`)

Fine-tunes the model on conversational data using a **masked loss** — only assistant tokens contribute to the loss.

**Conversation template:**
```text
<bos> <sys> {system} <eos> <user> {question} <eos> <assistant> {response} <eos>
```

---

## Results

### Pretraining (10K steps)

| Metric | Value |
|---|---|
| Validation loss | 3.6598 |
| Perplexity | 12.64 |

### Fine-tuning (4K steps)

| Metric | Value |
|---|---|
| Training loss | 1.7822 |
| Training time | ~97 minutes |

**Identity Check:**
> **Q:** Who are you?
> **A:** I am NanoMind, a 25-million-parameter language model developed by Abhishek Kapoor.

---

## Datasets

### Pretraining
Assembled from WikiText-103, FineWeb-Edu, OpenWebText, and TinyTextbooks.
- **Total trained:** ~655M tokens.

### Fine-tuning
- **Identity QA:** 1,500 hand-crafted pairs.
- **OpenHermes-2.5:** 200,000 GPT-4 quality samples.
- **Alpaca-Cleaned:** 52,000 instruction-following samples.

---

## How to Run

### 1. Requirements
```bash
pip install torch==2.3.1 tokenizers numpy tqdm
```

### 2. Prepare Data
```bash
python 1_prepare_data.py
```

### 3. Training
```bash
python 2_pretrain.py
python 3_finetune.py
```

### 4. Interactive Chat
```bash
python 4_inference.py
```

---

## Key Engineering Decisions

- **Memmap:** Avoids loading the full dataset into RAM.
- **Lazy Dataset:** Encodes JSONL samples on demand during SFT.
- **Weight Tying:** Shares weights between embedding and output heads.
- **RMSNorm & SwiGLU:** Modern architectural choices for better performance.

---

## Limitations

- Trained for only 1/3 of planned steps due to session constraints.
- Small vocabulary (16K) and context window (512).
- No RLHF or safety tuning.
- Single GPU constraints.

---

<p align="center">
  Built with ❤️ by <b>Abhishek Kapoor</b>
</p>
