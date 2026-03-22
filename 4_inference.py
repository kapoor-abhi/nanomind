import sys
import argparse
import torch
from pathlib import Path
from tokenizers import Tokenizer

sys.path.insert(0, ".")
from model import NanoMind, NanoMindConfig, load_checkpoint

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--temperature", type=float, default=0.75)
parser.add_argument("--top-k", type=int, default=50)
parser.add_argument("--top-p", type=float, default=0.92)
parser.add_argument("--max-tokens", type=int, default=256)
args = parser.parse_args()

DATA_DIR = Path("data")
CKPT_DIR = Path("checkpoints")

tok_path = DATA_DIR / "tokenizer.json"
assert tok_path.exists()

tok     = Tokenizer.from_file(str(tok_path))
BOS_ID  = tok.token_to_id("<bos>")
EOS_ID  = tok.token_to_id("<eos>")
SYS_ID  = tok.token_to_id("<sys>")
USR_ID  = tok.token_to_id("<user>")
ASST_ID = tok.token_to_id("<assistant>")

assert all(x is not None for x in [BOS_ID, EOS_ID, SYS_ID, USR_ID, ASST_ID])

device = "cuda" if torch.cuda.is_available() else "cpu"

if args.checkpoint:
    ckpt_path = Path(args.checkpoint)
elif (CKPT_DIR / "best_finetune.pt").exists():
    ckpt_path = CKPT_DIR / "best_finetune.pt"
elif (CKPT_DIR / "best_pretrain.pt").exists():
    ckpt_path = CKPT_DIR / "best_pretrain.pt"
else:
    sys.exit(1)

model, config, ckpt = load_checkpoint(str(ckpt_path), device)
model.eval()

SYSTEM_PROMPT = "You are NanoMind, a helpful AI assistant developed by Abhishek Kapoor."

def build_prompt(
    user_msg: str,
    history:  list,
    system:   str = SYSTEM_PROMPT,
) -> list:
    ids = [BOS_ID]
    ids += [SYS_ID] + tok.encode(system).ids + [EOS_ID]

    for u, a in history:
        ids += [USR_ID]  + tok.encode(u).ids + [EOS_ID]
        ids += [ASST_ID] + tok.encode(a).ids + [EOS_ID]

    ids += [USR_ID] + tok.encode(user_msg).ids + [EOS_ID]
    ids += [ASST_ID]

    return ids

def trim_to_context(
    ids:      list,
    max_ctx:  int,
    history:  list,
    system:   str,
) -> list:
    if len(ids) <= max_ctx:
        return ids

    for drop in range(len(history)):
        trimmed = build_prompt(
            user_msg=tok.decode(
                [i for i in ids[-100:] if i not in (USR_ID, ASST_ID, EOS_ID)]
            ),
            history=history[drop + 1:],
            system=system,
        )
        if len(trimmed) <= max_ctx:
            return trimmed

    return ids[: max_ctx]

@torch.no_grad()
def chat(
    user_msg:    str,
    history:     list,
    temperature: float = 0.75,
    top_k:       int   = 50,
    top_p:       float = 0.92,
    max_new:     int   = 256,
) -> str:
    ids = build_prompt(user_msg, history)

    max_ctx = config.block_size - max_new
    if len(ids) > max_ctx:
        ids = trim_to_context(ids, max_ctx, history, SYSTEM_PROMPT)

    idx = torch.tensor([ids], dtype=torch.long, device=device)

    out = model.generate(
        idx,
        max_new_tokens=max_new,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_id=EOS_ID,
    )

    new_token_ids = out[0, len(ids):].tolist()
    response      = tok.decode(new_token_ids).strip()

    for special in ["<eos>", "<pad>", "<user>", "<sys>", "<assistant>", "<bos>"]:
        response = response.replace(special, "").strip()

    return response if response else "..."

def main():
    print("-" * 55)
    print("NanoMind Chat")
    print("-" * 55)

    history     = []
    temperature = args.temperature
    top_k       = args.top_k
    top_p       = args.top_p
    max_new     = args.max_tokens

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd == "quit":
            break

        if cmd == "reset":
            history = []
            continue

        response = chat(
            user_input, history,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new=max_new,
        )
        print(f"\nNanoMind: {response}\n")
        history.append((user_input, response))

if __name__ == "__main__":
    main()
