import sys
import argparse
import torch
from pathlib import Path
from tokenizers import Tokenizer

sys.path.insert(0, ".")
from model import NanoMind, NanoMindConfig, load_checkpoint


# CLI
parser = argparse.ArgumentParser(description="NanoMind Chat")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="Path to checkpoint (default: best_finetune.pt or best_pretrain.pt)")
parser.add_argument("--temperature", type=float, default=0.75,
                    help="Sampling temperature (default: 0.75)")
parser.add_argument("--top-k", type=int, default=50,
                    help="Top-k sampling (default: 50)")
parser.add_argument("--top-p", type=float, default=0.92,
                    help="Nucleus sampling p (default: 0.92)")
parser.add_argument("--max-tokens", type=int, default=256,
                    help="Max new tokens per response (default: 256)")
args = parser.parse_args()


# Tokenizer
DATA_DIR = Path("data")
CKPT_DIR = Path("checkpoints")

tok_path = DATA_DIR / "tokenizer.json"
assert tok_path.exists(), (
    "data/tokenizer.json not found.\n"
    "Run 1_prepare_data.py to create it."
)

tok     = Tokenizer.from_file(str(tok_path))
BOS_ID  = tok.token_to_id("<bos>")
EOS_ID  = tok.token_to_id("<eos>")
SYS_ID  = tok.token_to_id("<sys>")        # system marker
USR_ID  = tok.token_to_id("<user>")
ASST_ID = tok.token_to_id("<assistant>")

for name, tid in [("<bos>", BOS_ID), ("<eos>", EOS_ID), ("<sys>", SYS_ID),
                  ("<user>", USR_ID), ("<assistant>", ASST_ID)]:
    assert tid is not None, (
        f"Special token '{name}' missing from tokenizer – "
        "was 1_prepare_data.py run correctly?"
    )


# Model
device = "cuda" if torch.cuda.is_available() else "cpu"

# Determine checkpoint to load
if args.checkpoint:
    ckpt_path = Path(args.checkpoint)
elif (CKPT_DIR / "best_finetune.pt").exists():
    ckpt_path = CKPT_DIR / "best_finetune.pt"
elif (CKPT_DIR / "best_pretrain.pt").exists():
    ckpt_path = CKPT_DIR / "best_pretrain.pt"
    print("WARNING: No fine-tuned checkpoint found. "
          "Using pre-trained weights (chat quality will be lower).")
else:
    print("ERROR: No checkpoint found.")
    print(f"  Looked in: {CKPT_DIR}/best_finetune.pt and best_pretrain.pt")
    print("  Run 2_pretrain.py and 3_finetune.py first.")
    sys.exit(1)

model, config, ckpt = load_checkpoint(str(ckpt_path), device)
model.eval()

ckpt_type = "fine-tuned" if "finetune" in str(ckpt_path) else "pre-trained only"
print(f"\n✔  NanoMind loaded ({ckpt_type})")
print(f"   Checkpoint : {ckpt_path}")
print(f"   Step       : {ckpt['step']:,}")
print(f"   Val loss   : {ckpt.get('val_loss', 'N/A')}")
print(f"   Params     : {sum(p.numel() for p in model.parameters())/1e6:.1f}M")


SYSTEM_PROMPT = "You are NanoMind, a helpful AI assistant developed by Abhishek Kapoor."


# Prompt Builder
def build_prompt(
    user_msg: str,
    history:  list,
    system:   str = SYSTEM_PROMPT,
) -> list:
    """
    Build the full token-id sequence for the current turn.
    history : list of (user_str, assistant_str) for past turns

    Format:
      <bos> <sys> SYSTEM <eos>
            <user> TURN1_USER <eos>  <asst> TURN1_ASST <eos>
            ...
            <user> CURRENT_USER <eos>  <asst>          ← model continues here
    """
    ids = [BOS_ID]
    # System prompt
    ids += [SYS_ID] + tok.encode(system).ids + [EOS_ID]

    # Past conversation history
    for u, a in history:
        ids += [USR_ID]  + tok.encode(u).ids + [EOS_ID]
        ids += [ASST_ID] + tok.encode(a).ids + [EOS_ID]

    # Current user turn (model will generate from ASST_ID onward)
    ids += [USR_ID] + tok.encode(user_msg).ids + [EOS_ID]
    ids += [ASST_ID]  # open-ended: model generates response

    return ids


def trim_to_context(
    ids:      list,
    max_ctx:  int,
    history:  list,
    system:   str,
) -> list:
    """
    If ids exceeds max_ctx, drop the oldest history turns one at a time
    until it fits, always keeping BOS + system prompt.
    """
    if len(ids) <= max_ctx:
        return ids

    # Try dropping history turns from oldest to newest
    for drop in range(len(history)):
        trimmed = build_prompt(
            # reconstruct from remaining history
            user_msg=tok.decode(
                [i for i in ids[-100:] if i not in (USR_ID, ASST_ID, EOS_ID)]
            ),
            history=history[drop + 1:],
            system=system,
        )
        if len(trimmed) <= max_ctx:
            return trimmed

    # Last resort: keep only system + current user turn (no history)
    return ids[: max_ctx]


# Chat
@torch.no_grad()
def chat(
    user_msg:    str,
    history:     list,
    temperature: float = 0.75,
    top_k:       int   = 50,
    top_p:       float = 0.92,
    max_new:     int   = 256,
) -> str:
    """
    Generate one assistant response.
    history : list of (user_str, asst_str) for multi-turn context.
    Returns the response string.
    """
    # Build prompt token ids
    ids = build_prompt(user_msg, history)

    # Trim if necessary (leave room for generation)
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

    # Slice off the new tokens (after the prompt)
    new_token_ids = out[0, len(ids):].tolist()
    response      = tok.decode(new_token_ids).strip()

    # Clean up any leaked special-token text
    for special in ["<eos>", "<pad>", "<user>", "<sys>", "<assistant>", "<bos>"]:
        response = response.replace(special, "").strip()

    return response if response else "..."


# Interactive Loop
def main():
    print("═" * 55)
    print("  🧠  NanoMind  –  AI Assistant")
    print("═" * 55)
    print("  Commands:")
    print("    reset      → clear conversation history")
    print("    temp X     → set temperature  (e.g.  temp 0.9)")
    print("    top_k X    → set top-k        (e.g.  top_k 40)")
    print("    top_p X    → set top-p        (e.g.  top_p 0.85)")
    print("    maxlen X   → set max tokens   (e.g.  maxlen 300)")
    print("    history    → print conversation history")
    print("    quit       → exit")
    print("═" * 55 + "\n")

    history     = []
    temperature = args.temperature
    top_k       = args.top_k
    top_p       = args.top_p
    max_new     = args.max_tokens

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        # Commands
        if cmd == "quit":
            print("Goodbye!")
            break

        if cmd == "reset":
            history = []
            print("(Conversation history cleared)\n")
            continue

        if cmd == "history":
            if not history:
                print("(No history yet)\n")
            else:
                print("\n─── Conversation History ───")
                for i, (u, a) in enumerate(history):
                    print(f"  Turn {i+1}")
                    print(f"    You      : {u[:100]}")
                    print(f"    NanoMind : {a[:100]}")
                print()
            continue

        if cmd.startswith("temp "):
            try:
                temperature = float(cmd.split()[1])
                print(f"(Temperature → {temperature})\n")
            except (ValueError, IndexError):
                print("Usage: temp 0.8\n")
            continue

        if cmd.startswith("top_k "):
            try:
                top_k = int(cmd.split()[1])
                print(f"(top_k → {top_k})\n")
            except (ValueError, IndexError):
                print("Usage: top_k 40\n")
            continue

        if cmd.startswith("top_p "):
            try:
                top_p = float(cmd.split()[1])
                print(f"(top_p → {top_p})\n")
            except (ValueError, IndexError):
                print("Usage: top_p 0.9\n")
            continue

        if cmd.startswith("maxlen "):
            try:
                max_new = int(cmd.split()[1])
                print(f"(max new tokens → {max_new})\n")
            except (ValueError, IndexError):
                print("Usage: maxlen 300\n")
            continue

        # Generate
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
