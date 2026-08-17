#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets>=4.0.0", "huggingface-hub>=0.34", "transformers>=4.56"]
# ///
"""Inspect a Hugging Face dataset before training.

Reports configs, splits, columns, a sample row, the inferred TRL dataset type,
and length percentiles — so `max_length` and the training stage are chosen from
evidence rather than habit.

    uv run inspect_dataset.py lianghsun/tw-legal-qa-chat
    uv run inspect_dataset.py twinkle-ai/tw-math-reasoning-2k --tokenizer Qwen/Qwen3-4B
    uv run inspect_dataset.py lianghsun/Formosa-bench --config Geography --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter

MAX_VAL = 400  # chars shown per field in the sample row


def _short(v, limit: int = MAX_VAL) -> str:
    s = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
    s = s.replace("\n", "\\n")
    return s if len(s) <= limit else s[:limit] + f"... <+{len(s) - limit} chars>"


# (required columns, TRL type, trainers, note)
SIGNATURES: list[tuple[set[str], str, str, str]] = [
    ({"prompt", "chosen", "rejected"}, "preference (explicit prompt)", "DPOTrainer, RewardTrainer", ""),
    ({"chosen", "rejected"}, "preference (implicit prompt)", "RewardTrainer, DPOTrainer", ""),
    ({"prompt", "completion", "label"}, "unpaired preference", "KTOTrainer", ""),
    ({"prompt", "completion"}, "prompt-completion", "SFTTrainer", "loss on completion only by default"),
    ({"messages"}, "conversational language modeling", "SFTTrainer", "chat template is applied"),
    ({"conversations"}, "ShareGPT conversational", "SFTTrainer",
     "NEEDS CONVERSION: from/value -> role/content, rename to `messages`"),
    ({"instruction", "output"}, "Alpaca-style", "SFTTrainer",
     "NEEDS CONVERSION: map to `prompt`/`completion` or `messages`"),
    ({"question", "answer"}, "QA pair", "SFTTrainer / GRPOTrainer",
     "map to prompt/completion for SFT, or prompt+ground_truth for GRPO"),
    ({"text"}, "language modeling", "SFTTrainer (packing=True)", "raw text -> use for CPT"),
    ({"prompt"}, "prompt-only", "GRPOTrainer, RLOOTrainer", "reward functions supply the signal"),
]

MCQ = {"question", "A", "B", "C", "D", "answer"}


def detect(columns: set[str]) -> list[tuple[str, str, str]]:
    """Return every matching dataset-type signature, most specific first."""
    hits = []
    if MCQ <= columns:
        hits.append((
            "multiple-choice benchmark",
            "hsun-eval (Twinkle Eval) or GRPO with a letter-accuracy reward",
            "columns question/A/B/C/D/answer",
        ))
    for required, kind, trainers, note in SIGNATURES:
        if required <= columns:
            hits.append((kind, trainers, note))
    return hits or [("unknown", "-", "no known signature matched; map columns manually")]


def load_head(name: str, config: str | None, split: str, n: int):
    """Pull the first n rows, streaming so huge corpora stay cheap."""
    from datasets import load_dataset

    try:
        it = load_dataset(name, config, split=split, streaming=True)
        rows = list(it.take(n))
        features = getattr(it, "features", None)
        return rows, features
    except Exception:
        ds = load_dataset(name, config, split=f"{split}[:{n}]")
        return list(ds), ds.features


def length_stats(rows: list[dict], columns: list[str], tokenizer_id: str | None) -> None:
    """Print char (and optionally token) percentiles for the dominant text column."""
    text_cols = [c for c in ("text", "completion", "output", "solution", "answer", "content")
                 if c in columns]
    struct_cols = [c for c in ("messages", "conversations") if c in columns]

    def render(row) -> str:
        if struct_cols:
            msgs = row.get(struct_cols[0]) or []
            parts = []
            for m in msgs:
                if isinstance(m, dict):
                    parts.append(str(m.get("content") or m.get("value") or ""))
            return "\n".join(parts)
        return "\n".join(str(row.get(c, "")) for c in (text_cols or columns[:1]))

    texts = [render(r) for r in rows]
    texts = [t for t in texts if t]
    if not texts:
        return

    basis = struct_cols[0] if struct_cols else (text_cols[0] if text_cols else columns[0])
    chars = sorted(len(t) for t in texts)

    def pct(seq, p):
        return seq[min(len(seq) - 1, int(len(seq) * p))]

    print(f"\nLENGTH  (n={len(texts)} sampled rows, basis column: {basis})")
    print(f"  chars   p50={pct(chars,0.5):>8,}  p90={pct(chars,0.9):>8,}  max={chars[-1]:>8,}")

    if tokenizer_id:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=False)
            toks = sorted(len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts)
            p50, p90, p99, mx = pct(toks, 0.5), pct(toks, 0.9), pct(toks, 0.99), toks[-1]
            print(f"  tokens  p50={p50:>8,}  p90={p90:>8,}  p99={p99:>8,}  max={mx:>8,}")
            rec = 1 << max(9, (max(p99, 1) - 1).bit_length())
            print(f"  -> suggested max_length: {rec} (covers p99; raising it costs O(n^2) attention)")
            if mx > rec:
                frac = sum(1 for t in toks if t > rec) / len(toks)
                print(f"  -> {frac:.1%} of rows would be truncated at {rec}")
        except Exception as exc:  # noqa: BLE001
            print(f"  tokens  <unavailable: {type(exc).__name__}: {exc}>")
    else:
        print("  (pass --tokenizer <model_id> for token-level percentiles)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect an HF dataset for training")
    ap.add_argument("dataset", help="e.g. lianghsun/tw-legal-qa-chat")
    ap.add_argument("--config", default=None, help="dataset config / subset name")
    ap.add_argument("--split", default=None, help="default: first available split")
    ap.add_argument("-n", "--num-rows", type=int, default=200, help="rows to sample (default 200)")
    ap.add_argument("--tokenizer", default=None, help="model id for token-length stats")
    ap.add_argument("--show", type=int, default=1, help="sample rows to print (default 1)")
    args = ap.parse_args()

    from datasets import get_dataset_config_names, get_dataset_split_names

    print("=" * 72)
    print(f"  {args.dataset}")
    print("=" * 72)

    try:
        configs = get_dataset_config_names(args.dataset)
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__
        print(f"[x] Cannot read dataset: {name}: {exc}")
        if "Gated" in name or "401" in str(exc) or "403" in str(exc):
            print("\n[!] This dataset is gated or private.")
            print("    1. `hf auth login`")
            print(f"    2. Accept terms at https://huggingface.co/datasets/{args.dataset}")
        return 1

    config = args.config or (configs[0] if configs else None)
    print(f"configs : {', '.join(configs) if configs else '(none)'}")
    if len(configs) > 1 and not args.config:
        print(f"          -> inspecting '{config}'; re-run with --config to see others")

    try:
        splits = get_dataset_split_names(args.dataset, config)
    except Exception as exc:  # noqa: BLE001
        print(f"[x] Cannot list splits: {exc}")
        return 1
    split = args.split or splits[0]
    print(f"splits  : {', '.join(splits)}  -> using '{split}'")

    try:
        rows, features = load_head(args.dataset, config, split, args.num_rows)
    except Exception as exc:  # noqa: BLE001
        print(f"[x] Failed to load rows: {type(exc).__name__}: {exc}")
        return 1

    if not rows:
        print("[x] Split is empty.")
        return 1

    columns = list(rows[0].keys())
    print(f"\nCOLUMNS ({len(columns)})")
    for c in columns:
        dtype = ""
        if features is not None:
            try:
                dtype = f"  {features[c]}"
            except Exception:  # noqa: BLE001
                dtype = ""
        print(f"  - {c}{_short(dtype, 90)}")

    print("\nDETECTED TYPE")
    for kind, trainers, note in detect(set(columns)):
        print(f"  * {kind}")
        print(f"      trainer : {trainers}")
        if note:
            print(f"      note    : {note}")

    # A `messages` column stored as a JSON *string* is the single most damaging
    # schema trap: TRL recognises the column name, fails to parse the rows, and
    # silently yields a 0-row training set instead of raising.
    stringly = [
        c for c in ("messages", "conversations", "chosen", "rejected", "tools")
        if c in columns and isinstance(rows[0].get(c), str)
    ]
    if stringly:
        print("\n[!] JSON-STRING COLUMNS DETECTED: " + ", ".join(stringly))
        print("    These hold serialized JSON, not native lists. TRL will drop every")
        print("    row and train on nothing. Decode them in the recipe:")
        print("      sources:")
        print(f"        - path: {args.dataset}")
        print(f"          json_columns: [{', '.join(stringly)}]")

    if "conversations" in columns and rows[0].get("conversations"):
        first = rows[0]["conversations"]
        if isinstance(first, list) and first and isinstance(first[0], dict):
            print(f"      keys    : {sorted(first[0].keys())}")
    if "messages" in columns and rows[0].get("messages"):
        roles = Counter()
        for r in rows[:50]:
            for m in r.get("messages") or []:
                if isinstance(m, dict) and "role" in m:
                    roles[m["role"]] += 1
        if roles:
            print(f"      roles   : {dict(roles)}")
        turns = [len(r.get("messages") or []) for r in rows]
        print(f"      turns   : min={min(turns)} median={sorted(turns)[len(turns)//2]} max={max(turns)}")

    for i, row in enumerate(rows[: max(0, args.show)]):
        print(f"\nSAMPLE ROW [{i}]")
        for k, v in row.items():
            print(f"  {k}: {_short(v)}")

    length_stats(rows, columns, args.tokenizer)

    print("\nRECIPE SNIPPET")
    print("  dataset:")
    print("    sources:")
    print(f"      - path: {args.dataset}")
    if config and len(configs) > 1:
        print(f"        config: {config}")
    print(f"        split: {split}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
