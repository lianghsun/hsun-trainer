#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["datasets>=4.0.0", "huggingface-hub>=0.34", "pyyaml>=6"]
# ///
"""Bridge Hugging Face benchmarks into ai-twinkle/Eval (Twinkle Eval).

Twinkle Eval evaluates an OpenAI-compatible endpoint against *local* dataset
directories. This exports zh-TW multiple-choice benchmarks from the Hub into
the layout it expects and writes a ready-to-run config.

    uv run scripts/make_eval_config.py \
        --model gemma3-12b-zhtw-sft \
        --base-url http://localhost:8000/v1 \
        --bench lianghsun/tw-legal-benchmark-v1 lianghsun/Formosa-bench \
        --out eval/

Then:
    twinkle-eval --validate --config eval/config.yaml
    twinkle-eval --config eval/config.yaml --export json csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

LETTERS = [chr(ord("A") + i) for i in range(26)]

# Benchmarks twinkle-eval downloads itself: `twinkle-eval --download-dataset X`
BUILTIN = {
    "mmlu": "box", "mmlu_pro": "box", "tmmluplus": "box", "mmlu_redux": "box",
    "supergpqa": "box", "gpqa": "box", "formosa_bench": "box",
    "gsm8k": "math", "aime2025": "math", "bbh": "regex_match",
    "ifeval": "ifeval", "ifbench": "ifbench", "bfcl": "bfcl_fc",
    "needlebench": "niah", "wikieval": "ragas",
}

ZH_SYSTEM = (
    "你是一位精通臺灣正體中文的助理。請仔細閱讀題目與所有選項，"
    "逐步思考後，將最終答案的選項字母放入 \\boxed{} 之中。"
)
EN_SYSTEM = (
    "You are a careful assistant. Read the question and all options, "
    "reason step by step, then put the final option letter inside \\boxed{}."
)


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def export_mcq(dataset_id: str, out_root: Path, split: str | None) -> Path | None:
    """Write an HF multiple-choice dataset as Twinkle Eval jsonl.

    Twinkle Eval accepts two MCQ shapes; this emits the letter-keyed one
    ({"question","A".."N","answer":"C"}), which the zh-TW benchmarks already
    use natively, so no answer remapping is needed.
    """
    from datasets import get_dataset_config_names, get_dataset_split_names, load_dataset

    target = out_root / "datasets" / safe(dataset_id.split("/")[-1])
    target.mkdir(parents=True, exist_ok=True)

    try:
        configs = get_dataset_config_names(dataset_id) or [None]
    except Exception as exc:  # noqa: BLE001
        print(f"  [x] {dataset_id}: cannot list configs ({type(exc).__name__}); skipped")
        return None

    written = 0
    for cfg in configs:
        try:
            splits = get_dataset_split_names(dataset_id, cfg)
        except Exception as exc:  # noqa: BLE001
            print(f"  [x] {dataset_id}/{cfg}: cannot list splits ({type(exc).__name__})")
            continue
        use = split or ("test" if "test" in splits else splits[0])
        ds = load_dataset(dataset_id, cfg, split=use)

        cols = set(ds.column_names)
        if "question" not in cols or "answer" not in cols:
            print(f"  [x] {dataset_id}/{cfg}: needs `question` + `answer`, has {sorted(cols)}")
            continue
        options = [L for L in LETTERS if L in cols]
        if len(options) < 2:
            print(f"  [x] {dataset_id}/{cfg}: no A/B/C... option columns; skipped")
            continue

        path = target / f"{safe(cfg) if cfg else 'test'}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in ds:
                rec = {"question": row["question"]}
                for L in options:
                    if row.get(L) is not None:
                        rec[L] = row[L]
                rec["answer"] = str(row["answer"]).strip().upper()[:1]
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  [ok] {dataset_id}/{cfg or '-'} [{use}] -> {path} ({len(ds)} items, {len(options)} options)")
        written += 1

    return target if written else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a Twinkle Eval config")
    ap.add_argument("--model", required=True, help="model name as the serving endpoint reports it")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="EMPTY", help="vLLM/SGLang ignore this; keep a placeholder")
    ap.add_argument("--bench", nargs="*", default=[], help="HF dataset ids to export locally")
    ap.add_argument("--builtin", nargs="*", default=[],
                    help=f"twinkle-eval registry names: {', '.join(sorted(BUILTIN))}")
    ap.add_argument("--split", default=None, help="force a split (default: test if present)")
    ap.add_argument("--method", default="box",
                    help="pattern | box | logit | math | regex_match (default box)")
    ap.add_argument("--repeat-runs", type=int, default=3,
                    help="repeat count; >1 reports mean +/- stddev (default 3)")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--out", default="eval", help="output directory (default eval/)")
    args = ap.parse_args()

    if not args.bench and not args.builtin:
        ap.error("give at least one --bench or --builtin")

    import yaml

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print("Exporting Hub benchmarks")
    dataset_paths, prompt_map = [], {}
    for ds_id in args.bench:
        target = export_mcq(ds_id, out_root, args.split)
        if target:
            rel = str(target.relative_to(out_root)) + "/"
            dataset_paths.append(rel)
            prompt_map[rel] = "zh"

    for name in args.builtin:
        if name not in BUILTIN:
            print(f"  [!] {name} is not a known twinkle-eval benchmark; skipped")
            continue
        rel = f"datasets/{name}/"
        dataset_paths.append(rel)
        prompt_map[rel] = "zh" if name in ("tmmluplus", "formosa_bench") else "en"
        print(f"  [ ] {name}: run `twinkle-eval --download-dataset {name}` inside {out_root}/")

    if not dataset_paths:
        print("\n[x] Nothing to evaluate - no dataset was exported or selected.")
        return 1

    config = {
        "llm_api": {
            "base_url": args.base_url,
            "api_key": args.api_key,
            "api_rate_limit": -1,
            "max_retries": 3,
            "timeout": 600,
        },
        "model": {
            "name": args.model,
            "temperature": args.temperature,
            "top_p": 0.95,
            "max_tokens": args.max_tokens,
        },
        "evaluation": {
            "dataset_paths": dataset_paths,
            "evaluation_method": args.method,
            "system_prompt": {"zh": ZH_SYSTEM, "en": EN_SYSTEM},
            "datasets_prompt_map": prompt_map,
            "shuffle_options": True,   # cancels position bias
            "repeat_runs": args.repeat_runs,
        },
        "logging": {"level": "INFO"},
    }

    cfg_path = out_root / "config.yaml"
    with cfg_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, allow_unicode=True, sort_keys=False, width=100)

    print(f"\nWrote {cfg_path}")
    print("=" * 70)
    print("Next steps")
    print("=" * 70)
    print("1. Serve the model with an OpenAI-compatible API:")
    print(f"     vllm serve <model-path> --served-model-name {args.model} --port 8000")
    print("   (a merged model serves faster than base+adapter; see hsun-eval SKILL.md)")
    for name in args.builtin:
        if name in BUILTIN:
            print(f"2. cd {out_root} && twinkle-eval --download-dataset {name}")
    print(f"3. twinkle-eval --validate --config {cfg_path}")
    print(f"4. twinkle-eval --dry-run  --config {cfg_path}")
    print(f"5. twinkle-eval --config {cfg_path} --export json csv")
    print("\nInstall if needed: pip install twinkle-eval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
