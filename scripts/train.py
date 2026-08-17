#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "trl>=1.10,<2",
#   "transformers>=5.0,<6",
#   "peft>=0.17",
#   "datasets>=4.0",
#   "accelerate>=1.10",
#   "huggingface-hub>=0.34",
#   "pyyaml>=6",
#   "bitsandbytes>=0.45 ; platform_system == 'Linux'",
#   "liger-kernel>=0.5 ; platform_system == 'Linux'",
# ]
# ///
"""Unified CPT / SFT / DPO trainer driven by one YAML recipe.

Self-contained by design: `hf jobs uv run` uploads a single file, so this
script carries everything it needs and takes its config as a path, URL, or
raw JSON string.

    # local, single GPU
    uv run scripts/train.py --config recipes/sft_zhtw.yaml --smoke-test
    uv run scripts/train.py --config recipes/sft_zhtw.yaml

    # local, multi GPU
    uv run --with accelerate accelerate launch --num_processes 4 \
        scripts/train.py --config recipes/sft_zhtw.yaml

    # Hugging Face Jobs
    hf jobs uv run --flavor a10g-large --secrets HF_TOKEN --timeout 6h \
        scripts/train.py --config https://.../sft_zhtw.yaml

Recipe schema: see recipes/ and skills/hsun-trainer/references/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

DEFAULTS: dict[str, Any] = {
    "stage": "sft",
    "model": {
        "name_or_path": None,
        "dtype": "bfloat16",
        "attn_implementation": "sdpa",
        "trust_remote_code": False,
        "chat_template_path": None,
        # Gemma 3/4 checkpoints above 1B are multimodal. For text-only training
        # the vision encoder is dead weight; freezing it saves memory and stops
        # gradients flowing into an encoder that never sees an image.
        "freeze_vision_tower": True,
    },
    "dataset": {"sources": [], "eval": None, "shuffle_seed": 42},
    "tuning": {
        "method": "lora",
        "lora": {
            "r": 32,
            "alpha": 64,
            "dropout": 0.05,
            "target_modules": "all-linear",
            "modules_to_save": None,
        },
    },
    "train": {
        "output_dir": "outputs/run",
        "max_length": 4096,
        "packing": None,  # None -> resolved per stage (on for CPT, off for SFT)
        "completion_only_loss": None,
        "assistant_only_loss": False,
        "num_train_epochs": 1,
        "max_steps": -1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 2.0e-5,
        "lr_scheduler_type": "cosine",
        "warmup_steps": 20,  # TRL 1.10 removed warmup_ratio; only warmup_steps exists
        "weight_decay": 0.0,
        "bf16": True,
        "gradient_checkpointing": True,
        "logging_steps": 10,
        "save_strategy": "steps",
        "save_steps": 500,
        "save_total_limit": 2,
        "seed": 42,
        "use_liger_kernel": False,
        "report_to": [],
    },
    "hub": {"push_to_hub": False, "hub_model_id": None, "private": True},
}


def sources_hint(cfg: dict) -> str:
    srcs = cfg.get("dataset", {}).get("sources") or []
    return srcs[0].get("path", "<dataset>") if srcs else "<dataset>"


def deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(spec: str) -> dict:
    """Accept a local path, an http(s) URL, or a raw JSON string."""
    import yaml

    text = spec.strip()
    if text.startswith("{"):
        raw = json.loads(text)
    elif text.startswith(("http://", "https://")):
        import urllib.request

        with urllib.request.urlopen(text, timeout=60) as resp:  # noqa: S310
            raw = yaml.safe_load(resp.read().decode("utf-8"))
    else:
        with open(text, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise SystemExit(f"[x] Config must be a mapping, got {type(raw).__name__}")
    return deep_merge(DEFAULTS, raw)


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def build_dataset(cfg: dict, smoke: bool):
    """Load, remap, and weight-mix every source into one dataset."""
    from datasets import concatenate_datasets, interleave_datasets, load_dataset

    sources = cfg["dataset"]["sources"]
    if not sources:
        raise SystemExit("[x] dataset.sources is empty")

    parts, weights = [], []
    for src in sources:
        path = src.get("path")
        if not path:
            raise SystemExit("[x] every dataset source needs `path`")
        split = src.get("split", "train")
        cap = 64 if smoke else src.get("max_samples")
        if cap:
            split = f"{split}[:{int(cap)}]"

        print(f"    loading {path} (config={src.get('config')}, split={split})")
        ds = load_dataset(path, src.get("config"), split=split)

        # Some Hub datasets store `messages` as a serialized JSON string. TRL
        # accepts the column name, fails to parse the rows, and silently trains
        # on zero examples - so decode before anything else touches the data.
        json_cols = src.get("json_columns") or []
        if json_cols:
            missing = [c for c in json_cols if c not in ds.column_names]
            if missing:
                raise SystemExit(f"[x] {path}: json_columns {missing} not in {ds.column_names}")

            def _decode(row, _cols=tuple(json_cols)):
                for col in _cols:
                    val = row[col]
                    if isinstance(val, str):
                        row[col] = json.loads(val)
                return row

            ds = ds.map(_decode, desc=f"decoding {', '.join(json_cols)}")
            print(f"      decoded JSON columns: {json_cols}")

        # ShareGPT ({"from": "human", "value": ...}) -> OpenAI ({"role", "content"}).
        if src.get("sharegpt_to_messages"):
            col = src.get("sharegpt_column", "conversations")
            if col not in ds.column_names:
                raise SystemExit(f"[x] {path}: no `{col}` column; have {ds.column_names}")
            role_map = {
                "human": "user", "user": "user",
                "gpt": "assistant", "assistant": "assistant", "chatgpt": "assistant",
                "system": "system",
            }

            def _convert(row, _col=col):
                turns = row[_col]
                if isinstance(turns, str):
                    turns = json.loads(turns)
                row["messages"] = [
                    {
                        "role": role_map.get(str(t.get("from", "")).lower(), "user"),
                        "content": t.get("value", ""),
                    }
                    for t in (turns or [])
                ]
                return row

            ds = ds.map(_convert, desc=f"sharegpt {col} -> messages")
            print(f"      converted ShareGPT `{col}` -> `messages`")

        rename = src.get("rename") or {}
        for old, new in rename.items():
            if old in ds.column_names and old != new:
                ds = ds.rename_column(old, new)

        keep = src.get("keep")
        if keep:
            missing = [c for c in keep if c not in ds.column_names]
            if missing:
                raise SystemExit(
                    f"[x] {path}: `keep` lists missing columns {missing}. "
                    f"Available: {ds.column_names}"
                )
            ds = ds.remove_columns([c for c in ds.column_names if c not in keep])

        parts.append(ds)
        weights.append(float(src.get("weight", 1.0)))
        print(f"      -> {len(ds):,} rows, columns={ds.column_names}")

    if len(parts) == 1:
        train = parts[0]
    else:
        cols = set(parts[0].column_names)
        if all(set(p.column_names) == cols for p in parts):
            if all(abs(w - weights[0]) < 1e-9 for w in weights):
                train = concatenate_datasets(parts)
            else:
                total = sum(weights)
                train = interleave_datasets(
                    parts,
                    probabilities=[w / total for w in weights],
                    seed=cfg["dataset"]["shuffle_seed"],
                    stopping_strategy="all_exhausted",
                )
        else:
            raise SystemExit(
                "[x] Sources have different columns; add `rename`/`keep` to align them.\n"
                + "\n".join(f"    {s['path']}: {p.column_names}" for s, p in zip(sources, parts))
            )

    train = train.shuffle(seed=cfg["dataset"]["shuffle_seed"])

    eval_ds = None
    ev = cfg["dataset"].get("eval")
    if ev and ev.get("path"):
        ev_split = ev.get("split", "test")
        if smoke:
            ev_split = f"{ev_split}[:32]"
        eval_ds = load_dataset(ev["path"], ev.get("config"), split=ev_split)

    print(f"    total training rows: {len(train):,}")
    return train, eval_ds


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------


def build_model_and_tokenizer(cfg: dict):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    mc = cfg["model"]
    name = mc["name_or_path"]
    if not name:
        raise SystemExit("[x] model.name_or_path is required")

    dtype = getattr(torch, mc["dtype"]) if isinstance(mc["dtype"], str) else mc["dtype"]
    method = cfg["tuning"]["method"].lower()

    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": mc["trust_remote_code"],
        "attn_implementation": mc["attn_implementation"],
    }

    if method == "qlora":
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        print("    quantization: 4-bit NF4 (double quant)")

    print(f"    loading model {name} [{method}, {mc['attn_implementation']}]")
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.config.use_cache = False

    if mc.get("freeze_vision_tower", True):
        frozen = 0
        for pname, param in model.named_parameters():
            if "vision_tower" in pname or "multi_modal_projector" in pname:
                param.requires_grad = False
                frozen += param.numel()
        if frozen:
            print(f"    froze vision tower + projector ({frozen/1e6:.0f} M params)")

    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=mc["trust_remote_code"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        print("    tokenizer had no pad_token -> using eos_token")

    if mc.get("chat_template_path"):
        with open(mc["chat_template_path"], encoding="utf-8") as fh:
            tok.chat_template = fh.read()
        print(f"    chat template overridden from {mc['chat_template_path']}")

    return model, tok


def build_peft_config(cfg: dict):
    method = cfg["tuning"]["method"].lower()
    if method == "full":
        return None
    if method not in ("lora", "qlora"):
        raise SystemExit(f"[x] tuning.method must be full|lora|qlora, got {method!r}")

    from peft import LoraConfig

    lc = cfg["tuning"]["lora"]
    return LoraConfig(
        r=lc["r"],
        lora_alpha=lc["alpha"],
        lora_dropout=lc["dropout"],
        target_modules=lc["target_modules"],
        modules_to_save=lc["modules_to_save"],
        bias="none",
        task_type="CAUSAL_LM",
    )


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

# SFTConfig fields that also exist on DPOConfig; anything else is stage-specific.
_SFT_ONLY = {"packing", "completion_only_loss", "assistant_only_loss", "max_length"}


def make_args(cfg: dict, stage: str, smoke: bool):
    from trl import DPOConfig, SFTConfig

    t = dict(cfg["train"])
    hub = cfg["hub"]

    common = {
        "output_dir": t["output_dir"],
        "num_train_epochs": t["num_train_epochs"],
        "max_steps": 5 if smoke else t["max_steps"],
        "per_device_train_batch_size": t["per_device_train_batch_size"],
        "gradient_accumulation_steps": 1 if smoke else t["gradient_accumulation_steps"],
        "learning_rate": t["learning_rate"],
        "lr_scheduler_type": t["lr_scheduler_type"],
        "warmup_steps": 0 if smoke else t["warmup_steps"],
        "weight_decay": t["weight_decay"],
        "bf16": t["bf16"],
        "gradient_checkpointing": t["gradient_checkpointing"],
        "logging_steps": 1 if smoke else t["logging_steps"],
        "save_strategy": "no" if smoke else t["save_strategy"],
        "save_steps": t["save_steps"],
        "save_total_limit": t["save_total_limit"],
        "seed": t["seed"],
        "report_to": [] if smoke else (t["report_to"] or []),
        "push_to_hub": False if smoke else hub["push_to_hub"],
        "use_liger_kernel": t["use_liger_kernel"],
    }
    if hub.get("hub_model_id") and not smoke:
        common["hub_model_id"] = hub["hub_model_id"]
        common["hub_private_repo"] = hub.get("private", True)

    if stage == "dpo":
        return DPOConfig(max_length=t["max_length"], beta=t.get("beta", 0.1), **common)

    return SFTConfig(
        max_length=t["max_length"],
        packing=t["packing"],
        completion_only_loss=t["completion_only_loss"],
        assistant_only_loss=t["assistant_only_loss"],
        dataset_text_field=cfg["dataset"].get("text_field", "text"),
        **common,
    )


def pin_single_gpu_if_not_distributed() -> None:
    """Keep an unlaunched `python train.py` off torch's DataParallel path.

    HF Trainer wraps the model in nn.DataParallel whenever it sees more than
    one visible GPU and no distributed launcher. DataParallel replicates the
    module per device, which breaks models holding device-bound buffers:
    Gemma 3's `embed_scale` stays on cuda:0 while replica 1 feeds it a cuda:1
    index, raising "Expected all tensors to be on the same device". Multi-GPU
    is supported, but only via `accelerate launch` / `torchrun` (see
    references/hardware.md). Must run before torch initialises CUDA.
    """
    if any(os.environ.get(v) for v in ("WORLD_SIZE", "RANK", "LOCAL_RANK")):
        return  # accelerate/torchrun owns device placement
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return  # caller already chose
    import subprocess

    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                             text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return
    count = len([ln for ln in out.stdout.splitlines() if ln.startswith("GPU ")])
    if count > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print(f"    [!] {count} GPUs visible but no distributed launcher; using GPU 0 only.")
        print("        For multi-GPU: uv run --with accelerate accelerate launch \\")
        print(f"            --num_processes {count} scripts/train.py --config <recipe>")


def main() -> int:
    ap = argparse.ArgumentParser(description="CPT / SFT / DPO trainer")
    ap.add_argument("--config", required=True, help="YAML path, https URL, or raw JSON")
    ap.add_argument("--stage", default=None, help="override config `stage`")
    ap.add_argument("--smoke-test", action="store_true",
                    help="5 steps on 64 rows, no save, no push - validates the pipeline")
    ap.add_argument("--dry-run", action="store_true", help="print resolved config and exit")
    args = ap.parse_args()

    cfg = load_config(args.config)
    stage = (args.stage or cfg["stage"]).lower()
    if stage not in ("cpt", "sft", "dpo"):
        raise SystemExit(f"[x] stage must be cpt|sft|dpo, got {stage!r} (GRPO -> train_grpo.py)")

    # CPT is language modeling over raw text: no chat template, and packing on
    # by default because unpacked short documents waste most of each sequence.
    if cfg["train"]["packing"] is None:
        cfg["train"]["packing"] = stage == "cpt"
    if stage == "cpt":
        cfg["train"]["completion_only_loss"] = False

    print("=" * 70)
    print(f"  hsun-trainer :: {stage.upper()}{'  [SMOKE TEST]' if args.smoke_test else ''}")
    print("=" * 70)

    if args.dry_run:
        print(json.dumps(cfg, indent=2, ensure_ascii=False, default=str))
        return 0

    pin_single_gpu_if_not_distributed()

    print("[1/4] dataset")
    train_ds, eval_ds = build_dataset(cfg, args.smoke_test)

    print("[2/4] model")
    model, tok = build_model_and_tokenizer(cfg)
    peft_cfg = build_peft_config(cfg)

    if stage == "sft" and cfg["train"]["assistant_only_loss"]:
        import re as _re

        tpl = getattr(tok, "chat_template", None) or ""
        if not _re.search(r"{%-?\s*generation\s*-?%}", tpl):
            raise SystemExit(
                "[x] train.assistant_only_loss=true, but this tokenizer's chat template\n"
                "    has no {% generation %} block, so assistant tokens cannot be located.\n"
                "    Gemma's stock template is affected. Either:\n"
                "      a) set train.assistant_only_loss: false, or\n"
                "      b) point model.chat_template_path at a patched template, e.g.\n"
                "         assets/gemma_chat_template_assistant_mask.jinja"
            )

    print("[3/4] trainer")
    targs = make_args(cfg, stage, args.smoke_test)

    if stage == "dpo":
        from trl import DPOTrainer

        trainer = DPOTrainer(
            model=model, args=targs, train_dataset=train_ds,
            eval_dataset=eval_ds, processing_class=tok, peft_config=peft_cfg,
        )
    else:
        from trl import SFTTrainer

        trainer = SFTTrainer(
            model=model, args=targs, train_dataset=train_ds,
            eval_dataset=eval_ds, processing_class=tok, peft_config=peft_cfg,
        )

    # TRL drops rows it cannot render; an empty result would otherwise "train"
    # for zero steps and report success.
    processed = getattr(trainer, "train_dataset", None)
    if processed is not None and hasattr(processed, "__len__") and len(processed) == 0:
        raise SystemExit(
            f"[x] After preprocessing, 0 of {len(train_ds):,} rows survived.\n"
            "    Usual causes:\n"
            "      - `messages` stored as a JSON string -> add `json_columns: [messages]`\n"
            "      - roles that the chat template rejects (Gemma requires strictly\n"
            "        alternating user/assistant turns)\n"
            "      - every row longer than train.max_length\n"
            f"    Inspect the source: uv run scripts/inspect_dataset.py {sources_hint(cfg)}"
        )
    print(f"    training rows after preprocessing: {len(processed):,}")

    if hasattr(trainer.model, "print_trainable_parameters"):
        trainer.model.print_trainable_parameters()

    print("[4/4] train")
    result = trainer.train()
    print(f"    loss={result.training_loss:.4f}  steps={result.global_step}")

    if args.smoke_test:
        print("\n[ok] Smoke test passed. Re-run without --smoke-test for the real run.")
        return 0

    trainer.save_model(cfg["train"]["output_dir"])
    tok.save_pretrained(cfg["train"]["output_dir"])
    print(f"    saved to {cfg['train']['output_dir']}")

    if cfg["hub"]["push_to_hub"]:
        if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
            print("[!] push_to_hub set but no HF_TOKEN in env; push will likely fail")
        trainer.push_to_hub()
        print(f"    pushed to {cfg['hub']['hub_model_id']}")

    print("=" * 70)
    print("Next: evaluate with the hsun-eval skill before starting the next stage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
