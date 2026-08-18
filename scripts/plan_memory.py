#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface-hub>=0.34"]
# ///
"""Estimate training VRAM and pick full / LoRA / QLoRA.

Reads the real parameter count from the model's safetensors index on the Hub,
then models weights + gradients + optimizer + activations + the logits tensor
(a frequently missed OOM source on large-vocabulary models).

    uv run plan_memory.py --model Qwen/Qwen3-4B --seq-len 4096
    uv run plan_memory.py --model Qwen/Qwen3-8B --seq-len 8192 --batch 2 --vram 80
    uv run plan_memory.py --model Qwen/Qwen3-8B --gpus 4

Estimates are deliberately conservative (~10-20% headroom vs observed peaks).
Treat them as a starting point, then confirm with `--smoke-test`.
"""

from __future__ import annotations

import argparse
import json

GB = 1024**3
DTYPE_BYTES = {
    "float32": 4, "float": 4, "fp32": 4,
    "bfloat16": 2, "bf16": 2, "float16": 2, "fp16": 2, "half": 2,
    "int8": 1, "float8_e4m3fn": 1, "uint8": 1,
}


def vocab_from_weights(model_id: str, revision: str | None) -> int | None:
    """Vocabulary size read off the embedding tensor's shape, or None."""
    from huggingface_hub import get_safetensors_metadata

    try:
        meta = get_safetensors_metadata(model_id, revision=revision)
    except Exception:  # noqa: BLE001 - offline, gated, or .bin-only repo
        return None
    for fmeta in meta.files_metadata.values():
        for name, tensor in fmeta.tensors.items():
            if name.endswith("embed_tokens.weight") and len(tensor.shape) == 2:
                return int(tensor.shape[0])
    return None


def fetch_model_facts(model_id: str, revision: str | None, vocab_override: int | None = None) -> dict:
    """Parameter count + architecture dims, read from the Hub."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    facts: dict = {"model_id": model_id}

    try:
        cfg_path = hf_hub_download(model_id, "config.json", revision=revision)
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"[x] Cannot read config.json for {model_id}: {exc}") from exc

    text_cfg = cfg.get("text_config") or cfg
    facts["hidden"] = text_cfg.get("hidden_size") or text_cfg.get("n_embd") or 4096
    facts["layers"] = (
        text_cfg.get("num_hidden_layers") or text_cfg.get("n_layer") or 32
    )
    # Multimodal Gemma 3 configs (4b/12b/27b) omit vocab_size entirely and let
    # transformers supply the default, so a plain .get() falls through to the
    # fallback and understates the logits term by 8x on the exact model family
    # this repo targets. Read the embedding row count from the safetensors
    # header instead - HTTP range requests, no weight download.
    facts["vocab"] = (
        vocab_override
        or text_cfg.get("vocab_size")
        or vocab_from_weights(model_id, revision)
        or 32000
    )
    # Needed for an exact LoRA parameter count: the MLP projections are far
    # wider than hidden, and GQA makes k/v narrower than q.
    facts["intermediate"] = text_cfg.get("intermediate_size") or 4 * facts["hidden"]
    facts["heads"] = text_cfg.get("num_attention_heads") or 32
    facts["kv_heads"] = text_cfg.get("num_key_value_heads") or facts["heads"]
    facts["head_dim"] = text_cfg.get("head_dim") or (facts["hidden"] // facts["heads"])
    facts["arch"] = (cfg.get("architectures") or ["?"])[0]
    dtype = str(cfg.get("dtype") or cfg.get("torch_dtype") or "bfloat16").replace("torch.", "")
    facts["dtype"] = dtype
    bpp = DTYPE_BYTES.get(dtype, 2)

    total_bytes = None
    for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        try:
            idx = hf_hub_download(model_id, index_name, revision=revision)
            with open(idx, encoding="utf-8") as fh:
                total_bytes = json.load(fh)["metadata"]["total_size"]
            break
        except Exception:  # noqa: BLE001, S112
            continue

    if total_bytes is None:
        try:
            info = api.model_info(model_id, revision=revision, files_metadata=True)
            weights = [
                s.size for s in info.siblings
                if s.rfilename.endswith((".safetensors", ".bin")) and s.size
            ]
            if weights:
                total_bytes = sum(weights)
        except Exception:  # noqa: BLE001
            pass

    if total_bytes:
        facts["params"] = total_bytes / bpp
        facts["params_source"] = "safetensors index"
    else:
        h, l, v = facts["hidden"], facts["layers"], facts["vocab"]
        inter = text_cfg.get("intermediate_size", 4 * h)
        facts["params"] = 2 * v * h + l * (4 * h * h + 3 * h * inter)
        facts["params_source"] = "estimated from config dims"
    return facts


def lora_params(facts: dict, rank: int) -> float:
    """Trainable parameters for LoRA over all linear projections.

    Each adapted Linear(in, out) adds rank * (in + out). Treating every
    projection as hidden x hidden undercounts by ~2x, because gate/up/down run
    at intermediate_size while GQA shrinks k/v.
    """
    h, l = facts["hidden"], facts["layers"]
    inter, kv = facts["intermediate"], facts["kv_heads"]
    q_out = facts["heads"] * facts["head_dim"]
    kv_out = kv * facts["head_dim"]
    per_layer = rank * (
        (h + q_out)          # q_proj
        + 2 * (h + kv_out)   # k_proj, v_proj
        + (q_out + h)        # o_proj
        + 2 * (h + inter)    # gate_proj, up_proj
        + (inter + h)        # down_proj
    )
    return l * per_layer


def activations(facts: dict, batch: int, seq: int, ckpt: bool) -> float:
    h, l = facts["hidden"], facts["layers"]
    per_layer_input = batch * seq * h * 2
    if ckpt:
        # Boundary inputs kept + one block recomputed at a time.
        return per_layer_input * l + per_layer_input * 12
    return per_layer_input * l * 12


def logits_mem(facts: dict, batch: int, seq: int) -> float:
    """Logits are upcast to fp32 for the loss; softmax needs a second copy."""
    return batch * seq * facts["vocab"] * 4 * 2


def plan(facts: dict, batch: int, seq: int, rank: int, ckpt: bool, gpus: int) -> list[dict]:
    p = facts["params"]
    act = activations(facts, batch, seq, ckpt)
    logits = logits_mem(facts, batch, seq)
    a = lora_params(facts, rank)
    rows = []

    # Full FT, bf16 mixed precision + AdamW: 2P weights, 2P grads, 4P fp32 master, 8P moments.
    full_static = 2 * p + 2 * p + 4 * p + 8 * p
    rows.append({
        "method": "full",
        "trainable": p,
        "static": full_static,
        "act": act,
        "logits": logits,
        "shardable": full_static * 0.75,  # ZeRO-3 shards grads + optimizer + params
    })

    # LoRA: frozen bf16 base + optimizer state on adapters only.
    lora_static = 2 * p + 2 * a + 4 * a + 8 * a
    rows.append({
        "method": f"lora(r={rank})",
        "trainable": a,
        "static": lora_static,
        "act": act,
        "logits": logits,
        "shardable": (lora_static - 2 * p) * 0.75,
    })

    # QLoRA: NF4 base (~0.5 bytes/param + quant constants) + adapters in bf16.
    qlora_static = 0.55 * p + 2 * a + 4 * a + 8 * a
    rows.append({
        "method": f"qlora(r={rank})",
        "trainable": a,
        "static": qlora_static,
        "act": act,
        "logits": logits,
        "shardable": 0.0,
    })

    for r in rows:
        total = r["static"] + r["act"] + r["logits"]
        r["total_gb"] = total / GB
        r["per_gpu_gb"] = (
            (total - r["shardable"] * (1 - 1 / gpus)) / GB if gpus > 1 else total / GB
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Estimate training VRAM")
    ap.add_argument("--model", required=True, help="HF model id, e.g. Qwen/Qwen3-4B")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=1, help="per-device batch size")
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--gpus", type=int, default=1, help="GPUs for ZeRO-3/FSDP sharding")
    ap.add_argument("--vram", type=float, default=None, help="VRAM per GPU in GB")
    ap.add_argument("--vocab", type=int, default=None,
                    help="override vocabulary size when the repo publishes neither")
    ap.add_argument("--no-grad-ckpt", action="store_true", help="disable gradient checkpointing")
    args = ap.parse_args()

    facts = fetch_model_facts(args.model, args.revision, args.vocab)
    ckpt = not args.no_grad_ckpt
    rows = plan(facts, args.batch, args.seq_len, args.lora_rank, ckpt, args.gpus)

    print("=" * 74)
    print(f"  {facts['model_id']}  ({facts['arch']})")
    print("=" * 74)
    print(
        f"params {facts['params']/1e9:.2f} B ({facts['params_source']}) | "
        f"hidden {facts['hidden']} | layers {facts['layers']} | vocab {facts['vocab']:,}"
    )
    print(
        f"seq_len {args.seq_len} | batch {args.batch} | grad_ckpt {ckpt} | "
        f"gpus {args.gpus} | base dtype {facts['dtype']}"
    )
    print("-" * 74)
    print(f"{'method':<14}{'trainable':>12}{'weights+opt':>13}{'acts':>9}{'logits':>9}{'PEAK/GPU':>13}")
    print("-" * 74)
    for r in rows:
        tr = f"{r['trainable']/1e6:.0f} M" if r["trainable"] < 1e9 else f"{r['trainable']/1e9:.2f} B"
        print(
            f"{r['method']:<14}{tr:>12}{r['static']/GB:>11.1f}G"
            f"{r['act']/GB:>8.1f}G{r['logits']/GB:>8.1f}G{r['per_gpu_gb']:>11.1f}G"
        )
    print("-" * 74)

    if r["logits"] / GB > 2:
        print(
            f"[!] Logits alone need {rows[0]['logits']/GB:.1f} GB "
            f"(vocab {facts['vocab']:,} x seq {args.seq_len} x batch {args.batch}, fp32).\n"
            f"    Cut per-device batch to 1 and raise gradient_accumulation_steps, or\n"
            f"    enable `use_liger_kernel=True` for a fused chunked cross-entropy."
        )

    if args.vram:
        print(f"\nAgainst {args.vram} GB/GPU:")
        chosen = None
        for r in rows:
            fits = r["per_gpu_gb"] <= args.vram * 0.9
            print(f"  {'[ok]' if fits else '[x] '} {r['method']:<14} {r['per_gpu_gb']:>6.1f} GB")
            if fits and chosen is None:
                chosen = r
        print()
        if chosen:
            print(f"  -> recommended: {chosen['method']}")
        else:
            print(
                "  -> nothing fits. Options: shorter seq_len, more GPUs with ZeRO-3,\n"
                "     CPU/NVMe offload, or run on HF Jobs (see references/hardware.md)."
            )
    else:
        print("\n(pass --vram <GB per GPU> for a fit verdict)")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
