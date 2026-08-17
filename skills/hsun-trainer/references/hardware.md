# Hardware, sharding, and where to run

Get numbers rather than guessing:

```bash
uv run scripts/preflight.py                                   # what you have
uv run scripts/plan_memory.py --model google/gemma-3-12b-it \
    --seq-len 8192 --vram 80 --gpus 2                         # what you need
```

## Choosing a tuning method

`plan_memory.py` models weights + gradients + optimizer + activations + the
logits tensor. Rough per-GPU peaks at seq 4096, batch 1, gradient checkpointing:

| Model | full FT | LoRA r=32 | QLoRA r=32 |
|---|---|---|---|
| ~1 B | ~18 GB | ~6 GB | ~4 GB |
| ~4 B | ~65 GB | ~14 GB | ~8 GB |
| ~12 B | ~190 GB | ~33 GB | ~16 GB |
| ~27 B | ~430 GB | ~66 GB | ~26 GB |

Full fine-tuning costs roughly **16 bytes per parameter** (2 weights + 2
gradients + 4 fp32 master + 8 Adam moments), which is why a 12B full FT does
not fit on a single 80 GB card without sharding or offload.

Decision order: full FT if it fits (best quality, required for real CPT) →
LoRA → QLoRA → shard across GPUs → HF Jobs.

## The logits trap

Gemma's vocabulary is ~262K entries. The loss upcasts logits to fp32:

```
batch × seq_len × vocab × 4 bytes × 2 (softmax copy)
1 × 8192 × 262144 × 4 × 2  ≈  17 GB
```

That single tensor can dwarf the model. It scales with **batch × seq_len**,
not parameters, so:

- keep `per_device_train_batch_size: 1` and raise `gradient_accumulation_steps`
- set `use_liger_kernel: true` (Linux/CUDA) for fused chunked cross-entropy
- shorten `max_length` to the p99 from `inspect_dataset.py`

This is the most common OOM that looks inexplicable.

## Multi-GPU on your own box

The launcher must run **inside the script's own PEP 723 environment**:

```bash
uv sync --script scripts/train.py
"$(uv python find --script scripts/train.py)" -m accelerate.commands.launch \
    --num_processes 4 scripts/train.py --config recipes/sft_gemma_zhtw.yaml
```

Do **not** use `uv run --with accelerate accelerate launch ...`. That builds an
environment for the `accelerate` command, never reads `train.py`'s PEP 723
header, and leaves the spawned ranks importing whatever `datasets` /
`transformers` happen to be installed system-wide. The failure is obscure - a
`TypeError` inside `datasets`' feature decoder - and if the system packages are
merely old rather than broken, the run silently trains against unpinned
versions. `accelerate>=1.10` is already in the header, so no `--with` is needed.

| Strategy | When | Notes |
|---|---|---|
| DDP | model fits on one GPU | fastest; pure throughput scaling |
| ZeRO-2 | gradients + optimizer too big | keeps params replicated |
| ZeRO-3 / FSDP | model itself too big | shards params; slower, big memory win |
| ZeRO-3 + CPU offload | still too big | very slow; last resort before renting |

ZeRO-3 shards weights, gradients, and optimizer state, so per-GPU memory falls
roughly linearly with GPU count — that is what `--gpus` models in
`plan_memory.py`. Activations and logits do **not** shard.

Configure with an accelerate config file (`accelerate config`) and launch as
above; TRL and Trainer pick it up automatically.

## Hugging Face Jobs

The same script runs remotely because `hf jobs uv run` accepts a local file:

```bash
hf jobs uv run --flavor a10g-large --secrets HF_TOKEN --timeout 6h \
    scripts/train.py --config https://raw.githubusercontent.com/<you>/<repo>/main/recipes/sft_gemma_zhtw.yaml
```

`--config` also takes a raw JSON string, so the job carries its own settings
with no shared filesystem.

Flavors accepted by the CLI:

```
cpu-basic  cpu-upgrade  cpu-xl  zero-a10g
t4-small  t4-medium              (T4, 16 GB)
l4x1  l4x4                       (L4, 24 GB each)
a10g-small  a10g-large  a10g-largex2  a10g-largex4   (A10G, 24 GB each)
l40sx1  l40sx4  l40sx8           (L40S, 48 GB each)
a100-large  a100x4  a100x8       (A100, 80 GB each)
```

Run `hf jobs hardware` for the authoritative list and current pricing.

Rules for jobs:

1. **Set `hub.push_to_hub: true`.** The machine is destroyed at the end; weights
   not pushed are lost.
2. **Pass `--secrets HF_TOKEN`.** Without it, gated models (all Gemma repos)
   fail to download and pushes fail.
3. **Set `--timeout` generously.** The default is far shorter than a real run,
   and a timeout kills the job mid-training.
4. **Smoke test locally first**, or on `t4-small`, before booking a big flavor.
5. `-d` detaches; follow with `hf jobs logs <id>` and `hf jobs ps`.

## macOS

Apple Silicon has no CUDA: no bitsandbytes (so no QLoRA), no flash-attention,
no DeepSpeed. MPS runs `recipes/smoke_gemma.yaml` with
`google/gemma-3-270m-it` in float32, which is enough to validate a dataset,
a chat template, and a recipe before renting a GPU. Do real training elsewhere.

## Attention implementation

| Value | When |
|---|---|
| `sdpa` | default; works everywhere |
| `flash_attention_2` | Linux/CUDA with a prebuilt wheel; do not add it to the script deps, it compiles for a long time |
| `eager` | debugging, MPS/CPU, or when a model requires it |
