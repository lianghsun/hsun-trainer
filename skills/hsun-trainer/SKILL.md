---
name: hsun-trainer
description: Plan and run an end-to-end LLM training pipeline — continued pretraining (CPT), SFT, GRPO/RLVR, and benchmarking — with a curated catalog of Traditional Chinese (zh-TW / Taiwan) datasets. Use when the user wants to train, fine-tune, post-train, align, or evaluate a language model; mentions TRL, pretrain, CPT, SFT, GRPO, RLVR, reward functions, LoRA/QLoRA, Twinkle Eval, TMMLU+, Formosa-bench, tw-legal-benchmark; or asks "which dataset should I use" for a Taiwan/zh-TW model. Routes to hsun-pretrain, hsun-sft, hsun-grpo, and hsun-eval.
license: MIT
---

# hsun-trainer

Entry point for training language models from Claude Code. Handles the whole
lifecycle — **CPT → SFT → GRPO → Eval** — with first-class support for
Traditional Chinese (zh-TW) models and datasets.

## When to use which skill

| User wants | Skill | Stage |
|---|---|---|
| Inject new domain knowledge / a new language into a base model | `hsun-pretrain` | CPT |
| Teach instruction-following, chat, tool use, reasoning style | `hsun-sft` | SFT |
| Optimize against a verifiable reward (accuracy, format, zh-TW purity) | `hsun-grpo` | GRPO / RLVR |
| Measure the model on benchmarks | `hsun-eval` | Eval |

Read only the SKILL.md you actually need. Each is self-contained.

## Core design

**One recipe, two targets.** Every training script is a self-contained
[PEP 723](https://peps.python.org/pep-0723/) `uv` script. The same file runs:

```bash
# local GPU box
uv run scripts/train_sft.py --config recipe.yaml

# Hugging Face Jobs (no local GPU needed)
hf jobs uv run --flavor a10g-large --secrets HF_TOKEN --timeout 6h \
  scripts/train_sft.py --config recipe.yaml
```

`--config` accepts a local path, an `https://` URL, or a raw JSON string, so the
job carries its own config with no shared filesystem. Nothing needs to be
`pip install`ed first — `uv` resolves dependencies from the script header.

## Always start here

**1. Preflight.** Before proposing any run, check what hardware and auth exist:

```bash
uv run scripts/preflight.py
```

It reports GPUs and VRAM, CUDA/MPS/CPU, installed versions, HF login state, and
disk. Use its `[!]` lines to pick between local training and HF Jobs.

> On macOS there is no CUDA. Apple Silicon (MPS) can run tiny smoke tests only.
> For anything real, route to a Linux CUDA box or to HF Jobs.

**2. Inspect the dataset before spending GPU time.** Never guess column names —
several datasets in the catalog are gated and many use non-obvious schemas:

```bash
uv run scripts/inspect_dataset.py lianghsun/tw-legal-qa-chat
```

It prints configs, splits, columns, a sample row, detected TRL dataset type
(`language_modeling` / `prompt_completion` / `conversational` / `preference` /
`prompt_only`), and token-length percentiles so you can set `max_length` from
evidence instead of habit.

**3. Size the run.** Decide full vs LoRA vs QLoRA and pick a batch size:

```bash
uv run scripts/plan_memory.py --model Qwen/Qwen3-4B --seq-len 4096
```

Then hand off to the stage skill.

## Datasets

The user maintains large zh-TW corpora on the Hub under
[`lianghsun`](https://huggingface.co/lianghsun) and
[`twinkle-ai`](https://huggingface.co/twinkle-ai). A curated, stage-tagged index
lives in `references/dataset-catalog.md` — read it when choosing data.

Quick orientation:

- **CPT corpora** — `lianghsun/tw-news-551M`, `tw-legal-qa-3M`,
  `wikipedia-zh-742M`, `Taiwan_c4`, `twinkle-ai/fineweb-zhtw-filtered`
- **SFT** — `twinkle-ai/tw-reasoning-instruct-50k`, `tw-function-call-reasoning-10k`,
  `tw-leetcode`, `lianghsun/tw-legal-qa-chat`
- **GRPO (verifiable)** — `twinkle-ai/tw-math-reasoning-2k`,
  `lianghsun/tw-legal-benchmark-v1`
- **Eval** — `lianghsun/Formosa-bench`, `tw-legal-benchmark-v1`,
  `tw-emergency-medicine-bench`, plus `ikala/tmmluplus`

Always confirm the schema with `inspect_dataset.py` before writing a recipe;
catalog entries record what was published, not what is loadable by your token.

## Pipeline recipes

For multi-stage plans (e.g. "make a Taiwanese legal model from Qwen3-8B"), read
`references/pipeline-recipes.md`. It has ordered, costed pipelines with the
checkpoints to evaluate between stages.

## Rules that prevent expensive mistakes

1. **Evaluate the base model first.** Run `hsun-eval` before training so there is
   a baseline. A post-training number with no baseline proves nothing.
2. **Smoke test before the real run.** Every training script takes
   `--smoke-test`, which forces a handful of steps on a tiny slice. Always run it
   first; it catches template, column, and OOM errors in ~2 minutes instead of
   after 6 hours.
3. **Push to the Hub.** HF Jobs machines are ephemeral — if `push_to_hub` is not
   set, the weights are destroyed when the job ends. Set `hub.push_to_hub: true`
   and pass `--secrets HF_TOKEN`.
4. **Match the chat template to the stage.** CPT trains on raw text with no
   template. SFT applies the tokenizer's chat template. Mixing them silently
   degrades the model.
5. **Don't stack quantization and merging carelessly.** A QLoRA adapter merged
   back into a 4-bit base loses quality; merge into the fp16/bf16 base instead.
6. **Verify TRL API names against the installed version.** TRL moves fast and
   renames arguments between minor releases. The scripts here pin known-good
   versions in their PEP 723 headers; if you edit them, check the field exists:
   ```bash
   uv run --with trl python -c "import dataclasses,trl; print([f.name for f in dataclasses.fields(trl.SFTConfig)])"
   ```

## Known API drift (TRL 1.10)

These changed from widely-copied older examples and will raise `TypeError` if
used from memory:

| Old | Current | Where |
|---|---|---|
| `SFTConfig(max_seq_length=...)` | `SFTConfig(max_length=...)` | SFT, CPT |
| `warmup_ratio=...` | **removed** — use `warmup_steps` | all stages |
| `GRPOConfig(max_prompt_length=...)` | **removed** | GRPO |
| `scale_rewards=True/False` | `scale_rewards="group" \| "batch" \| "none"` | GRPO |
| `loss_type="grpo"` default | default is now `"dapo"` | GRPO |
| `beta=0.04` default | default is now `0.0` (KL off) | GRPO |

Two schema traps cost more time than any of the above, both covered in
`references/dataset-catalog.md`: a `messages` column stored as a **JSON
string** (TRL silently trains on 0 rows), and ShareGPT `conversations` with
`from`/`value` keys.

## Related work

Hugging Face ships [`huggingface/skills`](https://github.com/huggingface/skills),
whose `huggingface-llm-trainer` covers SFT/DPO/GRPO **on HF Jobs only** and does
**not** cover continued pretraining, local multi-GPU, or zh-TW evaluation.
Install it alongside this plugin for HF Hub/Spaces/Gradio breadth; use
`hsun-trainer` for CPT, on-prem multi-GPU, and Traditional Chinese work.

## Reference files

- `references/dataset-catalog.md` — stage-tagged zh-TW dataset index
- `references/hardware.md` — GPU sizing, HF Jobs flavors, multi-GPU configs
- `references/pipeline-recipes.md` — end-to-end multi-stage plans
- `references/troubleshooting.md` — OOM, loss spikes, template bugs, NCCL hangs
