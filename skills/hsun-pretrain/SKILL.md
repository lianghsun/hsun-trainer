---
name: hsun-pretrain
description: Continued pretraining (CPT / domain-adaptive pretraining) of a base LLM on raw text — to add a new domain, a new language, or Traditional Chinese (zh-TW) knowledge. Use when the user says pretrain, continued pretraining, CPT, DAPT, "inject domain knowledge", "teach the model Taiwanese law/medicine", "train on raw corpus", or wants to extend a base model before SFT. Covers corpus mixing, replay ratios, learning rates, packing, and catastrophic forgetting.
license: MIT
---

# Continued pretraining (CPT)

Trains a **base** model on raw text with a plain next-token objective. Use it
to add knowledge the model does not have. It does not teach instruction
following — that is SFT's job, and it comes after.

## Decide whether CPT is the right tool

| Symptom | Right stage |
|---|---|
| Model doesn't know the facts / vocabulary / statutes | **CPT** |
| Model knows the facts but answers in the wrong style or format | SFT |
| Model produces correct-looking answers that are wrong in a checkable way | GRPO |
| Model writes Simplified Chinese | SFT, then GRPO with `zhtw_purity` |

CPT is the most expensive stage and the easiest to get wrong. If under
~100M tokens of domain text exist, SFT on synthesized QA usually beats CPT.

## Run it

```bash
# 1. verify columns and pick max_length from evidence
uv run scripts/inspect_dataset.py lianghsun/tw-legal-qa-3M --tokenizer google/gemma-3-4b-pt

# 2. size the run
uv run scripts/plan_memory.py --model google/gemma-3-4b-pt --seq-len 4096 --vram 80

# 3. validate the pipeline in ~2 minutes
uv run scripts/train.py --config recipes/cpt_gemma_zhtw.yaml --smoke-test

# 4. real run
uv run scripts/train.py --config recipes/cpt_gemma_zhtw.yaml
```

Start from `recipes/cpt_gemma_zhtw.yaml`.

## Rules that decide whether CPT works

**Always start from a base checkpoint.** `google/gemma-3-4b-pt`, not
`-it`. Raw-text training on an instruction-tuned model destroys its chat
ability, and you cannot get it back cheaply. Gemma base variants: Gemma 3 uses
the `-pt` suffix; Gemma 4 base is the unsuffixed id (`google/gemma-4-12B`).

**No chat template.** CPT is language modeling over `text`. The trainer sets
`packing: true` and `completion_only_loss: false` for `stage: cpt`
automatically. Do not add system prompts or role markers into the corpus.

**Known limitation: packing cross-contaminates under sdpa.** TRL prints this
on every CPT run:

> Packing gathers multiple samples into a single sequence, and only the
> following implementations are known to reliably support this:
> flash_attention_2, ... Using other implementations may lead to
> cross-contamination between samples.

Documents packed into one sequence attend across their boundaries, so the
model learns adjacency that does not exist. There is no free fix here:
`flash-attn` is excluded on purpose (it compiles for minutes), and
`kernels-community/flash-attn2` — which TRL lists as supported and which needs
no local compile — publishes no build variant for B200 / CUDA 13.1 (tested
2026-08, `FileNotFoundError: Cannot find a build variant for this system`).
Until a prebuilt kernel exists for your GPU, choose deliberately: keep
`packing: true` and accept the contamination, or set `packing: false` and pay
the throughput. Do not leave it undecided.

**Learning rate 5-10x below SFT.** 1e-5 or lower when updating all parameters. CPT at
SFT learning rates is the most common way to destroy a base model. Warm up
over ~100 steps and use a cosine schedule.

**One epoch.** Repeated passes over a domain corpus cause memorization rather
than generalization. If the corpus is too small for one meaningful epoch, that
is a signal to use SFT instead.

**Mix in replay data.** Training purely on domain text causes catastrophic
forgetting — the model gets better at your statutes and worse at everything,
including the reasoning you need it to keep. A workable split:

| Portion | Share | Example |
|---|---|---|
| Target domain | 50-70% | `lianghsun/tw-legal-qa-3M` |
| Broad in-language text | 20-40% | `lianghsun/tw-news-551M` |
| Replay / general | 10-15% | `lianghsun/wikipedia-zh-filtered` |

Set these with `weight` on each source; sources are interleaved with
`stopping_strategy="all_exhausted"`, so weights are sampling probabilities,
not row counts.

**Update all parameters, not LoRA.** These are two independent axes: CPT names
the *objective* (next-token prediction over raw text), while full vs LoRA names
*which weights move*. CPT is pretraining, not fine-tuning — the recipe just
happens to set `tuning.method: full`. LoRA absorbs style well and new knowledge
poorly. If VRAM forces LoRA, use a high rank (128+) and expect less knowledge
transfer. Only add `modules_to_save: [embed_tokens, lm_head]` if you actually
changed the tokenizer — it is expensive and usually unnecessary, since Gemma's
262K-entry vocabulary already covers Traditional Chinese well.

**Large effective batch.** CPT is noisier than SFT. Aim for an effective batch
of 128-512 sequences via `gradient_accumulation_steps`.

## Watch during training

- **Loss should fall smoothly then plateau.** A spike that does not recover
  means the LR is too high — stop, halve it, restart from the last checkpoint.
- **Compare against the base model's loss on held-out general text.** If that
  rises sharply, forgetting is happening; raise the replay share.
- Always evaluate with `hsun-eval` before and after. CPT that improves domain
  benchmarks while collapsing `Formosa-bench` is a regression, not a win.

## After CPT

The output is a base model — it cannot chat. Always follow with SFT
(`hsun-sft`) before evaluating anything instruction-shaped or serving it.

## Corpora

`skills/hsun-trainer/references/dataset-catalog.md` lists every verified
zh-TW `text` corpus with its quirks — including that `lianghsun/Taiwan_c4`
has no `train` split.
