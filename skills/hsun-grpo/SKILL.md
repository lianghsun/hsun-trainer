---
name: hsun-grpo
description: GRPO / RLVR reinforcement learning on an LLM with programmatic reward functions, including Traditional Chinese (zh-TW) rewards that punish Simplified-Chinese leakage and English drift. Use when the user says GRPO, RLVR, RLHF, reinforcement learning, reward function, verifiable reward, DAPO, "optimize for accuracy", "stop the model writing Simplified Chinese", or wants a model to improve on a checkable metric after SFT. Covers reward design, weighting, num_generations, vLLM generation, and reward hacking.
license: MIT
---

# GRPO / RLVR

Optimizes a model against **programmatic** rewards. Group Relative Policy
Optimization samples several completions per prompt, scores each, and pushes
the policy toward the above-average ones — no reward model or human labels.

## Precondition

GRPO only works when a program can check correctness. If you cannot write a
function that scores an answer, use DPO instead
(`scripts/train.py --stage dpo`) with a preference dataset.

Good GRPO targets: math with a final value, multiple choice, output format,
language purity, code that runs. Bad targets: "be more helpful", "write
better prose".

Run GRPO **after** SFT. It sharpens an existing ability; it cannot install one.

## Run it

```bash
# see the reward library and how it scores samples - no GPU needed
uv run scripts/train_grpo.py --list-rewards
uv run scripts/train_grpo.py --test-rewards

# validate, then train
uv run scripts/train_grpo.py --config recipes/grpo_gemma_zhtw.yaml --smoke-test
uv run scripts/train_grpo.py --config recipes/grpo_gemma_zhtw.yaml
```

## Reward library

| Reward | What it scores |
|---|---|
| `accuracy_math` | symbolic equivalence via `math-verify`, exact-match fallback |
| `accuracy_mcq` | boxed option letter vs `ground_truth` |
| `accuracy_exact` | normalized exact string match |
| `format_boxed` | exactly one non-empty `\boxed{...}` |
| `format_think` | exactly one well-formed `<think>...</think>` |
| `zhtw_purity` | **1.0 for pure Traditional Chinese; falls fast as Simplified characters appear** |
| `no_english_drift` | punishes English prose answering a Chinese prompt (code and LaTeX exempt) |
| `no_repetition` | punishes degenerate 4-gram loops |
| `length_target` | Gaussian bonus around a target length |

`zhtw_purity` matches against a curated set of Simplified-only characters,
deliberately excluding forms valid in both scripts (里, 后, 台, 只, 干, 面...)
so correct zh-TW output is never punished. Rewards return `None` when they do
not apply to a sample, and TRL skips them for that sample.

## Composing rewards

```yaml
grpo:
  rewards:
    - {name: accuracy_math,    weight: 3.0}   # the real objective dominates
    - {name: format_boxed,     weight: 0.5}   # make the answer extractable
    - {name: format_think,     weight: 0.5}
    - {name: zhtw_purity,      weight: 1.0}   # zh-TW constraint
    - {name: no_english_drift, weight: 0.5}
    - {name: no_repetition,    weight: 0.3}
```

Keep the correctness reward's weight clearly above the shaping rewards. If
format rewards outweigh accuracy, the model learns to emit beautifully
formatted wrong answers — the single most common GRPO failure.

## Dataset preparation

GRPO needs **prompt-only** data plus a verifiable target. Set `dataset_kind`
and the script builds `prompt` + `ground_truth`:

| `dataset_kind` | Expects | Example |
|---|---|---|
| `math` | `question_field`, `ground_truth_field` | `twinkle-ai/tw-math-reasoning-2k` |
| `mcq` | `question_field`, `choice_fields`, `ground_truth_field` | `lianghsun/tw-legal-benchmark-v1` |
| `chat` | `question_field` | anything open-ended |

For `math`, the preparer extracts the final `\boxed{...}` from the answer
field — necessary because `tw-math-reasoning-2k`'s `answer` column holds a
full worked solution, not a bare value.

Set `choice_fields: [A, B, C, D, E]` for
`lianghsun/tw-emergency-medicine-bench`, which has five options.

**Never GRPO on a benchmark you also report.** Hold it out.

## Configuration that matters

| Setting | Guidance |
|---|---|
| `num_generations` | 8-16. Below 4 the group advantage is too noisy |
| `per_device_train_batch_size` | must be divisible by `num_generations` |
| `learning_rate` | 1e-6 to 5e-6 — far lower than SFT |
| `max_completion_length` | long enough for full reasoning, or truncation looks like failure |
| `mask_truncated_completions` | `true`, so cut-off answers are not rewarded |
| `beta` | KL coefficient; TRL 1.10 defaults to `0.0` (off). Raise to 0.01-0.04 if the model drifts from its SFT behaviour |
| `use_vllm` + `vllm_mode: colocate` | large speedup; generation dominates GRPO wall-clock |

TRL 1.10 API notes: `scale_rewards` is a **string** (`"group"`/`"batch"`/
`"none"`), `loss_type` defaults to `"dapo"`, and `max_prompt_length` was
**removed** — do not pass it.

## Reward hacking — what it looks like

Watch the logged completions (`log_completions: true` is on by default here).

| Symptom | Cause | Fix |
|---|---|---|
| Answers get shorter and emptier, reward rises | format reward outweighs accuracy | raise accuracy weight |
| Perfect `\boxed{}`, wrong contents | same | same |
| Model emits `\boxed{}` many times | `format_boxed` rewards presence only | it already penalizes >1; check weights |
| Reward plateaus immediately | task too hard, or all generations score identically | easier data, or more `num_generations` |
| Reward rises, benchmarks fall | overfitting to the reward | add KL (`beta`), stop earlier |

## Adding a custom reward

Reward functions live inline in `scripts/train_grpo.py` so the file stays
single-upload for HF Jobs. Add one with the `@reward` decorator:

```python
@reward("cites_statute", "Rewards answers that cite a ROC statute article.")
def cites_statute(completions, **kwargs):
    import re
    return [1.0 if re.search(r"第\s*\d+\s*條", as_text(c)) else 0.0 for c in completions]
```

Then reference it by name in the recipe. Signature: accept `completions` plus
`**kwargs` — every dataset column arrives as a keyword argument, so
`ground_truth=None` works as a parameter. Verify with `--test-rewards`.

## Cost

GRPO is expensive: each step generates `num_generations` completions per
prompt before a single update. Budget roughly 5-10x an SFT run over the same
prompt count, and enable vLLM.
