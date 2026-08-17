# End-to-end pipelines

Ordered plans with the checkpoints to evaluate between stages. Every stage
ends with an eval — a pipeline without measurements between stages cannot be
debugged when the final model is worse than the first.

## A. zh-TW instruction model (most common)

Goal: Gemma that answers naturally in Traditional Chinese, reasons, and calls
tools. No new domain knowledge needed.

| # | Stage | Config | Roughly |
|---|---|---|---|
| 0 | Baseline eval | `--builtin tmmluplus` + `--bench Formosa-bench` | 20 min |
| 1 | SFT | `recipes/sft_gemma_zhtw.yaml` | 6-12 h on 1×A100 (12B LoRA) |
| 2 | Eval | same config as step 0 | 20 min |
| 3 | GRPO | `recipes/grpo_gemma_zhtw.yaml` | 12-24 h |
| 4 | Eval | same config as step 0 | 20 min |

Skip CPT entirely. Gemma already knows Traditional Chinese; the gap is
behavioural, and SFT plus GRPO closes it far more cheaply.

Stage 3 is worth it when step 2 shows the model still emits Simplified
characters or drifts into English — `zhtw_purity` and `no_english_drift` fix
what SFT data alone usually cannot.

## B. Taiwanese legal specialist

Goal: a model that knows ROC statutes and case law.

| # | Stage | Data | Notes |
|---|---|---|---|
| 0 | Baseline eval | `tw-legal-benchmark-v1`, `Formosa-bench` | hold both out of training |
| 1 | CPT | `tw-legal-qa-3M` (6) + `tw-news-551M` (3) + `wikipedia-zh-filtered` (1) | full FT from `google/gemma-3-4b-pt`, 1 epoch, lr 1e-5 |
| 2 | Eval | perplexity on held-out legal text **and** `Formosa-bench` | Formosa-bench must not collapse |
| 3 | SFT | `tw-legal-synthetic-qa` + `tw-judicial-wisdom` + `tw-legal-qa-chat` (`json_columns`) + general mix | restores chat ability |
| 4 | Eval | full benchmark set | |
| 5 | GRPO (optional) | `tw-legal-benchmark-v1`-style MCQ, held out from reporting | `accuracy_mcq` + `zhtw_purity` |

Step 2 is the decision point. If `Formosa-bench` dropped sharply, the replay
share was too low or the learning rate too high — fix that before spending on
SFT, because SFT will not undo the damage.

The base model after step 1 **cannot chat**. Do not evaluate it on
instruction benchmarks; use loss/perplexity there.

## C. Reasoning model

Goal: reliable `<think>` reasoning in zh-TW with checkable answers.

| # | Stage | Data |
|---|---|---|
| 1 | SFT | `tw-reasoning-instruct-50k` (`sharegpt_to_messages`) + `tw-math-reasoning-2k` |
| 2 | Eval | `gsm8k`, `tmmluplus` |
| 3 | GRPO | `tw-math-reasoning-2k` with `dataset_kind: math` |
| 4 | Eval | same as step 2 |

Reward mix: `accuracy_math` 3.0, `format_think` 0.5, `format_boxed` 0.5,
`zhtw_purity` 1.0. Keep accuracy dominant or the model learns to produce
well-formatted wrong answers.

## D. Tool-calling model

| # | Stage | Data |
|---|---|---|
| 1 | SFT | `tw-function-call-reasoning-10k` (`keep: [messages]`) + a general chat mix |
| 2 | Eval | `--builtin bfcl` (`pip install twinkle-eval[tool]`) |

Keep general chat data in the mix at 2-3x the tool data, or the model starts
calling functions for ordinary questions.

## Ordering rules

1. **CPT → SFT → GRPO.** Never SFT then CPT; raw-text training erases chat.
2. **Evaluate between every stage**, with one fixed benchmark set.
3. **Hold out anything you report.** Training on `Formosa-bench` makes your
   `Formosa-bench` score meaningless.
4. **Merge LoRA before the next stage** if stacking adapters, or explicitly
   continue training the existing adapter — do not silently start a fresh
   adapter over a merged one you never saved.
5. **One variable at a time.** Changing data mix and learning rate together
   makes a regression unattributable.

## Cost control

- Smoke test every recipe (`--smoke-test`) before booking a GPU.
- Run stage 0's baseline on a small flavor; it is inference only.
- LoRA first. Move to full FT only when LoRA is measurably the ceiling.
- GRPO last, and only against a reward that reflects a real, measured failure.
