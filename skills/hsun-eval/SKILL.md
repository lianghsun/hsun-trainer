---
name: hsun-eval
description: Benchmark an LLM with ai-twinkle/Eval (Twinkle Eval), including Traditional Chinese benchmarks — TMMLU+, Formosa-bench, tw-legal-benchmark-v1, tw-emergency-medicine-bench — plus MMLU, GSM8K, IFEval, BFCL and more. Use when the user says evaluate, benchmark, eval, "measure the model", "run TMMLU+", "how good is my model", twinkle-eval, or wants before/after scores around a training run. Covers serving the model, exporting Hub benchmarks locally, config generation, and reading the results.
license: MIT
---

# Evaluation with Twinkle Eval

[ai-twinkle/Eval](https://github.com/ai-twinkle/Eval) evaluates an
**OpenAI-compatible endpoint** with parallel requests (roughly 9-17x faster
than sequential harnesses), randomized option order, and multi-run stability
statistics.

## Always baseline first

Evaluate the model **before** training. A post-training number without a
matched baseline measures nothing. Use identical config for both runs.

## Workflow

### 1. Serve the model

Twinkle Eval calls an API; it does not load weights.

```bash
# merged full model (preferred - fastest)
vllm serve outputs/gemma3-12b-zhtw-sft --served-model-name my-model --port 8000

# LoRA adapter without merging
vllm serve google/gemma-3-12b-it --served-model-name my-model --port 8000 \
    --enable-lora --lora-modules my-model=outputs/gemma3-12b-zhtw-sft
```

Serve with the **same chat template used in training**. A mismatch shows up
as a large, misleading score drop.

### 2. Generate the config and export benchmarks

```bash
uv run scripts/make_eval_config.py \
    --model my-model \
    --base-url http://localhost:8000/v1 \
    --bench lianghsun/Formosa-bench lianghsun/tw-legal-benchmark-v1 \
    --builtin tmmluplus gsm8k \
    --out eval/
```

`--bench` pulls Hub datasets and writes them as Twinkle Eval jsonl
(`{"question","A".."E","answer":"C"}`), handling 4- and 5-option sets and
multi-config benchmarks such as Formosa-bench's four subjects.
`--builtin` refers to benchmarks Twinkle Eval downloads itself.

### 3. Run

```bash
pip install twinkle-eval

cd eval && twinkle-eval --download-dataset tmmluplus     # for --builtin entries
twinkle-eval --validate --config eval/config.yaml        # check paths and schema
twinkle-eval --dry-run  --config eval/config.yaml        # preview, no API calls
twinkle-eval --config eval/config.yaml --export json csv
```

## Benchmarks in the registry

| Name | Coverage | Method |
|---|---|---|
| `tmmluplus` | Traditional Chinese, 66 subjects | `box` |
| `formosa_bench` | Taiwan geography/government/history/society | `box` |
| `mmlu`, `mmlu_pro`, `mmlu_redux` | English general knowledge | `box` |
| `supergpqa`, `gpqa` | graduate-level science (`gpqa` is gated) | `box` |
| `gsm8k`, `aime2025` | math | `math` |
| `bbh` | reasoning | `regex_match` |
| `ifeval`, `ifbench` | instruction following | `ifeval` / `ifbench` |
| `bfcl` | function calling | `bfcl_fc` |
| `needlebench` | long context | `niah` |

Extras install per family: `pip install twinkle-eval[math,ifeval,tool]`.

## Evaluation methods

| Method | Use when |
|---|---|
| `pattern` | general regex answer extraction |
| `box` | the model emits `\boxed{}` — the default for reasoning models |
| `logit` | compare option log-probabilities; format-independent, needs a completions endpoint |
| `math` | `\boxed{}` plus symbolic equivalence |
| `regex_match` | free-form reasoning tasks |

**Pick the method from the model, not from habit.** `make_eval_config.py`
defaults to `box`, which is right for a model trained to emit `\boxed{}` and
wrong for a stock instruct checkpoint. Measured on `gemma-3-1b-it` against
`lianghsun/tw-legal-benchmark-v1` — same model, same 209 questions, same
system prompt:

| `--method` | accuracy | unparsed |
|---|---|---|
| `box` (default) | 4.78% | 83.7% |
| `pattern` | 29.67% | 0% |

The model was answering correctly and saying `最終答案：C` instead of
`\boxed{C}`. Nothing errors — a 6x wrong number is simply reported.

So: if a model scores near zero on multiple choice, the cause is almost always
**extraction, not knowledge**. Read the `無法解析` percentage twinkle-eval
prints on every dataset; above ~20% the score measures formatting. Confirm in
`eval_results_*.jsonl`, then switch to `pattern` (or `logit`, which is
format-independent but needs a completions endpoint).

## Trustworthy numbers

- `shuffle_options: true` — cancels position bias (models favour "C").
- `repeat_runs: 3` — reports mean and standard deviation. A 2-point gap with
  a 3-point standard deviation is not an improvement.
- `temperature: 0.0` for reproducibility.
- Report the same benchmark set before and after every stage.

## Output

Written to `results/`:

- `results_<timestamp>.json` — config, per-dataset accuracy, runtime
- `eval_results_<timestamp>_run<N>.jsonl` — per-question prediction and
  correctness. **Read this when a score looks wrong.**

Upload with `--hf-repo-id <user>/<model>-eval-logs-and-scores`. That matches
the convention used across `twinkle-ai/*-eval-logs-and-scores`, whose
published scores make useful comparison points for zh-TW models.

## Reference points

The `twinkle-ai` org publishes eval logs for Gemma 3 variants (including
`gemma-3-taide-12b-chat`), Llama-3-Taiwan, Breeze2, GPT-OSS, and others —
use them to judge whether your number is competitive rather than merely
higher than yesterday's.

## Python API

```python
from twinkle_eval import TwinkleEvalRunner

runner = TwinkleEvalRunner("eval/config.yaml")
runner.initialize()
results = runner.run_evaluation(export_formats=["json", "csv"])
```
