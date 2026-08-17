# zh-TW dataset catalog

Stage-tagged index of Traditional Chinese corpora on the Hub, focused on
[`lianghsun`](https://huggingface.co/lianghsun) and
[`twinkle-ai`](https://huggingface.co/twinkle-ai).

Every schema below was verified by loading the dataset. Sizes in the repo
names refer to **token counts** (`tw-news-551M` = ~551M tokens), not rows.

Still confirm before writing a recipe — repos change:

```bash
uv run scripts/inspect_dataset.py <dataset_id>
```

---

## Read this first: two schema traps

**1. `messages` stored as a JSON string.** TRL recognises the column name,
fails to parse the rows, and silently produces a **0-row** training set. Fix in
the recipe with `json_columns`.

Affected: `lianghsun/tw-legal-qa-chat`, `lianghsun/reasoning-base-20k-chat`
(also `tools` in `twinkle-ai/tw-function-call-reasoning-10k`).

```yaml
- path: lianghsun/tw-legal-qa-chat
  json_columns: [messages]
  keep: [messages]
```

**2. ShareGPT layout** — `conversations` with `{"from","value"}` instead of
`messages` with `{"role","content"}`.

Affected: `twinkle-ai/tw-reasoning-instruct-50k`, `lianghsun/tw-instruct`,
`lianghsun/tw-law-article-qa-DPO`.

```yaml
- path: twinkle-ai/tw-reasoning-instruct-50k
  sharegpt_to_messages: true
  sharegpt_column: conversations
  keep: [messages]
```

---

## Continued pretraining (CPT) — raw `text`

All of these expose a `text` column and are ready for `stage: cpt` with
`keep: [text]`. Several also carry a `cleaned_text` variant; prefer `text`
unless you have checked which is better filtered.

| Dataset | Domain | Notes |
|---|---|---|
| `lianghsun/tw-legal-qa-3M` | Legal QA prose | `text`, `token_count`, `url` |
| `lianghsun/tw-news-551M` | General news | largest general corpus here |
| `lianghsun/tw-novel-1.1B` | Fiction | strongest for fluent long-form zh-TW |
| `lianghsun/wikipedia-zh-742M` | Encyclopedia | good replay data |
| `lianghsun/wikipedia-zh-filtered` | Encyclopedia | smaller, filtered |
| `lianghsun/tw-book` | Textbooks | `id`, `text`, `src` |
| `lianghsun/tw-gov-news-90M` | Government | ROC public-sector register |
| `lianghsun/tw-society-88M` | Society, lifestyle | |
| `lianghsun/tw-finance-159M` | Finance | |
| `lianghsun/tw-health-43M` | Health, medical | |
| `lianghsun/tw-science-24M` | Science | |
| `lianghsun/tw-legal-news-24M` | Legal news | |
| `lianghsun/tw-processed-law-ctx` | Statute text | `text`, `name`, `level` |
| `lianghsun/tw-judgment-gist` | Judgment summaries | |
| `lianghsun/Taiwan_c4` | Gov web crawl | **no `train` split** — splits are per-domain (`moj.gov.tw`, `cy.gov.tw`, ...); name one explicitly |

Mixing guidance lives in `pipeline-recipes.md`.

---

## SFT — conversational

Native `messages` (`{"role","content"}`), usable with `keep: [messages]`:

| Dataset | Rows | Domain |
|---|---|---|
| `twinkle-ai/tw-math-reasoning-2k` | 2 K | Math CoT, has `<think>`; `problem_zhtw`, `answer` |
| `twinkle-ai/tw-function-call-reasoning-10k` | 10 K | Tool calling; `tools` is a JSON string |
| `lianghsun/tw-legal-synthetic-qa` | — | Legal QA, has `train`/`test` |
| `lianghsun/tw-judicial-wisdom` | — | Judicial reasoning |
| `lianghsun/my-sharegpt` | — | Mixed chat + `scores` |
| `lianghsun/tw-greeting` | — | Identity / greeting behaviour |

Needs a conversion flag:

| Dataset | Rows | Fix |
|---|---|---|
| `twinkle-ai/tw-reasoning-instruct-50k` | 50 K | `sharegpt_to_messages: true` |
| `lianghsun/tw-instruct` | — | `sharegpt_to_messages: true` |
| `lianghsun/tw-legal-qa-chat` | 527 | `json_columns: [messages]` |
| `lianghsun/reasoning-base-20k-chat` | 20 K | `json_columns: [messages]` |

Not conversational, map columns yourself:

| Dataset | Columns |
|---|---|
| `twinkle-ai/tw-leetcode` | `text`, `question`, `thought`, `answer`, `time_complexity` |

---

## Preference (DPO / reward modelling)

| Dataset | Shape |
|---|---|
| `lianghsun/tw-law-article-qa-DPO` | `conversations` (ShareGPT) + `chosen`/`rejected` **dicts** |
| `lianghsun/Everything-Instruct-Multilingual-DPO` | multilingual preference |
| `lianghsun/ultrafeedback-binarized-preferences-cleaned-multilingual` | UltraFeedback port |

`chosen`/`rejected` here are single message dicts, not strings — flatten to
text (or to a one-message list) before DPO.

---

## GRPO / RLVR — needs a checkable answer

GRPO only works where a program can verify correctness.

| Dataset | `dataset_kind` | Fields |
|---|---|---|
| `twinkle-ai/tw-math-reasoning-2k` | `math` | `problem_zhtw` + `answer` (final value sits in `\boxed{}` inside a full solution) |
| `lianghsun/tw-legal-benchmark-v1` | `mcq` | `question`, `A`-`D`, `answer` |
| `lianghsun/tw-emergency-medicine-bench` | `mcq` | `question`, `A`-`E` (**five** options) |
| `twinkle-ai/tw-leetcode` | custom | needs an execution-based reward |

Hold out any benchmark you train on — a GRPO'd benchmark stops being a
measurement.

---

## Evaluation benchmarks

| Dataset | Items | Shape |
|---|---|---|
| `lianghsun/Formosa-bench` | 349 | 4 configs (`geography` 102, `government` 116, `history` 83, `society` 48), split `test` |
| `lianghsun/tw-legal-benchmark-v1` | 209 | split is **`train`**, not `test` |
| `lianghsun/tw-emergency-medicine-bench` | 1,719 | 5 options, split `train` |
| `ikala/tmmluplus` | — | 66 subject configs, `test` split |

`Formosa-bench` is in Twinkle Eval's registry as `formosa_bench`; the others
export cleanly with `scripts/make_eval_config.py`. See the `hsun-eval` skill.

---

## Multimodal

| Dataset | Notes |
|---|---|
| `twinkle-ai/Formosa-Vision` | `images`, `text`, `messages` — zh-TW VQA |
| `twinkle-ai/tw-drug-labels-vision` | TFDA drug labels, OCR + structured extraction |
| `lianghsun/pokemon-blip-captions-en-zh_tw` | bilingual captions |
| `lianghsun/coco-caption-zh_tw-val` | COCO captions in zh-TW |

Gemma 3 4B+ and all Gemma 4 checkpoints are multimodal, so these are trainable
without swapping model family — but this plugin's scripts target text-only
training. Use TRL's VLM path or `huggingface-vision-trainer` for image work.

---

## Empty or unavailable repos

Verified as publishing no data files (do not put them in a recipe):
`twinkle-ai/fineweb-zhtw-filtered`, `twinkle-ai/finepdfs-zhtw`,
`twinkle-ai/finetranslations-zhtw`.

`twinkle-ai/tw-privacy-guides` needs `Pillow` installed to load.

The many `twinkle-ai/*-eval-logs-and-scores` repos are **evaluation outputs**
for published models, not training data — useful as reference scores when
judging your own eval numbers.
