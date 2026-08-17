---
name: hsun-sft
description: Supervised fine-tuning (SFT) of an LLM on instruction, chat, reasoning, or tool-calling data with TRL. Use when the user says SFT, fine-tune, instruction tuning, chat tuning, "teach the model to follow instructions", "train on my conversations", LoRA/QLoRA fine-tuning, or wants a model to answer in a particular style or format. Covers dataset formats (messages / ShareGPT / prompt-completion), chat templates, assistant-only loss, packing, and LoRA configuration.
license: MIT
---

# Supervised fine-tuning (SFT)

Teaches an instruction-tuned or base model *how to respond* — format, style,
language, tool use. It is the highest-value-per-GPU-hour stage; try it before
reaching for CPT or GRPO.

## Run it

```bash
# 1. always inspect first - schema traps are common (see below)
uv run scripts/inspect_dataset.py twinkle-ai/tw-reasoning-instruct-50k --tokenizer google/gemma-3-12b-it

# 2. size the run
uv run scripts/plan_memory.py --model google/gemma-3-12b-it --seq-len 8192 --vram 80

# 3. smoke test (~2 min, catches template and column errors)
uv run scripts/train.py --config recipes/sft_gemma_zhtw.yaml --smoke-test

# 4. real run
uv run scripts/train.py --config recipes/sft_gemma_zhtw.yaml
```

Start from `recipes/sft_gemma_zhtw.yaml`.

## Dataset formats TRL accepts

| Shape | Columns | Notes |
|---|---|---|
| Conversational | `messages` = `[{role, content}]` | most common; chat template applied |
| Language modeling | `text` | no template; loss on everything |
| Prompt-completion | `prompt`, `completion` | loss on completion only by default |

Anything else must be mapped. The recipe supports it without extra code:

```yaml
sources:
  - path: twinkle-ai/tw-reasoning-instruct-50k
    sharegpt_to_messages: true      # {"from","value"} -> {"role","content"}
    sharegpt_column: conversations
    keep: [messages]
  - path: lianghsun/tw-legal-qa-chat
    json_columns: [messages]        # messages stored as a JSON *string*
    keep: [messages]
  - path: some/alpaca-style
    rename: {instruction: prompt, output: completion}
    keep: [prompt, completion]
```

**The 0-row trap.** If `messages` is a JSON string, TRL accepts the column,
fails to parse it, and silently trains on **nothing**. `inspect_dataset.py`
flags this, and `train.py` aborts if preprocessing empties the dataset —
but only `json_columns` actually fixes it.

## Gemma specifics

**Chat template has no `{% generation %}` marker.** So `assistant_only_loss:
true` cannot locate assistant tokens and TRL raises. Two options:

```yaml
model:
  chat_template_path: assets/gemma_chat_template_assistant_mask.jinja
train:
  assistant_only_loss: true
```

That bundled template renders byte-identical output to Google's, but marks the
model turns so assistant-only loss works. Otherwise set
`assistant_only_loss: false` and accept loss on user turns too.

**Multimodal checkpoints.** Gemma 3 4B+ and all Gemma 4 models are multimodal.
For text-only SFT:

- keep `freeze_vision_tower: true` (the default)
- do **not** use `target_modules: all-linear` — on `gemma-3-4b` that targets
  401 Linear layers of which 162 are the vision tower, so ~40% of the adapter
  is spent on an encoder that never sees an image. Use the language-model
  regex from the shipped recipe instead.

**Strict role alternation.** Gemma's template raises unless turns alternate
user/assistant. A dataset with two consecutive assistant messages will fail.

## Hyperparameters that matter

| Setting | Guidance |
|---|---|
| `learning_rate` | 1e-5 to 2e-5 full FT; 1e-4 to 2e-4 for LoRA |
| `num_train_epochs` | 2-3. More overfits fast on small sets |
| `max_length` | from `inspect_dataset.py` p99, not from habit |
| `packing` | `false` for chat (keeps conversations intact), `true` for throughput on short data |
| LoRA `r` | 16-32 for style; 64-128 to move behaviour harder |
| `alpha` | conventionally `2 * r` |

Effective batch = `per_device_train_batch_size × grad_accum × num_gpus`.
Aim for 64-128 sequences.

## Mixing datasets

`weight` sets sampling probability across sources, interleaved with
`all_exhausted` so small high-quality sets get repeated rather than drowned.
Sources must share columns after `rename`/`keep` — the script fails loudly
with a diff if they do not.

Keep an identity/behaviour set such as `lianghsun/tw-greeting` in the mix at
low weight if the model should know who it is.

## Watch for

- **Loss near zero within a few hundred steps** — memorizing, not learning.
  Fewer epochs or more data.
- **`mean_token_accuracy` above ~0.95** — same problem.
- **Model answers in Simplified Chinese** — SFT data contamination. Filter the
  corpus, then use GRPO with `zhtw_purity` to finish the job.
- **Repeated `<end_of_turn>` at inference** — usually a chat-template mismatch
  between training and serving. Serve with the same template you trained with.

## After SFT

Evaluate with `hsun-eval` against the pre-SFT baseline. Then consider
`hsun-grpo` if any remaining failure is machine-checkable.
