# Troubleshooting

## Training silently does nothing

**`0 of N rows survived preprocessing`** — `train.py` raises this rather than
letting a no-op run report success. Causes, in order of likelihood:

1. `messages` is a JSON **string**, not a list. Add `json_columns: [messages]`.
   Affects `lianghsun/tw-legal-qa-chat`, `lianghsun/reasoning-base-20k-chat`.
2. ShareGPT layout (`conversations` with `from`/`value`). Add
   `sharegpt_to_messages: true`.
3. Roles the chat template rejects. Gemma requires strictly alternating
   user/assistant turns and raises on two consecutive assistant messages.
4. Every row longer than `train.max_length`.

Diagnose with `uv run scripts/inspect_dataset.py <dataset>` — it flags
JSON-string columns explicitly.

## TypeError on a config argument

TRL renames arguments between minor releases. Verified against **TRL 1.10**:

| Old | Current |
|---|---|
| `SFTConfig(max_seq_length=...)` | `max_length` |
| `warmup_ratio=...` | **removed** — use `warmup_steps` |
| `GRPOConfig(max_prompt_length=...)` | **removed** |
| `scale_rewards=True` | `"group"` / `"batch"` / `"none"` |
| `loss_type="grpo"` default | now `"dapo"` |
| `beta=0.04` default | now `0.0` |

Check any field before using it:

```bash
uv run --with trl python -c "import dataclasses,trl; print([f.name for f in dataclasses.fields(trl.SFTConfig)])"
```

## `assistant_only_loss` raises

Gemma's stock chat template has no `{% generation %}` marker, so assistant
tokens cannot be located. Either set `assistant_only_loss: false`, or point at
the bundled patched template:

```yaml
model:
  chat_template_path: assets/gemma_chat_template_assistant_mask.jinja
```

That template renders byte-identical text to Google's and marks model turns.

## Training runs on CPU on a machine that has a GPU

`uv` resolves the newest torch, and its default build targets the newest CUDA.
On an older driver that build cannot initialise, `torch.cuda.is_available()`
returns False, and Trainer quietly falls back to CPU — the run completes, the
loss curve looks fine, and it took orders of magnitude longer than it should.

Measured on an RTX 3090 with driver 555.42.06 (CUDA 12.5): uv installed
`torch 2.13.0+cu130`, CUDA was unavailable, and nothing errored.

`train.py` and `train_grpo.py` now abort instead. To fix it, patch the
script's own environment — **index flags do not work here**, because an
explicit index outranks `--default-index` and PyPI keeps winning the resolve:

```bash
uv pip install --python "$(uv python find --script scripts/train.py)" \
    --reinstall-package torch \
    --index-url https://download.pytorch.org/whl/cu126 \
    'torch==2.13.0+cu126'

"$(uv python find --script scripts/train.py)" scripts/train.py --config <recipe>
```

Pick the tag from `nvidia-smi`: CUDA 12.x needs `cu126`, 13.x needs `cu130`.
CUDA has minor-version compatibility, so a cu126 build runs on any 12.x driver;
crossing a major version (12 -> 13) does not work.

Things that look like they should help but do not:

| Attempt | Result |
|---|---|
| `UV_INDEX_URL` / `UV_EXTRA_INDEX_URL` env vars | ignored by `uv sync --script` |
| `--index-url` / `--default-index` on `uv run`/`uv sync` | PyPI still wins |
| `--index-strategy unsafe-best-match` | picks the *newest* build, i.e. the wrong one |
| `--torch-backend=auto` | only on `uv pip install`; sees 2.13.0 satisfied and no-ops |

Upgrading the driver is the real fix; the above is a per-machine workaround
that must be re-applied whenever the script environment is rebuilt.
`HSUN_ALLOW_CPU=1` forces CPU training if that is genuinely what you want.

## Out of memory

In order of cost to quality:

1. `per_device_train_batch_size: 1`, raise `gradient_accumulation_steps`
2. lower `max_length` to the p99 from `inspect_dataset.py`
3. `gradient_checkpointing: true`
4. `use_liger_kernel: true` (Linux/CUDA) — fixes large-vocabulary logits OOM
5. `tuning.method: lora`, then `qlora`
6. multi-GPU ZeRO-3
7. HF Jobs on a bigger flavor

If OOM happens at a seemingly small model size, it is usually the logits
tensor, not the weights — see `hardware.md`.

## Disk fills up during training

`save_strategy` and `train.save_final_model` are orthogonal, and each one
leaves the other's artefact behind. Measured on a 2-step run where
`save_steps: 500` was never reached:

| `save_final_model` | `save_strategy` | what lands in `output_dir` |
|---|---|---|
| `true` (default) | `steps` (default) | end-of-training checkpoint **and** final model |
| `false` | `steps` | end-of-training checkpoint only |
| `true` | `"no"` | final model only |
| `false` | `"no"` | **nothing** |

Trainer writes a checkpoint when training ends even if `save_steps` was never
hit, so `save_strategy: "no"` is required to suppress it — and it alone does
not stop the final model. For a throwaway experiment set both. `train.py`
prints a reminder when only one of the two is set.

A full fine-tune checkpoint carries optimizer state, so it is roughly 3x the
weight size (a 1B model: ~2 GB weights + ~4 GB Adam state per checkpoint).
`save_total_limit` caps how many are kept.

## Gated repo / 401 / 403

All Gemma repos are gated. Accept the licence on the model page, then:

```bash
hf auth login
uv run scripts/preflight.py     # confirms login and org membership
```

On HF Jobs, pass `--secrets HF_TOKEN` or downloads fail inside the job.

Several `twinkle-ai` repos publish no data files at all
(`fineweb-zhtw-filtered`, `finepdfs-zhtw`, `finetranslations-zhtw`) — that is
an empty repo, not an auth problem.

## Loss behaviour

| Symptom | Likely cause | Action |
|---|---|---|
| Loss spikes and never recovers | LR too high | halve LR, resume from checkpoint |
| Loss ~0 within a few hundred steps | overfitting / too few examples | fewer epochs, more data |
| `mean_token_accuracy` > 0.95 | memorizing | same |
| Loss flat from step 0 | LR too low, or nothing trainable | check `print_trainable_parameters` output |
| NaN in bf16 | unstable model or bad data | try `eager` attention, inspect extreme rows |

## Model outputs Simplified Chinese

1. Audit the SFT mix — most contamination arrives with the data.
2. Add GRPO with `zhtw_purity` (weight ~1.0) and `no_english_drift`.
3. Check the system prompt explicitly demands 臺灣正體中文.

Verify quickly: `uv run scripts/train_grpo.py --test-rewards`.

## Benchmark score is near zero

Almost always answer **extraction**, not knowledge. Open
`results/eval_results_*.jsonl` and read the raw predictions. If the model
answered correctly in prose but never wrote `\boxed{}`, either strengthen the
system prompt, switch `evaluation_method` to `pattern`, or use `logit`, which
ignores output format entirely.

## Scores differ between training and serving

The chat template must match. Serve with the same template you trained with —
a mismatch typically shows as repeated `<end_of_turn>` tokens or a large,
uniform score drop across every benchmark.

## Multi-GPU hangs at startup

- NCCL waiting on a mismatched world size: check `--num_processes` matches
  visible GPUs.
- `CUDA_VISIBLE_DEVICES` set inconsistently across ranks.
- Try `NCCL_P2P_DISABLE=1` on consumer cards without proper P2P support.

## HF Jobs finished but the model is gone

The machine is ephemeral. Set `hub.push_to_hub: true` and
`hub.hub_model_id`, and pass `--secrets HF_TOKEN`. There is no way to recover
weights from a finished job.
