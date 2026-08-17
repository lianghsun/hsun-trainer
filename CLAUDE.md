# CLAUDE.md

Context for Claude Code sessions working **in this repository**.

This repo is a **Claude Code plugin**, not a Python package. Nothing here is
imported; `scripts/*.py` are executed, `skills/*/SKILL.md` are read by the
model on demand. There is no `pyproject.toml`, no install step, and no test
suite — verification is done by running the scripts (see *Verifying changes*).

Docs split by audience: `README.md` is human-facing and written in
Traditional Chinese; everything under `skills/` is model-facing and written in
English. Keep that split when editing.

## Layout

```
.claude-plugin/   plugin.json + marketplace.json (plugin manifest)
skills/           5 skills: hsun-trainer (router) + pretrain/sft/grpo/eval
  hsun-trainer/references/   loaded on demand, not by default
scripts/          shared, executed via `uv run` - never imported
recipes/          example YAML configs
assets/           patched Gemma chat template
```

Scripts live at the repo root rather than inside each skill because all five
skills share them. SKILL.md files reference them as `scripts/<name>.py`.

## Using the plugin (what users do)

```
/plugin marketplace add lianghsun/hsun-trainer
/plugin install hsun-trainer@hsun-trainer
```

The repo is **private**, so only the owner and collaborators can install it
until `gh repo edit --visibility public`.

Then users ask in natural language ("用 tw-reasoning-instruct-50k 對 Gemma 做
SFT", "跑 Formosa-bench 評測") and the router skill dispatches. The core loop
is always: `preflight.py` → `inspect_dataset.py` → `plan_memory.py` →
`--smoke-test` → real run → `hsun-eval`.

## Hard constraints — breaking these breaks the plugin

**1. Training scripts must stay single-file and self-contained.**
`hf jobs uv run` uploads *one* file. Do not factor shared logic in
`train.py` / `train_grpo.py` out into an importable module — the deliberate
duplication between them (config loading, dataset building) is the cost of
running on HF Jobs. The GRPO reward library lives inline in `train_grpo.py`
for the same reason.

**2. Dependencies go in the PEP 723 header, not a requirements file.**
Platform-specific wheels need markers (`bitsandbytes ... ; platform_system ==
'Linux'`). Never add `flash-attn` — it compiles for many minutes.

**3. `--config` must keep accepting path | https URL | raw JSON.** That is how
a job carries its settings without a shared filesystem.

**4. Every training script keeps `--smoke-test`.** It is the only cheap guard
against wasting GPU hours, and it is what caught the `warmup_ratio` removal.

## Verified facts — do not "fix" these from memory

Checked by execution against **TRL 1.10 / transformers 5.15** (Aug 2026):

| Claim | Status |
|---|---|
| `SFTConfig.max_seq_length` | renamed to `max_length` |
| `warmup_ratio` | **removed from every TRL config**; only `warmup_steps` |
| `GRPOConfig.max_prompt_length` | **removed** |
| `scale_rewards` | string `"group"`/`"batch"`/`"none"`, not bool |
| `loss_type` default | `"dapo"` |
| `beta` default | `0.0` (KL off) |
| Gemma chat template | has **no** `{% generation %}` → `assistant_only_loss` raises |
| `AutoModelForCausalLM` on multimodal Gemma | works (maps to `Gemma*ForConditionalGeneration`) |
| `all-linear` on `gemma-3-4b` | targets 401 Linears, 162 in the vision tower |
| `lianghsun/tw-legal-qa-chat`, `reasoning-base-20k-chat` | `messages` is a **JSON string** → TRL silently trains on 0 rows |
| `twinkle-ai/fineweb-zhtw-filtered`, `finepdfs-zhtw`, `finetranslations-zhtw` | empty repos, no data files |

Before writing any TRL config argument, confirm it exists:

```bash
uv run --with trl python -c "import dataclasses,trl; print([f.name for f in dataclasses.fields(trl.SFTConfig)])"
```

Never state a dataset's schema from memory — run
`uv run scripts/inspect_dataset.py <id>`. The catalog in
`skills/hsun-trainer/references/dataset-catalog.md` records verified schemas;
update it when a repo changes rather than guessing.

## Verifying changes

No CI. After editing, run what you touched:

```bash
# structural
uv run --with pyyaml python3 -c "import yaml,glob;[yaml.safe_load(open(f)) for f in glob.glob('recipes/*.yaml')]"
python3 -m py_compile scripts/*.py

# behavioural - these actually exercise the code paths
uv run scripts/train.py --config recipes/smoke_gemma.yaml --smoke-test
uv run scripts/train_grpo.py --test-rewards
uv run scripts/train.py --config recipes/cpt_gemma_zhtw.yaml --dry-run
```

`smoke_gemma.yaml` uses `google/gemma-3-270m-it`, which is text-only and runs
on Apple Silicon (MPS) in float32 — so the full SFT path is testable on a Mac
without CUDA. GRPO smoke tests also run there, just slowly.

Gemma repos are gated; `hf auth login` is required even for the 270m model.

## Conventions

- **Model family: Gemma.** Note the id casing changes by generation —
  `gemma-3-12b-it` (lowercase) vs `gemma-4-12B-it` (uppercase); Gemma 3 base
  checkpoints use `-pt`, Gemma 4 base ids are unsuffixed.
- SKILL.md frontmatter `name` must equal its directory name.
- Skill `description` fields carry the trigger phrases that make the skill
  discoverable — edit them deliberately, they are the routing surface.
- Reference docs under `references/` are loaded only when the router points at
  them; keep SKILL.md lean and push detail there.
- Comments explain *why*, especially for non-obvious workarounds. A comment
  restating the code is noise.

## Adding things

- **A reward:** decorate a function in `scripts/train_grpo.py` with
  `@reward(name, doc)`; signature is `(completions, **kwargs)` returning
  `list[float | None]` (`None` = not applicable). Verify with `--test-rewards`.
- **A recipe:** copy the closest file in `recipes/`, then `--dry-run` and
  `--smoke-test` it before committing.
- **A skill:** new directory under `skills/` with SKILL.md; no manifest edit
  needed, skills are discovered by directory.
