#!/usr/bin/env bash
# CPT: inject Taiwanese legal knowledge into a base model.
source "$(dirname "$0")/lib.sh"

banner "1 · 繼續預訓練 (CPT)"
say "幫我用台灣法律語料，對 gemma-3-1b 做繼續預訓練"

step "先看這台機器能跑什麼"
run "preflight.py" uv run --quiet scripts/preflight.py
note "3090，15.6 GB 可用（另外 8 GB 是這台在跑的 embedding 服務）"

step "語料長什麼樣子 — 不看就訓練是在賭"
run "inspect_dataset.py lianghsun/tw-legal-qa-3M" \
    uv run --quiet scripts/inspect_dataset.py lianghsun/tw-legal-qa-3M -n 30
note "raw text，沒有對話結構 — 這正是 CPT 要的形狀"

step "全參數還是 LoRA？讓數字決定"
run "plan_memory.py --model gemma-3-1b-pt" \
    uv run --quiet scripts/plan_memory.py --model google/gemma-3-1b-pt --seq-len 1024 --batch 2 --vram 15.6
note "1B 全參數只要 8 GB — 放得下就不必用 LoRA"

step "開始訓練 · 全參數 · 20 步"
run_stage "train.py --stage cpt" \
  "$PY" scripts/train.py --config '{"stage":"cpt",
 "model":{"name_or_path":"google/gemma-3-1b-pt","dtype":"bfloat16","attn_implementation":"sdpa"},
 "dataset":{"sources":[{"path":"lianghsun/tw-legal-qa-3M","split":"train","keep":["text"],"max_samples":2000}]},
 "tuning":{"method":"full"},
 "train":{"output_dir":"/home/liang/out/demo_pt","max_length":1024,"packing":true,"bf16":true,
   "gradient_checkpointing":true,"per_device_train_batch_size":2,"gradient_accumulation_steps":1,
   "max_steps":20,"logging_steps":5,"save_strategy":"no","save_final_model":false}}'

ok "loss 有下降，峰值 8.3 GB — 跟預估的 8.1 差 0.2"
ok "但它現在不會聊天了：CPT 只教知識，沒教怎麼回答。下一步是 SFT。"
stage_verdict "CPT" "接著跑 ./2-sft.sh" "先檢查 GPU 有沒有殘留行程：nvidia-smi --query-compute-apps=pid,used_memory --format=csv"
