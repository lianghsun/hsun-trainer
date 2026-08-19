#!/usr/bin/env bash
# SFT: teach the model how to answer, not what to know.
source "$(dirname "$0")/lib.sh"

banner "2 · 監督式微調 (SFT)"
say "改用推理對話資料，教 gemma-3-1b 怎麼回答"

step "這批資料是什麼形狀？"
run "inspect_dataset.py twinkle-ai/tw-reasoning-instruct-50k" \
    uv run --quiet scripts/inspect_dataset.py twinkle-ai/tw-reasoning-instruct-50k -n 30
warn "是 ShareGPT 的 conversations（from/value），不是 TRL 要的 messages（role/content）"
note "直接餵下去，TRL 會認得欄位名、解不出內容，然後用 0 筆資料「訓練成功」"

step "所以要開轉換旗標 — 這是 recipe 裡一行的事"
printf "  ${D}⎿${R}  ${BLUE}sharegpt_to_messages: true${R}\n"
pause 1.2

step "開始訓練 · 全參數 · 20 步"
printf "  ${D}⎿${R}  ${BLUE}train.py --stage sft${R}\n\n"
"$PY" scripts/train.py --config '{"stage":"sft",
 "model":{"name_or_path":"google/gemma-3-1b-it","dtype":"bfloat16","attn_implementation":"sdpa"},
 "dataset":{"sources":[{"path":"twinkle-ai/tw-reasoning-instruct-50k","split":"train",
   "sharegpt_to_messages":true,"sharegpt_column":"conversations","keep":["messages"],"max_samples":2000}]},
 "tuning":{"method":"full"},
 "train":{"output_dir":"/home/liang/out/demo_sft","max_length":1024,"packing":false,"bf16":true,
   "gradient_checkpointing":true,"per_device_train_batch_size":2,"gradient_accumulation_steps":1,
   "max_steps":20,"logging_steps":5,"save_strategy":"no","save_final_model":false}}' 2>&1 \
 | grep -vE "^(Generating|Loading weights|Adding EOS|Truncating|Tokenizing|Map:)" \
 | sed "s/^/     /"

note "2000 筆全部存活 — 轉換對了；沒開旗標的話這裡會是 0"
note "跟 CPT 的差別：模型換成 -it、資料是對話、而且套了 chat template"
done_banner "SFT 完成" "接著跑 ./3-eval.sh"
