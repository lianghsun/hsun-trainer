#!/usr/bin/env bash
# hsun-trainer 訓練流程 demo（錄影用：固定進度表 + 即時 VRAM）
export PATH="$HOME/.local/bin:$PATH" HF_HOME="$HOME/hf-cache" TMPDIR="$HOME/tmp"
cd ~/hsun-trainer || exit 1

PY="$(uv python find --script scripts/train.py)"
PYG="$(uv python find --script scripts/train_grpo.py)"
LOG=$(mktemp); trap 'rm -f "$LOG"' EXIT

B=$'\033[1m'; D=$'\033[2m'; G=$'\033[32m'; C=$'\033[36m'; Y=$'\033[33m'; R=$'\033[0m'

NAMES=("環境檢查" "資料檢查" "記憶體規劃" "CPT 繼續預訓練" "SFT 監督式微調" "GRPO 可驗證獎勵")
DESC=("硬體 / 登入 / 版本" "欄位偵測 + token 長度" "full vs LoRA vs QLoRA" "raw text，全參數" "對話，全參數" "LoRA + 3 個 reward")
STATE=(idle idle idle idle idle idle); TIME=(0 0 0 0 0 0); PEAK=("" "" "" "" "" "")
CUR=-1; T0=0


# 依顯示寬度補齊（CJK 佔 2 欄）；printf 的 %-Ns 對中文會算錯
padw() {
  local str="$1" want="$2" w=0 i ch
  for ((i=0; i<${#str}; i++)); do
    ch="${str:i:1}"
    if [[ "$ch" > "\u2e7f" ]]; then w=$((w+2)); else w=$((w+1)); fi
  done
  printf "%s" "$str"
  for ((i=w; i<want; i++)); do printf " "; done
}

vram() { nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '{printf "%.1f/%.0f GB", $1/1024, $2/1024}'; }
bar() { local u t p i s=""; read u t < <(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr ',' ' ')
        p=$(( ${u:-0} / 10 )); for ((i=0;i<10;i++)); do [ $i -lt $p ] && s+="█" || s+="░"; done; printf "%s %3s%%" "$s" "${u:-0}"; }

draw() {
  printf '\033[H\033[J'
  printf "%s╭────────────────────────────────────────────────────────────────╮%s\n" "$C" "$R"
  printf "%s│%s  %shsun-trainer%s · CPT → SFT → GRPO · RTX 3090                 %s│%s\n" "$C" "$R" "$B" "$R" "$C" "$R"
  printf "%s╰────────────────────────────────────────────────────────────────╯%s\n\n" "$C" "$R"
  local i mark col el
  for i in "${!NAMES[@]}"; do
    case "${STATE[$i]}" in
      done) mark="${G}✓${R}"; col="$R" ;;
      run)  mark="${Y}▸${R}"; col="$B" ;;
      fail) mark="${Y}✗${R}"; col="$R" ;;
      *)    mark="${D}·${R}"; col="$D" ;;
    esac
    el=""
    [ "${STATE[$i]}" = "run" ] && el=$(printf "%4ds  %s" $(( $(date +%s) - T0 )) "$(bar)")
    [ "${STATE[$i]}" = "done" ] && el=$(printf "%4ds  %s" "${TIME[$i]}" "${PEAK[$i]}")
    printf "  %b  %s%d  %s%s %s%s%s %s\n" "$mark" "$col" "$i" "$(padw "${NAMES[$i]}" 20)" "$R" "$D" "$(padw "${DESC[$i]}" 26)" "$R" "$el"
  done
  printf "\n  %sVRAM%s %s\n" "$D" "$R" "$(vram)"
  printf "  %s────────────────────────── 輸出 ──────────────────────────%s\n" "$D" "$R"
  tail -n 7 "$LOG" 2>/dev/null | cut -c1-100 | sed "s/^/  ${D}/;s/\$/${R}/"
}

stage() { # idx  command...
  local idx=$1; shift
  CUR=$idx; STATE[$idx]=run; T0=$(date +%s); : > "$LOG"
  ( "$@" >> "$LOG" 2>&1 ) & local pid=$!
  while kill -0 $pid 2>/dev/null; do draw; sleep 1; done
  wait $pid; local rc=$?
  TIME[$idx]=$(( $(date +%s) - T0 ))
  PEAK[$idx]=$(grep -oE "peak VRAM: [0-9.]+ GB" "$LOG" | tail -1 | sed 's/peak VRAM: /峰值 /')
  [ $rc -eq 0 ] && STATE[$idx]=done || STATE[$idx]=fail
  draw; sleep 2
}

CPT_CFG='{"stage":"cpt","model":{"name_or_path":"google/gemma-3-1b-pt","dtype":"bfloat16","attn_implementation":"sdpa"},
 "dataset":{"sources":[{"path":"lianghsun/tw-legal-qa-3M","split":"train","keep":["text"],"max_samples":2000}]},
 "tuning":{"method":"full"},
 "train":{"output_dir":"/home/liang/out/live_cpt","report_to":["trackio"],"project":"hsun-trainer-demo","max_length":1024,"packing":true,"bf16":true,
   "gradient_checkpointing":true,"per_device_train_batch_size":2,"gradient_accumulation_steps":1,
   "max_steps":20,"logging_steps":2,"save_strategy":"no","save_final_model":false}}'
SFT_CFG='{"stage":"sft","model":{"name_or_path":"google/gemma-3-1b-it","dtype":"bfloat16","attn_implementation":"sdpa"},
 "dataset":{"sources":[{"path":"twinkle-ai/tw-reasoning-instruct-50k","split":"train","sharegpt_to_messages":true,
   "sharegpt_column":"conversations","keep":["messages"],"max_samples":2000}]},
 "tuning":{"method":"full"},
 "train":{"output_dir":"/home/liang/out/live_sft","report_to":["trackio"],"project":"hsun-trainer-demo","max_length":1024,"packing":false,"bf16":true,
   "gradient_checkpointing":true,"per_device_train_batch_size":2,"gradient_accumulation_steps":1,
   "max_steps":20,"logging_steps":2,"save_strategy":"no","save_final_model":false}}'
GRPO_CFG='{"stage":"grpo","model":{"name_or_path":"google/gemma-3-1b-it","dtype":"bfloat16","attn_implementation":"sdpa"},
 "dataset":{"sources":[{"path":"twinkle-ai/tw-math-reasoning-2k","split":"train","max_samples":64}]},
 "grpo":{"dataset_kind":"math","question_field":"problem_zhtw","ground_truth_field":"answer",
   # 256 is deliberately too small: it makes the clipped-ratio guard fire, which
   # is the point of this stage. Raise to 3072 to watch GRPO actually train.
   "num_generations":4,"max_completion_length":256,
   "rewards":[{"name":"accuracy_math","weight":3.0},{"name":"format_boxed","weight":0.5},{"name":"zhtw_purity","weight":1.0}]},
 "tuning":{"method":"lora","lora":{"r":16,"alpha":32,"dropout":0.0,"target_modules":"all-linear"}},
 "train":{"output_dir":"/home/liang/out/live_grpo","report_to":["trackio"],"project":"hsun-trainer-demo","bf16":true,"gradient_checkpointing":true,
   "per_device_train_batch_size":4,"gradient_accumulation_steps":1,"max_steps":5,"logging_steps":1,"save_strategy":"no"}}'

draw
stage 0 uv run --quiet scripts/preflight.py
stage 1 uv run --quiet scripts/inspect_dataset.py twinkle-ai/tw-math-reasoning-2k -n 50 --tokenizer google/gemma-3-1b-it
stage 2 uv run --quiet scripts/plan_memory.py --model google/gemma-3-1b-it --seq-len 1024 --batch 2 --vram 15.6
stage 3 "$PY"  scripts/train.py      --config "$CPT_CFG"
stage 4 "$PY"  scripts/train.py      --config "$SFT_CFG"
stage 5 "$PYG" scripts/train_grpo.py --config "$GRPO_CFG"

printf "\n  %sTrackio%s  儀表板： trackio show   （或 http://localhost:7860）\n" "$D" "$R"
printf "  %s全部完成%s   總計 %ds\n\n" "$G$B" "$R" "$(( ${TIME[0]}+${TIME[1]}+${TIME[2]}+${TIME[3]}+${TIME[4]}+${TIME[5]} ))"
