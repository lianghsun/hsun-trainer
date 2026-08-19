#!/usr/bin/env bash
# Eval: the only stage that answers "did it get better".
source "$(dirname "$0")/lib.sh"

VLLM_DIR="$HOME/evaltools"
PORT=${PORT:-8011}
MODEL=${MODEL:-google/gemma-3-1b-it}
SERVED=demo-model

banner "3 · 評測 (Twinkle Eval)"
say "訓練完怎麼知道有沒有變好？先把台灣的題庫拉下來"

step "把 Hub 上的繁中題庫轉成 Twinkle Eval 的格式"
run "make_eval_config.py --bench Formosa-bench tw-legal-benchmark-v1" \
    uv run --quiet scripts/make_eval_config.py \
      --model "$SERVED" --base-url "http://localhost:$PORT/v1" \
      --bench lianghsun/Formosa-bench lianghsun/tw-legal-benchmark-v1 \
      --out eval
note "地理、政府、歷史、社會、法律 — 全是台灣在地題目，不是翻譯的英文題庫"

step "起一個 OpenAI 相容的服務（Twinkle Eval 打 API，不載權重）"
printf "  ${D}⎿${R}  ${BLUE}vllm serve %s --port %s${R}\n" "$MODEL" "$PORT"
if [ -x "$VLLM_DIR/.venv/bin/vllm" ]; then
  ( "$VLLM_DIR/.venv/bin/vllm" serve "$MODEL" --served-model-name "$SERVED" \
      --port "$PORT" --max-model-len 2048 --gpu-memory-utilization 0.55 \
      > /tmp/vllm.log 2>&1 & )
  printf "     ${GREY}啟動中"
  for i in $(seq 1 90); do
    curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break
    printf "."; sleep 2
  done
  printf "${R}\n"
  curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1 \
    && note "服務就緒" \
    || { warn "vLLM 未就緒，跳過實際評測"; SKIP=1; }
else
  warn "這台沒有 vllm，跳過實際評測（設定檔已產生，可在有服務的機器上跑）"; SKIP=1
fi

if [ -z "$SKIP" ]; then
  step "跑評測 · 209 題法律 + 349 題台灣常識"
  printf "  ${D}⎿${R}  ${BLUE}twinkle-eval --config eval/config.yaml${R}\n\n"
  ( cd eval && "$VLLM_DIR/.venv/bin/twinkle-eval" --config config.yaml 2>&1 \
      | grep -vE "^\s*$" | tail -40 | sed "s/^/     /" )
  note "這就是基準線。沒有它，之後任何「變好了」都是感覺"
  warn "抽取方法選錯會安靜地報出錯 6 倍的分數 — box 適合會寫 \\boxed{} 的模型，原廠 instruct 要用 pattern"
  pkill -f "vllm serve" 2>/dev/null
else
  step "設定檔已就緒"
  run "cat eval/config.yaml" head -20 eval/config.yaml
fi

done_banner "評測完成" "訓練前先量一次，訓練後再量一次 — 這才是完整的流程"
