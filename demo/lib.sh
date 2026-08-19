#!/usr/bin/env bash
# Presentation helpers for the hsun-trainer stage demos.
#
# The visual grammar (prompt -> tool call -> indented result) mirrors how an
# agent session reads, because that is the shape of the work. This is a
# scripted replay, not a live agent, and the header says so - every command
# below really runs and every number on screen is its real output.

export PATH="$HOME/.local/bin:$PATH" HF_HOME="$HOME/hf-cache" TMPDIR="$HOME/tmp"
cd ~/hsun-trainer 2>/dev/null || cd "$(dirname "$0")/.." || exit 1

PY="$(uv python find --script scripts/train.py 2>/dev/null)"
PYG="$(uv python find --script scripts/train_grpo.py 2>/dev/null)"

R=$'\033[0m'; B=$'\033[1m'; D=$'\033[2m'
ORANGE=$'\033[38;5;179m'; GREEN=$'\033[38;5;71m'; BLUE=$'\033[38;5;110m'
RED=$'\033[38;5;167m'; GREY=$'\033[38;5;245m'

PACE=${PACE:-1}                       # PACE=0 removes all pauses (for retakes)
pause() { [ "$PACE" = "0" ] || sleep "${1:-1}"; }

# Typewriter, so the ask lands as something a person said
say() {
  printf "\n${B}> ${R}"
  local s="$1" i
  for ((i=0; i<${#s}; i++)); do
    printf "%s" "${s:i:1}"
    [ "$PACE" = "0" ] || sleep 0.022
  done
  printf "\n"
  pause 1.2
}

banner() {
  clear
  printf "${ORANGE}╭──────────────────────────────────────────────────────────────╮${R}\n"
  printf "${ORANGE}│${R}  ${B}hsun-trainer${R}  ·  %-40s${ORANGE}│${R}\n" "$1"
  printf "${ORANGE}╰──────────────────────────────────────────────────────────────╯${R}\n"
  printf "${D}  scripted replay · every command below really runs${R}\n"
}

# Training stages are run back to back, and a python process that has exited its
# main loop can still hold VRAM for a moment. Waiting is cheaper than an OOM
# three seconds into the next stage - which is exactly how this was found.
wait_for_gpu() {
  local need=${1:-11000} free i
  for i in $(seq 1 30); do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    [ -z "$free" ] && return 0
    [ "$free" -ge "$need" ] && return 0
    [ "$i" = 1 ] && printf "     ${GREY}等待前一個行程釋放 VRAM${R}"
    printf "."
    sleep 2
  done
  printf "\n"
  warn "VRAM 仍不足（${free} MiB）。檢查殘留行程："
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>/dev/null | sed "s/^/     /"
  return 1
}

# step "narration"  -- what this step is for, in the speaker's words
step() { printf "\n${ORANGE}⏺${R} ${B}%s${R}\n" "$1"; pause 0.7; }

# run "label" cmd...  -- show the call, run it, indent the output
run() {
  local label="$1"; shift
  printf "  ${D}⎿${R}  ${BLUE}%s${R}\n" "$label"
  pause 0.4
  "$@" 2>&1 | sed "s/^/     ${GREY}/;s/\$/${R}/"
}

# note "text"  -- the conclusion a human would draw from what just printed
note() { printf "\n     ${GREEN}▸${R} %s\n" "$1"; pause 1.4; }
warn() { printf "\n     ${RED}▸${R} %s\n" "$1"; pause 1.4; }

done_banner() {
  printf "\n${GREEN}  ✓ %s${R}   ${D}%s${R}\n\n" "$1" "${2:-}"
}

# Run a training stage and remember whether it worked. Narration that claims a
# result must never print when the stage failed - a script that says "✓ done"
# over an OOM traceback is the exact failure this project exists to catch.
STAGE_OK=0
run_stage() {
  local label="$1"; shift
  wait_for_gpu 11000
  printf "  ${D}⎿${R}  ${BLUE}%s${R}\n\n" "$label"
  set -o pipefail
  "$@" 2>&1 \
    | grep -vE "^(Generating|Loading weights|Packing|Adding EOS|Truncating|Tokenizing|Map:)" \
    | sed "s/^/     /"
  local rc=${PIPESTATUS[0]}
  set +o pipefail
  STAGE_OK=$([ "$rc" -eq 0 ] && echo 1 || echo 0)
  return 0
}

# ok "text" -- only prints when the stage actually succeeded
ok() { [ "$STAGE_OK" = "1" ] && note "$1"; }

fail_banner() {
  printf "\n${RED}  ✗ %s${R}\n" "$1"
  printf "${D}     %s${R}\n\n" "${2:-}"
}

stage_verdict() {
  if [ "$STAGE_OK" = "1" ]; then done_banner "$1" "$2"; else fail_banner "$1 失敗" "$3"; fi
}
