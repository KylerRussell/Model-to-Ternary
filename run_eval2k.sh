#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# TRUE eval2k on every model from the pipeline-validation campaign.
#
# WHY: run_full_pipeline.sh's Gate A scored each model on its OWN held-out (`$WORK/calib_eval.json`),
# which is drawn from that run's own training mixture. That measures "does the model fit its own
# distribution" and is blind to the failure we are hunting: V1 (50% chat throughout) and FINAL (chat only
# at the last pass) BOTH fit their own held-out, yet the frozen eval2k referee separated them by 7.7 pt
# (72.77 vs 80.46) — that gap IS the generic-capability loss. §7 also documents fixed-eval FLOORING while
# own-held-out kept falling. Only the frozen same-mixture draw is comparable to the 77% gate / prior runs.
#
# eval2k = kl_flips_eval.py against output_4b/eval2k.json (frozen 1946-seq draw, seed 9001), NP=1946,
# SEQ=1024 — exactly as run_final_chain.sh invoked it.
#
# REFERENCE POINTS (§8b/§8c):  V1 (chat throughout) 72.77 · combined-2560 79.76 · FINAL 80.46 · gate >=77
#
#   bash run_eval2k.sh
# ──────────────────────────────────────────────────────────────────────────────
(
# SINGLE-INSTANCE GUARD. A duplicate launch (two watchers both firing) once ran this script twice: two
# independent workers with 20.8GB PRIVATE each, 48.9GB PSS on a 60GB host, both writing the SAME --out.
# Refuse to start if a trainer is already live. MUST match comm=="python" AND the args via ps/awk: plain
# `pgrep -f` also matches any SHELL whose command text contains the pattern (e.g. a terminal running a grep
# for it), which false-fired and blocked a legitimate launch; and `pgrep -x python -f PATTERN` is invalid
# (pgrep accepts only one pattern) so it fails OPEN. Root cause of the original incident: `kill $!` after
# `setsid`,
# which forks when not already a process-group leader, so $! was not the running script.
if ps -eo comm=,args= | awk '$1=="python" && /e2e_qp_distill\.py --train/{f=1} END{exit !f}'; then
  echo "FATAL: a training process is already running -- refusing to start a second one."; exit 1
fi
set -e
export PYTHONUNBUFFERED=1
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
ORIG_MODEL=${ORIG_MODEL:-$MODEL_4B}
export ORIG_MODEL
export TERNARY_BLOCK_SIZE=${BLK:-64}
ROT=${ROT:-output_4bpipe/rotbase/modified_model}
EVAL2K=${EVAL2K:-output_4b/eval2k.json}
RES=${RES:-logs/eval2k_campaign.txt}

TOTAL_GB=$(free -g | awk '/^Mem:/{print $2}')
DEF_MAX=$(( TOTAL_GB>=48 ? TOTAL_GB-18 : (TOTAL_GB>=24 ? TOTAL_GB-10 : TOTAL_GB*2/3) ))
MEM_MAX=${MEM_MAX:-${DEF_MAX}G}; MM_NUM=${MEM_MAX%G}
MEM_HIGH=${MEM_HIGH:-$(( MM_NUM>8 ? MM_NUM-4 : MM_NUM ))G}
systemd-run --user --scope --quiet -p MemoryMax=64M -p MemorySwapMax=0 /bin/true 2>/dev/null \
  || { echo "FATAL: no systemd --user scope"; exit 1; }
MEMGUARD="systemd-run --user --scope --quiet -p MemoryHigh=$MEM_HIGH -p MemoryMax=$MEM_MAX -p MemorySwapMax=0 -E TERNARY_BLOCK_SIZE=${BLK:-64}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[ -f "$EVAL2K" ] || { echo "FATAL: $EVAL2K missing"; exit 1; }
# most important first, so a crash still leaves the decisive number
MODELS=("curriculum   output_4bpipe/curr/e2e2/modified_model"
        "cold-run     output_4bpipe/e2eqp/modified_model"
        "pass2        output_4bpipe/e2eqp2/modified_model")

{ echo "TRUE eval2k (frozen 1946-seq referee, NP=1946 SEQ=1024) — $(date)"
  echo "refs: V1(chat-throughout) 72.77 · combined-2560 79.76 · FINAL 80.46 · gate >=77"
  printf "%-12s %-12s %s\n" "model" "agreement%" "meanKL"; } | tee "$RES"

for m in "${MODELS[@]}"; do
  set -- $m; NAME=$1; M=$2
  [ -f "$M/model.safetensors.index.json" ] || { printf "%-12s %-12s %s\n" "$NAME" "ABSENT" "-" | tee -a "$RES"; continue; }
  LOG=logs/eval2k_${NAME}.log
  echo "--- $NAME ---"
  $MEMGUARD env ORIG="$ORIG_MODEL" FP_DIR="$ROT" E2E_MODEL="$M" \
      EVAL_DATA="$EVAL2K" NP=1946 SEQ=1024 \
      ./.venv/bin/python src/kl_flips_eval.py > "$LOG" 2>&1 || echo "  (eval failed for $NAME)"
  AG=$(grep -aoE "top-1 agreement *: *[0-9.]+" "$LOG" | grep -oE "[0-9.]+$" | tail -1)
  KL=$(grep -aoE "mean KL\(fp\|\|tern\) *: *[0-9.]+" "$LOG" | grep -oE "[0-9.]+$" | tail -1)
  printf "%-12s %-12s %s\n" "$NAME" "${AG:-NA}" "${KL:-NA}" | tee -a "$RES"
done
echo; cat "$RES"
echo "########## eval2k DONE -> $RES ##########"
) > "${LOG_FILE:-logs/eval2k_campaign.log}" 2>&1 &
echo "eval2k campaign started in the background (PID $!)."
