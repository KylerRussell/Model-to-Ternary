#!/bin/bash
# 8-LAYER SEQUENTIAL CHAIN, RE-RUN WITH THE CORRECTED TALR RAMP.
# The first attempt (§8aj) is invalid as a test of "does the whole sequence help": TALR's old 1.3x-per-
# measurement ramp compounded to ~8x within 40 steps, so stages BURST before any correction landed —
# measured gain 17.92 on the attn stage, and every stage showed a damage-then-recover trajectory.
# Now the gain may at most double per 40 steps (--tr-ramp-2x-steps 40), so a stage can still climb to
# whatever flip rate it needs (a fresh axis can start at ZERO flips) but cannot overshoot.
# Chain: down -> up -> gate -> attn, 8 layers (stride 4), 240 steps/stage, scales FROZEN, each stage
# warm-starting from the previous stage's restored-best checkpoint.
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
export PYTHONUNBUFFERED=1; export TERNARY_BLOCK_SIZE=64
ORIG=/home/kyler/Documents/Model-to-Ternary/output_4b/untied_4b
SRC=output_4bpipe; W=$SRC/chain8; mkdir -p "$W"
source ./lib_timing.sh
echo "########## 8L CHAIN (corrected ramp) — $(date) ##########"
echo "  old chain: down 0.4935 -> up 0.4555 -> gate 0.4580 -> attn 0.4536  (all with the bursty ramp)"
run_stage () {  # tag scope lr student
  local OUT=$W/$1/modified_model
  stage "$1 (scope=$2 lr=$3)"
  [ -f "$W/$1/.done" ] && { echo "=== [skip] $1 ==="; return 0; }
  mkdir -p "$W/$1"
  systemd-run --user --scope --quiet -p MemoryHigh=40G -p MemoryMax=44G -p MemorySwapMax=0 \
    -E TERNARY_BLOCK_SIZE=64 -E ORIG_MODEL=$ORIG -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -E SAVE_MAX_SHARD_GB=2 \
    ./.venv/bin/python -m torch.distributed.run --nproc_per_node=1 --standalone \
      src/e2e_qp_distill.py --train --student-path "$4" --orig-config-path output_4b/untied_4b \
      --calib $SRC/calibration_data.json --teacher-cache $SRC/teacher_topk.pt --out "$OUT" \
      --seq 2560 --epochs 2 --max-samples 120 \
      --lr 0 --latent-lr "$3" --latent-init fp-spread --fp-model $SRC/rotbase/modified_model \
      --latent-offload --latent-grad-release --tw-layer-stride 4 --latent-warmup-steps 20 \
      --target-tr 1.25e-4 --tr-every 5 --tr-final-frac 0.2 --tr-ramp-2x-steps 40 \
      --lr-schedule linear --scale-ema-decay 0 --select final --loss-fn cakld --decision-gamma 2 \
      --ce-weight 0.1 --ce-positions 128 --train-weights "$2" --scale-qat-bits 8 \
      --heldout-n 4 --eval-every 40 --ckpt-every 0 || { echo "  $1 FAILED"; return 1; }
  touch "$W/$1/.done"; echo "  >>> $1 done"
}
run_stage c1_down down 8e-7  "$SRC/modified_model"       || exit 1
run_stage c2_up   up   8e-7  "$W/c1_down/modified_model" || exit 1
run_stage c3_gate gate 8e-7  "$W/c2_up/modified_model"   || exit 1
run_stage c4_attn attn 2.5e-7 "$W/c3_gate/modified_model" || exit 1
echo; echo "=== 8L CHAIN RESULT (corrected ramp) ==="
stage_end
) > "${LOG_FILE:-logs/chain8.log}" 2>&1 &
echo "chain8 started (PID $!)."
