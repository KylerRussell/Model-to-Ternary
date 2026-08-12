#!/bin/bash
# JOINT {gate,up} vs SEQUENTIAL up->gate, at 8 layers.
# run_chain8.sh established that `gate` after `up` contributes EXACTLY ZERO (0.4543 -> 0.4543, best recorded
# during the ramp), reproducing the old chain's net -0.0025. No burst anywhere in that run, so it is a real
# property. Mechanism: SwiGLU computes down(silu(gate(x)) * up(x)) — gate and up are MULTIPLICATIVELY coupled,
# so a stage that tunes `up` against the current gate assignments makes those exact assignments the thing
# `gate` must then move away from. Sequential ordering cannot fix a co-adaptation it just created.
# This arm trains them JOINTLY instead, warm-starting from the SAME c1_down (so the comparison is exactly
# "one joint {gate,up} stage" vs "up then gate", holding the first stage fixed).
#   sequential reference: c2_up 0.4543 -> c3_gate 0.4543 -> c4_attn <pending>
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
SRC=output_4bpipe; W=$SRC/chain8b; C1=$SRC/chain8/c1_down/modified_model; mkdir -p "$W"
source ./lib_timing.sh
echo "########## 8L JOINT {gate,up} CHAIN — $(date) ##########"
echo "  warm-starting from chain8/c1_down (KL 0.4891)"
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
run_stage j2_gateup gateup 8e-7  "$C1"                       || exit 1
run_stage j3_attn   attn   5e-6   "$W/j2_gateup/modified_model" || exit 1
echo; echo "=== 8L JOINT-MLP CHAIN RESULT ==="
stage_end
) > "${LOG_FILE:-logs/chain8b.log}" 2>&1 &
echo "chain8b started (PID $!)."
