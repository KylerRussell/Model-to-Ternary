#!/bin/bash
# TEST 3 (matched pair): does SCOPE pay at fixed coverage?
#   A) down @8L  = 0.189B latents  (1 projection)
#   B) mlp  @8L  = 0.566B latents  (3 projections, 3x the trainable assignments)
# EVERYTHING else identical: 8 layers (stride 4), TR 1.25e-4 annealed, 360 steps, scales frozen,
# fp-spread init, grad-release, offload. Only --train-weights and the calibrated latent-lr differ.
# lr is CALIBRATED not extrapolated (§8z): want first post-warmup assign-moved <= 0.05%.
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
SRC=output_4bpipe; W=$SRC/scopepair; mkdir -p "$W"
source ./lib_timing.sh
echo "########## SCOPE PAIR @8L — $(date) ##########"
run_one () {  # tag scope lr
  local OUT=$W/$1/modified_model
  stage "$1 (scope=$2, latent-lr=$3)"
  [ -f "$W/$1/.done" ] && { echo "=== [skip] $1 ==="; return 0; }
  mkdir -p "$W/$1"
  systemd-run --user --scope --quiet -p MemoryHigh=40G -p MemoryMax=44G -p MemorySwapMax=0 \
    -E TERNARY_BLOCK_SIZE=64 -E ORIG_MODEL=$ORIG -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ./.venv/bin/python -m torch.distributed.run --nproc_per_node=1 --standalone \
      src/e2e_qp_distill.py --train --student-path $SRC/modified_model --orig-config-path output_4b/untied_4b \
      --calib $SRC/calibration_data.json --teacher-cache $SRC/teacher_topk.pt --out "$OUT" \
      --seq 2560 --epochs 3 --max-samples 120 \
      --lr 0 --latent-lr "$3" --latent-init fp-spread --fp-model $SRC/rotbase/modified_model \
      --latent-offload --latent-grad-release --tw-layer-stride 4 --latent-warmup-steps 20 \
      --target-tr 1.25e-4 --tr-every 5 --tr-final-frac 0.2 \
      --lr-schedule linear --scale-ema-decay 0 --select final --loss-fn cakld --decision-gamma 2 \
      --ce-weight 0.1 --ce-positions 128 --train-weights "$2" --scale-qat-bits 8 \
      --heldout-n 4 --eval-every 40 || { echo "  $1 FAILED"; return 0; }
  touch "$W/$1/.done"; echo "  >>> $1 done"
}
# down@8L at 0.189B: the 32L/0.755B point used 5e-7; 4x fewer latents -> ~1.2e-6 by N^-1.66. Start conservative.
run_one down8 down 8e-7
# mlp@8L at 0.566B: 3x down8's latents -> ~1.5e-7. (9e-7 burst KL to 1.08 — too hot, confirmed.)
run_one mlp8  mlp  1.5e-7
echo; echo "=== SCOPE PAIR RESULT (both 8 layers, TR 1.25e-4, 360 steps) ==="
stage_end
) > "${LOG_FILE:-logs/scope_pair.log}" 2>&1 &
echo "Scope pair started (PID $!)."
