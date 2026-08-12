#!/bin/bash
# TR SWEEP (Test 2). Holds the winning 32L config fixed and varies ONLY --target-tr.
# Reference: TR 1.25e-4 -> KL 0.4493 (from a 0.7355 baseline), 0.80% of trits moved, 360 steps.
# Now meaningful for the first time: earlier runs were dominated by the pre-TALR burst, so TR barely mattered
# (5e-4 and 1.25e-4 gave nearly identical results). With base latent-lr sized to the coverage, TALR is in real
# two-sided control. Annealed schedule kept (--tr-final-frac 0.2) since that is what we would deploy.
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
export TERNARY_BLOCK_SIZE=64
ORIG=/home/kyler/Documents/Model-to-Ternary/output_4b/untied_4b
SRC=output_4bpipe
W=$SRC/trsweep
mkdir -p "$W"
source ./lib_timing.sh
echo "########## TR SWEEP — $(date) ##########"
for TR in 5e-5 5e-4; do
  OUT=$W/tr${TR}/modified_model
  stage "TR $TR"
  [ -f "$W/tr${TR}/.done" ] && { echo "=== [skip] $TR ==="; continue; }
  mkdir -p "$W/tr${TR}"
  systemd-run --user --scope --quiet -p MemoryHigh=47G -p MemoryMax=50G -p MemorySwapMax=0 \
    -E TERNARY_BLOCK_SIZE=64 -E ORIG_MODEL=$ORIG -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ./.venv/bin/python -m torch.distributed.run --nproc_per_node=1 --standalone \
      src/e2e_qp_distill.py --train --student-path $SRC/modified_model --orig-config-path output_4b/untied_4b \
      --calib $SRC/calibration_data.json --teacher-cache $SRC/teacher_topk.pt --out "$OUT" \
      --seq 2560 --epochs 3 --max-samples 120 \
      --lr 0 --latent-lr 5e-7 --latent-init fp-spread --fp-model $SRC/rotbase/modified_model \
      --latent-offload --tw-layer-stride 1 --latent-warmup-steps 20 \
      --target-tr "$TR" --tr-every 5 --tr-final-frac 0.2 \
      --lr-schedule linear --scale-ema-decay 0 --select final --loss-fn cakld --decision-gamma 2 \
      --ce-weight 0.1 --ce-positions 128 --train-weights down --scale-qat-bits 8 \
      --heldout-n 4 --eval-every 40 || { echo "  TR $TR FAILED"; continue; }
  touch "$W/tr${TR}/.done"; echo "  >>> TR $TR done"
done
echo; echo "=== TR SWEEP RESULT (ref TR 1.25e-4 = 0.4493, 0.80% moved) ==="
for TR in 5e-5 5e-4; do
  echo "  TR $TR: $(grep -aoE '\[held-out\] step 360 KL=[0-9.]+ flips=[0-9.]+% assign-moved=[0-9.]+%' $W/tr${TR}.log 2>/dev/null | tail -1)"
done
stage_end
) > "${LOG_FILE:-logs/tr_sweep.log}" 2>&1 &
echo "TR sweep started (PID $!)."
