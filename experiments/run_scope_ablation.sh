#!/bin/bash
# SCOPE ABLATION @8 layers (stride 4). Which single projection is worth training, and does chaining beat it?
# ARMS: down / gate / up / attn individually, then SEQUENTIAL down->up->gate->attn (each warm-starting
# from the previous arm's saved model).
# The three MLP projections are IDENTICALLY sized (0.189B latents @8L) so they share lr 8e-7 — a clean
# three-way comparison. attn is 0.630B (3.3x) so it gets its own calibrated lr.
# Everything else fixed: TR 1.25e-4 annealed, 240 steps, scales FROZEN, fp-spread init, grad-release, offload.
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
SRC=output_4bpipe; W=$SRC/ablation; mkdir -p "$W"
source ./lib_timing.sh
echo "########## SCOPE ABLATION @8L — $(date) ##########"
run_arm () {  # tag scope lr student_path
  local OUT=$W/$1/modified_model
  stage "$1 (scope=$2 lr=$3 from $(basename $(dirname $4)))"
  [ -f "$W/$1/.done" ] && { echo "=== [skip] $1 ==="; return 0; }
  mkdir -p "$W/$1"
  systemd-run --user --scope --quiet -p MemoryHigh=40G -p MemoryMax=44G -p MemorySwapMax=0 \
    -E TERNARY_BLOCK_SIZE=64 -E ORIG_MODEL=$ORIG -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    ./.venv/bin/python -m torch.distributed.run --nproc_per_node=1 --standalone \
      src/e2e_qp_distill.py --train --student-path "$4" --orig-config-path output_4b/untied_4b \
      --calib $SRC/calibration_data.json --teacher-cache $SRC/teacher_topk.pt --out "$OUT" \
      --seq 2560 --epochs 2 --max-samples 120 \
      --lr 0 --latent-lr "$3" --latent-init fp-spread --fp-model $SRC/rotbase/modified_model \
      --latent-offload --latent-grad-release --tw-layer-stride 4 --latent-warmup-steps 20 \
      --target-tr 1.25e-4 --tr-every 5 --tr-final-frac 0.2 \
      --lr-schedule linear --scale-ema-decay 0 --select final --loss-fn cakld --decision-gamma 2 \
      --ce-weight 0.1 --ce-positions 128 --train-weights "$2" --scale-qat-bits 8 \
      --heldout-n 4 --eval-every 40 || { echo "  $1 FAILED"; return 1; }
  touch "$W/$1/.done"; echo "  >>> $1 done"
}
SK=$SRC/modified_model            # the raw block-AP skeleton (all arms start here except the chain)
# ── individual arms (all from the same skeleton) ──
run_arm a_down down 8e-7  "$SK" || true
run_arm a_gate gate 8e-7  "$SK" || true
run_arm a_up   up   8e-7  "$SK" || true
run_arm a_attn attn 2.5e-7 "$SK" || true
# ── sequential chain: down -> up -> gate -> attn, each warm-starting from the previous.
# NOTE: stage 1 IS a_down (identical scope/lr/start), so reuse it instead of re-running (saves ~35min).
run_arm s2_up   up   8e-7  "$W/a_down/modified_model"    || true
run_arm s3_gate gate 8e-7  "$W/s2_up/modified_model"     || true
run_arm s4_attn attn 2.5e-7 "$W/s3_gate/modified_model"  || true
echo; echo "=== ABLATION RESULT (baseline 0.7355; best-so-far down@32L 0.4493) ==="
for t in a_down a_gate a_up a_attn s2_up s3_gate s4_attn; do
  k=$(grep -aoE "\[held-out\] step 240 KL=[0-9.]+" "$W/../ablation_$t.log" 2>/dev/null | tail -1)
  echo "  $t: ${k:-see log}"
done
stage_end
) > "${LOG_FILE:-logs/ablation.log}" 2>&1 &
echo "Scope ablation started (PID $!)."
