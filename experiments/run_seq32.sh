#!/bin/bash
# ★ SEQUENTIAL SCOPE @ 32 LAYERS — combines the two levers that were measured to work:
#     coverage 8L->32L        = -0.022 KL  (0.4718 -> 0.4493, down scope)
#     sequential scope        = -0.038 KL  (down 0.4935 -> +up 0.4555, @8L)
# Both are memory-cheap: only ONE projection's latents are resident at a time, so the peak stays at the
# down@32L level (~26GB) no matter how many stages are chained. That is what makes this possible where
# simultaneous mlp@32L was OOM-killed at 48GB (§8ag).
# Order follows the @8L ablation: down (best MLP) -> up -> gate -> attn (best absolute, largest, last).
# Each stage warm-starts from the previous stage's SAVED model. Scales FROZEN throughout (V-phase).
# lr per stage from the §8z rule, calibrated at 32L: MLP 0.755B -> 5e-7 ; attn 2.52B -> 1.2e-7.
# --ckpt-every 0 is ESSENTIAL: the default (100) triggers a mid-training save_student(), which builds the
# whole fp16 model dict (~10.6GB) + save_file's serialisation copy = a ~20GB SPIKE on top of ~30GB of
# training state -> crosses the cgroup limit and wedges the run in D state at exactly step 100. We use
# --select final, so mid-training checkpoints are useless here anyway.
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
SRC=output_4bpipe; W=$SRC/seq32; mkdir -p "$W"
source ./lib_timing.sh
echo "########## SEQUENTIAL SCOPE @32L — $(date) ##########"
echo "  refs: down@32L 0.4493 (best so far) | chain@8L down->up 0.4555 | baseline 0.7355"
run_stage () {  # tag scope lr student [stride]
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
      --seq 2560 --epochs 3 --max-samples 120 \
      --lr 0 --latent-lr "$3" --latent-init fp-spread --fp-model $SRC/rotbase/modified_model \
      --latent-offload --latent-grad-release --tw-layer-stride "${5:-1}" --latent-warmup-steps 20 \
      --target-tr 1.25e-4 --tr-every 5 --tr-final-frac 0.2 \
      --lr-schedule linear --scale-ema-decay 0 --select final --loss-fn cakld --decision-gamma 2 \
      --ce-weight 0.1 --ce-positions 128 --train-weights "$2" --scale-qat-bits 8 \
      --heldout-n 4 --eval-every 40 --ckpt-every 0 || { echo "  $1 FAILED"; return 1; }
  touch "$W/$1/.done"; echo "  >>> $1 done -> $OUT"
}
# STAGE 3 (gate) DROPPED: the @8L chain showed gate adds nothing after down->up (0.4555 -> 0.4580, i.e.
# marginally WORSE). The MLP projections are largely substitutable, so two stages capture the benefit.
run_stage t1_down down 5e-7   "$SRC/modified_model"        || exit 1
run_stage t2_up   up   5e-7   "$W/t1_down/modified_model"  || exit 1
# attn@32L = 2.52B latents ~= 77GB by the measured 29.1 B/latent model -> WILL NOT FIT.
# Run it at stride 2 (16 layers, 1.26B ~= 41GB) so the stage is at least possible; noted as a coverage
# asymmetry in the result rather than silently dropped.
run_stage t3_attn16 attn 2.5e-7 "$W/t2_up/modified_model" 2 || exit 1
echo; echo "=== SEQ32 RESULT (each stage's step-360 KL) ==="
stage_end
) > "${LOG_FILE:-logs/seq32.log}" 2>&1 &
echo "seq32 started (PID $!)."
