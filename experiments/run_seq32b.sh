#!/bin/bash
# ★ SEQUENTIAL SCOPE @ 32 LAYERS — combines the two levers that were measured to work:
#     coverage 8L->32L        = -0.022 KL  (0.4718 -> 0.4493, down scope)
#     sequential scope        = -0.038 KL  (down 0.4935 -> +up 0.4555, @8L)
# Both are memory-cheap: only ONE projection's latents are resident at a time, so the peak stays at the
# down@32L level (~26GB) no matter how many stages are chained. That is what makes this possible where
# simultaneous mlp@32L was OOM-killed at 48GB (§8ag).
# Order: down (best MLP) -> up -> attn (best absolute, largest, last). GATE IS DROPPED, see below.
# Each stage warm-starts from the previous stage's SAVED model. Scales FROZEN throughout (V-phase).
# MLP lr stays 5e-7. The cold-start ramp (--latent-lr-ramp-steps, default 4x--tr-every) now phases the latent
# lr in from 0 and TALR is gated until it completes, so the base lr only needs to be within ~1 order of
# magnitude -- it is no longer the burst trigger it was. Paired @8L proof: 0.4935 -> 0.4891 from 11% fewer
# flips in 17% fewer effective steps.
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
SRC=output_4bpipe; W=$SRC/seq32b; mkdir -p "$W"
source ./lib_timing.sh
echo "########## SEQUENTIAL SCOPE @32L (v2: cold-start ramp + recalibrated attn lr) — $(date) ##########"
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
# GATE DROPPED. Re-confirmed on the corrected @8L chain: gate after up moved 0.517% of its assignments and
# finished at EXACTLY its entry KL (0.4543 -> 0.4543), its best having been recorded during the ramp before
# anything moved. Mechanism: SwiGLU is down(silu(gate(x))*up(x)), so gate and up multiply -- tuning up first
# makes the very assignments gate must move away from. Training them JOINTLY does beat sequential (0.4538 in
# ONE stage vs 0.4543 in two), but joint {gate,up}@32L = 1510M latents ~= 45.7GB > the 42GB host ceiling and
# would need --latent-state-nvme (+12GB memmap, ~3x step time). Not worth the operational risk at 27B scale,
# so gate is simply forgone here.
run_stage t1_down down 5e-7   "$SRC/modified_model"        || exit 1
run_stage t2_up   up   5e-7   "$W/t1_down/modified_model"  || exit 1
# attn stays at stride 2 (16 layers): that configuration is EMPIRICALLY known to fit (the v1 run completed
# at this setting). Full 32L attn is not attempted -- the 8L sample gives 40 modules over 8 layers (5/layer,
# not 4), so layer widths are non-uniform and neither x4 extrapolation nor the old 2.52B note is trustworthy.
# lr 2.5e-7 -> 7.5e-7: v1's attn stage was starved. Its TALR gain climbed to 41 while the measured flip rate
# sat at ZERO, i.e. the servo spent the whole run trying to reach a usable lr and never got there. 7.5e-7 is
# the servo's own answer, read from the @8L attn stage AFTER it plateaued (5e-6 clamped to gain 0.13-0.22).
# NB: a still-CLIMBING gain is not a calibration reading -- inferring 5e-6 from v1's unplateaued 41 cost a
# stage (KL 0.4538 -> 0.5212, 2.1% of assignments displaced).
run_stage t3_attn16 attn 7.5e-7 "$W/t2_up/modified_model" 2 || exit 1
echo; echo "=== SEQ32 RESULT (each stage's step-360 KL) ==="
stage_end
) > "${LOG_FILE:-logs/seq32b.log}" 2>&1 &
echo "seq32b started (PID $!)."
