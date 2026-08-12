#!/bin/bash
# ══ FULL ASSIGNMENT CHAIN @32L: down -> up -> attn, 3240 steps/stage ═══════════════════════════════
# STAGE 1 (down) IS ARM F, ALREADY DONE: 6244 steps x 6244 unique (16M tok) = 78.35% / KL 0.3637 eval2k.
# That is a better `down` than a fresh 3240-step stage would be (arm D, 8.3M tok = 76.21%), so we warm-start
# from it rather than spending 11h reproducing a worse stage-1.
#
# The 360-step evidence that `up` and `attn` contribute ~0 at 32L is NOT trusted here: the data-scaling
# sweep showed 360 steps is undertrained by ~5 pt of agreement for `down`, so those nulls may have been
# starvation. 3240 steps/stage is the budget the user set.
#   up  @32L  5e-7   (seq32b: servo reached target by step 100 at this base)
#   attn@16L  7.5e-7 stride 2 -- the servo's PLATEAUED answer; 2.5e-7 starved it, 5e-6 burst it.
#     attn is the memory-tight stage: ~1.26B latents x 27.3 B/latent + 4.5GB base ~= 39GB vs the 42GB
#     ceiling. Proven (it ran in seq32/seq32b) but nothing else may run alongside it.
# ABORT PATIENCE 30 IS LOAD-BEARING. The abort threshold is measured in STEPS
# (ho_worse x eval_every >= 100 x abort_patience => 300 steps at the default patience of 3), tuned for
# --eval-every 40 where it means TEN evals. At --eval-every 400 it means ONE, and the first attempt at this
# chain died on it: c2_up ABORTED at step 800 of 3240 ("held-out KL 0.3396 > init 0.3118 for 1 evals"), so
# the `up` null was never tested at the intended budget. Attention's damage-then-recover arc alone runs
# 1000-2000 steps. 30 => 3000 steps of tolerance, effectively off for a 3240-step stage; --select still
# restores the best checkpoint, and with the step-0 ENTRY BASELINE now logged, a stage that never beats its
# entry restores that entry (a true no-op rather than shipping something worse).
# Reference to beat after E2E: scale-only E2E 16M_2ep_colscale_v2 = 80.86% / KL 0.2977.
(
# SINGLE-INSTANCE GUARD. A duplicate launch once ran a script twice: two workers with 20.8GB PRIVATE each,
# 48.9GB PSS on a 60GB host, both writing the SAME --out. Refuse to start if a trainer is already live.
# MUST match comm=="python" AND the args: plain `pgrep -f` also matches any SHELL whose command text
# contains the pattern (e.g. a terminal running a grep for it), which false-fired and blocked a launch.
if ps -eo comm=,args= | awk '$1=="python" && /e2e_qp_distill\.py --train/{f=1} END{exit !f}'; then
  echo "FATAL: a training process is already running -- refusing to start a second one."; exit 1
fi
set -e
export PYTHONUNBUFFERED=1; export TERNARY_BLOCK_SIZE=64
ORIG=/home/kyler/Documents/Model-to-Ternary/output_4b/untied_4b
SRC=output_4bpipe; W=$SRC/chain32; mkdir -p "$W"
F=$SRC/dscale/f_6244s_16M/modified_model
[ -f "$F/model.safetensors.index.json" ] || { echo "FATAL: arm F model missing at $F"; exit 1; }
source ./lib_timing.sh
echo "########## CHAIN32: (down=armF 78.35%) -> up -> attn — $(date) ##########"
run_stage () {  # tag scope lr student [stride]
  local OUT=$W/$1/modified_model
  stage "$1 (scope=$2 lr=$3 stride=${5:-1})"
  [ -f "$W/$1/.done" ] && { echo "=== [skip] $1 ==="; return 0; }
  mkdir -p "$W/$1"
  systemd-run --user --scope --quiet -p MemoryHigh=40G -p MemoryMax=44G -p MemorySwapMax=0 \
    -E TERNARY_BLOCK_SIZE=64 -E ORIG_MODEL=$ORIG -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -E SAVE_MAX_SHARD_GB=2 \
    ./.venv/bin/python -m torch.distributed.run --nproc_per_node=1 --standalone \
      src/e2e_qp_distill.py --train --student-path "$4" --orig-config-path output_4b/untied_4b \
      --calib $SRC/calibration_data.json --teacher-cache $SRC/teacher_topk.pt --out "$OUT" \
      --seq 2560 --epochs 1 --max-samples 3240 \
      --lr 0 --latent-lr "$3" --latent-init fp-spread --fp-model $SRC/rotbase/modified_model \
      --latent-offload --latent-grad-release --tw-layer-stride "${5:-1}" --latent-warmup-steps 20 \
      --target-tr 1.25e-4 --tr-every 5 --tr-final-frac 0.2 \
      --lr-schedule linear --scale-ema-decay 0 --select final --loss-fn cakld --decision-gamma 2 \
      --ce-weight 0.1 --ce-positions 128 --train-weights "$2" --scale-qat-bits 8 \
      --heldout-n 4 --eval-every 200 --ckpt-every 0 --abort-patience 30 || { echo "  $1 FAILED"; return 1; }
  touch "$W/$1/.done"; echo "  >>> $1 done"
}
run_stage c2_up   up   5e-7   "$F"                       || exit 1
run_stage c3_attn attn 7.5e-7 "$W/c2_up/modified_model" 2 || exit 1

echo; echo "########## eval2k: chain stages vs arm F (78.35 / 0.3637) ##########"
RES=logs/chain32_eval2k.txt
{ echo "chain32 @32L — eval2k (NP=1946 SEQ=1024) — $(date)"
  echo "  stage1 = arm F (down, 16M tok) = 78.35 / 0.3637 | target after E2E: 80.86 / 0.2977"
  printf "%-10s %-12s %s\n" "stage" "agreement%" "meanKL"; } | tee "$RES"
for s in c2_up c3_attn; do
  M=$W/$s/modified_model
  [ -f "$M/model.safetensors.index.json" ] || { printf "%-10s %-12s %s\n" "$s" "ABSENT" "-" | tee -a "$RES"; continue; }
  systemd-run --user --scope --quiet -p MemoryHigh=38G -p MemoryMax=42G -p MemorySwapMax=0 -E TERNARY_BLOCK_SIZE=64 \
      env ORIG="$ORIG" FP_DIR="$SRC/rotbase/modified_model" E2E_MODEL="$M" \
      EVAL_DATA=output_4b/eval2k.json NP=1946 SEQ=1024 \
      ./.venv/bin/python src/kl_flips_eval.py > "logs/chain32_e2k_${s}.log" 2>&1 || echo "  (eval failed $s)"
  AG=$(grep -aoE "top-1 agreement *: *[0-9.]+" "logs/chain32_e2k_${s}.log" | grep -oE "[0-9.]+$" | tail -1)
  KL=$(grep -aoE "mean KL\(fp\|\|tern\) *: *[0-9.]+" "logs/chain32_e2k_${s}.log" | grep -oE "[0-9.]+$" | tail -1)
  printf "%-10s %-12s %s\n" "$s" "${AG:-NA}" "${KL:-NA}" | tee -a "$RES"
done
echo; cat "$RES"
stage_end
) > "${LOG_FILE:-logs/chain32.log}" 2>&1 &
echo "chain32 started (PID $!)."
