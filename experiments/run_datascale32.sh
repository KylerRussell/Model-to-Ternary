#!/bin/bash
# ══ DOES THE ASSIGNMENT STAGE SCALE WITH DATA? (down@32L, the only stage that reliably contributes) ══
# Scope scheduling is exhausted: at 32L, `down` alone 0.4493, v1 3-stage chain 0.4488, v2 clean 3-stage
# 0.4495 -- a 0.0007 spread across three very different paths. That is a FLOOR, and the open question is
# whether it is a data floor or a mechanism floor.
#
# CRITICAL: accum=1, world=1 => ONE SEQUENCE PER OPTIMIZER STEP, and steps = epochs x n_samples. So 360
# steps sees exactly 360 sequences no matter how big --max-samples is. Holding steps fixed while raising
# --max-samples does NOT add data, it only changes which subset is drawn. The data axis therefore REQUIRES
# scaling steps, which is why C/D cost proportionally more.
#
#   A 360 steps / 120 uniq x3ep  0.31M tok  <- anchor, must reproduce seq32b t1_down = 0.4521
#   B 360 steps / 360 uniq x1ep  0.92M tok  <- 3x DATA at IDENTICAL compute (the clean A/B)
#   C 1080      / 1080     x1ep  2.76M tok
#   D 3240      / 3240     x1ep  8.29M tok
#
# All arms: same lr 5e-7, same cold-start ramp, same target-tr, from the SAME raw skeleton.
# Per-run held-out is NOT comparable across arms (held_idx = last 4 of the LOADED subset, so each arm
# scores itself on different sequences). It is used ONLY for checkpoint selection here; the cross-arm
# numbers come from the frozen eval2k referee at the end.
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
SRC=output_4bpipe; W=$SRC/dscale; mkdir -p "$W"
source ./lib_timing.sh
echo "########## DATA-SCALING @32L down — $(date) ##########"
run_arm () {  # tag max_samples epochs
  local OUT=$W/$1/modified_model
  stage "$1 (n=$2 epochs=$3 => $(( $2 * 2560 / 1000 ))k tok)"
  [ -f "$W/$1/.done" ] && { echo "=== [skip] $1 ==="; return 0; }
  mkdir -p "$W/$1"
  systemd-run --user --scope --quiet -p MemoryHigh=40G -p MemoryMax=44G -p MemorySwapMax=0 \
    -E TERNARY_BLOCK_SIZE=64 -E ORIG_MODEL=$ORIG -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -E SAVE_MAX_SHARD_GB=2 \
    ./.venv/bin/python -m torch.distributed.run --nproc_per_node=1 --standalone \
      src/e2e_qp_distill.py --train --student-path "$SRC/modified_model" \
      --orig-config-path output_4b/untied_4b \
      --calib $SRC/calibration_data.json --teacher-cache $SRC/teacher_topk.pt --out "$OUT" \
      --seq 2560 --epochs "$3" --max-samples "$2" \
      --lr 0 --latent-lr 5e-7 --latent-init fp-spread --fp-model $SRC/rotbase/modified_model \
      --latent-offload --latent-grad-release --tw-layer-stride 1 --latent-warmup-steps 20 \
      --target-tr 1.25e-4 --tr-every 5 --tr-final-frac 0.2 \
      --lr-schedule linear --scale-ema-decay 0 --select final --loss-fn cakld --decision-gamma 2 \
      --ce-weight 0.1 --ce-positions 128 --train-weights down --scale-qat-bits 8 \
      --heldout-n 4 --eval-every 40 --ckpt-every 0 || { echo "  $1 FAILED"; return 1; }
  touch "$W/$1/.done"; echo "  >>> $1 done"
}
run_arm a_360s_120n  120  3      || exit 1
run_arm b_360s_360n  360  1      || exit 1
run_arm c_1080s      1080 1      || exit 1
run_arm d_3240s      3240 1      || exit 1

echo; echo "########## eval2k on every arm (frozen 1946-seq referee) ##########"
RES=logs/dscale_eval2k.txt
{ echo "data-scaling @32L down — eval2k (NP=1946 SEQ=1024) — $(date)"
  printf "%-14s %-8s %-12s %s\n" "arm" "tokens" "agreement%" "meanKL"; } | tee "$RES"
for a in "a_360s_120n 307k" "b_360s_360n 922k" "c_1080s 2765k" "d_3240s 8294k"; do
  set -- $a; NAME=$1; TOK=$2; M=$W/$NAME/modified_model
  [ -f "$M/model.safetensors.index.json" ] || { printf "%-14s %-8s %-12s %s\n" "$NAME" "$TOK" "ABSENT" "-" | tee -a "$RES"; continue; }
  systemd-run --user --scope --quiet -p MemoryHigh=38G -p MemoryMax=42G -p MemorySwapMax=0 \
      -E TERNARY_BLOCK_SIZE=64 \
      env ORIG="$ORIG" FP_DIR="$SRC/rotbase/modified_model" E2E_MODEL="$M" \
      EVAL_DATA=output_4b/eval2k.json NP=1946 SEQ=1024 \
      ./.venv/bin/python src/kl_flips_eval.py > "logs/dscale_e2k_${NAME}.log" 2>&1 \
      || echo "  (eval failed for $NAME)"
  AG=$(grep -aoE "top-1 agreement *: *[0-9.]+" "logs/dscale_e2k_${NAME}.log" | grep -oE "[0-9.]+$" | tail -1)
  KL=$(grep -aoE "mean KL\(fp\|\|tern\) *: *[0-9.]+" "logs/dscale_e2k_${NAME}.log" | grep -oE "[0-9.]+$" | tail -1)
  printf "%-14s %-8s %-12s %s\n" "$NAME" "$TOK" "${AG:-NA}" "${KL:-NA}" | tee -a "$RES"
done
echo; cat "$RES"
stage_end
) > "${LOG_FILE:-logs/dscale.log}" 2>&1 &
echo "datascale started (PID $!)."
