#!/bin/bash
# ══ STAGE 0, REC 1: ALL-MLP JOINT WITHIN AN 8-LAYER GROUP, SEQUENTIAL ACROSS GROUPS (Block-AP style) ══
# THE DECISIVE EXPERIMENT. Everything we know says scope-sequential is broken (a 2nd MLP projection after
# the 1st damages and never recovers, 4x confirmed) while scope-JOINT works where it fits (gateup@8L beat
# sequential up->gate in HALF the compute; all-MLP@8L 0.4631 beat down-only 0.4718). The decomposition that
# respects this is: JOINT across scopes WITHIN a layer group (SwiGLU's gate/up multiply => must be joint),
# SEQUENTIAL across groups (cross-layer coupling is additive through the residual stream => separable).
#
# MEMORY: all-MLP on 8 of 32 layers = 3 x 8 x (2560x9216) = 566M latents x 27.3 B/latent + 4.5GB = 20.0 GB.
# Fits today at the CURRENT 27.3 B/latent -- no optimizer-state work needed to run this test.
#
# COMPUTE-MATCHED to arm F on purpose: 4 groups x 1561 steps = 6244 steps = 16M token-steps, exactly the
# budget that gave down-only 78.35%. So a win here is a win at EQUAL compute, not a bigger-budget artifact.
#
# DECISION THRESHOLD (from the report): if group-joint does not clearly beat down-only 78.35%, the joint-
# scope program is not worth the memory engineering -- ship scale-only + down and spend compute on data.
#
# --abort-patience is DISABLED. Every stage here dips before recovering (attn and E2E both did), and the
# abort is threshold-in-STEPS: it killed E2E at step 3000/12488 while it was improving monotonically.
(
if ps -eo comm=,args= | awk '$1=="python" && /e2e_qp_distill\.py --train/{f=1} END{exit !f}'; then
  echo "FATAL: a training process is already running -- refusing to start a second one."; exit 1
fi
set -e
export PYTHONUNBUFFERED=1; export TERNARY_BLOCK_SIZE=64
ORIG=/home/kyler/Documents/Model-to-Ternary/output_4b/untied_4b
SRC=output_4bpipe; W=$SRC/groupjoint; mkdir -p "$W"
STRIDE=${STRIDE:-4}; NSTEP=${NSTEP:-1561}; LR=${LR:-5e-7}
source ./lib_timing.sh
echo "########## GROUP-JOINT all-MLP (stride $STRIDE, $NSTEP steps/group, lr $LR) — $(date) ##########"
echo "  refs: down-only@16M 78.35% / KL 0.3637 | scale-only E2E 80.86% / 0.2977 | assign+E2E 81.32% / 0.3042"
CUR=$SRC/modified_model                      # start from the RAW block-AP skeleton, like arm F did
for ((off=0; off<STRIDE; off++)); do
  OUT=$W/g$off/modified_model
  stage "group $off/$((STRIDE-1)) (layers idx%$STRIDE==$off, all-MLP joint)"
  if [ -f "$W/g$off/.done" ]; then echo "=== [skip] g$off ==="; CUR=$OUT; continue; fi
  mkdir -p "$W/g$off"
  systemd-run --user --scope --quiet -p MemoryHigh=38G -p MemoryMax=42G -p MemorySwapMax=0 \
    -E TERNARY_BLOCK_SIZE=64 -E ORIG_MODEL=$ORIG -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -E SAVE_MAX_SHARD_GB=2 \
    ./.venv/bin/python -m torch.distributed.run --nproc_per_node=1 --standalone \
      src/e2e_qp_distill.py --train --student-path "$CUR" --orig-config-path "$ORIG" \
      --calib $SRC/calibration_data.json --teacher-cache $SRC/teacher_topk.pt --out "$OUT" \
      --seq 2560 --epochs 1 --max-samples "$NSTEP" \
      --lr 0 --latent-lr "$LR" --latent-init fp-spread --fp-model $SRC/rotbase/modified_model \
      --latent-offload --latent-grad-release \
      --tw-layer-stride "$STRIDE" --tw-layer-offset "$off" --latent-warmup-steps 20 \
      --target-tr 1.25e-4 --tr-every 5 --tr-final-frac 0.2 \
      --lr-schedule linear --scale-ema-decay 0 --select final --loss-fn cakld --decision-gamma 2 \
      --ce-weight 0.1 --ce-positions 128 --train-weights mlp --scale-qat-bits 8 \
      --heldout-n 4 --eval-every 200 --ckpt-every 0 --abort-patience 1000000 \
    || { echo "  group $off FAILED"; exit 1; }
  touch "$W/g$off/.done"; CUR=$OUT; echo "  >>> group $off done -> $OUT"
done
echo; echo "########## eval2k: group-joint final ##########"
RES=logs/groupjoint_eval2k.txt
{ echo "group-joint all-MLP (stride $STRIDE x $NSTEP steps = $((STRIDE*NSTEP)) steps, 16M token-steps) — $(date)"
  echo "  down-only @16M (SAME budget): 78.35% / KL 0.3637"
  echo "  scale-only E2E:               80.86% / KL 0.2977"
  printf "%-14s %-12s %s\n" "model" "agreement%" "meanKL"; } | tee "$RES"
systemd-run --user --scope --quiet -p MemoryHigh=38G -p MemoryMax=42G -p MemorySwapMax=0 -E TERNARY_BLOCK_SIZE=64 \
    env ORIG="$ORIG" FP_DIR="$SRC/rotbase/modified_model" E2E_MODEL="$CUR" \
    EVAL_DATA=output_4b/eval2k.json NP=1946 SEQ=1024 \
    ./.venv/bin/python src/kl_flips_eval.py > logs/groupjoint_e2k.log 2>&1 || echo "  (eval failed)"
AG=$(grep -aoE "top-1 agreement *: *[0-9.]+" logs/groupjoint_e2k.log | grep -oE "[0-9.]+$" | tail -1)
KL=$(grep -aoE "mean KL\(fp\|\|tern\) *: *[0-9.]+" logs/groupjoint_e2k.log | grep -oE "[0-9.]+$" | tail -1)
printf "%-14s %-12s %s\n" "groupjoint" "${AG:-NA}" "${KL:-NA}" | tee -a "$RES"
echo; cat "$RES"
stage_end
) > "${LOG_FILE:-logs/groupjoint.log}" 2>&1 &
echo "groupjoint started (PID $!)."
