#!/bin/bash
# ══ E2E SCALE DISTILLATION ON TOP OF THE ASSIGNMENT CHAIN ═══════════════════════════════════════════
# Uses run_full_pipeline.sh's SETTLED Phase-5 recipe verbatim (2ep x lr2e-5 x linear x final-select x
# cakld x col-scale x scale-qat-8 x commit-beta 1.5) so the result is directly comparable to the
# scale-only anchor: 16M_2ep_colscale_v2 = 80.86% / KL 0.2977, which ran that recipe from the RAW
# block-AP skeleton. The only difference here is the starting point: assignments already trained.
#   assignment-only reference: arm F (down, 16M tok) 78.35% / 0.3637; + attn = chain32 result.
# E2E trains SCALES ONLY (--train-weights defaults to none); the trained assignments are already folded
# to hard ternary in the saved student and are carried through unchanged.
(
if ps -eo comm=,args= | awk '$1=="python" && /e2e_qp_distill\.py --train/{f=1} END{exit !f}'; then
  echo "FATAL: a training process is already running -- refusing to start a second one."; exit 1
fi
set -e
export PYTHONUNBUFFERED=1; export TERNARY_BLOCK_SIZE=64
ORIG=/home/kyler/Documents/Model-to-Ternary/output_4b/untied_4b
SRC=output_4bpipe; W=$SRC/e2e_on_assign; mkdir -p "$W"
# start from the assignment chain's final model; fall back to arm F if attn produced nothing usable
START=${START:-$SRC/attn32/c3_attn/modified_model}
[ -f "$START/model.safetensors.index.json" ] || START=$SRC/chain32/c3_attn/modified_model
[ -f "$START/model.safetensors.index.json" ] || START=$SRC/dscale/f_6244s_16M/modified_model
echo "########## E2E ON ASSIGNMENTS — start=$START — $(date) ##########"
source ./lib_timing.sh
stage "E2E (2ep, settled Phase-5 recipe)"
if [ ! -f "$W/.done" ]; then
systemd-run --user --scope --quiet -p MemoryHigh=38G -p MemoryMax=42G -p MemorySwapMax=0 \
  -E TERNARY_BLOCK_SIZE=64 -E ORIG_MODEL=$ORIG -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -E SAVE_MAX_SHARD_GB=2 \
  ./.venv/bin/python -m torch.distributed.run --nproc_per_node=1 --standalone \
    src/e2e_qp_distill.py --train --student-path "$START" --orig-config-path "$ORIG" \
    --calib $SRC/calibration_data.json --teacher-cache $SRC/teacher_topk.pt --out "$W/modified_model" \
    --seq 2560 --epochs 2 --max-samples 6244 \
    --lr 2e-5 --lr-schedule linear --scale-ema-decay 0 --select final \
    --loss-fn cakld --feat-weight 0 --decision-gamma 2 \
    --col-scale --scale-qat-bits 8 \
    --commit-beta 1.5 --commit-pre 16 --commit-post 12 \
    --heldout-n 48 --eval-every 300 --abort-patience 1000000 || { echo "  E2E FAILED"; exit 1; }
touch "$W/.done"; fi
echo "  >>> E2E done"
echo; echo "########## eval2k ##########"
RES=logs/e2e_on_assign_eval2k.txt
{ echo "E2E-on-assignments — eval2k (NP=1946 SEQ=1024) — $(date)"
  echo "  scale-only anchor (same recipe, raw skeleton): 80.86% / KL 0.2977"
  echo "  assignment-only (arm F down 16M):              78.35% / KL 0.3637"
  printf "%-22s %-12s %s\n" "model" "agreement%" "meanKL"; } | tee "$RES"
systemd-run --user --scope --quiet -p MemoryHigh=38G -p MemoryMax=42G -p MemorySwapMax=0 -E TERNARY_BLOCK_SIZE=64 \
    env ORIG="$ORIG" FP_DIR="$SRC/rotbase/modified_model" E2E_MODEL="$W/modified_model" \
    EVAL_DATA=output_4b/eval2k.json NP=1946 SEQ=1024 \
    ./.venv/bin/python src/kl_flips_eval.py > logs/e2e_on_assign_e2k.log 2>&1 || echo "  (eval failed)"
AG=$(grep -aoE "top-1 agreement *: *[0-9.]+" logs/e2e_on_assign_e2k.log | grep -oE "[0-9.]+$" | tail -1)
KL=$(grep -aoE "mean KL\(fp\|\|tern\) *: *[0-9.]+" logs/e2e_on_assign_e2k.log | grep -oE "[0-9.]+$" | tail -1)
printf "%-22s %-12s %s\n" "assign+E2E" "${AG:-NA}" "${KL:-NA}" | tee -a "$RES"
echo; cat "$RES"
stage_end
) > "${LOG_FILE:-logs/e2e_on_assign.log}" 2>&1 &
echo "e2e_on_assign started (PID $!)."
