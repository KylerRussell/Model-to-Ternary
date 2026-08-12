#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# SEQUENTIAL GROUP ASSIGNMENT PASSES (fix for the depth-compounding problem, §8u).
#
# Moving ALL layers' assignments at once compounds error through depth: at the SAME per-layer flip rate,
# first-eval damage vs a 0.735 baseline was 8L 0.58 (improves) · 16L 0.75 · 32L 1.91 (2.6x worse), with
# per-layer flip rates UNIFORM ⇒ compounding, not imbalance. Worse at 64 layers (27B).
#
# So: cover every layer, but only ever perturb 1/STRIDE of them at a time. STRIDE sequential passes
# (offset 0..STRIDE-1), each WARM-STARTING from the previous pass's output, letting the model re-stabilise
# between groups — the same reason block-AP goes layer-by-layer.
#
# Scales stay FROZEN throughout (--lr 0): this is the V-phase. No E2E afterwards (per user).
# Reference to beat: single 8-layer pass reached KL 0.4625 (baseline 0.7358) but touched only 8/32 layers.
#
#   bash run_seq_assign.sh            # ~4 x 40min on 2 GPUs
# ──────────────────────────────────────────────────────────────────────────────
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
ORIG_MODEL=${ORIG_MODEL:-/home/kyler/Documents/Model-to-Ternary/output_4b/untied_4b}
export ORIG_MODEL
SRC=${SRC:-output_4bpipe}
FP=$SRC/rotbase/modified_model
CALIB=$SRC/calibration_data.json
TEACHER=$SRC/teacher_topk.pt
W=${W:-$SRC/seqassign}
STRIDE=${STRIDE:-4}
STEPS_SAMPLES=${STEPS_SAMPLES:-120}     # x epochs / world = optimizer steps
EPOCHS=${EPOCHS:-4}
TR=${TR:-5e-4}                          # per-layer flip rate — matches the 8L reference
LATENT_LR=${LATENT_LR:-5e-6}
NGPU=${NGPU:-2}
export TERNARY_BLOCK_SIZE=${BLK:-64}

TOTAL_GB=$(free -g | awk '/^Mem:/{print $2}'); DEF=$(( TOTAL_GB-18 ))
MEM_MAX=${MEM_MAX:-${DEF}G}; MM=${MEM_MAX%G}
systemd-run --user --scope --quiet -p MemoryMax=64M -p MemorySwapMax=0 /bin/true 2>/dev/null \
  || { echo "FATAL: no systemd --user scope"; exit 1; }
GUARD="systemd-run --user --scope --quiet -p MemoryHigh=$((MM-4))G -p MemoryMax=$MEM_MAX -p MemorySwapMax=0 \
       -E TERNARY_BLOCK_SIZE=${BLK:-64} -E ORIG_MODEL=$ORIG_MODEL -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
mkdir -p "$W"
source ./lib_timing.sh
echo "########## SEQUENTIAL GROUP ASSIGNMENT PASSES — $(date) ##########"
echo "  stride $STRIDE ⇒ $STRIDE passes of $((32/STRIDE)) layers each; TR $TR; scales FROZEN; NO E2E"

CUR=$SRC/modified_model                 # start from the raw block-AP skeleton
for ((off=0; off<STRIDE; off++)); do
  OUT=$W/pass${off}/modified_model
  stage "pass $off/$((STRIDE-1)): layers idx%${STRIDE}==${off}"
  if [ -f "$W/pass${off}/.done" ]; then echo "=== [skip] pass $off exists ==="; CUR=$OUT; continue; fi
  mkdir -p "$W/pass${off}"
  $GUARD ./.venv/bin/python -m torch.distributed.run --nproc_per_node="$NGPU" --standalone \
      src/e2e_qp_distill.py --train --student-path "$CUR" --orig-config-path "$ORIG_MODEL" \
      --calib "$CALIB" --teacher-cache "$TEACHER" --out "$OUT" \
      --seq 2560 --epochs "$EPOCHS" --max-samples "$STEPS_SAMPLES" \
      --lr 0 --latent-lr "$LATENT_LR" --latent-init fp-spread --fp-model "$FP" \
      --tw-layer-stride "$STRIDE" --tw-layer-offset "$off" \
      --latent-warmup-steps 20 --target-tr "$TR" --tr-every 5 --tr-final-frac 0.2 \
      --lr-schedule linear --scale-ema-decay 0 --select final --loss-fn cakld --decision-gamma 2 \
      --ce-weight 0.1 --ce-positions 128 --train-weights down --scale-qat-bits 8 \
      --heldout-n 4 --eval-every 20 || { echo "  pass $off FAILED"; exit 1; }
  touch "$W/pass${off}/.done"
  CUR=$OUT                              # next pass warm-starts from this one
  echo "  >>> pass $off done -> $OUT"
done

echo; echo "=== SEQUENTIAL PASSES COMPLETE — final model: $CUR ==="
echo "  (each pass logs its own [held-out] trajectory above; the LAST pass's final KL is the result)"
echo "  reference: single 8-layer pass = 0.4625 from a 0.7358 baseline, but only 8/32 layers touched"
stage_end
) > "${LOG_FILE:-logs/seq_assign.log}" 2>&1 &
echo "Sequential assignment passes started in the background (PID $!)."
