#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# GRANULARITY CORROBORATOR (assignment-vs-capacity diagnostic; researcher report 2026-08-05, Q5).
#
# Refit the ternary SCALES at g64 → g32 → g16 with the trit ASSIGNMENTS FROZEN (scale-only E2E = the fast
# full-logits path, no CE / no latents / no instability), warm-started from the 77.53% model, then score each
# on the frozen eval2k referee. Reading:
#   eval2k RISES with finer grids (g64 < g32 < g16)  → capacity/GRANULARITY-limited (finer scales recover FP)
#                                                       ⇒ redirect budget to finer granularity + CE-on-scales.
#   eval2k FLAT across grids                          → residual is NOT scale-granularity ⇒ it's in the
#                                                       ASSIGNMENTS ⇒ the flip-control program is worth building.
# fp16 scales (--scale-qat-bits 0) to avoid the 4-bit-cancels-granularity confound (§8b). Assignments stay
# exactly the g64-placed trits (extract_ternary_scale re-derives finer scales, same signs). Diagnostic only —
# g32/g16 scales are more bits/weight, not a deploy format.
#
#   bash run_granularity_test.sh          # background; ~3h (3 × ~45min refit + 3 evals)
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
WARM=$SRC/e2eqp/modified_model            # the 77.53% pure-distillation model (assignments to freeze)
CALIB=$SRC/calibration_data.json
TEACHER=$SRC/teacher_topk.pt              # FP teacher logits — granularity-independent, reused
ROT=$SRC/rotbase/modified_model
EVAL2K=${EVAL2K:-output_4b/eval2k.json}
W=${W:-$SRC/granularity}
TOKENS=${TOKENS:-4000000}; SEQ=2560
NSAMP=$(( TOKENS / SEQ ))
NGPU=${NGPU:-2}

TOTAL_GB=$(free -g | awk '/^Mem:/{print $2}'); DEF=$(( TOTAL_GB-18 ))
MEM_MAX=${MEM_MAX:-${DEF}G}; MM=${MEM_MAX%G}; MEM_HIGH=$(( MM-4 ))G
systemd-run --user --scope --quiet -p MemoryMax=64M -p MemorySwapMax=0 /bin/true 2>/dev/null \
  || { echo "FATAL: no systemd --user scope"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FORCE_MEM_EFF=1        # chunked-KL path (no full-vocab logits) — g32/g16 have 2-4x scales and OOM the
                             # full-logits path; run ALL grids this way so the comparison is on one recipe.
mkdir -p "$W"
source ./lib_timing.sh
echo "########## GRANULARITY CORROBORATOR — $(date) ##########"
echo "  warm-start $WARM (eval2k baseline 77.53% = g64) | ${TOKENS} tok | fp16 scales | assignments FROZEN"

for BLK in 64 32 16; do
  OUT=$W/g${BLK}/modified_model
  GUARD="systemd-run --user --scope --quiet -p MemoryHigh=$MEM_HIGH -p MemoryMax=$MEM_MAX -p MemorySwapMax=0 -E TERNARY_BLOCK_SIZE=$BLK"
  stage "refit g$BLK scales (assignments frozen)"
  if [ ! -f "$W/g${BLK}/.done" ]; then
    mkdir -p "$W/g${BLK}"
    $GUARD ./.venv/bin/python -m torch.distributed.run --nproc_per_node="$NGPU" --standalone \
        src/e2e_qp_distill.py --train --student-path "$WARM" --orig-config-path "$ORIG_MODEL" \
        --calib "$CALIB" --teacher-cache "$TEACHER" --out "$OUT" \
        --seq "$SEQ" --epochs 2 --max-samples "$NSAMP" \
        --lr 2e-5 --lr-schedule linear --scale-ema-decay 0 --select final \
        --loss-fn cakld --decision-gamma 2 --scale-qat-bits 0 \
        --heldout-n 48 --eval-every 300 --abort-patience 30 || { echo "  g$BLK refit FAILED"; continue; }
    touch "$W/g${BLK}/.done"
  else echo "=== [skip] g$BLK exists ==="; fi
  stage "eval2k g$BLK"
  $GUARD env ORIG="$ORIG_MODEL" FP_DIR="$ROT" E2E_MODEL="$OUT" EVAL_DATA="$EVAL2K" NP=1946 SEQ=1024 \
      ./.venv/bin/python src/kl_flips_eval.py 2>&1 | tee "$W/g${BLK}_eval2k.log" | grep -aE "agreement|mean KL"
  AG=$(grep -aoE "top-1 agreement *: *[0-9.]+" "$W/g${BLK}_eval2k.log" | grep -oE "[0-9.]+$" | tail -1)
  echo "  >>> g$BLK eval2k = ${AG:-NA}%"
done

echo; echo "=== GRANULARITY RESULT (vs g64 baseline 77.53%) ==="
for BLK in 64 32 16; do
  AG=$(grep -aoE "top-1 agreement *: *[0-9.]+" "$W/g${BLK}_eval2k.log" 2>/dev/null | grep -oE "[0-9.]+$" | tail -1)
  printf "  g%-3s eval2k %s%%\n" "$BLK" "${AG:-NA}"
done
echo "  RISING with finer grid ⇒ granularity/capacity-limited; FLAT ⇒ assignment-limited (build the flip fix)."
stage_end
echo "########## GRANULARITY TEST DONE ##########"
) > "${LOG_FILE:-logs/granularity_test.log}" 2>&1 &
echo "Granularity test started in the background (PID $!)."
