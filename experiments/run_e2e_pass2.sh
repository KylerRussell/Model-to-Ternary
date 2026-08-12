#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# E2E PASS 2 — warm-start refinement, testing the §8i undertraining hypothesis.
#
# The first end-to-end run of run_full_pipeline.sh (§8i) produced a model that FAILED Gate B
# (commit 41.7% vs target 68%, trunc 58.3%) with held-out KL still DESCENDING at the last eval.
# The validated reference FINAL was never a single pass: it was skeleton -> E2E(combined-2560)
# -> SECOND E2E at lr 1e-5, i.e. ~2x the E2E exposure, landing at held-out 0.1691.
#
# This runs that missing second pass: same calib, same teacher cache (NO regeneration), warm-started
# from pass 1, lr 1e-5, then re-runs both gates. If Gate B passes, Phase 5 of run_full_pipeline.sh
# must become two passes.
#
#   bash run_e2e_pass2.sh          # background; logs to logs/e2e_pass2.log
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
WORK=${WORK:-output_4bpipe}
BLK=${BLK:-64}; SBITS=${SBITS:-8}; SEQ=${SEQ:-2560}; NGPU=${NGPU:-2}
export TERNARY_BLOCK_SIZE=$BLK

ROT=$WORK/rotbase/modified_model          # FP reference for Gate A
CALIB=$WORK/calibration_data.json
CALIB_EVAL=$WORK/calib_eval.json
TEACHER=$WORK/teacher_topk.pt
PASS1=$WORK/e2eqp/modified_model          # warm-start source
OUT=$WORK/e2eqp2/modified_model

EPOCHS=${EPOCHS:-2}
LR=${LR:-1e-5}                            # the reference's second-pass LR (pass 1 used 2e-5)
COMMIT_BETA=${COMMIT_BETA:-1.5}
GATE_MAXNEW=${GATE_MAXNEW:-2048}

# same fail-closed host-safety cap as the pipeline (physical-18G); N concurrent scopes would each get
# the full cap, so everything below runs under ONE scope at a time.
TOTAL_GB=$(free -g | awk '/^Mem:/{print $2}')
DEF_MAX=$(( TOTAL_GB>=48 ? TOTAL_GB-18 : (TOTAL_GB>=24 ? TOTAL_GB-10 : TOTAL_GB*2/3) ))
MEM_MAX=${MEM_MAX:-${DEF_MAX}G}; MM_NUM=${MEM_MAX%G}
MEM_HIGH=${MEM_HIGH:-$(( MM_NUM>8 ? MM_NUM-4 : MM_NUM ))G}
if systemd-run --user --scope --quiet -p MemoryMax=64M -p MemorySwapMax=0 /bin/true 2>/dev/null; then
  MEMGUARD="systemd-run --user --scope --quiet -p MemoryHigh=$MEM_HIGH -p MemoryMax=$MEM_MAX -p MemorySwapMax=0 -E TERNARY_BLOCK_SIZE=$BLK"
  echo "[memguard] high=$MEM_HIGH max=$MEM_MAX swap=off (physical ${TOTAL_GB}G)"
else echo "[memguard] FATAL: no systemd --user scope; refusing to run unguarded."; exit 1; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$MEMGUARD ./.venv/bin/python"

for p in "$PASS1/model.safetensors.index.json" "$CALIB" "$TEACHER" "$CALIB_EVAL"; do
  [ -e "$p" ] || { echo "FATAL: missing prerequisite $p"; exit 1; }
done
NSAMP=$(./.venv/bin/python -c "import json;print(len(json.load(open('$CALIB'))))")
echo "########## E2E PASS 2 — $(date) ##########"
echo "  warm-start $PASS1 -> $OUT | lr $LR | ${EPOCHS}ep | $NSAMP seqs | g$BLK/${SBITS}bit | beta $COMMIT_BETA"
source ./lib_timing.sh

stage "Pass 2: E2E warm-start (lr $LR)"
mkdir -p "$WORK/e2eqp2"
$PY -m torch.distributed.run --nproc_per_node="$NGPU" --standalone \
    src/e2e_qp_distill.py --train --student-path "$PASS1" --orig-config-path "$ORIG_MODEL" \
    --calib "$CALIB" --teacher-cache "$TEACHER" --out "$OUT" \
    --seq "$SEQ" --epochs "$EPOCHS" --max-samples "$NSAMP" \
    --lr "$LR" --lr-schedule linear --scale-ema-decay 0 --select final \
    --loss-fn cakld --feat-weight 0 --decision-gamma 2 \
    --col-scale --scale-qat-bits "$SBITS" \
    --commit-beta "$COMMIT_BETA" --commit-pre 16 --commit-post 12 \
    --heldout-n 48 --eval-every 300 --abort-patience 30

stage "Pass 2: gates"
echo "=== Gate A: teacher-forced agreement (tracking only) ==="
$MEMGUARD env ORIG="$ORIG_MODEL" FP_DIR="$ROT" E2E_MODEL="$OUT" \
    EVAL_DATA="$CALIB_EVAL" NP="${GATE_NP:-512}" SEQ=1024 \
    ./.venv/bin/python src/kl_flips_eval.py || echo "(agreement gate failed; continuing)"
echo "=== Gate B: FREE-GEN loop/commit @${GATE_MAXNEW} (THE deploy gate) ==="
echo "    targets: commit >=68% · loop <=30% · compR <=3.1   [FP 75/25/2.40 | pass1 41.7/33.3/3.12]"
$MEMGUARD env ORIG="$ORIG_MODEL" E2E_MODEL="$OUT" \
    N_PREFIX=48 MAXNEW="$GATE_MAXNEW" THINK=1 TEMP=0.6 BATCH=8 GATE_OUT="$WORK/loop_gate_pass2.json" \
    ./.venv/bin/python src/loop_gate.py || echo "(free-gen gate failed; continuing)"

stage_end
echo "########## PASS 2 DONE -> $OUT ##########"
) > "${LOG_FILE:-logs/e2e_pass2.log}" 2>&1 &
echo "Pass 2 started in the background (PID $!)."
echo "Watch it:   tail -f ${LOG_FILE:-logs/e2e_pass2.log}"
