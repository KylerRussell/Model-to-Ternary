#!/bin/bash
# RE-BASELINE under the fixed harness (13aq/13ap-iii). Changes vs every previous number:
#   * no discarded loop_gate sweep inside each run -> different RNG stream -> numbers are NOT
#     comparable to any pre-13aq result. That is the point of re-baselining.
#   * DRY is OFF (13ap-ii, curbed).
#   * the gate now scores CORRECTNESS and cannot return PASS without it.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
GPU=$1; TAG=$2; KIND=$3; MODEL=$4; C=$5; DENSE=${DENSE:-1}
O=output_sweep/rb_${TAG}.json
[ -f "$O" ] && { echo "SKIP $TAG"; exit 0; }
[ "$KIND" = "fp" ] && DENSE=0
echo "  [$(date +%F' '%H:%M:%S)] START rb_$TAG" >> output_sweep/rebase_results.txt
CUDA_VISIBLE_DEVICES=$GPU env ORIG=$PWD/output_4b/untied_4b E2E_MODEL="$MODEL" MODEL_KIND=$KIND \
  DENSE_INFER=$DENSE THINK_ROW_SCALE=$C DRY_MULT=0 \
  N_PREFIX=48 MAXNEW=2048 THINK=1 TEMP=0.6 BATCH=8 SEED=${SEED:-0} \
  N_SCORE=${N_SCORE:-24} SCORE_MAXNEW=2048 \
  ${TEACHER_THINK_LEN:+TEACHER_THINK_LEN=$TEACHER_THINK_LEN} \
  GATE_OUT="$O" SCORE_ROWS=output_sweep/rb_${TAG}_rows.json ./.venv/bin/python src/loop_gate.py > output_sweep/rb_${TAG}.log 2>&1
echo "  [$(date +%F' '%H:%M:%S)] END rb_$TAG rc=$?" >> output_sweep/rebase_results.txt
sed -n '/SCORED SUBSET/,/VERDICT/p' output_sweep/rb_${TAG}.log >> output_sweep/rebase_results.txt
touch output_sweep/.rb_${TAG}_done
