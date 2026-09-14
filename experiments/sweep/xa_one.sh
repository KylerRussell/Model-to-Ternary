#!/bin/bash
# One exaccerr arm on one GPU. Both models land on the single visible device (exaccerr.py picks
# 1-GPU mode when device_count==1): FP bf16 ~8 GB + packed ternary ~1.5 GB + two [L,V] bf16 logit
# tensors ~2.1 GB fits inside 24 GB, so an arm can run on the free card while the other still
# generates. THINK_ROW_SCALE stays 1.30 on EVERY arm -- the student is the fixed instrument here and
# only the token source varies.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
GPU=$1
TAG=$2
SRC=$3
O=output_sweep/xa_${TAG}.json
if [ -f "$O" ]; then echo "SKIP $TAG"; exit 0; fi
echo "  [$(date +%F' '%H:%M:%S)] START xa_$TAG on GPU$GPU (tokens from $SRC)" >> output_sweep/exacc_results.txt
CUDA_VISIBLE_DEVICES=$GPU \
env ORIG=$PWD/output_4b/untied_4b E2E_MODEL=output_sweep/opsa/modified_model \
    FP_DIR=$PWD/output_4bpipe/rotbase/modified_model \
    ROWS=output_sweep/${SRC}_rows.json THINK_ROW_SCALE=1.30 SKIP_PREFIX=${SKIP_PREFIX:-0} OUT="$O" \
    ./.venv/bin/python src/exaccerr.py > output_sweep/xa_${TAG}.log 2>&1
echo "  [$(date +%F' '%H:%M:%S)] END xa_$TAG rc=$?" >> output_sweep/exacc_results.txt
tail -16 output_sweep/xa_${TAG}.log >> output_sweep/exacc_results.txt
