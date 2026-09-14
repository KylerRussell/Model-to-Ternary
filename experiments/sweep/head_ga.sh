#!/bin/bash
# Gate-A agreement, PAIRED across head arms on the SAME eval sequences. Paired is the point (§4:
# paired deltas resolve ~9x finer than absolutes). Also the co-adaptation discriminator: the body was
# trained against a ternary head, so if agreement DROPS when the head improves, body and head
# co-adapted and a null accuracy result cannot be read as "the head is innocent".
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
GPU=$1; TAG=$2; MODE=$3; NP=${NP:-16}
O=output_sweep/ga_${TAG}.json
[ -f "$O" ] && { echo "SKIP $TAG"; exit 0; }
echo "  [$(date +%F' '%H:%M:%S)] START ga_$TAG mode=${MODE:-ternary}" >> output_sweep/head_results.txt
CUDA_VISIBLE_DEVICES=$GPU \
env ORIG=$PWD/output_4b/untied_4b FP_DIR=$PWD/output_4bpipe/rotbase/modified_model \
    E2E_MODEL=output_sweep/opsa/modified_model EVAL_DATA=$PWD/output_4b/eval2k.json \
    NP=$NP SEQ=1024 ${MODE:+HEAD_MODE=$MODE} ${EMB:+EMBED_MODE=$EMB} PER_SEQ_OUT="$O" \
    ./.venv/bin/python src/kl_flips_eval.py > output_sweep/ga_${TAG}.log 2>&1
echo "  [$(date +%F' '%H:%M:%S)] END ga_$TAG rc=$?" >> output_sweep/head_results.txt
grep -E "mean KL|flips|agreement|head_swap" output_sweep/ga_${TAG}.log >> output_sweep/head_results.txt
