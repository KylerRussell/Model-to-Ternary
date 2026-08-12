#!/bin/bash
# run_gate2048.sh — re-measure the free-gen gate at a 2048-token budget (was 768) to separate genuine
# "won't stop" from "budget too small". 2048 <= the 2560 training window, so we never test beyond what we
# trained. Models carry DIFFERENT group sizes -> TERNARY_BLOCK_SIZE must match each (mismatch = garbage).
set -u; cd "$(dirname "$0")"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY=./.venv/bin/python; ORIG=output_4b/untied_4b
run () {  # tag  kind  model  blocksize
  local tag=$1 kind=$2 model=$3 blk=$4
  [ -e "$model/model.safetensors" ] || [ -e "$model/model.safetensors.index.json" ] || { echo "SKIP $tag (missing)"; return; }
  echo "### $tag (g$blk) ###"
  TERNARY_BLOCK_SIZE=$blk MODEL_KIND=$kind ORIG=$ORIG E2E_MODEL="$model" \
    N_PREFIX=48 MAXNEW=2048 THINK=1 TEMP=0.6 BATCH=8 GATE_OUT=logs/g2048_${tag}.json \
    $PY src/loop_gate.py > logs/g2048_${tag}.log 2>&1
  grep -aE 'loop_rate|trunc_rate|commit_rate|comp_ratio' logs/g2048_${tag}.log | sed 's/^/  /'
}
run fp       fp   output_4b/rot/modified_model                      256
run v1       tern output_4b/chatfix_v2/e2e_v1/modified_model        256
run comb2560 tern output_4b/combined2560_g64q8/e2e/modified_model    64
run stageB   tern output_4b/stageB_g64q8/e2e/modified_model          64
echo "########## GATE-2048 DONE $(date) ##########"
