#!/bin/bash
# run_loopgate_compare.sh — FP vs V1 free-gen, interleaved prompts (easy/reason/open-ended), temp 0 & 0.6.
set -u; cd "$(dirname "$0")"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY=./.venv/bin/python; ORIG=output_4b/untied_4b
FP=output_4b/rot/modified_model; V1=output_4b/chatfix_v2/e2e_v1/modified_model
run () {  # tag kind model temp
  local tag=$1 kind=$2 model=$3 temp=$4
  MODEL_KIND=$kind ORIG=$ORIG E2E_MODEL=$model N_PREFIX=48 MAXNEW=768 THINK=1 TEMP=$temp BATCH=8 \
    GATE_OUT=logs/cmp_${tag}.json GATE_SAMPLES=logs/cmp_${tag}_s.json \
    $PY src/loop_gate.py > logs/cmp_${tag}.log 2>&1
  echo "### $tag ###"; grep -aE 'loop_rate|trunc_rate|commit_rate|comp_ratio' logs/cmp_${tag}.log | tail -4
}
run fp_t0  fp   "$FP" 0.0
run v1_t0  tern "$V1" 0.0
run fp_t06 fp   "$FP" 0.6
run v1_t06 tern "$V1" 0.6
echo "########## COMPARE DONE $(date) ##########"
