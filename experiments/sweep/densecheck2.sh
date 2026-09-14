#!/bin/bash
# densify validation under the NEW harness. DENSE_INFER must be a pure speedup: identical tokens.
# Must be re-done here because the loop_gate refactor changed the RNG stream, so the old baseline
# (tr_c1.30) is no longer the right comparison.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
run(){
  local gpu=$1
  local tag=$2
  local dense=$3
  CUDA_VISIBLE_DEVICES=$gpu env ORIG=$PWD/output_4b/untied_4b \
    E2E_MODEL=output_sweep/opsa/modified_model MODEL_KIND=tern DENSE_INFER=$dense \
    THINK_ROW_SCALE=1.30 DRY_MULT=0 N_PROB=8 MAXNEW=2048 THINK=1 TEMP=0.6 BATCH=8 SEED=0 \
    OUT=output_sweep/dc_${tag}.json ROWS=output_sweep/dc_${tag}_rows.json \
    ./.venv/bin/python src/math_correct.py > output_sweep/dc_${tag}.log 2>&1
}
run 1 packed 0
run 1 dense  1

touch output_sweep/.dc_done
