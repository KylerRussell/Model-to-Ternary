#!/bin/bash
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
CUDA_VISIBLE_DEVICES=0 env ORIG=$PWD/output_4b/untied_4b E2E_MODEL=output_sweep/opsa/modified_model \
  MODEL_KIND=tern DENSE_INFER=1 THINK_ROW_SCALE=1.30 DRY_MULT=0 N_PROB=8 MAXNEW=2048 THINK=1 \
  TEMP=0.6 BATCH=8 SEED=0 OUT=output_sweep/dc_dense.json ROWS=output_sweep/dc_dense_rows.json \
  ./.venv/bin/python src/math_correct.py > output_sweep/dc_dense.log 2>&1
touch output_sweep/.dc_done
