#!/bin/bash
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
CUDA_VISIBLE_DEVICES=0 env ORIG=$PWD/output_4b/untied_4b E2E_MODEL=output_sweep/opsa/modified_model \
  MODEL_KIND=tern THINK_ROW_SCALE=1.30 DRY_MULT=0.8 DRY_BASE=1.75 DRY_ALLOWED=2 \
  N_PREFIX=48 MAXNEW=2048 THINK=1 TEMP=0.6 BATCH=8 SEED=0 GATE_OUT=output_sweep/lg_regress.json \
  ./.venv/bin/python src/loop_gate.py > output_sweep/lg_regress.log 2>&1
touch output_sweep/.lg_regress_done
