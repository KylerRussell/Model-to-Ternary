#!/bin/bash
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${GPU:-0} N_T=${N_T:-3} \
  OUT=output_sweep/family_sweep.json ./.venv/bin/python src/run_family_sweep.py \
  > output_sweep/family_sweep.log 2>&1
touch output_sweep/.family_done
