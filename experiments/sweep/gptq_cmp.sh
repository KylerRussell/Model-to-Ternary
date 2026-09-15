#!/bin/bash
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=${GPU:-0} NB=${NB:-16} \
  ./.venv/bin/python src/run_gptq_compare.py > output_sweep/gptq_compare.log 2>&1
touch output_sweep/.gptq_done
