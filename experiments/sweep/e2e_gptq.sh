#!/bin/bash
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$1 FMT=$2 \
  OUT=output_sweep/e2egptq_$2.json ./.venv/bin/python src/e2e_gptq.py \
  > output_sweep/e2egptq_$2.log 2>&1
touch output_sweep/.e2egptq_$2_done
