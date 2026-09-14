#!/bin/bash
# Format head-to-head, stage 1: weighted reconstruction error on REAL weights.
# Cheap signal before spending on a full end-to-end eval. All arms RTN, same weights, same objective.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
CUDA_VISIBLE_DEVICES=${GPU:-0} ./.venv/bin/python src/fmt_recon.py "$@"
