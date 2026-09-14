#!/bin/bash
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
until [ -f output_sweep/.tc_done ]; do sleep 180; done   # artifact-gated, never pgrep (13ah-i)
./experiments/sweep/drytune.sh
