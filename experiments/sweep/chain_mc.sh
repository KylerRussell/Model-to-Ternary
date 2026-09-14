#!/bin/bash
set -u; cd /home/kasm-user/Documents/Model-to-Ternary
until [ -f output_sweep/.tcc_done ]; do sleep 180; done   # artifact-gated (13ah-i)
./experiments/sweep/mathcorrect.sh
