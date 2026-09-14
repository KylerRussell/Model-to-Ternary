#!/bin/bash
set -u; cd /home/kasm-user/Documents/Model-to-Ternary
until [ -f output_sweep/.dt_done ]; do sleep 180; done   # artifact-gated (13ah-i)
./experiments/sweep/tc_confirm.sh
