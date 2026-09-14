#!/bin/bash
cd /home/kasm-user/Documents/Model-to-Ternary
until grep -q "QUALITY SWEEP DONE" output_sweep/quality_results.txt 2>/dev/null; do sleep 60; done
sleep 30
bash output_sweep/quality_A.sh
