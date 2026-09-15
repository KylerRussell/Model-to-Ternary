#!/bin/bash
# Gate on the completion ARTIFACT, never on the process -- zombies match pgrep forever (13ah).
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
while [ ! -f output_sweep/.e2e_trellis1_done ]; do sleep 30; done
PYTHONUNBUFFERED=1 ./.venv/bin/python src/fill_band.py > output_sweep/band_fill.log 2>&1
touch output_sweep/.band_done
