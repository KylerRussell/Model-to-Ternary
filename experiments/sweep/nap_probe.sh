#!/bin/bash
# Fast trust-region calibration: run the preconditioning pass at the REAL step budget and read the
# gain-ratio band off the first few layers. Unbounded gave [0.875, 1.124] at 512 steps; --col-scale
# (S7) put the useful optimum for this axis at [0.994, 1.006].
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
ROT=$PWD/output_4bpipe/rotbase/modified_model
ORIG=$PWD/output_4b/untied_4b
W=$PWD/output_4b_napprobe
rm -rf "$W"; mkdir -p "$W"
ln -sfn "$PWD/output_4bpipe/calibration_data.json" "$W/calibration_data.json"
ORIG_MODEL="$ORIG" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
TERNARY_BLOCK_SIZE=64 ./.venv/bin/python src/block_ap_recovery.py \
    --model-path "$ROT" --orig-config-path "$ORIG" --output-dir "$W" \
    --block-size 64 --samples $(( 640 * 1024 / 2560 )) --nap-epochs 2 --nap-lr 1e-3 \
    --nap-trust "${NAP_TRUST:-1.0}" > output_sweep/nap_probe.log 2>&1
