#!/bin/bash
# NAP preconditioning plumbing smoke: tiny config, just enough to prove the pass runs, that the
# norm gains actually MOVE, and that the gain-ratio range lands near [0.994, 1.006] (the band S7
# measured for --col-scale on this same axis). A range far outside that is a bug signal.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
ROT=$PWD/output_4bpipe/rotbase/modified_model
ORIG=$PWD/output_4b/untied_4b
W=$PWD/output_4b_napsmoke
rm -rf "$W"; mkdir -p "$W"
ln -sfn "$PWD/output_4bpipe/calibration_data.json" "$W/calibration_data.json"
ORIG_MODEL="$ORIG" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
TERNARY_BLOCK_SIZE=64 ./.venv/bin/python src/block_ap_recovery.py \
    --model-path "$ROT" --orig-config-path "$ORIG" --output-dir "$W" \
    --block-size 64 --samples 32 --nap-epochs 1 --nap-lr 1e-3 \
    > output_sweep/nap_smoke.log 2>&1
echo "rc=$?" >> output_sweep/nap_smoke.log
