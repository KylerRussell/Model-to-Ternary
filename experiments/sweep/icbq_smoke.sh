#!/bin/bash
# ICBQ plumbing smoke. Tiny config (samples 32, 1 epoch/pair, K=4) run ONLY to prove the
# schedule executes and the pair objective is in the same band as the single-block sweep.
# The matched control is output_sweep/smoke_gptq.log (identical --samples/--qat-epochs, ICBQ off):
#   L0 8.760e-05  L1 6.681e-05  L2 1.246e-04  L3 2.062e-04 ... L17 1.634e-03
# Building the control FIRST is the 13z lesson: without it "is 1.85e+01 bad?" is unanswerable.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
ROT=$PWD/output_4bpipe/rotbase/modified_model
ORIG=$PWD/output_4b/untied_4b
W=$PWD/output_4b_smokei
rm -rf "$W"; mkdir -p "$W"
ln -sfn "$PWD/output_4bpipe/calibration_data.json" "$W/calibration_data.json"
ICBQ_SELFTEST=1 ORIG_MODEL="$ORIG" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
TERNARY_BLOCK_SIZE=64 ./.venv/bin/python src/block_ap_recovery.py \
    --model-path "$ROT" --orig-config-path "$ORIG" --output-dir "$W" \
    --block-size 64 --samples 32 --qat --qat-gptq-init --qat-keep-best --qat-attn-only \
    --qat-lr 1e-4 --qat-scale-lr 1e-4 --qat-epochs 1 --quant-embed-head \
    --icbq-chunk 4 > output_sweep/icbq_smoke.log 2>&1
echo "rc=$?" >> output_sweep/icbq_smoke.log
