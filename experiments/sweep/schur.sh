#!/bin/bash
# SchurOpt grid-refit A/B at the SKELETON stage (arXiv 2608.15567 Eq. 16).
# Only Phase 3 changes, so we score the skeleton on the DETERMINISTIC eval2k referee (teacher-forced,
# no sampling) against our recorded baseline: skeleton 62.92% agreement / KL 0.9222. That avoids the
# 28 h assignment + 6 h E2E, and avoids Gate B, which needs multi-seed to resolve anything.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
W=output_4b_schur
ROT=$PWD/output_4bpipe/rotbase/modified_model
ORIG=$PWD/output_4b/untied_4b
R=${REFIT:-4}
SEQ=2560; NS=$(( 640 * 1024 / SEQ ))
echo "  [$(date +%F' '%H:%M:%S)] START schur refit=$R samples=$NS" >> output_sweep/schur_results.txt
# ORIG_MODEL MUST be exported before python starts: config.py binds NUM_HIDDEN_LAYERS at IMPORT
# time and defaults to the 27B's 64 layers, so without it a 4B run walks off the end at layer 33.
ORIG_MODEL="$ORIG" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 TERNARY_BLOCK_SIZE=64 \
./.venv/bin/python src/block_ap_recovery.py --model-path "$ROT" --orig-config-path "$ORIG" \
    --output-dir "$W" --block-size 64 --samples "$NS" \
    --qat --qat-gptq-init --qat-keep-best --qat-attn-only \
    --qat-lr 1e-4 --qat-scale-lr 1e-4 --qat-epochs 4 --quant-embed-head \
    --gptq-refit-iters "$R" > output_sweep/schur_blockap.log 2>&1
rc=$?
echo "  [$(date +%F' '%H:%M:%S)] block-AP rc=$rc" >> output_sweep/schur_results.txt
[ $rc -ne 0 ] && exit 1
env ORIG_MODEL="$ORIG" ORIG="$ORIG" FP_DIR="$ROT" E2E_MODEL="$W/modified_model" \
    EVAL_DATA="$PWD/output_4b/eval2k.json" NP=1946 SEQ=1024 \
    ./.venv/bin/python src/kl_flips_eval.py > output_sweep/schur_eval2k.log 2>&1
AG=$(grep -aoE "top-1 agreement *: *[0-9.]+" output_sweep/schur_eval2k.log | grep -oE "[0-9.]+$" | tail -1)
KL=$(grep -aoE "mean KL\(fp\|\|tern\) *: *[0-9.]+" output_sweep/schur_eval2k.log | grep -oE "[0-9.]+$" | tail -1)
echo "  SCHUR refit=$R  agreement=${AG:-NA}%  meanKL=${KL:-NA}   (baseline 62.92% / 0.9222)" >> output_sweep/schur_results.txt
echo DONE > output_sweep/.schur_done
