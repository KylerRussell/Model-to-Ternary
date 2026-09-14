#!/bin/bash
# CAT-Q soft ternarization (ScaleQ-1.58, arXiv:2608.01078 Eq. 2) A/B at the skeleton.
# Same harness as the control, only --qat-catq differs.
#
# READ THE NOISE FLOOR FIRST (13ac): control n=3 is 56.273% +/- 0.309 pp, KL 1.1897 +/- 0.0155.
# A single run here resolves ~0.9 pp / ~0.05 KL and NOTHING smaller. This is a screen for a LARGE
# effect; a sub-1 pp result means "not resolvable", not "small win".
#
# Scope note: the OTHER half of that paper (AYOT CoT-aware calibration) is already in this pipeline
# and was derived here independently -- our calib is 50.0% <think>-bearing. CAT-Q is the only
# untested piece of paper #12.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
ROT=$PWD/output_4bpipe/rotbase/modified_model
ORIG=$PWD/output_4b/untied_4b
NS=$(( 640 * 1024 / 2560 ))
W=$PWD/output_4b_catq
RES=output_sweep/catq_results.txt
rm -rf "$W"; mkdir -p "$W"
ln -sfn "$PWD/output_4bpipe/calibration_data.json" "$W/calibration_data.json"
echo "  [$(date +%F' '%H:%M:%S)] START catq s=${CATQ_S:-20}" >> $RES
ORIG_MODEL="$ORIG" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
TERNARY_BLOCK_SIZE=64 ./.venv/bin/python src/block_ap_recovery.py \
    --model-path "$ROT" --orig-config-path "$ORIG" --output-dir "$W" \
    --block-size 64 --samples "$NS" --qat --qat-gptq-init --qat-keep-best --qat-attn-only \
    --qat-lr 1e-4 --qat-scale-lr 1e-4 --qat-epochs 4 --quant-embed-head \
    --qat-catq --catq-sharpness "${CATQ_S:-20}" > output_sweep/catq_blockap.log 2>&1
rc=$?
NL=$(./.venv/bin/python -c "import json;print(len(json.load(open('$W/recovery_report.json'))['layers']))" 2>/dev/null || echo 0)
echo "  [$(date +%F' '%H:%M:%S)] block-AP rc=$rc layers=$NL" >> $RES
if [ "$rc" -eq 0 ] && [ "$NL" -ge 32 ]; then
  env ORIG_MODEL="$ORIG" ORIG="$ORIG" FP_DIR="$ROT" E2E_MODEL="$W/modified_model" \
      EVAL_DATA="$PWD/output_4b/eval2k.json" NP=1946 SEQ=1024 \
      ./.venv/bin/python src/kl_flips_eval.py > output_sweep/catq_eval2k.log 2>&1
  AG=$(grep -aoE "top-1 agreement *: *[0-9.]+" output_sweep/catq_eval2k.log | grep -oE "[0-9.]+$" | tail -1)
  KL=$(grep -aoE "mean KL\(fp\|\|tern\) *: *[0-9.]+" output_sweep/catq_eval2k.log | grep -oE "[0-9.]+$" | tail -1)
  echo "  RESULT catq  agreement=${AG:-NA}%  meanKL=${KL:-NA}   (control n=3: 56.273+/-0.309% / 1.1897+/-0.0155)" >> $RES
else
  echo "  ABORT catq" >> $RES
fi
echo DONE > output_sweep/.catq_done
