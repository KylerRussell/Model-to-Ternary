#!/bin/bash
# QUASAR loss-aware reconstruction (arXiv:2608.13966) A/B at the skeleton.
# Same harness as the control, only --qat-quasar differs.
#
# READ THE NOISE FLOOR FIRST (13ac): control n=3 is 56.273% +/- 0.309 pp, KL 1.1897 +/- 0.0155.
# A single run here resolves ~0.9 pp / ~0.05 KL and NOTHING smaller. This is a screen for a LARGE
# effect; a sub-1 pp result means "not resolvable", not "small win".
#
# Screened at BLOCK scope, not the ~28h assignment stage: QUASAR is a QAT-loop method and block-AP's
# QAT is a legitimate instance of that loop. Promote to assignment training only if it clears the bar.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
ROT=$PWD/output_4bpipe/rotbase/modified_model
ORIG=$PWD/output_4b/untied_4b
NS=$(( 640 * 1024 / 2560 ))
W=$PWD/output_4b_quasar
RES=output_sweep/quasar_results.txt
rm -rf "$W"; mkdir -p "$W"
ln -sfn "$PWD/output_4bpipe/calibration_data.json" "$W/calibration_data.json"
echo "  [$(date +%F' '%H:%M:%S)] START quasar every=${QUASAR_EVERY:-50}" >> $RES
ORIG_MODEL="$ORIG" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
TERNARY_BLOCK_SIZE=64 ./.venv/bin/python src/block_ap_recovery.py \
    --model-path "$ROT" --orig-config-path "$ORIG" --output-dir "$W" \
    --block-size 64 --samples "$NS" --qat --qat-gptq-init --qat-keep-best --qat-attn-only \
    --qat-lr 1e-4 --qat-scale-lr 1e-4 --qat-epochs 4 --quant-embed-head \
    --qat-quasar "${QUASAR_EVERY:-50}" > output_sweep/quasar_blockap.log 2>&1
rc=$?
NL=$(./.venv/bin/python -c "import json;print(len(json.load(open('$W/recovery_report.json'))['layers']))" 2>/dev/null || echo 0)
echo "  [$(date +%F' '%H:%M:%S)] block-AP rc=$rc layers=$NL" >> $RES
if [ "$rc" -eq 0 ] && [ "$NL" -ge 32 ]; then
  env ORIG_MODEL="$ORIG" ORIG="$ORIG" FP_DIR="$ROT" E2E_MODEL="$W/modified_model" \
      EVAL_DATA="$PWD/output_4b/eval2k.json" NP=1946 SEQ=1024 \
      ./.venv/bin/python src/kl_flips_eval.py > output_sweep/quasar_eval2k.log 2>&1
  AG=$(grep -aoE "top-1 agreement *: *[0-9.]+" output_sweep/quasar_eval2k.log | grep -oE "[0-9.]+$" | tail -1)
  KL=$(grep -aoE "mean KL\(fp\|\|tern\) *: *[0-9.]+" output_sweep/quasar_eval2k.log | grep -oE "[0-9.]+$" | tail -1)
  echo "  RESULT quasar  agreement=${AG:-NA}%  meanKL=${KL:-NA}   (control n=3: 56.273+/-0.309% / 1.1897+/-0.0155)" >> $RES
else
  echo "  ABORT quasar" >> $RES
fi
echo DONE > output_sweep/.quasar_done
