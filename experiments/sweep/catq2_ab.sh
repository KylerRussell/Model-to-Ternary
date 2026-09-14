#!/bin/bash
# CAT-Q, ISOLATING run: alpha frozen at the GPTQ level, mu clamped. The soft-vs-hard forward (and its
# gradient) is then the ONLY difference from the 56.273% control -- a true single-variable test of the
# mechanism, after two confounded attempts (runaway mu -> wrong diagnosis; then runaway alpha).
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
ROT=$PWD/output_4bpipe/rotbase/modified_model; ORIG=$PWD/output_4b/untied_4b
NS=$(( 640 * 1024 / 2560 )); W=$PWD/output_4b_catq2; RES=output_sweep/catq2_results.txt
rm -rf "$W"; mkdir -p "$W"; ln -sfn "$PWD/output_4bpipe/calibration_data.json" "$W/calibration_data.json"
echo "  [$(date +%F' '%H:%M:%S)] START catq2 (alpha frozen, mu capped)" >> $RES
ORIG_MODEL="$ORIG" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
TERNARY_BLOCK_SIZE=64 ./.venv/bin/python src/block_ap_recovery.py \
    --model-path "$ROT" --orig-config-path "$ORIG" --output-dir "$W" \
    --block-size 64 --samples "$NS" --qat --qat-gptq-init --qat-keep-best --qat-attn-only \
    --qat-lr 1e-4 --qat-scale-lr 1e-4 --qat-epochs 4 --quant-embed-head \
    --qat-catq --catq-freeze-alpha > output_sweep/catq2_blockap.log 2>&1
rc=$?; NL=$(./.venv/bin/python -c "import json;print(len(json.load(open('$W/recovery_report.json'))['layers']))" 2>/dev/null || echo 0)
echo "  [$(date +%F' '%H:%M:%S)] block-AP rc=$rc layers=$NL" >> $RES
if [ "$rc" -eq 0 ] && [ "$NL" -ge 32 ]; then
  env ORIG_MODEL="$ORIG" ORIG="$ORIG" FP_DIR="$ROT" E2E_MODEL="$W/modified_model" \
      EVAL_DATA="$PWD/output_4b/eval2k.json" NP=1946 SEQ=1024 \
      ./.venv/bin/python src/kl_flips_eval.py > output_sweep/catq2_eval2k.log 2>&1
  AG=$(grep -aoE "top-1 agreement *: *[0-9.]+" output_sweep/catq2_eval2k.log | grep -oE "[0-9.]+$" | tail -1)
  KL=$(grep -aoE "mean KL\(fp\|\|tern\) *: *[0-9.]+" output_sweep/catq2_eval2k.log | grep -oE "[0-9.]+$" | tail -1)
  echo "  RESULT catq2  agreement=${AG:-NA}%  meanKL=${KL:-NA}   (control n=3: 56.273+/-0.309% / 1.1897+/-0.0155)" >> $RES
else echo "  ABORT catq2" >> $RES; fi
echo DONE > output_sweep/.catq2_done
