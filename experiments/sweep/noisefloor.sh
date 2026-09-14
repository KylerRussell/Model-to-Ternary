#!/bin/bash
# NOISE FLOOR for skeleton A/Bs in this harness.
#
# Every paper result so far is N=1 per arm, and the only surviving positive (ICBQ, +0.44 pp) sits at
# a magnitude we cannot currently distinguish from run-to-run variance. block-AP has no global seed:
# the QAT loop's torch.randperm draws from the unseeded global RNG, so re-running the IDENTICAL
# config with the IDENTICAL data samples exactly the variance we need.
#
# Two repeats + the banked control (56.11% / 1.1988) gives n=3 and an SD estimate, which calibrates
# every future skeleton A/B here -- retroactively including ICBQ and NAP.
#
# Config is byte-identical to the control arm of schur_ab.sh (gptq optimizer, no icbq, no nap).
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
ROT=$PWD/output_4bpipe/rotbase/modified_model
ORIG=$PWD/output_4b/untied_4b
NS=$(( 640 * 1024 / 2560 ))
RES=output_sweep/noisefloor_results.txt
run () {
  local TAG=$1 W=$PWD/output_4b_nf$1
  rm -rf "$W"; mkdir -p "$W"
  ln -sfn "$PWD/output_4bpipe/calibration_data.json" "$W/calibration_data.json"
  echo "  [$(date +%F' '%H:%M:%S)] START repeat $TAG" >> $RES
  ORIG_MODEL="$ORIG" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
  TERNARY_BLOCK_SIZE=64 ./.venv/bin/python src/block_ap_recovery.py \
      --model-path "$ROT" --orig-config-path "$ORIG" --output-dir "$W" \
      --block-size 64 --samples "$NS" --qat --qat-gptq-init --qat-keep-best --qat-attn-only \
      --qat-lr 1e-4 --qat-scale-lr 1e-4 --qat-epochs 4 --quant-embed-head \
      > output_sweep/nf_blockap_$TAG.log 2>&1
  local rc=$? NL
  NL=$(./.venv/bin/python -c "import json;print(len(json.load(open('$W/recovery_report.json'))['layers']))" 2>/dev/null || echo 0)
  echo "  [$(date +%F' '%H:%M:%S)] repeat $TAG block-AP rc=$rc layers=$NL" >> $RES
  [ "$rc" -ne 0 ] || [ "$NL" -lt 32 ] && { echo "  ABORT repeat $TAG" >> $RES; return 1; }
  env ORIG_MODEL="$ORIG" ORIG="$ORIG" FP_DIR="$ROT" E2E_MODEL="$W/modified_model" \
      EVAL_DATA="$PWD/output_4b/eval2k.json" NP=1946 SEQ=1024 \
      ./.venv/bin/python src/kl_flips_eval.py > output_sweep/nf_eval2k_$TAG.log 2>&1
  local AG KL
  AG=$(grep -aoE "top-1 agreement *: *[0-9.]+" output_sweep/nf_eval2k_$TAG.log | grep -oE "[0-9.]+$" | tail -1)
  KL=$(grep -aoE "mean KL\(fp\|\|tern\) *: *[0-9.]+" output_sweep/nf_eval2k_$TAG.log | grep -oE "[0-9.]+$" | tail -1)
  echo "  RESULT repeat $TAG  agreement=${AG:-NA}%  meanKL=${KL:-NA}   (ctl run1: 56.11% / 1.1988)" >> $RES
}
run r2
run r3
echo DONE > output_sweep/.nf_done
