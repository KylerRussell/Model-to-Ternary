#!/bin/bash
# ICBQ (arXiv:2608.09595) A/B at the skeleton. Same harness as schur_ab.sh so the numbers are
# directly comparable to the banked control:  refit=0 / gptq  ->  56.11% agreement, meanKL 1.1988.
#
# Arms (set by $ARMS, default "4"):
#   0   ICBQ off              -- rebuild of the control, only if you need it again
#   4   ICBQ K=4              -- the paper's default chunk size
#   99  ICBQ K>=n_layers      -- their "K = L" Sequential-CBQ baseline: one window, NO seam.
#                                Run this ONLY if K=4 beats the control, to attribute the gain to
#                                the seam schedule rather than to the two-block objective alone.
# The three hard-won guards from 13y are kept: export ORIG_MODEL (config.py binds
# NUM_HIDDEN_LAYERS at import), always start from a clean output dir (stale _recovery_staging
# silently yields an UNRECOVERED model), and assert layers_recovered == 32.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
ROT=$PWD/output_4bpipe/rotbase/modified_model
ORIG=$PWD/output_4b/untied_4b
NS=$(( 640 * 1024 / 2560 ))
RES=output_sweep/icbq_results.txt
run () {   # $1 = icbq chunk K (0 = off), $2 = work dir
  local K=$1 W=$2
  rm -rf "$W"; mkdir -p "$W"
  ln -sfn "$PWD/output_4bpipe/calibration_data.json" "$W/calibration_data.json"
  echo "  [$(date +%F' '%H:%M:%S)] START icbq_K=$K -> $W" >> $RES
  ICBQ_SELFTEST=1 ORIG_MODEL="$ORIG" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
  TERNARY_BLOCK_SIZE=64 ./.venv/bin/python src/block_ap_recovery.py \
      --model-path "$ROT" --orig-config-path "$ORIG" --output-dir "$W" \
      --block-size 64 --samples "$NS" --qat --qat-gptq-init --qat-keep-best --qat-attn-only \
      --qat-lr 1e-4 --qat-scale-lr 1e-4 --qat-epochs 4 --quant-embed-head \
      --icbq-chunk "$K" > output_sweep/icbq_blockap_K$K.log 2>&1
  local rc=$?
  local NL=$(./.venv/bin/python -c "import json;print(len(json.load(open('$W/recovery_report.json'))['layers']))" 2>/dev/null || echo 0)
  local PR=$(./.venv/bin/python -c "import json;d=json.load(open('$W/recovery_report.json')).get('icbq');print('%s pairs, %s seam revisits'%(d['pairs'],d['seam_visits']) if d else 'n/a')" 2>/dev/null || echo n/a)
  echo "  [$(date +%F' '%H:%M:%S)] block-AP K=$K rc=$rc layers_recovered=$NL  icbq: $PR" >> $RES
  [ "$rc" -ne 0 ] && return 1
  [ "$NL" -lt 32 ] && { echo "  ABORT K=$K: only $NL layers recovered" >> $RES; return 1; }
  env ORIG_MODEL="$ORIG" ORIG="$ORIG" FP_DIR="$ROT" E2E_MODEL="$W/modified_model" \
      EVAL_DATA="$PWD/output_4b/eval2k.json" NP=1946 SEQ=1024 \
      ./.venv/bin/python src/kl_flips_eval.py > output_sweep/icbq_eval2k_K$K.log 2>&1
  local AG=$(grep -aoE "top-1 agreement *: *[0-9.]+" output_sweep/icbq_eval2k_K$K.log | grep -oE "[0-9.]+$" | tail -1)
  local KL=$(grep -aoE "mean KL\(fp\|\|tern\) *: *[0-9.]+" output_sweep/icbq_eval2k_K$K.log | grep -oE "[0-9.]+$" | tail -1)
  echo "  RESULT icbq_K=$K  agreement=${AG:-NA}%  meanKL=${KL:-NA}   (control: 56.11% / 1.1988)" >> $RES
}
for K in ${ARMS:-4}; do run "$K" "$PWD/output_4b_icbq$K"; done
echo DONE > output_sweep/.icbq_done
