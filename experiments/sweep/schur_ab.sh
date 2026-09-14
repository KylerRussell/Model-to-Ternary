#!/bin/bash
# CLEAN A/B: same harness, same inputs, ONLY --gptq-refit-iters differs.
# Two prior invalidations forced this design:
#   1. ORIG_MODEL must be EXPORTED (config.py binds NUM_HIDDEN_LAYERS at import; default 64 = the 27B).
#      Without it a 4B run walks off the end at layer 33 AND leaves stale _recovery_staging.
#   2. That stale staging then satisfies the resume check (inputs_after_l + layer_l for l<32), so the
#      next run sets start_layer=32, runs an EMPTY recovery loop and saves an UNRECOVERED model
#      (recovery_report layers:{}, eval2k 3.75%). Always start from a clean output dir.
#   3. The recorded 62.92% skeleton came from output_4bpipe, a DIFFERENT earlier run, so it is not a
#      valid control for this harness. Run refit=0 here too.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
ROT=$PWD/output_4bpipe/rotbase/modified_model
ORIG=$PWD/output_4b/untied_4b
NS=$(( 640 * 1024 / 2560 ))
run () {   # $1 = optimizer (gptq|schuropt), $2 = work dir
  local R=$1 W=$2
  rm -rf "$W"; mkdir -p "$W"
  ln -sfn "$PWD/output_4bpipe/calibration_data.json" "$W/calibration_data.json"
  echo "  [$(date +%F' '%H:%M:%S)] START opt=$R -> $W" >> output_sweep/schur_results.txt
  ORIG_MODEL="$ORIG" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
  TERNARY_BLOCK_SIZE=64 ./.venv/bin/python src/block_ap_recovery.py \
      --model-path "$ROT" --orig-config-path "$ORIG" --output-dir "$W" \
      --block-size 64 --samples "$NS" --qat --qat-gptq-init --qat-keep-best --qat-attn-only \
      --qat-lr 1e-4 --qat-scale-lr 1e-4 --qat-epochs 4 --quant-embed-head \
      --quant-optimizer "$R" --schuropt-refine 8 > output_sweep/schur_blockap_$R.log 2>&1
  local rc=$?
  local NL=$(./.venv/bin/python -c "import json;print(len(json.load(open('$W/recovery_report.json'))['layers']))" 2>/dev/null || echo 0)
  echo "  [$(date +%F' '%H:%M:%S)] block-AP opt=$R rc=$rc layers_recovered=$NL" >> output_sweep/schur_results.txt
  [ "$rc" -ne 0 ] && return 1
  [ "$NL" -lt 32 ] && { echo "  ABORT opt=$R: only $NL layers recovered" >> output_sweep/schur_results.txt; return 1; }
  env ORIG_MODEL="$ORIG" ORIG="$ORIG" FP_DIR="$ROT" E2E_MODEL="$W/modified_model" \
      EVAL_DATA="$PWD/output_4b/eval2k.json" NP=1946 SEQ=1024 \
      ./.venv/bin/python src/kl_flips_eval.py > output_sweep/schur_eval2k_$R.log 2>&1
  local AG=$(grep -aoE "top-1 agreement *: *[0-9.]+" output_sweep/schur_eval2k_$R.log | grep -oE "[0-9.]+$" | tail -1)
  local KL=$(grep -aoE "mean KL\(fp\|\|tern\) *: *[0-9.]+" output_sweep/schur_eval2k_$R.log | grep -oE "[0-9.]+$" | tail -1)
  echo "  RESULT opt=$R  agreement=${AG:-NA}%  meanKL=${KL:-NA}" >> output_sweep/schur_results.txt
}
run schuropt output_4b_schuropt
echo DONE > output_sweep/.schur_done
