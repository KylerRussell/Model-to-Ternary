#!/bin/bash
# NAP stage 2: the IDENTICAL block-AP recipe as the 56.11% control, run on the PRECONDITIONED FP
# model produced by the trust-regioned preconditioning pass. The only difference vs the control is
# the norm gains the recovery receives.
#
# The referee scores against the ORIGINAL rotated FP model (FP_DIR=$ROT), not the preconditioned
# one, so NAP has to overcome its own FP drift rather than get credit for moving the reference.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
ROT=$PWD/output_4bpipe/rotbase/modified_model
ORIG=$PWD/output_4b/untied_4b
NAPDIR=${NAPDIR:-$PWD/output_4b_napprobe}
W=$PWD/output_4b_nap
NS=$(( 640 * 1024 / 2560 ))
RES=output_sweep/nap_results.txt

echo "  [$(date +%F' '%H:%M:%S)] START block-AP on preconditioned model ($NAPDIR)" >> $RES
rm -rf "$W"; mkdir -p "$W"
ln -sfn "$PWD/output_4bpipe/calibration_data.json" "$W/calibration_data.json"
ORIG_MODEL="$ORIG" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
TERNARY_BLOCK_SIZE=64 ./.venv/bin/python src/block_ap_recovery.py \
    --model-path "$NAPDIR/modified_model" --orig-config-path "$ORIG" --output-dir "$W" \
    --block-size 64 --samples "$NS" --qat --qat-gptq-init --qat-keep-best --qat-attn-only \
    --qat-lr 1e-4 --qat-scale-lr 1e-4 --qat-epochs 4 --quant-embed-head \
    > output_sweep/nap_blockap.log 2>&1
rc=$?
NL=$(./.venv/bin/python -c "import json;print(len(json.load(open('$W/recovery_report.json'))['layers']))" 2>/dev/null || echo 0)
echo "  [$(date +%F' '%H:%M:%S)] block-AP rc=$rc layers_recovered=$NL" >> $RES
[ "$rc" -ne 0 ] && { echo DONE > output_sweep/.nap_done; exit 1; }
[ "$NL" -lt 32 ] && { echo "  ABORT: only $NL layers recovered" >> $RES; echo DONE > output_sweep/.nap_done; exit 1; }

env ORIG_MODEL="$ORIG" ORIG="$ORIG" FP_DIR="$ROT" E2E_MODEL="$W/modified_model" \
    EVAL_DATA="$PWD/output_4b/eval2k.json" NP=1946 SEQ=1024 \
    ./.venv/bin/python src/kl_flips_eval.py > output_sweep/nap_eval2k.log 2>&1
AG=$(grep -aoE "top-1 agreement *: *[0-9.]+" output_sweep/nap_eval2k.log | grep -oE "[0-9.]+$" | tail -1)
KL=$(grep -aoE "mean KL\(fp\|\|tern\) *: *[0-9.]+" output_sweep/nap_eval2k.log | grep -oE "[0-9.]+$" | tail -1)
echo "  RESULT nap(trust)  agreement=${AG:-NA}%  meanKL=${KL:-NA}   (control: 56.11% / 1.1988 | icbq: 56.55% / 1.1767)" >> $RES
echo DONE > output_sweep/.nap_done
