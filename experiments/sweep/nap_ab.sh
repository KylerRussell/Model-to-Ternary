#!/bin/bash
# NAP (arXiv:2608.03919) normalization-affine PRECONDITIONING, A/B at the skeleton.
# Same harness as icbq_ab.sh / schur_ab.sh, so it is directly comparable to the banked control:
#   block-AP on rotbase -> 56.11% agreement, meanKL 1.1988.
#
# Two stages:
#   1. precondition: freeze the backbone at its ternary dequant, tune ONLY the norm gains on the
#      FP model, write a preconditioned FP model. Recovery is skipped in this pass.
#   2. recover:      run the IDENTICAL block-AP recipe on that preconditioned model.
# So the only difference vs the control is the norm gains of the model block-AP receives.
#
# What to watch: S7 already worked this axis (--col-scale, trained gains inside [0.994, 1.006],
# paired dKL -0.00038). If NAP's reported gain-ratio range comes out far outside that, treat it as
# a bug, not a win -- and note QuaRot folds these affines away by design (stored w == 0), so the
# preconditioning lives in the ROTATED basis where the paper's per-channel alpha_c is not the same
# object it is in their CNN/BN setting.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
ROT=$PWD/output_4bpipe/rotbase/modified_model
ORIG=$PWD/output_4b/untied_4b
NS=$(( 640 * 1024 / 2560 ))
EP=${NAP_EPOCHS:-2}
LR=${NAP_LR:-1e-3}
NAPDIR=$PWD/output_4b_napmodel
W=$PWD/output_4b_nap
RES=output_sweep/nap_results.txt

echo "  [$(date +%F' '%H:%M:%S)] START nap precondition ep=$EP lr=$LR" >> $RES
rm -rf "$NAPDIR"; mkdir -p "$NAPDIR"
ln -sfn "$PWD/output_4bpipe/calibration_data.json" "$NAPDIR/calibration_data.json"
ORIG_MODEL="$ORIG" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
TERNARY_BLOCK_SIZE=64 ./.venv/bin/python src/block_ap_recovery.py \
    --model-path "$ROT" --orig-config-path "$ORIG" --output-dir "$NAPDIR" \
    --block-size 64 --samples "$NS" --nap-epochs "$EP" --nap-lr "$LR" --nap-trust "${NAP_TRUST:-1.0}" \
    > output_sweep/nap_precond.log 2>&1
rc=$?
echo "  [$(date +%F' '%H:%M:%S)] precondition rc=$rc" >> $RES
[ "$rc" -ne 0 ] && { echo "  ABORT: preconditioning failed" >> $RES; exit 1; }
grep -a "effective gain ratio" output_sweep/nap_precond.log | tail -3 >> $RES

# stage 2: the SAME block-AP recipe as the control, on the preconditioned model
echo "  [$(date +%F' '%H:%M:%S)] START block-AP on preconditioned model" >> $RES
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
[ "$rc" -ne 0 ] && exit 1
[ "$NL" -lt 32 ] && { echo "  ABORT: only $NL layers recovered" >> $RES; exit 1; }

# NOTE: the referee compares against the ORIGINAL rotated FP model, not the preconditioned one,
# so the NAP arm gets no credit for moving the reference.
env ORIG_MODEL="$ORIG" ORIG="$ORIG" FP_DIR="$ROT" E2E_MODEL="$W/modified_model" \
    EVAL_DATA="$PWD/output_4b/eval2k.json" NP=1946 SEQ=1024 \
    ./.venv/bin/python src/kl_flips_eval.py > output_sweep/nap_eval2k.log 2>&1
AG=$(grep -aoE "top-1 agreement *: *[0-9.]+" output_sweep/nap_eval2k.log | grep -oE "[0-9.]+$" | tail -1)
KL=$(grep -aoE "mean KL\(fp\|\|tern\) *: *[0-9.]+" output_sweep/nap_eval2k.log | grep -oE "[0-9.]+$" | tail -1)
echo "  RESULT nap ep=$EP lr=$LR  agreement=${AG:-NA}%  meanKL=${KL:-NA}   (control: 56.11% / 1.1988)" >> $RES
echo DONE > output_sweep/.nap_done
