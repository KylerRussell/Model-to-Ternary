#!/bin/bash
# E2E feature-distillation A/B -- SQuaT's own BASELINE, which has never been evaluated in this
# pipeline (13ae: --feat-weight was disabled by a 34GB cache-loader memory bug, not by a result).
# SQuaT itself is null here (no student feature lattice; weight-only), so this asks the prior
# question: does feature-KD at the E2E stage help AT ALL? If not, no SQuaT variant could improve it.
#
# Design: ONE teacher cache built WITH hidden states, used by BOTH arms, so --feat-weight is the
# only difference. Both start from the banked control skeleton (assignment stage skipped -- it is
# ~28h and identical for both arms anyway).
#
# Caveats recorded up front: E2E-QP trains only SCALES, so feature-KD has weaker leverage here than
# it would in block QAT; and block-AP's objective ALREADY is hidden-state MSE at all 32 layers, so
# this tests the marginal value of adding it to the logit-KD stage on top.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
ROT=$PWD/output_4bpipe/rotbase/modified_model
ORIG=$PWD/output_4b/untied_4b
SKEL=$PWD/output_4b_ctl/modified_model
CALIB=$PWD/output_4bpipe/calibration_data.json
CACHE=$PWD/output_4b_feat/teacher_hidden.pt
NS=${FEAT_NSAMP:-1000}          # seq stays 2560; 1000 seqs -> ~13GB hidden cache (mmap-backed)
RES=output_sweep/feat_results.txt
mkdir -p "$PWD/output_4b_feat"

if [ ! -f "$CACHE" ]; then
  echo "  [$(date +%F' '%H:%M:%S)] building teacher cache WITH hidden (n=$NS, seq=2560)" >> $RES
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
  ./.venv/bin/python src/e2e_qp_distill.py --precompute-teacher --teacher-path "$ROT" \
      --orig-config-path "$ORIG" --calib "$CALIB" --teacher-cache "$CACHE" --seq 2560 --topk 64 \
      --cache-batch 1 --teacher-dp 1 --gpu-mem 20GiB --cpu-mem 30GiB \
      --max-samples "$NS" --cache-hidden > output_sweep/feat_cache.log 2>&1
  echo "  [$(date +%F' '%H:%M:%S)] cache rc=$? size=$(du -h "$CACHE" 2>/dev/null | cut -f1)" >> $RES
fi

run () {   # $1 = feat weight, $2 = tag
  local FW=$1 TAG=$2 OUT=$PWD/output_4b_feat/$2
  rm -rf "$OUT"; mkdir -p "$OUT"
  echo "  [$(date +%F' '%H:%M:%S)] START e2e feat=$FW" >> $RES
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
  ./.venv/bin/python -m torch.distributed.run --nproc_per_node=2 --standalone \
      src/e2e_qp_distill.py --train --student-path "$SKEL" --orig-config-path "$ORIG" \
      --calib "$CALIB" --teacher-cache "$CACHE" --out "$OUT" \
      --seq 2560 --epochs 2 --max-samples "$NS" \
      --lr 2e-5 --lr-schedule linear --scale-ema-decay 0 --select final \
      --loss-fn cakld --feat-weight "$FW" --decision-gamma 2 \
      --col-scale --scale-qat-bits 8 \
      --commit-beta 1.5 --commit-pre 16 --commit-post 12 \
      --heldout-n 48 --eval-every 300 --abort-patience 30 \
      > output_sweep/feat_e2e_$TAG.log 2>&1
  local rc=$?
  echo "  [$(date +%F' '%H:%M:%S)] e2e feat=$FW rc=$rc" >> $RES
  [ "$rc" -ne 0 ] && return 1
  env ORIG_MODEL="$ORIG" ORIG="$ORIG" FP_DIR="$ROT" E2E_MODEL="$OUT" \
      EVAL_DATA="$PWD/output_4b/eval2k.json" NP=1946 SEQ=1024 \
      ./.venv/bin/python src/kl_flips_eval.py > output_sweep/feat_eval_$TAG.log 2>&1
  local AG KL
  AG=$(grep -aoE "top-1 agreement *: *[0-9.]+" output_sweep/feat_eval_$TAG.log | grep -oE "[0-9.]+$" | tail -1)
  KL=$(grep -aoE "mean KL\(fp\|\|tern\) *: *[0-9.]+" output_sweep/feat_eval_$TAG.log | grep -oE "[0-9.]+$" | tail -1)
  echo "  RESULT e2e feat=$FW  agreement=${AG:-NA}%  meanKL=${KL:-NA}" >> $RES
}
run 0 base
run "${FEAT_W:-1.0}" feat
echo DONE > output_sweep/.feat_done
