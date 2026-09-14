#!/bin/bash
# CORRECTNESS VALIDATION at a real learning rate.
#
# Everything in the throughput sweeps ran at --lr 0, so NOTHING trained and the fused path's optimizer
# step was never exercised for real. The fused path moves WHERE the update happens: instead of
# autograd's AccumulateGrad filling param.grad and a post-accumulate hook stepping, the transfer's
# backward stages into a host arena, sets param.grad, calls the SAME hook, and returns None. Unit
# tests say the gradients are bit-identical. This checks the TRAJECTORY at scale.
#
# The specific risk: if _UseTransferred.backward fires more than once per microbatch (gradient
# checkpointing recomputes the forward), the fused path would apply TWO optimizer steps where the old
# path accumulated into one grad and stepped once. That divergence is invisible at --lr 0.
#
# Also exercises, for the first time: the PERIODIC pipelined held-out eval (--eval-every 4) and
# pp_sync_for_save (final save NOT skipped, so rank 1's trained half must cross to rank 0).
#
# SCOPE IS DELIBERATELY SMALL (`--train-weights down --tw-layer-stride 4`, not `all`/stride 2).
# Correctness does not need production scale, and production scale is dangerous here: EVERY throughput
# arm ran at --lr 0, where the optimizer step is SKIPPED, so Adam state was never allocated in any of
# them. The 244 GB "safe" peak they established does not apply to real training. At lr 5e-7 the first
# backward allocates ~30.7 GB of exp_avg on rank 0 plus ~18 GB on rank 1, and a full-scope attempt was
# still climbing through 269 GB when the container died. This scope exercises the identical code paths
# at a fraction of the footprint.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
W=/home/kasm-user/Documents/Model-to-Ternary/output_sweep
P=$W/validate_results.txt; mkdir -p $W; touch $P
LOCK=$W/validate.lock
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "  [lock] validation already running (pid $(cat $LOCK))" | tee -a $P; exit 1
fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
preflight () {
  for i in $(seq 1 160); do
    local tr=$(ps -eo args= | grep "[e]2e_qp_distill.py --train" | grep -vc torchrun)
    local g0=$(timeout 20 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "$tr" -eq 0 ] && [ "${g0:-9999}" -lt 500 ] && return 0
    sleep 15
  done
  echo "  [preflight] ABORT" >> $P; return 1
}
BASE="--train --student-path $HOME/_scratch_naive27b --orig-config-path $MODEL_27B \
 --calib output_4bpipe/calibration_data.json --teacher-cache output_4bpipe/teacher_topk.pt \
 --seq 2560 --epochs 1 --max-samples 20 --lr 0 --latent-lr 5e-7 --latent-opt adam-blockv \
 --latent-init center --latent-offload --latent-grad-release --train-weights down --tw-layer-stride 4 \
 --pipe-parallel --pipe-parallel-mb 2 --latent-pin --latent-warmup-steps 0 --lr-schedule linear \
 --scale-ema-decay 0 --select final --loss-fn cakld --decision-gamma 2 --ce-weight 0.1 \
 --ce-positions 128 --scale-qat-bits 8 --heldout-n 4 --eval-every 4 --ckpt-every 0 \
 --abort-patience 1000000"
run_v () {   # tag  [extra flags]
  local tag=$1; shift
  grep -q "^  $tag " $P 2>/dev/null && { echo "  [skip] $tag"; return 0; }
  preflight || return 1
  echo "  [$(date +%F' '%H:%M:%S)] START $tag ($*)" >> $P
  MP_SPLIT=40 PYTHONUNBUFFERED=1 ORIG_MODEL="$MODEL_27B" \
  ./.venv/bin/torchrun --nproc_per_node=2 --no-python --master_port=29681 \
    $W/numa_colocate.sh $BASE --out $W/$tag/m/modified_model "$@" > $W/$tag.log 2>&1
  local rc=$?
  local saved=$(grep -c "pulled .* stage-1 modules from rank 1" $W/$tag.log)
  local evals=$(grep -c "\[held-out\] step" $W/$tag.log)
  rm -rf $W/$tag/m 2>/dev/null; sync
  echo "  $tag  rc=$rc  periodic_evals=$evals  save_gather_ran=$saved" >> $P
  grep -oE "step [0-9]+/[0-9]+  pp rank 0 mb=2 kl=[0-9.]+" $W/$tag.log \
    | sed 's/.*step \([0-9]*\).*kl=/    step \1 kl=/' >> $P
}
run_v fused
run_v nofused --no-fused-latent-grad
echo "=== VALIDATION DONE $(date +%F' '%H:%M:%S) ===" >> $P
