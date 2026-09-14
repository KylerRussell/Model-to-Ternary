#!/bin/bash
# PLACEMENT SWEEP: co-located (cpunodebind+membind, one node per rank) vs interleaved.
# Target: recover the ~175 s/step seen before --no-final-save, WITH the low spread we now have.
# Counterbalanced A,B,B,A so slot position cancels; 2 reps each so spread is visible.
# Same fused path, same --no-final-save, same 8 steps as the main sweep, so numbers are comparable.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
W=/home/kasm-user/Documents/Model-to-Ternary/output_sweep
P=$W/place_results.txt; mkdir -p $W; touch $P
LOCK=$W/place.lock
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "  [lock] placement sweep already running (pid $(cat $LOCK))" | tee -a $P; exit 1
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
watch_arm () {
  local pid=""
  for i in $(seq 1 400); do
    pid=$(ps -eo pid=,args= | grep "[e]2e_qp_distill.py --train" | grep -v torchrun | awk 'NR==1{print $1}')
    [ -n "$pid" ] && [ -d /proc/$pid ] && break; sleep 10
  done
  [ -n "$pid" ] || return
  while [ -d /proc/$pid ] && [ "$(grep -c 'pp rank 0' $1 2>/dev/null)" -lt 2 ]; do sleep 5; done
  numastat -p $pid 2>/dev/null | awk '/^Total/{print}' > $2.numa
  local f0=$(awk '{print $10}' /proc/$pid/stat 2>/dev/null) s0=$(grep -c 'pp rank 0' $1 2>/dev/null)
  local f1=$f0 s1=$s0
  while [ -d /proc/$pid ]; do
    f1=$(awk '{print $10}' /proc/$pid/stat 2>/dev/null || echo $f1)
    s1=$(grep -c 'pp rank 0' $1 2>/dev/null || echo $s1)
    sleep 10
  done
  [ "${s1:-0}" -gt "${s0:-0}" ] && echo "$(( (f1-f0)/(s1-s0) ))" > $2
}
BASE="--train --student-path $HOME/_scratch_naive27b --orig-config-path $MODEL_27B \
 --calib output_4bpipe/calibration_data.json --teacher-cache output_4bpipe/teacher_topk.pt \
 --seq 2560 --epochs 1 --max-samples 20 --lr 0 --latent-lr 0 --latent-opt adam-blockv \
 --latent-init center --latent-offload --latent-grad-release --train-weights all --tw-layer-stride 2 \
 --pipe-parallel --pipe-parallel-mb 2 --latent-pin --latent-warmup-steps 0 --lr-schedule linear \
 --scale-ema-decay 0 --select final --loss-fn cakld --decision-gamma 2 --ce-weight 0.1 \
 --ce-positions 128 --scale-qat-bits 8 --heldout-n 4 --eval-every 100000 --ckpt-every 0 \
 --abort-patience 1000000 --no-final-save"
run_p () {   # rep  wrapper-name
  local rep=$1 wname=$2; local tag="p${rep}_${wname}"
  grep -q "^  $tag " $P 2>/dev/null && { echo "  [skip] $tag"; return 0; }
  preflight || return 1
  echo "  [$(date +%F' '%H:%M:%S)] START $tag" >> $P
  watch_arm $W/$tag.log $W/$tag.faults &
  local fw=$!
  MP_SPLIT=40 PYTHONUNBUFFERED=1 ORIG_MODEL="$MODEL_27B" \
  ./.venv/bin/torchrun --nproc_per_node=2 --no-python --master_port=29671 \
    $W/numa_$wname.sh $BASE --out $W/$tag/m/modified_model > $W/$tag.log 2>&1
  local rc=$?
  wait $fw 2>/dev/null; rm -rf $W/$tag/m 2>/dev/null; sync
  local flt=$(cat $W/$tag.faults 2>/dev/null || echo "?")
  local nod=$(awk '{mx=0;mi=0;for(i=2;i<=5;i++) if($i+0>mx){mx=$i+0;mi=i-2}; printf "n%d(%.0f%%)", mi, 100*mx/($6+0.001)}' $W/$tag.faults.numa 2>/dev/null || echo "?")
  grep -oE "^\[[0-9:]+\].*step [0-9]+/[0-9]+  pp rank 0" $W/$tag.log | awk -v t="$tag" -v r="$rc" -v f="$flt" -v n="$nod" '
    function secs(x,a){gsub(/[\[\]]/,"",x);split(x,a,":");return a[1]*3600+a[2]*60+a[3]}
    {tt=secs($1); if(p){d=tt-p; if(d<0)d+=86400; c++; if(c>2){s+=d;m++}} p=tt}
    END{ if(m) printf "  %-16s %6.1f s/step  faults=%-10s mem=%-10s rc=%s\n", t, s/m, f, n, r;
         else printf "  %-16s NO_MEASUREMENT rc=%s\n", t, r }' >> $P
}
run_p 1 colocate
run_p 1 mem          # interleave=0,1,2 control (numa_mem.sh)
run_p 2 mem
run_p 2 colocate
echo "=== PLACEMENT SWEEP DONE $(date +%F' '%H:%M:%S) ===" >> $P
