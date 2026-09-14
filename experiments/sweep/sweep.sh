#!/bin/bash
# THE SWEEP -- durable and RESUMABLE.
#
# Every throughput claim in the campaign was measured without NUMA placement control and is
# unreliable (RESULTS_SUMMARY 13j). This re-measures all of them under strict membind, two reps per
# config in counterbalanced order (forward then reverse) so any time trend cancels. One variable per
# config -- three earlier results were ruined by bundling two.
#
# Lives under output_sweep/ (gitignored, DURABLE). The container resets and wipes /tmp; a previous
# 12-arm run kept everything there and lost all of it. Completed arms are recorded in results.txt and
# SKIPPED on relaunch, so a reset costs one arm, not the sweep.
#   bash experiments/sweep/sweep.sh          # run / resume
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
W=/home/kasm-user/Documents/Model-to-Ternary/output_sweep
P=$W/results.txt; mkdir -p $W; touch $P
# SINGLE-INSTANCE LOCK. Relaunching after a reset while an older driver was still alive (blocked in
# preflight, waiting for an idle box) put TWO drivers in the sweep: they ran the same arm twice and
# then collided on the torchrun master_port (EADDRINUSE), corrupting results.txt and killing an arm.
LOCK=$W/sweep.lock
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "  [lock] a sweep driver (pid $(cat "$LOCK")) is already running -- refusing to start a second" | tee -a $P
  exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
preflight () {
  for i in $(seq 1 160); do
    local tr=$(ps -eo args= | grep "[e]2e_qp_distill.py --train" | grep -vc torchrun)
    local g0=$(timeout 20 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    local av=$(free -g | awk 'NR==2{print $7}')
    [ "$tr" -eq 0 ] && [ "${g0:-9999}" -lt 500 ] && [ "${av:-0}" -gt 300 ] && return 0
    sleep 15
  done
  echo "  [preflight] ABORT: box not idle" >> $P; return 1
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
run_cfg () {   # rep name split spike [extra flags...]
  local rep=$1 name=$2 split=$3 spike=$4; shift 4
  local tag="r${rep}_${name}"
  grep -q "^  $tag " $P 2>/dev/null && { echo "  [skip] $tag already recorded"; return 0; }
  preflight || return 1
  echo "  [$(date +%F' '%H:%M:%S)] START $tag (split=$split spike=$spike $*)" >> $P
  # reset the cgroup peak counter so each arm's high-water mark is its own
  CGD="/sys/fs/cgroup$(awk -F: '{print $3}' /proc/self/cgroup | head -1)"
  [ -w "$CGD/memory.peak" ] && echo 0 > "$CGD/memory.peak" 2>/dev/null
  watch_arm $W/$tag.log $W/$tag.faults &
  local fw=$!
  SPIKE_NO_H2D=$spike MP_SPLIT=$split PYTHONUNBUFFERED=1 ORIG_MODEL="$MODEL_27B" \
  ./.venv/bin/torchrun --nproc_per_node=2 --no-python --master_port=29661 \
    $W/numa_mem.sh $BASE --out $W/$tag/m/modified_model "$@" > $W/$tag.log 2>&1
  local rc=$?
  local cgpk=$(( $(cat "$CGD/memory.peak" 2>/dev/null || echo 0) / 1073741824 ))
  wait $fw 2>/dev/null
  rm -rf $W/$tag/m 2>/dev/null            # nothing should be here now (--no-final-save)
  sync                                     # flush dirty pages before the next arm allocates
  local flt=$(cat $W/$tag.faults 2>/dev/null || echo "?")
  local nod=$(awk '{mx=0;mi=0;for(i=2;i<=5;i++) if($i+0>mx){mx=$i+0;mi=i-2}; printf "n%d(%.0f%%)", mi, 100*mx/($6+0.001)}' $W/$tag.faults.numa 2>/dev/null || echo "?")
  grep -oE "^\[[0-9:]+\].*step [0-9]+/[0-9]+  pp rank 0" $W/$tag.log | awk -v t="$tag" -v r="$rc" -v f="$flt" -v n="$nod" -v k="$cgpk" '
    function secs(x,a){gsub(/[\[\]]/,"",x);split(x,a,":");return a[1]*3600+a[2]*60+a[3]}
    {tt=secs($1); if(p){d=tt-p; if(d<0)d+=86400; c++; if(c>2){s+=d;m++}} p=tt}
    END{ if(m) printf "  %-16s %6.1f s/step  faults=%-10s mem=%-10s cgpeak=%sGB rc=%s\n", t, s/m, f, n, k, r;
         else printf "  %-16s NO_MEASUREMENT rc=%s\n", t, r }' >> $P
}
# ORDER MATTERS: the container has been OOM-killed repeatedly, so the sweep must answer its most
# important question FIRST and treat the rest as bonus. Arms 1-4 settle the fused latent-grad path
# (base/nofused x2, counterbalanced base,nofused | nofused,base so position cancels). Everything
# after that is a lever we merely doubt. Completed arms are skipped on resume, so this reordering is
# safe to apply mid-sweep.
run_cfg 1 base     40 0
run_cfg 1 nofused  40 0 --no-fused-latent-grad
run_cfg 2 nofused  40 0 --no-fused-latent-grad
run_cfg 2 base     40 0
run_cfg 1 split36  36 0
run_cfg 1 budget8  40 0 --latent-gpu-budget 8
run_cfg 1 spike    40 1
run_cfg 2 spike    40 1
run_cfg 2 budget8  40 0 --latent-gpu-budget 8
run_cfg 2 split36  36 0
run_cfg 1 blocking 40 0 --pipe-blocking-handoff
run_cfg 2 blocking 40 0 --pipe-blocking-handoff
echo "=== SWEEP DONE $(date +%F' '%H:%M:%S) ===" >> $P
