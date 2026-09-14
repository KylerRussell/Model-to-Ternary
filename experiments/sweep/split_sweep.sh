#!/bin/bash
# Re-tune MP_SPLIT under the 13q cost structure. The old optimum (40) was chosen when rank 0's Adam
# was 236.6 s; at 40.1 s the balance point moved. §13 recorded split36 as +2.2% WORSE -- that result
# is from the OLD structure and is exactly what this re-tests.
#
# PID LOCK: two sweep drivers were once launched concurrently and collided on the torchrun
# master_port, corrupting results and killing an arm. Never remove this.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
LOCK=output_sweep/.split_sweep.lock
if [ -e "$LOCK" ] && kill -0 "$(cat $LOCK 2>/dev/null)" 2>/dev/null; then
  echo "another split_sweep is live (pid $(cat $LOCK)); refusing"; exit 1
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

source ./env.sh
W=/home/kasm-user/Documents/Model-to-Ternary/output_sweep
P=$W/split_results.txt; mkdir -p $W; touch $P
PORT=29711

for SPLIT in 36 34 32; do
  echo "  [$(date +%F' '%H:%M:%S)] START split=$SPLIT threads=8 (control split40 = 234.7 s/step)" >> $P
  LOG=$W/split${SPLIT}.log
  MP_SPLIT=$SPLIT PP_PROF=1 PYTHONUNBUFFERED=1 ORIG_MODEL="$MODEL_27B" \
  ./.venv/bin/torchrun --nproc_per_node=2 --no-python --master_port=$PORT \
    $W/numa_colocate.sh --train \
    --student-path $HOME/_scratch_naive27b --orig-config-path "$MODEL_27B" \
    --calib output_4bpipe/calibration_data.json --teacher-cache output_4bpipe/teacher_topk.pt \
    --seq 2560 --epochs 1 --max-samples 12 --lr 0 --latent-lr 5e-7 --latent-opt adam-blockv \
    --latent-init center --latent-offload --latent-grad-release \
    --train-weights all --tw-layer-stride 1 \
    --pipe-parallel --pipe-parallel-mb 2 --latent-warmup-steps 0 --lr-schedule linear \
    --scale-ema-decay 0 --select final --loss-fn cakld --decision-gamma 2 --ce-weight 0.1 \
    --ce-positions 128 --scale-qat-bits 8 --heldout-n 4 --eval-every 100000 --ckpt-every 0 \
    --abort-patience 1000000 --no-final-save --no-latent-snapshot --cpu-threads 8 \
    --out $W/split$SPLIT/m/modified_model > $LOG 2>&1
  rc=$?
  echo "  [$(date +%F' '%H:%M:%S)] END split=$SPLIT rc=$rc steps=$(grep -c 'pp rank 0 mb=2 kl=' $LOG)" >> $P
  grep -oE "^\[[0-9:]+\].*step [0-9]+/[0-9]+  pp rank 0" $LOG | awk -v s=$SPLIT '
    function secs(x,a){gsub(/[\[\]]/,"",x);split(x,a,":");return a[1]*3600+a[2]*60+a[3]}
    {t=secs($1); if(p){d=t-p; if(d<0)d+=86400; n++; if(n>2){acc+=d;m++}} p=t}
    END{ if(m) printf "  SPLIT%s %.1f s/step (n=%d steady) vs 234.7 control = %+.1f%%\n", s, acc/m, m, 100*(acc/m-234.7)/234.7 }' >> $P
  grep -E "lat-prof|pp-prof" $LOG | tail -2 | sed 's/^/    /' >> $P
  rm -rf $W/split$SPLIT/m 2>/dev/null
  PORT=$((PORT+2))
done
echo "=== SPLIT SWEEP DONE ===" >> $P
