#!/bin/bash
# QUALITY comparison for the --mb-seqs 4 config (13t), which is NOT config-identical to G<=3:
# it adds --latent-code-transfer and --latent-bf16-compute, making the latent GRADIENT bf16.
#
# Three arms separate the two variables that G=3-vs-G=4 otherwise conflates:
#   A  G=3 fp32                     conservative baseline (13s config)
#   B  G=3 + code + bf16            isolates PRECISION at fixed batch   <- the crux
#   C  G=4 + code + bf16            the candidate
# A vs B = precision effect.  B vs C = batch effect.  A vs C = the practical question.
#
# EQUAL DATA, not equal steps: n_train=96 sequences for every arm, so G=3 runs 16 steps and G=4 runs
# 12, and all three see the same 96 sequences. held-out is the last 8, disjoint from n_train.
# --lr 0 is NOT a shortcut here: arm B freezes scales by design (the V-phase, §8r), so the bf16
# SCALE-gradient path is moot and only the bf16 LATENT gradient is under test.
# Same allocator/chunking env in every arm so memory settings are not a confound (both are
# numerically exact: the prealloc-vs-cat rewrite was verified bit-exact in value and both gradients).
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
LOCK=output_sweep/.quality.lock
if [ -e "$LOCK" ] && kill -0 "$(cat $LOCK 2>/dev/null)" 2>/dev/null; then
  echo "another quality sweep is live (pid $(cat $LOCK)); refusing"; exit 1
fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT
source ./env.sh
W=/home/kasm-user/Documents/Model-to-Ternary/output_sweep
P=$W/quality_results.txt; mkdir -p $W; touch $P
PORT=29811

run_arm () {          # $1=name  $2=mb-seqs  $3=extra flags
  local NAME=$1 G=$2 EXTRA=$3 LOG=$W/q_$1.log
  echo "  [$(date +%F' '%H:%M:%S)] START $NAME: --mb-seqs $G $EXTRA" >> $P
  MP_SPLIT=40 PP_PROF=1 PYTHONUNBUFFERED=1 ORIG_MODEL="$MODEL_27B" \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  LATENT_DEQUANT_MIN_ELEMS=1048576 LATENT_DEQUANT_CHUNK_BLOCKS=16384 \
  ./.venv/bin/torchrun --nproc_per_node=2 --no-python --master_port=$PORT \
    $W/numa_colocate.sh --train \
    --student-path $HOME/_scratch_naive27b --orig-config-path "$MODEL_27B" \
    --calib output_4bpipe/calibration_data.json --teacher-cache output_4bpipe/teacher_topk.pt \
    --seq 2560 --epochs 1 --max-samples 104 --lr 0 --latent-lr 5e-7 --latent-opt adam-blockv \
    --latent-init center --latent-offload --latent-grad-release \
    --train-weights all --tw-layer-stride 1 \
    --pipe-parallel --pipe-parallel-mb 2 --latent-warmup-steps 0 --lr-schedule linear \
    --scale-ema-decay 0 --select final --loss-fn cakld --decision-gamma 2 --ce-weight 0.1 \
    --ce-positions 128 --scale-qat-bits 8 --heldout-n 8 --eval-every 4 --ckpt-every 0 \
    --abort-patience 1000000 --no-final-save --no-latent-snapshot --cpu-threads 8 \
    --mb-seqs $G $EXTRA \
    --out $W/q_$NAME/m/modified_model > $LOG 2>&1
  local rc=$?
  echo "  [$(date +%F' '%H:%M:%S)] END $NAME rc=$rc steps=$(grep -c 'pp rank 0 mb=2 kl=' $LOG)" >> $P
  grep -E "\[held-out\]" $LOG | sed 's/^/    /' >> $P
  grep -oE "^\[[0-9:]+\].*step [0-9]+/[0-9]+  pp rank 0" $LOG | awk -v n=$NAME '
    function secs(x,a){gsub(/[\[\]]/,"",x);split(x,a,":");return a[1]*3600+a[2]*60+a[3]}
    {t=secs($1); if(p){d=t-p; if(d<0)d+=86400; c++; if(c>1){s+=d;m++}} p=t}
    END{ if(m) printf "    %s steady %.1f s/step\n", n, s/m }' >> $P
  rm -rf $W/q_$NAME/m 2>/dev/null
  PORT=$((PORT+2))
}

run_arm A 3 ""
run_arm B 3 "--latent-code-transfer --latent-bf16-compute"
run_arm C 4 "--latent-code-transfer --latent-bf16-compute"
echo "=== QUALITY SWEEP DONE ===" >> $P
