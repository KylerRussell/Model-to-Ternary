#!/bin/bash
# STRIDE 1, intended to RUN (not a probe). No NVMe offload -- there is none on this box.
#
# Budget, all in RAM (SWAP IS 0, so anon can NEVER be reclaimed; only page cache can):
#   latents fp32      106 GB   (fp32 master is mandatory; bf16 is swamped by the Adam update)
#   Adam exp_avg      106 GB   (adam-blockv already drops exp_avg_sq -- that IS the saving)
#   packed model x2    13.5 GB
#   Wlm bf16            1.3 GB
#   misc/activations   ~20 GB
#   ---- anon floor   ~247 GB of 620, leaving ~373 GB for page cache and transients.
#
# Changes vs the run that OOM'd:
#   1. --latent-pin REMOVED. mlocked pages cannot be reclaimed OR migrated, and pin_memory() COPIES,
#      so it also caused a transient 2x during setup. Measured worth ~2% (H2D 3.31 vs 3.23 GB/s).
#   2. page cache dropped immediately before launch, so a burst starts with room rather than racing
#      reclaim -- reclaim is what lost the race at >=5.5 GB/s.
#   3. host_mem_audit() at 7 phases, so if it still fails we learn WHICH phase allocates.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
W=/home/kasm-user/Documents/Model-to-Ternary/output_sweep
P=$W/g4_results.txt; mkdir -p $W; touch $P
echo "  [$(date +%F' '%H:%M:%S)] START g4: --mb-seqs 4 + expandable_segments + chunked dequant" >> $P
MP_SPLIT=40 VRAM_AUDIT=1 PP_PROF=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True LATENT_DEQUANT_MIN_ELEMS=1048576 LATENT_DEQUANT_CHUNK_BLOCKS=16384 PYTHONUNBUFFERED=1 ORIG_MODEL="$MODEL_27B" \
./.venv/bin/torchrun --nproc_per_node=2 --no-python --master_port=29791 \
  $W/numa_colocate.sh --train \
  --student-path $HOME/_scratch_naive27b --orig-config-path "$MODEL_27B" \
  --calib output_4bpipe/calibration_data.json --teacher-cache output_4bpipe/teacher_topk.pt \
  --seq 2560 --epochs 1 --max-samples 48 --lr 0 --latent-lr 5e-7 --latent-opt adam-blockv \
  --latent-init center --latent-offload --latent-grad-release \
  --train-weights all --tw-layer-stride 1 \
  --pipe-parallel --pipe-parallel-mb 2 --latent-warmup-steps 0 --lr-schedule linear \
  --scale-ema-decay 0 --select final --loss-fn cakld --decision-gamma 2 --ce-weight 0.1 \
  --ce-positions 128 --scale-qat-bits 8 --heldout-n 4 --eval-every 100000 --ckpt-every 0 \
  --abort-patience 1000000 --no-final-save --no-latent-snapshot --cpu-threads 8 --mb-seqs 4 \
  --out $W/g4/m/modified_model > $W/g4.log 2>&1
rc=$?
echo "  [$(date +%F' '%H:%M:%S)] END rc=$rc  steps=$(grep -c 'step [0-9]*/[0-9]*  pp rank 0 mb=' $W/g4.log)" >> $P
grep -oE "^\[[0-9:]+\].*step [0-9]+/[0-9]+  pp rank 0" $W/g4.log | awk '
  function secs(x,a){gsub(/[\[\]]/,"",x);split(x,a,":");return a[1]*3600+a[2]*60+a[3]}
  {t=secs($1); if(p){d=t-p; if(d<0)d+=86400; n++; printf "%d ", d; if(n>2){s+=d;m++}} p=t}
  END{ if(m) printf "\n  G4 %.1f s/step = %.1f s/microbatch (n=%d steady)\n", s/m, s/m/2, m }' >> $P
grep -oE "step [0-9]+/[0-9]+  pp rank 0 mb=2 kl=[0-9.]+" $W/g4.log | sed 's/^/    /' >> $P
echo "  peak cur=$(grep -oE 'cur=[0-9]+G' $W/cgmon.log | tr -d 'curG=' | sort -n | tail -1)G/620G  peak anon=$(grep -oE 'anon=[0-9]+G' $W/cgmon.log | tr -d 'anoG=' | sort -n | tail -1)G" >> $P
grep "\[hostmem\]" $W/g4.log >> $P
rm -rf $W/g4/m 2>/dev/null
echo "=== G4 DONE ===" >> $P
