#!/bin/bash
# MINIMAL-SCOPE probe: --tw-layer-stride 64 trains only layer 0, so the expected footprint is small
# and exact. If measured RAM greatly exceeds expectation even here, the excess is structural and the
# tensor inventory will name it -- without needing a run that can OOM the box.
#
# Expected: packed model ~13.5 GB (both ranks hold the full module tree) + ~1.5 GB latents
# + ~1.5 GB Adam = ~17 GB anon. Anything far above that is the bug we are hunting.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
W=/home/kasm-user/Documents/Model-to-Ternary/output_sweep
P=$W/stride64_results.txt; mkdir -p $W; touch $P
echo "  [$(date +%F' '%H:%M:%S)] START stride64 (layer 0 only), lr 5e-7" >> $P
MP_SPLIT=40 PYTHONUNBUFFERED=1 ORIG_MODEL="$MODEL_27B" \
./.venv/bin/torchrun --nproc_per_node=2 --no-python --master_port=29699 \
  $W/numa_colocate.sh --train \
  --student-path $HOME/_scratch_naive27b --orig-config-path "$MODEL_27B" \
  --calib output_4bpipe/calibration_data.json --teacher-cache output_4bpipe/teacher_topk.pt \
  --seq 2560 --epochs 1 --max-samples 12 --lr 0 --latent-lr 5e-7 --latent-opt adam-blockv \
  --latent-init center --latent-offload --latent-grad-release \
  --train-weights all --tw-layer-stride 64 \
  --pipe-parallel --pipe-parallel-mb 2 --latent-warmup-steps 0 --lr-schedule linear \
  --scale-ema-decay 0 --select final --loss-fn cakld --decision-gamma 2 --ce-weight 0.1 \
  --ce-positions 128 --scale-qat-bits 8 --heldout-n 4 --eval-every 100000 --ckpt-every 0 \
  --abort-patience 1000000 --no-final-save \
  --out $W/stride64/m/modified_model > $W/stride64.log 2>&1
rc=$?
echo "  [$(date +%F' '%H:%M:%S)] END rc=$rc steps=$(grep -c 'step [0-9]*/[0-9]*  pp rank 0 mb=' $W/stride64.log)" >> $P
echo "  latents: $(grep -oE '[0-9]+ latents \([0-9.]+M\)' $W/stride64.log | head -1)" >> $P
echo "  peak cur=$(grep -oE 'cur=[0-9]+G' $W/cgmon.log | tr -d 'curG=' | sort -n | tail -1)G peak anon=$(grep -oE 'anon=[0-9]+G' $W/cgmon.log | tr -d 'anoG=' | sort -n | tail -1)G" >> $P
grep -E "\[hostmem\]|\[tensors\]|GiB " $W/stride64.log >> $P
rm -rf $W/stride64/m 2>/dev/null
echo "=== STRIDE64 DONE ===" >> $P
