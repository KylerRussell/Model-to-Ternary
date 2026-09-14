#!/bin/bash
# MEMORY PROBE, not a measurement run. Question: does --train-weights all --tw-layer-stride 1 fit
# under the 620 GB cgroup limit at a REAL learning rate?
#
# Why a probe and not just the real run: every throughput arm ran at --lr 0, where the optimizer step
# is skipped entirely, so Adam state was NEVER allocated in any of them. Peak memory is set by scope
# plus the FIRST optimizer step -- both of which happen in step 1 -- so 2 steps reveal the peak that a
# 10-step run would. The documented 18.9 B/latent model under-predicted the stride-2 run by >=42%
# (predicted 134 GB anon, measured 191 GB and still climbing), so extrapolation is not trustworthy.
#
# Guards: watchdog SIGKILLs the trainer at 85% of the limit (527 GB) so the CONTAINER survives to tell
# us the answer; --no-final-save keeps the ~108 GB save transient out of the picture for now.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
W=/home/kasm-user/Documents/Model-to-Ternary/output_sweep
P=$W/probe_results.txt; mkdir -p $W; touch $P
echo "  [$(date +%F' '%H:%M:%S)] PROBE START: all / stride 1 / latent-lr 5e-7 / 2 steps" >> $P
MP_SPLIT=40 PYTHONUNBUFFERED=1 ORIG_MODEL="$MODEL_27B" \
./.venv/bin/torchrun --nproc_per_node=2 --no-python --master_port=29691 \
  $W/numa_colocate.sh --train \
  --student-path $HOME/_scratch_naive27b --orig-config-path "$MODEL_27B" \
  --calib output_4bpipe/calibration_data.json --teacher-cache output_4bpipe/teacher_topk.pt \
  --seq 2560 --epochs 1 --max-samples 8 --lr 0 --latent-lr 5e-7 --latent-opt adam-blockv \
  --latent-init center --latent-offload --latent-grad-release --train-weights all --tw-layer-stride 1 \
  --pipe-parallel --pipe-parallel-mb 2 --latent-pin --latent-warmup-steps 0 --lr-schedule linear \
  --scale-ema-decay 0 --select final --loss-fn cakld --decision-gamma 2 --ce-weight 0.1 \
  --ce-positions 128 --scale-qat-bits 8 --heldout-n 4 --eval-every 100000 --ckpt-every 0 \
  --abort-patience 1000000 --no-final-save \
  --out $W/probe/m/modified_model > $W/probe.log 2>&1
rc=$?
rm -rf $W/probe/m 2>/dev/null
echo "  [$(date +%F' '%H:%M:%S)] PROBE END rc=$rc" >> $P
echo "  latents: $(grep -oE 'training [0-9]+ scale tensors \([0-9.]+M\) \+ [0-9]+ latents \([0-9.]+M\)' $W/probe.log | head -1)" >> $P
echo "  peak cgroup: $(grep -oE 'cur=[0-9]+G' $W/cgmon.log | tr -d 'curG=' | sort -n | tail -1) GB of 620" >> $P
echo "  peak anon:   $(grep -oE 'anon=[0-9]+G' $W/cgmon.log | tr -d 'anoG=' | sort -n | tail -1) GB" >> $P
echo "=== PROBE DONE ===" >> $P
