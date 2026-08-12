#!/bin/bash
# ══ ARM E: SEPARATE COMPUTE FROM DATA ══════════════════════════════════════════
# The A-D sweep confounded them: C and D scaled steps AND unique sequences together.
#   A 360s /120n  0.31M tok -> 71.03      B 360s /360n  0.92M -> 71.49   (+0.46 = data @ fixed compute)
#   C 1080s/1080n 2.76M     -> 73.72      D 3240s/3240n 8.29M  -> 76.21  (+4.7 over B, but 9x the STEPS)
# E holds DATA at arm B's 360 unique sequences and gives it arm D's 3240 steps (9 epochs).
#   E ~= D (76.2)  => the curve is COMPUTE-driven; unique tokens are cheap, re-reading works.
#   E ~= B (71.5)  => the curve is DATA-driven; steps without fresh tokens buy nothing.
# Everything else identical to A-D: down@32L, lr 5e-7, same cold-start ramp, same raw skeleton.
(
# SINGLE-INSTANCE GUARD. A duplicate launch (two watchers both firing) once ran this script twice: two
# independent workers with 20.8GB PRIVATE each, 48.9GB PSS on a 60GB host, both writing the SAME --out.
# Refuse to start if a trainer is already live. MUST match comm=="python" AND the args via ps/awk: plain
# `pgrep -f` also matches any SHELL whose command text contains the pattern (e.g. a terminal running a grep
# for it), which false-fired and blocked a legitimate launch; and `pgrep -x python -f PATTERN` is invalid
# (pgrep accepts only one pattern) so it fails OPEN. Root cause of the original incident: `kill $!` after
# `setsid`,
# which forks when not already a process-group leader, so $! was not the running script.
if ps -eo comm=,args= | awk '$1=="python" && /e2e_qp_distill\.py --train/{f=1} END{exit !f}'; then
  echo "FATAL: a training process is already running -- refusing to start a second one."; exit 1
fi
set -e
export PYTHONUNBUFFERED=1; export TERNARY_BLOCK_SIZE=64
ORIG=/home/kyler/Documents/Model-to-Ternary/output_4b/untied_4b
SRC=output_4bpipe; W=$SRC/dscale; mkdir -p "$W/e_3240s_360n"
source ./lib_timing.sh
echo "########## ARM E: 3240 steps x 360 unique (9 epochs) — $(date) ##########"
stage "e_3240s_360n"
if [ ! -f "$W/e_3240s_360n/.done" ]; then
systemd-run --user --scope --quiet -p MemoryHigh=40G -p MemoryMax=44G -p MemorySwapMax=0 \
  -E TERNARY_BLOCK_SIZE=64 -E ORIG_MODEL=$ORIG -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -E SAVE_MAX_SHARD_GB=2 \
  ./.venv/bin/python -m torch.distributed.run --nproc_per_node=1 --standalone \
    src/e2e_qp_distill.py --train --student-path "$SRC/modified_model" \
    --orig-config-path output_4b/untied_4b \
    --calib $SRC/calibration_data.json --teacher-cache $SRC/teacher_topk.pt \
    --out "$W/e_3240s_360n/modified_model" \
    --seq 2560 --epochs 9 --max-samples 360 \
    --lr 0 --latent-lr 5e-7 --latent-init fp-spread --fp-model $SRC/rotbase/modified_model \
    --latent-offload --latent-grad-release --tw-layer-stride 1 --latent-warmup-steps 20 \
    --target-tr 1.25e-4 --tr-every 5 --tr-final-frac 0.2 \
    --lr-schedule linear --scale-ema-decay 0 --select final --loss-fn cakld --decision-gamma 2 \
    --ce-weight 0.1 --ce-positions 128 --train-weights down --scale-qat-bits 8 \
    --heldout-n 4 --eval-every 120 --ckpt-every 0 || { echo "  ARM E FAILED"; exit 1; }
touch "$W/e_3240s_360n/.done"; fi
echo "  >>> arm E done"
echo; echo "########## eval2k arm E ##########"
RES=logs/dscale_eval2k.txt
M=$W/e_3240s_360n/modified_model
systemd-run --user --scope --quiet -p MemoryHigh=38G -p MemoryMax=42G -p MemorySwapMax=0 \
    -E TERNARY_BLOCK_SIZE=64 \
    env ORIG="$ORIG" FP_DIR="$SRC/rotbase/modified_model" E2E_MODEL="$M" \
    EVAL_DATA=output_4b/eval2k.json NP=1946 SEQ=1024 \
    ./.venv/bin/python src/kl_flips_eval.py > logs/dscale_e2k_e_3240s_360n.log 2>&1 \
    || echo "  (eval failed)"
AG=$(grep -aoE "top-1 agreement *: *[0-9.]+" logs/dscale_e2k_e_3240s_360n.log | grep -oE "[0-9.]+$" | tail -1)
KL=$(grep -aoE "mean KL\(fp\|\|tern\) *: *[0-9.]+" logs/dscale_e2k_e_3240s_360n.log | grep -oE "[0-9.]+$" | tail -1)
printf "%-14s %-8s %-12s %s\n" "e_3240s_360n" "922k*9ep" "${AG:-NA}" "${KL:-NA}" | tee -a "$RES"
echo; cat "$RES"
stage_end
) > "${LOG_FILE:-logs/dscale_e.log}" 2>&1 &
echo "arm E started (PID $!)."
