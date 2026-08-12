#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# QAT DATA-SCALING SWEEP — does the Block-AP SKELETON actually improve with more data?
#
# WHY THIS IS NOT A CALIB_TOKENS SWEEP: block_ap_recovery.py does `samples[:n]` with
# n = BLOCKAP_SAMPLES, which run_full_pipeline.sh pins to 640*1024/SEQ = 256. The skeleton therefore
# sees ~0.65M tokens whether the calib is 4M or 64M — sweeping CALIB_TOKENS would have produced four
# identical skeletons and looked like "no scaling". The real knob is --samples.
#
# RAM: SOLVED via NVMe activation spill. Block-AP used to hold the whole propagated stream (+ the FP-target
# and next-layer streams) in CPU RAM, linear in tokens — the 2026-08-03 host OOM (fixed overhead ~24 GB +
# 2-3 streams crossed the 42 GB cap around 512 samples). ACT_SPILL_GB (set below) routes every activation
# stream to NVMe with a shared bounded RAM window, so activation RAM is CONSTANT in token count and QAT can
# no longer OOM the host regardless of sample count. Reachable ceiling is now disk, not RAM.
#
# WHY GPTQ IS NOT SWEPT: the Hessian is X^T X, a d x d second-moment estimate (d=2560), which converges
# in ~O(d log d) tokens; GPTQ itself uses 128x2048 = 0.26M and is flat there. Our 655k-token Hessian is
# already saturated. QAT is the slow-saturating part, so QAT data is the axis that matters.
#
# THE CONTROL (point E) IS THE POINT: raising --samples raises BOTH distinct data AND gradient updates.
# E runs 256 samples x 16 epochs = the SAME update count as D with 1/4 the distinct data.
#   D > E  => genuinely data-hungry (27B needs a big corpus)
#   D ~ E  => merely under-optimized (27B can use far fewer tokens + more epochs) <- the cheap outcome
#
# ALL POINTS MUST SHARE ONE CORPUS or they are not comparable. Set CALIB to whichever corpus the
# validated recipe ends up using (low-density if the curriculum run passes).
#
#   CALIB=<calib.json> EVAL=<eval.json> bash run_qat_scaling.sh
# ──────────────────────────────────────────────────────────────────────────────
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
export PYTHONUNBUFFERED=1

ORIG_MODEL=${ORIG_MODEL:-/home/kyler/Documents/Model-to-Ternary/output_4b/untied_4b}
export ORIG_MODEL
SRC=${SRC:-output_4bpipe}
ROT=${ROT:-$SRC/rotbase/modified_model}
CALIB=${CALIB:?set CALIB to the shared corpus (e.g. output_4bpipe/curr/calibration_data.json)}
EVAL=${EVAL:?set EVAL to the fixed held-out set used for ALL points}
W=${W:-output_4bpipe/qatscale}
BLK=${BLK:-64}; SEQ=${SEQ:-2560}
export TERNARY_BLOCK_SIZE=$BLK

# NVMe activation spill: bound resident activation RAM. COUNTERINTUITIVE — keep this SMALL (4 GB), not large.
# Fixed overhead is ~18 GB; a big cache does NOT just risk the MemoryHigh line, it makes the torch.save/load
# churn's working set large enough that glibc allocator fragmentation balloons RSS (measured: 10 GB cache →
# 39.5 GB RSS + progressive reclaim throttle, L0 propagate 20→73 s/it climbing; 4 GB cache → 21.7 GB RSS,
# stable, 7.9 it/s). 4 GB keeps total ≈ 22 GB, well under MemoryHigh=38, full speed at any token count.
ACT_SPILL_GB=${ACT_SPILL_GB:-4}
export ACT_SPILL_GB

TOTAL_GB=$(free -g | awk '/^Mem:/{print $2}')
DEF_MAX=$(( TOTAL_GB>=48 ? TOTAL_GB-18 : (TOTAL_GB>=24 ? TOTAL_GB-10 : TOTAL_GB*2/3) ))
MEM_MAX=${MEM_MAX:-${DEF_MAX}G}; MM_NUM=${MEM_MAX%G}
MEM_HIGH=${MEM_HIGH:-$(( MM_NUM>8 ? MM_NUM-4 : MM_NUM ))G}
if systemd-run --user --scope --quiet -p MemoryMax=64M -p MemorySwapMax=0 /bin/true 2>/dev/null; then
  MEMGUARD="systemd-run --user --scope --quiet -p MemoryHigh=$MEM_HIGH -p MemoryMax=$MEM_MAX -p MemorySwapMax=0 -E TERNARY_BLOCK_SIZE=$BLK -E ACT_SPILL_GB=$ACT_SPILL_GB"
  echo "[memguard] high=$MEM_HIGH max=$MEM_MAX swap=off (physical ${TOTAL_GB}G) | ACT_SPILL_GB=$ACT_SPILL_GB"
else echo "[memguard] FATAL: no systemd --user scope; refusing to run unguarded."; exit 1; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$MEMGUARD ./.venv/bin/python"

mkdir -p "$W"
NSAMP=$(./.venv/bin/python -c "import json;print(len(json.load(open('$CALIB'))))")
DENS=$(./.venv/bin/python -c "import json;d=json.load(open('$CALIB'));print(f'{100*sum(1 for s in d if 248069 in s)/len(d):.1f}')")
echo "########## QAT DATA-SCALING SWEEP — $(date) ##########"
echo "  corpus $CALIB ($NSAMP seqs, </think> density ${DENS}%) | eval $EVAL | g$BLK seq $SEQ"

#            name        samples epochs
POINTS=("A   128   4"
        "B   256   4"
        "C   512   4"
        "D   1024  4"
        "E   256   16")

source ./lib_timing.sh
for p in "${POINTS[@]}"; do
  set -- $p; NAME=$1; S=$2; E=$3
  [ "$S" -gt "$NSAMP" ] && { echo "=== [skip] point $NAME: needs $S samples, corpus has $NSAMP ==="; continue; }
  DISK=$(awk -v s="$S" -v q="$SEQ" 'BEGIN{printf "%.1f", 2*s*q*2560*2/1073741824}')   # NVMe spill footprint (~2 live streams)
  OUT=$W/${NAME}_s${S}_e${E}
  stage "point $NAME: samples=$S epochs=$E (~$((S*SEQ/1000))k tok, ~${DISK}GB NVMe spill, RAM capped ${ACT_SPILL_GB}GB)"
  if [ -f "$OUT/modified_model/model.safetensors.index.json" ]; then echo "=== [skip] $NAME exists ==="; continue; fi
  mkdir -p "$OUT"
  cp "$CALIB" "$OUT/calibration_data.json"      # block_ap reads <output-dir>/calibration_data.json
  # identical to the pipeline's Phase 3 except --samples / --qat-epochs
  $PY src/block_ap_recovery.py --model-path "$ROT" --orig-config-path "$ORIG_MODEL" \
      --output-dir "$OUT" --block-size "$BLK" --samples "$S" \
      --qat --qat-gptq-init --qat-keep-best --qat-attn-only \
      --qat-lr 1e-4 --qat-scale-lr 1e-4 --qat-epochs "$E" --quant-embed-head \
      || { echo "  point $NAME FAILED — continuing to the next point"; continue; }
done

# ─── paired evaluation: same held-out, same settings, every point ───
stage "evaluate all points (fixed held-out)"
RES=$W/results.txt
{ echo "QAT data-scaling sweep — corpus $CALIB (density ${DENS}%), eval $EVAL"
  echo "point  samples  epochs   tokens   agreement%   KL"; } > "$RES"
for p in "${POINTS[@]}"; do
  set -- $p; NAME=$1; S=$2; E=$3
  M=$W/${NAME}_s${S}_e${E}/modified_model
  [ -f "$M/model.safetensors.index.json" ] || continue
  echo "--- evaluating point $NAME ---"
  OUTLOG=$W/${NAME}_eval.log
  $MEMGUARD env ORIG="$ORIG_MODEL" FP_DIR="$ROT" E2E_MODEL="$M" \
      EVAL_DATA="$EVAL" NP="${GATE_NP:-512}" SEQ=1024 \
      ./.venv/bin/python src/kl_flips_eval.py > "$OUTLOG" 2>&1 || echo "  (eval failed for $NAME)"
  AG=$(grep -aoE "top-1 agreement *: *[0-9.]+" "$OUTLOG" | grep -oE "[0-9.]+$" | tail -1)
  KL=$(grep -aoE "mean KL\(fp\|\|tern\) *: *[0-9.]+" "$OUTLOG" | grep -oE "[0-9.]+$" | tail -1)
  printf "%-6s %-8s %-8s %-8s %-12s %s\n" "$NAME" "$S" "$E" "$((S*SEQ/1000))k" "${AG:-NA}" "${KL:-NA}" >> "$RES"
done
echo; cat "$RES"
echo
echo "READ IT LIKE THIS:  A->D rising = skeleton is data-hungry."
echo "                    D vs E (same updates, 4x data): D>E = data matters; D~E = just needs more epochs."
stage_end
echo "########## SWEEP DONE -> $RES ##########"
) > "${LOG_FILE:-logs/qat_scaling.log}" 2>&1 &
echo "QAT scaling sweep started in the background (PID $!)."
echo "Watch it:   tail -f ${LOG_FILE:-logs/qat_scaling.log}"
