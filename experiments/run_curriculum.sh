#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# CURRICULUM RUN — reproduce the reference recipe's TWO-CORPUS structure (§8i follow-up).
#
# WHY: the first cold run of run_full_pipeline.sh used the 47.5%-chat corpus for EVERY stage and
# failed Gate B at commit 41.7% — WORSE than the 4.3%-density combined-2560 run (56.2%) despite 11x
# the commit density. Density was supposed to be the lever (4.3->56%, 30.9->62.5%, 47.2->75%), so
# dosage is not the problem: the PATH is. Reading final_g64q8 back through §8c + the surviving calib
# files shows the reference introduced chat data ONLY at the last pass:
#
#     Block-AP skeleton  <- combined-2560   4.3% density
#     E2E pass 1         <- combined-2560   4.3% density
#     E2E pass 2 (1e-5)  <- final          47.2% density
#
# Leading hypothesis: Block-AP Hessians are X^T X over calibration activations, so collecting them on
# a corpus that is half model-generated reasoning traces optimizes the trit placement for THAT
# distribution at the expense of general text. E2E can retune scales afterward but cannot re-place
# trits — the damage is upstream of where the chat data is meant to help.
#
# NOTE: this deliberately changes three stages at once (skeleton corpus + pass-1 corpus + 2 passes).
# It tests "does the reference recipe reproduce", NOT which stage is responsible. Ablate afterward.
#
#   bash run_curriculum.sh          # background; logs to logs/curriculum.log
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
CTM_DATA=${CTM_DATA:-$HOME/Documents/CTM-Transformer/data_cache}
export ORIG_MODEL
SRC=${SRC:-output_4bpipe}                 # the completed cold run: reuse its rotation, chat pool,
W=${W:-output_4bpipe/curr}                # 47.5% calib and 47.5% teacher cache
BLK=${BLK:-64}; SBITS=${SBITS:-8}; SEQ=${SEQ:-2560}; NGPU=${NGPU:-2}
export TERNARY_BLOCK_SIZE=$BLK

ROT=$SRC/rotbase/modified_model
CHATPOOL=$SRC/chat_pool.json
CALIB_HI=$SRC/calibration_data.json       # 47.5% — pass 2 (already built)
TEACHER_HI=$SRC/teacher_topk.pt           # matches CALIB_HI (already built)
CALIB_LO=$W/calibration_data.json         # 4.3%  — skeleton + pass 1 (block_ap reads this by convention)
CALIB_LO_EVAL=$W/calib_eval.json
TEACHER_LO=$W/teacher_topk.pt
SKEL=$W/modified_model
E2E1=$W/e2e1/modified_model
E2E2=$W/e2e2/modified_model

CALIB_TOKENS=${CALIB_TOKENS:-16000000}
CHAT_FRAC_LO=${CHAT_FRAC_LO:-0.10}        # the reference's pass-1 mix (10% chat -> 4.3% density)
LR1=${LR1:-2e-5}; LR2=${LR2:-1e-5}; EPOCHS=${EPOCHS:-2}
COMMIT_BETA=${COMMIT_BETA:-1.5}; GATE_MAXNEW=${GATE_MAXNEW:-2048}

TOTAL_GB=$(free -g | awk '/^Mem:/{print $2}')
DEF_MAX=$(( TOTAL_GB>=48 ? TOTAL_GB-18 : (TOTAL_GB>=24 ? TOTAL_GB-10 : TOTAL_GB*2/3) ))
MEM_MAX=${MEM_MAX:-${DEF_MAX}G}; MM_NUM=${MEM_MAX%G}
MEM_HIGH=${MEM_HIGH:-$(( MM_NUM>8 ? MM_NUM-4 : MM_NUM ))G}
if systemd-run --user --scope --quiet -p MemoryMax=64M -p MemorySwapMax=0 /bin/true 2>/dev/null; then
  MEMGUARD="systemd-run --user --scope --quiet -p MemoryHigh=$MEM_HIGH -p MemoryMax=$MEM_MAX -p MemorySwapMax=0 -E TERNARY_BLOCK_SIZE=$BLK"
  echo "[memguard] high=$MEM_HIGH max=$MEM_MAX swap=off (physical ${TOTAL_GB}G)"
else echo "[memguard] FATAL: no systemd --user scope; refusing to run unguarded."; exit 1; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$MEMGUARD ./.venv/bin/python"

for p in "$ROT" "$CHATPOOL" "$CALIB_HI" "$TEACHER_HI"; do
  [ -e "$p" ] || { echo "FATAL: missing reusable artifact $p"; exit 1; }
done
mkdir -p "$W"
echo "########## CURRICULUM RUN — $(date) ##########"
echo "  skeleton+pass1 <- ${CHAT_FRAC_LO} chat (target ~4.3% density) | pass2 <- $CALIB_HI (47.5%)"
source ./lib_timing.sh

# ─── C1: low-density calib (reuses the existing chat pool; no generation) ───
stage "C1: low-density calib (${CHAT_FRAC_LO} chat)"
if [ ! -f "$CALIB_LO" ]; then
  $PY src/build_diverse_calib.py --orig-model "$ORIG_MODEL" --ctm-data "$CTM_DATA" \
      --tokens "$CALIB_TOKENS" --seq "$SEQ" --chat-frac "$CHAT_FRAC_LO" --chat-src "$CHATPOOL" \
      --out "$CALIB_LO" --eval-out "$CALIB_LO_EVAL"
else echo "=== [skip] C1: $CALIB_LO exists ==="; fi
NSAMP_LO=$(./.venv/bin/python -c "import json;print(len(json.load(open('$CALIB_LO'))))")
DENS_LO=$(./.venv/bin/python -c "import json;d=json.load(open('$CALIB_LO'));print(f'{100*sum(1 for s in d if 248069 in s)/len(d):.1f}')")
echo "  low-density calib: $NSAMP_LO seqs | </think> density ${DENS_LO}%  (reference pass-1 was 4.3%)"
# guard: if this lands near the 47% corpus the whole experiment is void
awk -v d="$DENS_LO" 'BEGIN{ if (d > 15.0) { print "FATAL: density " d "% is not low — check --chat-frac"; exit 1 } }'
NSAMP_HI=$(./.venv/bin/python -c "import json;print(len(json.load(open('$CALIB_HI'))))")
BLOCKAP_SAMPLES=${BLOCKAP_SAMPLES:-$(( 640 * 1024 / SEQ ))}
[ "$BLOCKAP_SAMPLES" -gt "$NSAMP_LO" ] && BLOCKAP_SAMPLES=$NSAMP_LO

# ─── C2: skeleton on the LOW-density corpus (the key change) ───
stage "C2: Block-AP skeleton (low-density Hessians)"
if [ ! -f "$SKEL/model.safetensors.index.json" ]; then
  $PY src/block_ap_recovery.py --model-path "$ROT" --orig-config-path "$ORIG_MODEL" \
      --output-dir "$W" --block-size "$BLK" --samples "$BLOCKAP_SAMPLES" \
      --qat --qat-gptq-init --qat-keep-best --qat-attn-only \
      --qat-lr 1e-4 --qat-scale-lr 1e-4 --qat-epochs 4 --quant-embed-head
else echo "=== [skip] C2: $SKEL exists ==="; fi

# ─── C3: teacher cache for the low-density corpus (each pass needs its own) ───
stage "C3: teacher cache (low-density)"
if [ ! -f "$TEACHER_LO" ]; then
  $PY src/e2e_qp_distill.py --precompute-teacher --teacher-path "$ROT" --orig-config-path "$ORIG_MODEL" \
      --calib "$CALIB_LO" --teacher-cache "$TEACHER_LO" --seq "$SEQ" --topk 64 \
      --cache-batch "${CACHE_BATCH:-1}" --teacher-dp "${TEACHER_DP:-1}" \
      --gpu-mem 20GiB --cpu-mem 30GiB --max-samples "$NSAMP_LO"
else echo "=== [skip] C3: $TEACHER_LO exists ==="; fi

e2e () {  # e2e <student> <calib> <teacher> <out> <lr> <nsamp>
  $PY -m torch.distributed.run --nproc_per_node="$NGPU" --standalone \
      src/e2e_qp_distill.py --train --student-path "$1" --orig-config-path "$ORIG_MODEL" \
      --calib "$2" --teacher-cache "$3" --out "$4" \
      --seq "$SEQ" --epochs "$EPOCHS" --max-samples "$6" \
      --lr "$5" --lr-schedule linear --scale-ema-decay 0 --select final \
      --loss-fn cakld --feat-weight 0 --decision-gamma 2 \
      --col-scale --scale-qat-bits "$SBITS" \
      --commit-beta "$COMMIT_BETA" --commit-pre 16 --commit-post 12 \
      --heldout-n 48 --eval-every 300 --abort-patience 30
}

stage "C4: E2E pass 1 (low-density, lr $LR1)"
if [ ! -f "$W/e2e1/.done" ]; then
  mkdir -p "$W/e2e1"; e2e "$SKEL" "$CALIB_LO" "$TEACHER_LO" "$E2E1" "$LR1" "$NSAMP_LO"; touch "$W/e2e1/.done"
else echo "=== [skip] C4 ==="; fi

stage "C5: E2E pass 2 (47.5% chat, lr $LR2)"
if [ ! -f "$W/e2e2/.done" ]; then
  mkdir -p "$W/e2e2"; e2e "$E2E1" "$CALIB_HI" "$TEACHER_HI" "$E2E2" "$LR2" "$NSAMP_HI"; touch "$W/e2e2/.done"
else echo "=== [skip] C5 ==="; fi

stage "C6: gates"
# Gate A must use the FROZEN eval2k referee, not a per-run held-out (see RESULTS_SUMMARY §8l / run_full_pipeline.sh).
# refs: V1(chat-throughout) 72.77 · combined-2560 79.76 · reference FINAL 80.46 · single-corpus v2 77.53 · gate >=77
echo "=== Gate A: eval2k agreement (frozen 1946-seq referee; gate >=77%) ==="
$MEMGUARD env ORIG="$ORIG_MODEL" FP_DIR="$ROT" E2E_MODEL="$E2E2" \
    EVAL_DATA="${EVAL2K:-output_4b/eval2k.json}" NP="${GATE_NP:-1946}" SEQ=1024 \
    ./.venv/bin/python src/kl_flips_eval.py || echo "(eval2k gate failed; continuing)"
echo "=== Gate B: FREE-GEN loop/commit @${GATE_MAXNEW} (THE deploy gate) ==="
echo "    targets: commit >=68% · loop <=30% · compR <=3.1"
echo "    refs: FP 75.0/25.0/2.40 · reference-FINAL 75.0/29.2/3.08 · cold-run 41.7/33.3/3.12"
$MEMGUARD env ORIG="$ORIG_MODEL" E2E_MODEL="$E2E2" \
    N_PREFIX=48 MAXNEW="$GATE_MAXNEW" THINK=1 TEMP=0.6 BATCH=8 GATE_OUT="$W/loop_gate_curriculum.json" \
    ./.venv/bin/python src/loop_gate.py || echo "(free-gen gate failed; continuing)"

stage_end
echo "########## CURRICULUM RUN DONE -> $E2E2 ##########"
) > "${LOG_FILE:-logs/curriculum.log}" 2>&1 &
echo "Curriculum run started in the background (PID $!)."
echo "Watch it:   tail -f ${LOG_FILE:-logs/curriculum.log}"
