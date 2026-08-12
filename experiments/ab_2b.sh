#!/usr/bin/env bash
# ab_2b.sh — GENERIC A/B harness on the Qwen3.5-2B testbed.
# Builds the 2B upstream once (untie -> rotation -> calib -> Block-AP -> teacher; skipped if present),
# then runs E2E with TWO configs and compares them on HELD-OUT text vs the FP teacher (eval_ternary.py),
# sharing one FP-teacher pass. Point A_FLAGS / B_FLAGS at whatever two E2E recipes you want to compare.
#
#   bash ab_2b.sh                              # defaults: current method  vs  the winning recipe
#   A_NAME=const B_NAME=d2z \
#     A_FLAGS="--lr 5e-5 --lr-schedule constant --select final" \
#     B_FLAGS="--lr 5e-5 --lr-schedule linear   --select final" \
#     bash ab_2b.sh
#
# Env: WORK(output_2b_ab) CALIB_TOKENS(500000, only when building upstream) E2E_STEPS(2000)
#      EVAL_SAMPLES(64) A_NAME/A_FLAGS B_NAME/B_FLAGS FRESH(1=rerun both cells)
set -u
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ./lib_timing.sh

# ── host-safety RAM cap ──
TOTAL_GB=$(free -g | awk '/^Mem:/{print $2}')
MEM_MAX=$(( TOTAL_GB>=48 ? TOTAL_GB-18 : (TOTAL_GB>=24 ? TOTAL_GB-10 : TOTAL_GB*2/3) ))
if systemd-run --user --scope --quiet -p MemoryMax=1G true 2>/dev/null; then
  MEMGUARD="systemd-run --user --scope --quiet -p MemoryHigh=$((MEM_MAX-4))G -p MemoryMax=${MEM_MAX}G -p MemorySwapMax=0"
else MEMGUARD=""; fi
PY="$MEMGUARD ./.venv/bin/python"

# ── config ──
TIED_2B=$(ls -d "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/*/ 2>/dev/null | head -1)
CTM_DATA=${CTM_DATA:-$HOME/Documents/CTM-Transformer/data_cache}
WORK=${WORK:-output_2b_ab}
ORIG_MODEL=$WORK/untied_2b; export ORIG_MODEL
ROT=$WORK/rot/modified_model
CALIB=$WORK/calibration_data.json; CALIB_EVAL=$WORK/calib_eval.json
TEACHER=$WORK/teacher_topk.pt;     RECOVERED=$WORK/modified_model
SHARED_REF=$WORK/_eval_ref.pt
SEQ=1024; CALIB_TOKENS=${CALIB_TOKENS:-500000}; E2E_STEPS=${E2E_STEPS:-2000}; EVAL_SAMPLES=${EVAL_SAMPLES:-64}

# the two cells (override via env). Defaults = current method vs the 2B-screen winner (ema@5e-5).
A_NAME=${A_NAME:-current}; A_FLAGS=${A_FLAGS:---lr 2e-5 --lr-schedule constant --select best}
B_NAME=${B_NAME:-test};    B_FLAGS=${B_FLAGS:---lr 5e-5 --lr-schedule constant --scale-ema-decay 0.999 --scale-ema-start-frac 0.5 --select ema}
COMMON="--student-path $RECOVERED --orig-config-path $ORIG_MODEL --calib $CALIB --teacher-cache $TEACHER \
        --seq $SEQ --steps $E2E_STEPS --loss-fn cakld --feat-weight 0 --max-samples 0"

[ -n "$TIED_2B" ] || { echo "FATAL: Qwen3.5-2B snapshot not in HF cache"; exit 1; }
mkdir -p "$WORK"
echo "########## 2B A/B — $(date) ##########"
echo "  A[$A_NAME]: $A_FLAGS"
echo "  B[$B_NAME]: $B_FLAGS"

# ── build the 2B upstream once (each stage skipped if already present) ──
build_upstream () {
  if [ ! -f "$ORIG_MODEL/config.json" ]; then
    stage "untie 2B"; $PY src/untie_embeddings.py --src "$TIED_2B" --out "$ORIG_MODEL" || exit 1; fi
  if [ ! -d "$ROT" ]; then
    stage "rotation"; $PY src/convert.py --model-path "$ORIG_MODEL" --output-dir "$WORK/rot" --rotation-only --skip-gguf || exit 1; fi
  if [ ! -f "$CALIB" ]; then
    stage "calib ($CALIB_TOKENS tok)"; $PY src/build_diverse_calib.py --orig-model "$ORIG_MODEL" \
        --ctm-data "$CTM_DATA" --tokens "$CALIB_TOKENS" --out "$CALIB" --eval-out "$CALIB_EVAL" || exit 1; fi
  NSAMP=$(./.venv/bin/python -c "import json;print(len(json.load(open('$CALIB'))))")
  local cap; cap=$([ "$NSAMP" -lt 512 ] && echo "$NSAMP" || echo 512)
  if [ ! -f "$RECOVERED/model.safetensors.index.json" ]; then
    stage "Block-AP ($cap seqs)"; $PY src/block_ap_recovery.py --model-path "$ROT" --orig-config-path "$ORIG_MODEL" \
        --output-dir "$WORK" --samples "$cap" --iters 200 --lr 1e-3 || exit 1; fi
  if [ ! -f "$TEACHER" ]; then
    stage "teacher cache"; $PY src/e2e_qp_distill.py --precompute-teacher --teacher-path "$ORIG_MODEL" \
        --calib "$CALIB" --teacher-cache "$TEACHER" --seq "$SEQ" --topk 64 --cache-batch 1 --teacher-dp 0 \
        --gpu-mem 20GiB --cpu-mem 30GiB --max-samples "$NSAMP" || exit 1; fi
}
build_upstream

run_cell () {  # $1 name  $2 flags
  local out="$WORK/ab/$1/modified_model" log="$WORK/ab/$1.eval.log"
  stage "E2E [$1]"
  rm -rf "$WORK/ab/$1"; mkdir -p "$WORK/ab/$1"
  $PY src/e2e_qp_distill.py --train $COMMON --out "$out" $2 || { echo "FATAL: E2E $1"; exit 1; }
  stage "eval [$1]"
  $PY src/eval_ternary.py --fp-path "$ORIG_MODEL" --ternary-path "$out" --orig-config-path "$ORIG_MODEL" \
      --calib "$CALIB_EVAL" --seq "$SEQ" --max-samples "$EVAL_SAMPLES" \
      --ref-file "$SHARED_REF" --reuse-ref 2>&1 | tee "$log"
}

[ "${FRESH:-1}" = "1" ] && rm -rf "$WORK/ab"
rm -f "$SHARED_REF"     # one fresh FP-teacher pass, shared by both cells
run_cell "$A_NAME" "$A_FLAGS"
run_cell "$B_NAME" "$B_FLAGS"
stage_end

echo; echo "=== A/B COMPARISON (held-out vs FP teacher) ==="
./.venv/bin/python - "$WORK/ab/$A_NAME.eval.log" "$WORK/ab/$B_NAME.eval.log" "$A_NAME" "$B_NAME" <<'PY'
import re, sys, math
la, lb, na, nb = sys.argv[1:5]
def parse(p):
    t = open(p).read()
    g = lambda pat: (float(re.search(pat, t).group(1)) if re.search(pat, t) else float('nan'))
    return dict(ppl=g(r'ppl ratio \(ternary/FP\)\s*:\s*([0-9.]+)'),
                kl=g(r'mean top-\d+ KL\(FP\|\|tern\):\s*([0-9.]+)'),
                ag=g(r'top-1 agreement\s*:\s*([0-9.]+)'),
                cf=g(r'confident-flip rate\s*:\s*([0-9.]+)'))
a, b = parse(la), parse(lb)
print(f"\n  {'metric':<22}{na:>12}{nb:>12}{'Δ(B-A)':>12}")
print("  " + "-" * 58)
def row(name, k, better='lower', fmt='{:.4f}'):
    d = b[k] - a[k]
    win = nb if ((d < 0) == (better == 'lower') and abs(d) > 1e-9) else (na if abs(d) > 1e-9 else 'tie')
    print(f"  {name:<22}{fmt.format(a[k]):>12}{fmt.format(b[k]):>12}{('%+.4f'%d):>12}  {win}")
row('held-out ppl ratio', 'ppl'); row('top-k KL (nats)', 'kl')
row('top-1 agreement %', 'ag', 'higher', '{:.2f}'); row('confident-flip %', 'cf', 'lower', '{:.2f}')
if not math.isnan(a['ppl']) and not math.isnan(b['ppl']):
    gap = a['ppl'] - 1.0
    print(f"\n  {nb} vs {na} on the gate (held-out ppl ratio): {a['ppl']:.4f} -> {b['ppl']:.4f}"
          + (f"  ({100*(a['ppl']-b['ppl'])/gap:+.1f}% of A's quant gap)" if gap > 1e-9 else ""))
PY
echo; echo "########## A/B done — logs: $WORK/ab/$A_NAME.eval.log  $WORK/ab/$B_NAME.eval.log ##########"
