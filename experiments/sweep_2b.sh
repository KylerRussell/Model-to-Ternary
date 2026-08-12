#!/usr/bin/env bash
# sweep_2b.sh — GENERIC one-parameter sweep on the Qwen3.5-2B testbed.
# Builds the 2B upstream once (untie -> rotation -> calib -> Block-AP -> teacher; skipped if present),
# then runs E2E once per swept value (everything else held at BASE_FLAGS) and ranks them on HELD-OUT
# text vs the FP teacher, sharing one FP-teacher pass.
#
#   bash sweep_2b.sh                                            # default: sweep --lr over {2e-5,5e-5,1e-4}
#   SWEEP_FLAG=--lr SWEEP_VALUES="2e-5 5e-5 1e-4 2e-4" bash sweep_2b.sh
#   SWEEP_FLAG=--scale-ema-decay SWEEP_VALUES="0.9 0.99 0.999" \
#     BASE_FLAGS="--lr 5e-5 --lr-schedule constant --select ema --scale-ema-start-frac 0.5" bash sweep_2b.sh
#
# Env: WORK(output_2b_ab) CALIB_TOKENS(500000) E2E_STEPS(2000) EVAL_SAMPLES(64)
#      SWEEP_FLAG SWEEP_VALUES BASE_FLAGS FRESH(1=rerun cells)
set -u
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source ./lib_timing.sh

TOTAL_GB=$(free -g | awk '/^Mem:/{print $2}')
MEM_MAX=$(( TOTAL_GB>=48 ? TOTAL_GB-18 : (TOTAL_GB>=24 ? TOTAL_GB-10 : TOTAL_GB*2/3) ))
if systemd-run --user --scope --quiet -p MemoryMax=1G true 2>/dev/null; then
  MEMGUARD="systemd-run --user --scope --quiet -p MemoryHigh=$((MEM_MAX-4))G -p MemoryMax=${MEM_MAX}G -p MemorySwapMax=0"
else MEMGUARD=""; fi
PY="$MEMGUARD ./.venv/bin/python"

TIED_2B=$(ls -d "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/*/ 2>/dev/null | head -1)
CTM_DATA=${CTM_DATA:-$HOME/Documents/CTM-Transformer/data_cache}
WORK=${WORK:-output_2b_ab}
ORIG_MODEL=$WORK/untied_2b; export ORIG_MODEL
ROT=$WORK/rot/modified_model
CALIB=$WORK/calibration_data.json; CALIB_EVAL=$WORK/calib_eval.json
TEACHER=$WORK/teacher_topk.pt;     RECOVERED=$WORK/modified_model
SHARED_REF=$WORK/_eval_ref.pt
SEQ=1024; CALIB_TOKENS=${CALIB_TOKENS:-500000}; E2E_STEPS=${E2E_STEPS:-2000}; EVAL_SAMPLES=${EVAL_SAMPLES:-64}

SWEEP_FLAG=${SWEEP_FLAG:---lr}
SWEEP_VALUES=${SWEEP_VALUES:-2e-5 5e-5 1e-4}
# fixed recipe for the non-swept dimensions (default = winning recipe minus --lr)
BASE_FLAGS=${BASE_FLAGS:---lr-schedule constant --scale-ema-decay 0.999 --scale-ema-start-frac 0.5 --select ema}
COMMON="--student-path $RECOVERED --orig-config-path $ORIG_MODEL --calib $CALIB --teacher-cache $TEACHER \
        --seq $SEQ --steps $E2E_STEPS --loss-fn cakld --feat-weight 0 --max-samples 0"

[ -n "$TIED_2B" ] || { echo "FATAL: Qwen3.5-2B snapshot not in HF cache"; exit 1; }
mkdir -p "$WORK"
echo "########## 2B SWEEP — $(date) ##########"
echo "  sweeping $SWEEP_FLAG over: $SWEEP_VALUES"
echo "  base flags: $BASE_FLAGS"

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

[ "${FRESH:-1}" = "1" ] && rm -rf "$WORK/sweep"
mkdir -p "$WORK/sweep"
rm -f "$SHARED_REF"     # one fresh FP-teacher pass, shared across all cells
for v in $SWEEP_VALUES; do
  safe=$(echo "$v" | tr -c 'A-Za-z0-9.-' '_')
  out="$WORK/sweep/$safe/modified_model"; log="$WORK/sweep/$safe.eval.log"
  stage "E2E [$SWEEP_FLAG=$v]"
  rm -rf "$WORK/sweep/$safe"; mkdir -p "$WORK/sweep/$safe"
  $PY src/e2e_qp_distill.py --train $COMMON $BASE_FLAGS $SWEEP_FLAG "$v" --out "$out" \
     || { echo "FATAL: cell $SWEEP_FLAG=$v"; exit 1; }
  stage "eval [$SWEEP_FLAG=$v]"
  $PY src/eval_ternary.py --fp-path "$ORIG_MODEL" --ternary-path "$out" --orig-config-path "$ORIG_MODEL" \
      --calib "$CALIB_EVAL" --seq "$SEQ" --max-samples "$EVAL_SAMPLES" \
      --ref-file "$SHARED_REF" --reuse-ref 2>&1 | tee "$log"
done
stage_end

echo; echo "=== SWEEP RESULTS ($SWEEP_FLAG, held-out vs FP teacher; lower ppl = better) ==="
./.venv/bin/python - "$WORK/sweep" "$SWEEP_FLAG" $SWEEP_VALUES <<'PY'
import re, os, sys
d, flag = sys.argv[1], sys.argv[2]; vals = sys.argv[3:]
def parse(p):
    if not os.path.exists(p): return None
    t = open(p).read()
    g = lambda pat: (float(re.search(pat, t).group(1)) if re.search(pat, t) else float('nan'))
    return dict(ppl=g(r'ppl ratio \(ternary/FP\)\s*:\s*([0-9.]+)'),
                kl=g(r'mean top-\d+ KL\(FP\|\|tern\):\s*([0-9.]+)'),
                ag=g(r'top-1 agreement\s*:\s*([0-9.]+)'))
import re as _re
rows = []
print(f"\n  {flag:>14}{'ppl':>10}{'KL':>9}{'agree%':>9}")
print("  " + "-" * 42)
for v in vals:
    safe = _re.sub(r'[^A-Za-z0-9.-]', '_', v)
    m = parse(os.path.join(d, f"{safe}.eval.log"))
    if not m: print(f"  {v:>14}{'(no log)':>10}"); continue
    print(f"  {v:>14}{m['ppl']:>10.4f}{m['kl']:>9.4f}{m['ag']:>9.2f}")
    if m['ppl'] == m['ppl']: rows.append((v, m['ppl']))
if rows:
    bv, bp = min(rows, key=lambda x: x[1])
    print("  " + "-" * 42)
    print(f"  BEST: {flag}={bv}  ->  ppl ratio {bp:.4f}")
PY
echo; echo "########## sweep done — logs under $WORK/sweep/*.eval.log ##########"
