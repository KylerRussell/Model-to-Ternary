#!/usr/bin/env bash
# sweep_27b_lr.sh — sweep the E2E learning rate on the 27B, REUSING an existing pipeline upstream
# (rotation + Block-AP + teacher cache already built by run_full_pipeline.sh). Each cell is just
# E2E (DDP x NGPU) + a held-out eval that REUSES the upstream's FP-teacher reference — no FP recompute.
# New recipe held fixed (constant LR + scale-EMA, no greedy restart); only --lr varies.
#
#   bash sweep_27b_lr.sh                       # default LRs 1e-5 5e-6 1e-6 on output_scale_0p5M
#   WORK=output_scale_0p5M LR_VALUES="1e-5 5e-6 1e-6" bash sweep_27b_lr.sh
#
# Env: WORK(output_scale_0p5M) LR_VALUES("1e-5 5e-6 1e-6") E2E_STEPS(2000) NGPU(2) EVAL_SAMPLES(96)
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

ORIG_MODEL=${ORIG_MODEL:-/home/kyler/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9}
export ORIG_MODEL
WORK=${WORK:-output_scale_0p5M}
RECOVERED=$WORK/modified_model            # Block-AP output = E2E student
TEACHER=$WORK/teacher_topk.pt
CALIB=$WORK/calibration_data.json; CALIB_EVAL=$WORK/calib_eval.json
SHARED_REF=$WORK/_eval_ref.pt             # FP-teacher reference already staged by the pipeline run
SEQ=1024; NGPU=${NGPU:-2}; E2E_STEPS=${E2E_STEPS:-2000}; EVAL_SAMPLES=${EVAL_SAMPLES:-96}
LR_VALUES=${LR_VALUES:-1e-5 5e-6 1e-6}
RECIPE="--lr-schedule constant --scale-ema-decay 0.999 --scale-ema-start-frac 0.5 --select ema"
NSAMP=$(./.venv/bin/python -c "import json;print(len(json.load(open('$CALIB'))))")

for f in "$RECOVERED/model.safetensors.index.json" "$TEACHER" "$CALIB_EVAL" "$SHARED_REF"; do
  [ -e "$f" ] || { echo "FATAL: missing upstream artifact $f (run run_full_pipeline.sh for $WORK first)"; exit 1; }
done
echo "########## 27B E2E LR SWEEP — $(date) ##########"
echo "  WORK=$WORK  steps=$E2E_STEPS  NSAMP=$NSAMP  recipe: $RECIPE"
echo "  LRs: $LR_VALUES   (reference: lr 5e-5 already gave in-domain 1.1001x)"

for lr in $LR_VALUES; do
  safe=$(printf '%s' "$lr" | tr -c 'A-Za-z0-9.-' '_')   # printf (not echo) — no trailing newline -> no trailing '_'
  out="$WORK/lrsweep/lr_$safe/modified_model"; log="$WORK/lrsweep/lr_$safe.eval.log"
  stage "E2E lr=$lr"
  rm -rf "$WORK/lrsweep/lr_$safe"; mkdir -p "$WORK/lrsweep/lr_$safe"
  $PY -m torch.distributed.run --nproc_per_node="$NGPU" --standalone \
      src/e2e_qp_distill.py --train --student-path "$RECOVERED" --orig-config-path "$ORIG_MODEL" \
      --calib "$CALIB" --teacher-cache "$TEACHER" --out "$out" \
      --seq "$SEQ" --steps "$E2E_STEPS" --lr "$lr" --max-samples "$NSAMP" --loss-fn cakld --feat-weight 0 \
      $RECIPE || { echo "FATAL: E2E lr=$lr"; exit 1; }
  stage "eval lr=$lr"
  $PY src/eval_ternary.py --fp-path "$ORIG_MODEL" --ternary-path "$out" --orig-config-path "$ORIG_MODEL" \
      --calib "$CALIB_EVAL" --seq "$SEQ" --max-samples "$EVAL_SAMPLES" \
      --ref-file "$SHARED_REF" --reuse-ref 2>&1 | tee "$log"
done
stage_end

echo; echo "=== LR SWEEP RESULTS (27B 0.5M, held-out vs FP; lower ppl = better; target < 1.087x) ==="
./.venv/bin/python - "$WORK" $LR_VALUES <<'PY'
import re, os, sys
W, lrs = sys.argv[1], sys.argv[2:]
def ppl(p):
    if not os.path.exists(p): return None
    t = open(p).read()
    g = lambda pat: (float(re.search(pat, t).group(1)) if re.search(pat, t) else float('nan'))
    return dict(ppl=g(r'ppl ratio \(ternary/FP\)\s*:\s*([0-9.]+)'),
                kl=g(r'mean top-\d+ KL\(FP\|\|tern\):\s*([0-9.]+)'),
                ag=g(r'top-1 agreement\s*:\s*([0-9.]+)'))
print(f"\n  {'lr':>8}{'ppl':>10}{'KL':>9}{'agree%':>9}")
print("  " + "-" * 38)
rows = []
# include the already-run 5e-5 from the main pipeline eval if findable
m = ppl(os.path.join(W, "lrsweep", "_ref5e-5.eval.log"))
for lr in lrs:
    safe = re.sub(r'[^A-Za-z0-9.-]', '_', lr)
    r = ppl(os.path.join(W, "lrsweep", f"lr_{safe}.eval.log"))
    if not r: print(f"  {lr:>8}{'(no log)':>10}"); continue
    print(f"  {lr:>8}{r['ppl']:>10.4f}{r['kl']:>9.4f}{r['ag']:>9.2f}")
    if r['ppl'] == r['ppl']: rows.append((lr, r['ppl']))
print(f"  {'5e-5':>8}{1.1001:>10.4f}{'':>9}{'(main run)':>9}")
if rows:
    bv, bp = min(rows, key=lambda x: x[1])
    print("  " + "-" * 38)
    print(f"  BEST swept: lr={bv} -> {bp:.4f}" + ("  ✅ beats 1.087x target" if bp < 1.087 else "  (still above 1.087x)"))
PY
echo; echo "########## LR sweep done — logs under $WORK/lrsweep/*.eval.log ##########"
