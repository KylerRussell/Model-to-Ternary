#!/usr/bin/env bash
# qat_sweep_2b.sh — sweep the distillation-QAT pipeline on the Qwen3.5-2B testbed.
# Builds/reuses the 2B upstream (untie -> rotation -> calib -> teacher; NO Block-AP — QAT is FP-init),
# then for each swept value runs:
#     block-wise QAT (block_ap_recovery --qat)  ->  IN-DOMAIN ppl eval
#     E2E polish (e2e_qp_distill)                ->  IN-DOMAIN ppl + OOD (WikiText-2) ppl
# so we see what each stage buys. All evals vs the FP-2B; in-domain shares one FP ref.
#
#   bash qat_sweep_2b.sh                                  # default: sweep --qat-epochs over {2,4}
#   SWEEP_FLAG=--samples SWEEP_VALUES="512 1024 2048" bash qat_sweep_2b.sh
#
# Env: WORK(output_2b_ab) CALIB_TOKENS(4000000) E2E_STEPS(2000) EVAL_SAMPLES(64)
#      SWEEP_FLAG(--qat-epochs) SWEEP_VALUES("2 4") QAT_BASE E2E_FLAGS FRESH(1=rerun cells)
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
PLAIN=./.venv/bin/python

WORK=${WORK:-output_2b_ab}
TIED_2B=$(printf '%s' "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/*/)
CTM_DATA=${CTM_DATA:-$HOME/Documents/CTM-Transformer/data_cache}
ORIG_MODEL=$WORK/untied_2b; export ORIG_MODEL
ROT=$WORK/rot/modified_model
CALIB_TOKENS=${CALIB_TOKENS:-4000000}
CALIB=$WORK/calibration_data_4M.json
TEACHER=$WORK/teacher_topk_4M.pt
CALIB_EVAL=$WORK/calib_eval.json            # held-out (kept consistent across all cells/runs)
SHARED_REF=$WORK/_qatsweep_ref.pt           # one shared FP-teacher pass for in-domain evals
SEQ=1024; E2E_STEPS=${E2E_STEPS:-2000}; EVAL_SAMPLES=${EVAL_SAMPLES:-64}
SWEEP_FLAG=${SWEEP_FLAG:---qat-epochs}
SWEEP_VALUES=${SWEEP_VALUES:-2 4}
QAT_BASE=${QAT_BASE:---samples 1024 --qat-lr 1e-4 --qat-scale-lr 1e-5}
E2E_FLAGS=${E2E_FLAGS:---lr 5e-5 --lr-schedule constant --scale-ema-decay 0.999 --scale-ema-start-frac 0.5 --select ema --loss-fn cakld --feat-weight 0}
OUT=$WORK/qatsweep; mkdir -p "$OUT"

# ── build the upstream once (FP-init QAT needs rotation+calib+teacher; no Block-AP) ──
[ -n "$TIED_2B" ] || { echo "FATAL: Qwen3.5-2B snapshot not in HF cache"; exit 1; }
[ -f "$ORIG_MODEL/config.json" ] || { stage "untie 2B"; $PY src/untie_embeddings.py --src "$TIED_2B" --out "$ORIG_MODEL" || exit 1; }
[ -d "$ROT" ] || { stage "rotation"; $PY src/convert.py --model-path "$ORIG_MODEL" --output-dir "$WORK/rot" --rotation-only --skip-gguf || exit 1; }
[ -f "$CALIB_EVAL" ] || { stage "calib_eval (held-out, one-time)"; $PY src/build_diverse_calib.py --orig-model "$ORIG_MODEL" --ctm-data "$CTM_DATA" --tokens 500000 --out /tmp/_junk_train.json --eval-out "$CALIB_EVAL" || exit 1; }
[ -f "$CALIB" ] || { stage "calib ($CALIB_TOKENS tok, train only — keep existing eval)"; $PY src/build_diverse_calib.py --orig-model "$ORIG_MODEL" --ctm-data "$CTM_DATA" --tokens "$CALIB_TOKENS" --out "$CALIB" --eval-out /tmp/_junk_eval.json || exit 1; }
NSAMP=$($PLAIN -c "import json;print(len(json.load(open('$CALIB'))))")
[ -f "$TEACHER" ] || { stage "teacher cache ($NSAMP seqs)"; $PY src/e2e_qp_distill.py --precompute-teacher --teacher-path "$ORIG_MODEL" --calib "$CALIB" --teacher-cache "$TEACHER" --seq "$SEQ" --topk 64 --teacher-dp 0 --gpu-mem 20GiB --cpu-mem 30GiB --max-samples "$NSAMP" || exit 1; }

# Evals run with PLAIN python (no memguard scope — they're small, no host-freeze risk; the scope+pipe
# was silently eating output) and redirect straight to the log (no tee>/dev/null that masked failures).
indomain_eval () {  # $1 model-dir  $2 out-log  $3 extra (e.g. --plain-hf)
  $PLAIN src/eval_ternary.py --fp-path "$ORIG_MODEL" --ternary-path "$1" --orig-config-path "$ORIG_MODEL" \
      --calib "$CALIB_EVAL" --seq "$SEQ" --max-samples "$EVAL_SAMPLES" \
      --ref-file "$SHARED_REF" --reuse-ref ${3:-} > "$2" 2>&1 \
      && grep -q 'ppl ratio' "$2" || echo "(IN-DOMAIN EVAL FAILED: $1 — see $2)"
}
ood_eval () {       # $1 e2e-model-dir  $2 out-log
  ORIG="$ORIG_MODEL" FP_DIR="$ORIG_MODEL" E2E_MODEL="$1" $PLAIN src/ood_ppl.py > "$2" 2>&1 \
      && grep -q 'OOD ppl ratio' "$2" || echo "(OOD EVAL FAILED: $1 — see $2)"
}

[ "${FRESH:-1}" = "1" ] && rm -f "$SHARED_REF"
echo "########## 2B QAT SWEEP — $(date) ##########"
echo "  sweep $SWEEP_FLAG over: $SWEEP_VALUES   QAT_BASE: $QAT_BASE"
echo "  E2E: $E2E_FLAGS  (steps $E2E_STEPS)"

for v in $SWEEP_VALUES; do
  safe=$(printf '%s' "$v" | tr -c 'A-Za-z0-9.-' '_')
  QD=$OUT/qat_$safe; ED=$OUT/qat_${safe}_e2e
  [ "${FRESH:-1}" = "1" ] && rm -rf "$QD" "$ED"
  mkdir -p "$QD"
  ln -sf "$(realpath "$CALIB")" "$QD/calibration_data.json"     # block_ap reads <output-dir>/calibration_data.json

  if [ ! -f "$QD/modified_model/model.safetensors.index.json" ]; then
    stage "[$v] QAT block-AP"
    $PY src/block_ap_recovery.py --model-path "$ROT" --orig-config-path "$ORIG_MODEL" \
        --output-dir "$QD" --qat $QAT_BASE $SWEEP_FLAG "$v" || { echo "FATAL: QAT $v"; continue; }
  else stage "[$v] QAT block-AP [skip — exists]"; fi
  stage "[$v] in-domain eval (after QAT)"
  indomain_eval "$QD/modified_model" "$OUT/qat_$safe.indomain.log" "--plain-hf"

  if [ ! -f "$ED/modified_model/model.safetensors.index.json" ]; then
    stage "[$v] E2E polish"
    $PY src/e2e_qp_distill.py --train --student-path "$QD/modified_model" --orig-config-path "$ORIG_MODEL" \
        --calib "$CALIB" --teacher-cache "$TEACHER" --out "$ED/modified_model" \
        --seq "$SEQ" --steps "$E2E_STEPS" --max-samples "$NSAMP" $E2E_FLAGS --log-every 50 || { echo "FATAL: E2E $v"; continue; }
  else stage "[$v] E2E polish [skip — exists]"; fi
  stage "[$v] in-domain + OOD eval (after E2E)"
  indomain_eval "$ED/modified_model" "$OUT/qat_${safe}_e2e.indomain.log"
  ood_eval "$ED/modified_model" "$OUT/qat_${safe}_e2e.ood.log"
done
stage_end

echo; echo "=== QAT SWEEP SUMMARY ($SWEEP_FLAG; ppl ratio ternary/FP, lower = better) ==="
$PLAIN - "$OUT" "$SWEEP_FLAG" $SWEEP_VALUES <<'PY'
import re, os, sys
OUT, flag, vals = sys.argv[1], sys.argv[2], sys.argv[3:]
def grab(p, pat):
    if not os.path.exists(p): return float('nan')
    m = re.search(pat, open(p).read()); return float(m.group(1)) if m else float('nan')
ID = r'ppl ratio \(ternary/FP\)\s*:\s*([0-9.]+)'
OD = r'OOD ppl ratio\s+([0-9.]+)'
print(f"\n  {flag:>14}{'QAT in-dom':>12}{'E2E in-dom':>12}{'E2E OOD':>10}")
print("  " + "-" * 48)
for v in vals:
    s = re.sub(r'[^A-Za-z0-9.-]', '_', v)
    qi = grab(f"{OUT}/qat_{s}.indomain.log", ID)
    ei = grab(f"{OUT}/qat_{s}_e2e.indomain.log", ID)
    eo = grab(f"{OUT}/qat_{s}_e2e.ood.log", OD)
    print(f"  {v:>14}{qi:>12.4f}{ei:>12.4f}{eo:>10.4f}")
print("  " + "-" * 48)
print("  refs: post-hoc+E2E in-dom 1.7752x ; bare-QAT(0.5M,2ep) blockAP 2.16x / +E2E 1.5224x")
PY
echo; echo "########## QAT sweep done — logs under $OUT/ ##########"
