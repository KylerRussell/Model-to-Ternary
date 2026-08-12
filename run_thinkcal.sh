#!/bin/bash
# run_thinkcal.sh — POST-HOC, ZERO-TRAINING `</think>`-row calibration (researcher round-6 RQ4).
# Multiplies all per-block ternary scales of the </think> row of lm_head by c>1: a fully-ternary, foldable
# per-token logit gain (assignments untouched -> TQ2_0-exact). Sweeps c and reports commit/loop/think_len at the
# 2048 gate. PICK THE SMALLEST c that reaches ~FP-parity commit WITHOUT collapsing think_len (= premature
# closing) and WITHOUT raising loop. It CANNOT fix looping — that is the trained model's job.
#   MODEL=<dir> bash run_thinkcal.sh          (defaults to the final run's output)
set -u; cd "$(dirname "$0")"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY=./.venv/bin/python; ORIG=output_4b/untied_4b
MODEL=${MODEL:-output_4b/final_g64q8/e2e/modified_model}; BLK=${BLK:-64}
CS=${CS:-"1.0 1.05 1.10 1.20 1.40"}
[ -e "$MODEL/model.safetensors" ] || { echo "FATAL: $MODEL missing"; exit 1; }
echo "########## </think>-ROW CALIBRATION $(date) | $MODEL (g$BLK) ##########"
echo "  refs: FP commit 75.0% loop 25.0% | target = smallest c at ~FP commit, no think_len collapse"
for c in $CS; do
  echo "--- c=$c ---"
  TERNARY_BLOCK_SIZE=$BLK ORIG=$ORIG E2E_MODEL="$MODEL" THINK_ROW_SCALE=$c \
    N_PREFIX=48 MAXNEW=2048 THINK=1 TEMP=0.6 BATCH=8 GATE_OUT=logs/thinkcal_c${c}.json \
    $PY src/loop_gate.py > logs/thinkcal_c${c}.log 2>&1
  grep -aE 'commit_rate|loop_rate|trunc_rate|comp_ratio|think_len' logs/thinkcal_c${c}.log | sed 's/^/    /'
done
echo ""
echo "=== SUMMARY (pick smallest c at FP-parity commit, think_len stable) ==="
$PY - <<'PY'
import json, glob, re
rows=[]
for f in sorted(glob.glob("logs/thinkcal_c*.json")):
    c=float(re.search(r'_c([0-9.]+)\.json',f).group(1)); d=json.load(open(f))
    rows.append((c,d.get('commit_rate'),d.get('loop_rate'),d.get('mean_think_len'),d.get('mean_comp_ratio')))
print(f"  {'c':>5} {'commit':>8} {'loop':>7} {'think_len':>10} {'compR':>7}")
for c,cm,lp,tl,cr in sorted(rows):
    tl = f"{tl:.0f}" if tl else "-"
    print(f"  {c:>5.2f} {100*cm:>7.1f}% {100*lp:>6.1f}% {tl:>10} {cr:>7.2f}")
print("  (FP: commit 75.0%  loop 25.0%  compR 2.40)")
PY
echo "########## CALIBRATION DONE $(date) ##########"
