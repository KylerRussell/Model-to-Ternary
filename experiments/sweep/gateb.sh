#!/bin/bash
# Re-run Gate B (the deploy gate) on an arbitrary model dir. Same knobs the pipeline uses, so the
# numbers are directly comparable to output_4b_g4/loop_gate.json.
#   usage: MODEL=<dir> OUT=<json> [NP=48] bash experiments/sweep/gateb.sh
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
M=${MODEL:?set MODEL=<model dir>}
O=${OUT:-output_sweep/loop_gate_opsa.json}
echo "=== Gate B on $M  (N_PREFIX=${NP:-48}, MAXNEW=2048, THINK=1, TEMP=0.6) ==="
echo "    targets: commit >=0.68 | loop <=0.30 | comp-ratio <=3.1   [FP teacher .75/.25/2.40]"
env ORIG="$PWD/output_4b/untied_4b" E2E_MODEL="$M" SEED="${SEED:-0}" \
    N_PREFIX="${NP:-48}" MAXNEW=2048 THINK=1 TEMP=0.6 BATCH=8 GATE_OUT="$O" \
    ${GATE_SAMPLES:+GATE_SAMPLES="$GATE_SAMPLES"} \
    ./.venv/bin/python src/loop_gate.py
echo "--- $O ---"
./.venv/bin/python - "$O" <<'PY'
import json,sys,math
d=json.load(open(sys.argv[1])); n=d.get("n",48)
for k,tgt,op in (("commit_rate",0.68,">="),("loop_rate",0.30,"<="),("mean_comp_ratio",3.1,"<=")):
    v=d[k]; ok = (v>=tgt) if op==">=" else (v<=tgt)
    extra=""
    if k in ("commit_rate","loop_rate"):
        se=math.sqrt(v*(1-v)/n); extra=f"  ({v*n:.0f}/{n}, SE {se*100:.1f}pts, {abs(v-tgt)/se:.2f} SE from gate)"
    print(f"  {k:<17} {v:.4f} {op} {tgt}  {'PASS' if ok else 'FAIL'}{extra}")
PY
