#!/bin/bash
# Stage 0 trace capture. Re-runs the GSM8K arms with the patched row dump so every rollout keeps its
# exact prompt_ids / gen_ids / text. Nothing about the decoding config changes -- same STU, same
# DRY, same SEED, same MAXNEW -- so each arm's accuracy MUST reproduce its mc_* baseline:
#     teacher 18/48 (14/15 closed) | c1.30 2/48 (2/28 closed) | c1.40 1/48 (1/46 closed)
# That reproduction is itself the first result: it tells us whether SEED=0 actually pins the
# rollouts, which the per-problem ExAccErr validation depends on (we correlate a per-rollout metric
# against a per-rollout `ok` label, so the labels have to belong to the traces we analyse).
#
# Arms chosen for what they control, not for coverage:
#   teacher  the on-distribution oracle -- supplies epsilon, and is the positive control (its own
#            traces must NOT show the drift profile, or the metric is measuring something else)
#   c1.30    the primary student operating point (28 closed = the largest scorable student subset)
#   c1.40    known-collapsed positive control: passes every Gate B bar at 1/46. If the metric cannot
#            separate c1.40 from c1.30 it is not measuring chain survival.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
STU=${STU:-output_sweep/opsa/modified_model}
FP=$PWD/output_4bpipe/rotbase/modified_model
N=${N_PROB:-48}
RES=output_sweep/traces_results.txt
run(){
  local tag=$1
  local kind=$2
  local model=$3
  local c=$4
  local O=output_sweep/tr_${tag}.json
  if [ -f "$O" ]; then echo "  SKIP $tag (exists)" >> $RES; return 0; fi
  echo "  [$(date +%F' '%H:%M:%S)] START $tag" >> $RES
  env ORIG=$PWD/output_4b/untied_4b E2E_MODEL="$model" MODEL_KIND=$kind \
      THINK_ROW_SCALE=$c DRY_MULT=0.8 DRY_BASE=1.75 DRY_ALLOWED=2 \
      N_PROB=$N MAXNEW=2048 THINK=1 TEMP=0.6 BATCH=8 SEED=0 \
      OUT="$O" ROWS=output_sweep/tr_${tag}_rows.json \
      ./.venv/bin/python src/math_correct.py > output_sweep/tr_${tag}.log 2>&1
  echo "  [$(date +%F' '%H:%M:%S)] END   $tag rc=$?" >> $RES
  ./.venv/bin/python - "$O" "$tag" <<'PY' >> $RES
import json,sys
d=json.load(open(sys.argv[1]))
print(f"  RESULT {sys.argv[2]:8s} acc={d['accuracy']:.4f} ({d['n_correct']}/{d['n']})  "
      f"closed={d['closed_rate']:.4f}  think_len={d['mean_think_len']:.0f}")
PY
}
run c1.30    tern "$STU" 1.30
run c1.40    tern "$STU" 1.40
run teacher  fp   "$FP"  1.0
echo DONE > output_sweep/.tr_done
