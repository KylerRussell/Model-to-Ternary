#!/bin/bash
# GSM8K accuracy under the Gate B decoding config. Settles the c=1.25 vs c=1.30 judgement that
# think_len can only proxy for: is the 16% shorter thinking at c=1.30 free, or is it accuracy?
#
# Arms:
#   teacher            the FP reference (MODEL_KIND=fp REQUIRED -- see 13ah-i)
#   c=1.0              DRY only, the current student operating point
#   c=1.25 / c=1.30    the two candidates
#   c=1.40             the known-collapsed setting. Included as a POSITIVE CONTROL: it passed every
#                      Gate B bar, so if accuracy scoring is meaningful it must show damage here. If
#                      c=1.40 scores fine, the accuracy metric is not measuring what I think it is.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
STU=${STU:-output_sweep/opsa/modified_model}
FP=$PWD/output_4bpipe/rotbase/modified_model
N=${N_PROB:-48}
RES=output_sweep/mathcorrect_results.txt
run(){
  local tag=$1; local kind=$2; local model=$3; local c=$4
  local O=output_sweep/mc_${tag}.json
  [ -f "$O" ] && return 0
  echo "  [$(date +%F' '%H:%M:%S)] START $tag" >> $RES
  env ORIG=$PWD/output_4b/untied_4b E2E_MODEL="$model" MODEL_KIND=$kind \
      THINK_ROW_SCALE=$c DRY_MULT=0.8 DRY_BASE=1.75 DRY_ALLOWED=2 \
      N_PROB=$N MAXNEW=2048 THINK=1 TEMP=0.6 BATCH=8 SEED=0 \
      OUT="$O" ROWS=output_sweep/mc_${tag}_rows.json \
      ./.venv/bin/python src/math_correct.py > output_sweep/mc_${tag}.log 2>&1
  ./.venv/bin/python - "$O" "$tag" <<'PY' >> $RES
import json,sys
d=json.load(open(sys.argv[1]))
print(f"  RESULT {sys.argv[2]:10s} acc={d['accuracy']:.4f} ({d['n_correct']}/{d['n']})  "
      f"closed={d['closed_rate']:.4f}  think_len={d['mean_think_len']:.0f}")
PY
}
run teacher  fp   "$FP" 1.0
run c1.00    tern "$STU" 1.0
run c1.25    tern "$STU" 1.25
run c1.30    tern "$STU" 1.30
run c1.40    tern "$STU" 1.40
echo DONE > output_sweep/.mc_done
