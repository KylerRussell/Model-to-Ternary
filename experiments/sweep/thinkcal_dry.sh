#!/bin/bash
# </think>-row calibration sweep, run AT THE NEW OPERATING POINT (DRY on).
#
# 13ah isolated the remaining Gate B blocker: commit_rate 0.4931 vs a 0.68 bar, and it is NOT caused
# by looping (DRY removed 76% of the looping and moved commit by +0.049; truncation-without-looping
# went -0.09 -> +0.25). The model simply does not emit </think>.
#
# run_thinkcal.sh already targets exactly that, and its own docstring says it "CANNOT fix looping --
# that is the trained model's job". So the two are complementary: DRY for loop/comp, row-gain for
# commit. It was shelved in S8e because commit was at FP parity THEN; it is not now.
#
# Format: multiplies the per-block scales of lm_head row 248069 by c. Assignments untouched ->
# on-grid, TQ2_0-exact, 0 bpw, foldable at deploy via src/fold_think_scale.py.
#
# WATCH FOR: think_len COLLAPSE (premature closing) and loop_rate rising. The right c is the SMALLEST
# that reaches the commit bar without either. A c that closes the trace instantly would "pass" commit
# while destroying the answer.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
M=${MODEL:-output_sweep/opsa/modified_model}
RES=output_sweep/thinkcal_results.txt
SEEDS=${SEEDS:-0}
for c in ${CS:-1.0 1.05 1.10 1.20 1.40}; do
  for SEED in $SEEDS; do
    O=output_sweep/tc_c${c}_s${SEED}.json
    [ -f "$O" ] && continue
    echo "  [$(date +%F' '%H:%M:%S)] START c=$c seed=$SEED (DRY on)" >> $RES
    env DRY_MULT=0.8 DRY_BASE=1.75 DRY_ALLOWED=2 THINK_ROW_SCALE=$c SEED=$SEED NP=48 \
        MODEL="$M" OUT="$O" bash experiments/sweep/gateb.sh > output_sweep/tc_c${c}_s${SEED}.log 2>&1
    ./.venv/bin/python - "$O" "$c" "$SEED" <<'PY' >> $RES
import json,sys
d=json.load(open(sys.argv[1]))
print(f"  RESULT c={sys.argv[2]} seed={sys.argv[3]}  commit={d['commit_rate']:.4f} "
      f"loop={d['loop_rate']:.4f} comp={d['mean_comp_ratio']:.4f} trunc={d['trunc_rate']:.4f} "
      f"think_len={d.get('mean_think_len',0):.0f} n_closed={d.get('n_closed',0)}")
PY
  done
done
echo DONE > output_sweep/.tc_done
