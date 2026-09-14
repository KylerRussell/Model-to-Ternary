#!/bin/bash
# Confirm c=1.20 on 3 paired seeds, and bracket the collapse boundary between 1.20 and 1.40.
# c=1.40 passed ALL THREE Gate B bars while think_len fell to 224 vs the teacher's 586-648 -- a fake
# pass the gate cannot see. The rule is the smallest c reaching the bar with think_len ABOVE the
# teacher's, so 1.25/1.30 are scanned to find where that stops being true.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
M=${MODEL:-output_sweep/opsa/modified_model}
RES=output_sweep/thinkcal_results.txt
run(){
  local c=$1; local s=$2                      # separate statements: under `set -u`, bash expands
  local O=output_sweep/tc_c${c}_s${s}.json    # every arg of one `local` BEFORE assigning any of them,
                                              # so ${c} in the same statement is unbound and aborts
  [ -f "$O" ] && return 0
  echo "  [$(date +%F' '%H:%M:%S)] START c=$c seed=$s" >> $RES
  env DRY_MULT=0.8 DRY_BASE=1.75 DRY_ALLOWED=2 THINK_ROW_SCALE=$c SEED=$s NP=48 \
      MODEL="$M" OUT="$O" bash experiments/sweep/gateb.sh > output_sweep/tc_c${c}_s${s}.log 2>&1
  ./.venv/bin/python - "$O" "$c" "$s" <<'PY' >> $RES
import json,sys
d=json.load(open(sys.argv[1]))
print(f"  RESULT c={sys.argv[2]} seed={sys.argv[3]}  commit={d['commit_rate']:.4f} "
      f"loop={d['loop_rate']:.4f} comp={d['mean_comp_ratio']:.4f} trunc={d['trunc_rate']:.4f} "
      f"think_len={d.get('mean_think_len',0):.0f} n_closed={d.get('n_closed',0)}")
PY
}
run 1.25 0; run 1.30 0          # bracket the collapse boundary
run 1.20 1; run 1.20 2          # confirm the safe value on paired seeds
echo DONE > output_sweep/.tcc_done
