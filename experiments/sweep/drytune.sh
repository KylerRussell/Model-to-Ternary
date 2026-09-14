#!/bin/bash
# DRY strength sweep. 13ai measured a reproducible commit COST from stock llama.cpp settings
# (teacher commit -0.1111 +/- 0.0241, 3/3 seeds), because committing means RESTATING and DRY
# penalises exactly the continuations that extend earlier context. Commit is our binding constraint,
# so the stock config is mistuned for us: we want most of the -0.43 loop benefit at less commit cost.
#
# Two axes, cheapest first:
#   mult    lowers the penalty everywhere
#   allowed raises the match length before any penalty applies, which spares SHORT legitimate
#           restatements ("the answer is 60") while still catching long verbatim loops
# Seed 0 scan; the winner gets confirmed on 3 paired seeds before anything is claimed.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
M=${MODEL:-output_sweep/opsa/modified_model}
RES=output_sweep/drytune_results.txt
for CFG in "0.8 2" "0.4 2" "0.8 4" "0.8 8" "0.3 4"; do
  set -- $CFG; MULT=$1; ALLOW=$2
  O=output_sweep/dt_m${MULT}_a${ALLOW}_s0.json
  [ -f "$O" ] && continue
  echo "  [$(date +%F' '%H:%M:%S)] START mult=$MULT allowed=$ALLOW" >> $RES
  env DRY_MULT=$MULT DRY_BASE=1.75 DRY_ALLOWED=$ALLOW SEED=0 NP=48 MODEL="$M" OUT="$O" \
      bash experiments/sweep/gateb.sh > output_sweep/dt_m${MULT}_a${ALLOW}.log 2>&1
  ./.venv/bin/python - "$O" "$MULT" "$ALLOW" <<'PY' >> $RES
import json,sys
d=json.load(open(sys.argv[1]))
print(f"  RESULT mult={sys.argv[2]} allowed={sys.argv[3]}  commit={d['commit_rate']:.4f} "
      f"loop={d['loop_rate']:.4f} comp={d['mean_comp_ratio']:.4f} trunc={d['trunc_rate']:.4f}")
PY
done
echo DONE > output_sweep/.dt_done
