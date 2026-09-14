#!/bin/bash
# DRY sampler A/B on Gate B — PAIRED BY SEED, one variable, no stacking.
#
# Why DRY first (13ag): PLAER = 0.400, so only ~40% of our loops hold an extractable answer. DRY is
# the only candidate whose value does NOT scale with PLAER — it suppresses verbatim continuation
# whether or not an answer was ever derived.
#
# Why seeds: an UNCHANGED model swung loop_rate by 0.396 across reruns (§13w). Looping is bistable
# per prompt and the prompt set sits near that boundary, so a single run is uninterpretable. Paired
# seeds (same prompts, same base RNG stream) are far more powerful than unpaired means.
#
# Why the teacher arm: Gate B's bars (commit .75 / loop .25 / comp 2.40) come from the FP TEACHER.
# Changing the sampler for the ternary model only would compare under different rules, so the
# teacher is run with the same sampler. Queued after the ternary arms so the cheap question resolves
# first.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
M=${MODEL:-output_sweep/opsa/modified_model}
RES=output_sweep/dry_results.txt
for SEED in 0 1 2; do
  for ARM in off on; do
    if [ "$ARM" = "on" ]; then EXTRA="DRY_MULT=0.8 DRY_BASE=1.75 DRY_ALLOWED=2"; else EXTRA="DRY_MULT=0"; fi
    O=output_sweep/dry_${ARM}_s${SEED}.json
    [ -f "$O" ] && { echo "  [skip] $O" >> $RES; continue; }
    echo "  [$(date +%F' '%H:%M:%S)] START dry=$ARM seed=$SEED" >> $RES
    env $EXTRA SEED=$SEED NP=48 MODEL="$M" OUT="$O" \
        bash experiments/sweep/gateb.sh > output_sweep/dry_${ARM}_s${SEED}.log 2>&1
    ./.venv/bin/python - "$O" "$ARM" "$SEED" <<'PY' >> $RES
import json,sys
d=json.load(open(sys.argv[1]))
print(f"  RESULT dry={sys.argv[2]} seed={sys.argv[3]}  loop={d['loop_rate']:.4f} "
      f"commit={d['commit_rate']:.4f} comp={d['mean_comp_ratio']:.4f} trunc={d['trunc_rate']:.4f}")
PY
  done
done
echo DONE > output_sweep/.dry_done
