#!/bin/bash
# FP TEACHER with the SAME sampler, paired by seed.
#
# NOTE: MODEL_KIND=fp is REQUIRED. loop_gate defaults to build_student(), which ternarises
# whatever model it is handed -- a first attempt without it scored the "FP teacher" at commit 0.0000 /
# trunc 0.9792 because it had RTN-ternarised the teacher with no recovery.
# Necessary, not optional: Gate B's bars (commit .75 / loop .25 / comp 2.40) were measured on the FP
# teacher WITHOUT DRY. Seed 0 already has ternary+DRY at loop 0.1042 -- better than the teacher's
# no-DRY 0.25 -- so scoring a DRY'd student against a non-DRY'd teacher is not like-for-like. If DRY
# is part of the shipped inference config it applies to both, and the bar must move with it.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
FP=$PWD/output_4bpipe/rotbase/modified_model     # the rotated FP teacher the pipeline distils toward
RES=output_sweep/dry_results.txt
for SEED in 0 1 2; do
  for ARM in off on; do
    if [ "$ARM" = "on" ]; then EXTRA="DRY_MULT=0.8 DRY_BASE=1.75 DRY_ALLOWED=2"; else EXTRA="DRY_MULT=0"; fi
    O=output_sweep/dryT_${ARM}_s${SEED}.json
    [ -f "$O" ] && continue
    echo "  [$(date +%F' '%H:%M:%S)] START teacher dry=$ARM seed=$SEED" >> $RES
    env $EXTRA MODEL_KIND=fp SEED=$SEED NP=48 MODEL="$FP" OUT="$O" \
        bash experiments/sweep/gateb.sh > output_sweep/dryT_${ARM}_s${SEED}.log 2>&1
    ./.venv/bin/python - "$O" "$ARM" "$SEED" <<'PY' >> $RES
import json,sys
d=json.load(open(sys.argv[1]))
print(f"  RESULT TEACHER dry={sys.argv[2]} seed={sys.argv[3]}  loop={d['loop_rate']:.4f} "
      f"commit={d['commit_rate']:.4f} comp={d['mean_comp_ratio']:.4f} trunc={d['trunc_rate']:.4f}")
PY
  done
done
echo DONE > output_sweep/.dryT_done
