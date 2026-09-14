#!/bin/bash
# Chain gated on the ARTIFACT, never on pgrep: zombie processes match `pgrep -f` forever in this
# container and cost 6.5h of idle GPU when the teacher chain waited on a defunct dry_ab.sh.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
until [ -f output_sweep/.dryT_done ]; do sleep 120; done
SEEDS=0 ./experiments/sweep/thinkcal_dry.sh
