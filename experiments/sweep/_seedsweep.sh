#!/bin/bash
# Gate B across SEEDS. One unseeded run swings loop_rate by 0.396 on an unchanged model, so a single
# number cannot compare two models. Report mean +/- sd over seeds instead.
cd /home/kasm-user/Documents/Model-to-Ternary
for S in 0 1 2 3 4; do
  for M in base:output_4b_g4/e2eqp/modified_model opsa:output_sweep/opsa/modified_model; do
    TAG=${M%%:*}; DIR=${M#*:}
    O=output_sweep/gb_${TAG}_s${S}.json
    [ -f "$O" ] && continue
    SEED=$S NP=48 MODEL="$DIR" OUT="$O" bash experiments/sweep/gateb.sh \
      > output_sweep/gb_${TAG}_s${S}.log 2>&1
  done
done
echo DONE > output_sweep/.seedsweep_done
