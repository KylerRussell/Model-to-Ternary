#!/bin/bash
cd /home/kasm-user/Documents/Model-to-Ternary
NP=96 MODEL=output_sweep/opsa/modified_model        OUT=output_sweep/loop_gate_opsa96.json \
  bash experiments/sweep/gateb.sh > output_sweep/gateb_opsa96.log 2>&1
NP=96 MODEL=output_4b_g4/e2eqp/modified_model       OUT=output_sweep/loop_gate_base96.json \
  bash experiments/sweep/gateb.sh > output_sweep/gateb_base96.log 2>&1
echo DONE > output_sweep/.gateb_pair_done
