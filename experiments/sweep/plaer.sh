#!/bin/bash
# Dump FULL free-gen rollouts so PLAER (Pre-Loop Answer Extraction Rate) can be measured offline.
# PLAER decides between two remediations that have nothing in common:
#   answer present before loop onset -> COMMITMENT failure -> decoding-time control can fix it
#   no answer ever derived           -> PATH-FINDING failure -> no sampler will help
# Existing evidence already leans commitment: trunc_rate 0.625, n_closed 17/48, mean_think_len 880.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
export ORIG=$PWD/output_4b/untied_4b
NP=${NP:-48} SEED=${SEED:-0} \
MODEL=${MODEL:-output_sweep/opsa/modified_model} \
GATE_OUT=output_sweep/plaer_gate.json \
GATE_SAMPLES=output_sweep/plaer_samples.json \
  bash experiments/sweep/gateb.sh > output_sweep/plaer.log 2>&1
echo "rc=$? $(date)" >> output_sweep/plaer.log
