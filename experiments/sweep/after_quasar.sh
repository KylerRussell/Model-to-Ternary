#!/bin/bash
# Wait for the running chain (QUASAR) to finish, then run the isolating CAT-Q.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
while pgrep -f "chain_catq_quasar.sh" >/dev/null 2>&1; do sleep 60; done
./experiments/sweep/catq2_ab.sh
