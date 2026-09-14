#!/bin/bash
# Chain: corrected CAT-Q (mu clamped) -> QUASAR. Both screened against the measured noise floor
# (13ac): control n=3 = 56.273 +/- 0.309 % / KL 1.1897 +/- 0.0155, 3-SD limit 0.93 pp.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
./experiments/sweep/catq_ab.sh
./experiments/sweep/quasar_ab.sh
echo DONE > output_sweep/.chain_done
