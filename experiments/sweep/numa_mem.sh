#!/bin/bash
# torchrun --no-python wrapper: INTERLEAVED placement across nodes 0,1,2.
#
# Measured tradeoff on this box (2 GB pinned buffer -> GPU0, and a CPU copy over it):
#   policy              H2D GB/s   CPU-copy GB/s   deterministic
#   unbound             3.3-3.9    19-24           NO (node lottery: node 3 costs 55%)
#   interleave=all      3.3-3.6    12-14           yes
#   interleave=0,1,2    4.37       10.70           yes      <- chosen
#   membind=1,2         5.42        5.28           yes
# Concentrating memory helps H2D and starves CPU bandwidth (one memory controller instead of four).
# The step needs BOTH: latent H2D streaming and heavy CPU-side Adam/accumulate. membind=1,2 sits at
# the starved extreme and measured 271.7 s/step, slower than any unbound run.
#
# The sweep's purpose is COMPARING CONFIGS, so what it needs above all is the SAME placement every
# arm. Interleaving is deterministic by construction (round-robin), excludes node 3, and sits
# mid-range on both axes rather than at either extreme. Absolute numbers are not claimed optimal.
echo "[numa] LOCAL_RANK=${LOCAL_RANK:-?} -> interleave=0,1,2" >&2
exec numactl --interleave=0,1,2 \
  /home/kasm-user/Documents/Model-to-Ternary/.venv/bin/python3 -u src/e2e_qp_distill.py "$@"
