#!/bin/bash
# Per-rank NUMA co-location. Both GPUs are on node 1; rank 0 is the critical path so it gets the
# GPU-local node and rank 1 goes to node 0 (distance 20), since one 161 GB node cannot hold both.
#
# NO RLIMIT_DATA CAP. One was tried and REMOVED: safetensors mmaps the model shards PRIVATE, and per
# getrlimit(2) RLIMIT_DATA has covered private file mappings since Linux 4.7, so the cap counted the
# 52 GB student checkpoint against it. build_student()'s loader swallows load exceptions
# (`except Exception: continue`), so the failure surfaced as "310 tensors did not load" -- every norm
# weight stranded on meta -- rather than an allocation error. The cap existed only to catch the
# ~211 GB duplicate-Adam allocation, and that is now fixed at source (_opt_step_scales_only), so it
# was costing correctness for no remaining benefit.
case "${LOCAL_RANK:-0}" in
  0) NODE=1 ;;   # GPU-local, critical path
  *) NODE=0 ;;   # distance 20
esac
echo "[numa] LOCAL_RANK=${LOCAL_RANK:-?} -> cpunodebind=$NODE membind=$NODE (no RLIMIT_DATA)" >&2
exec numactl --cpunodebind=$NODE --membind=$NODE \
  /home/kasm-user/Documents/Model-to-Ternary/.venv/bin/python3 -u src/e2e_qp_distill.py "$@"
