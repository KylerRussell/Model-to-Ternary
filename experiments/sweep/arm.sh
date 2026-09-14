#!/bin/bash
# Run ONE math_correct arm pinned to ONE physical GPU. math_correct.py hardcodes "cuda:0";
# CUDA_VISIBLE_DEVICES remaps it, so no code change is needed to place an arm on GPU1.
# Arms are independent processes with their own SEED, so running two concurrently cannot change
# either one's result -- it only stops half the machine sitting idle (GPU1 was at 0% for 2.4 h).
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
GPU=$1
TAG=$2
KIND=$3
MODEL=$4
C=$5
STU=${STU:-output_sweep/opsa/modified_model}
O=output_sweep/tr_${TAG}.json
if [ -f "$O" ]; then echo "SKIP $TAG (exists)"; exit 0; fi
echo "  [$(date +%F' '%H:%M:%S)] START $TAG on GPU$GPU" >> output_sweep/traces_results.txt
CUDA_VISIBLE_DEVICES=$GPU \
env ORIG=$PWD/output_4b/untied_4b E2E_MODEL="$MODEL" MODEL_KIND=$KIND \
    THINK_ROW_SCALE=$C DRY_MULT=0.8 DRY_BASE=1.75 DRY_ALLOWED=2 \
    N_PROB=${N_PROB:-48} MAXNEW=2048 THINK=1 TEMP=0.6 BATCH=8 SEED=0 \
    OUT="$O" ROWS=output_sweep/tr_${TAG}_rows.json \
    ./.venv/bin/python src/math_correct.py > output_sweep/tr_${TAG}.log 2>&1
rc=$?
echo "  [$(date +%F' '%H:%M:%S)] END $TAG rc=$rc" >> output_sweep/traces_results.txt
./.venv/bin/python - "$O" "$TAG" <<'PY' >> output_sweep/traces_results.txt
import json,sys
d=json.load(open(sys.argv[1]))
print(f"  RESULT {sys.argv[2]:8s} acc={d['accuracy']:.4f} ({d['n_correct']}/{d['n']})  "
      f"closed={d['closed_rate']:.4f}  think_len={d['mean_think_len']:.0f}")
PY
touch output_sweep/.arm_${TAG}_done
