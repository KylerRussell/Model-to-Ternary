#!/bin/bash
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
GPU=$1; TAG=$2; MODE=$3; EMB=${EMB:-}
O=output_sweep/hg_${TAG}.json
[ -f "$O" ] && { echo "SKIP $TAG"; exit 0; }
echo "  [$(date +%F' '%H:%M:%S)] START hg_$TAG mode=${MODE:-ternary}" >> output_sweep/head_results.txt
CUDA_VISIBLE_DEVICES=$GPU \
env ORIG=$PWD/output_4b/untied_4b E2E_MODEL=output_sweep/opsa/modified_model MODEL_KIND=tern \
    ${MODE:+HEAD_MODE=$MODE} ${EMB:+EMBED_MODE=$EMB} HEAD_SRC=$PWD/output_4bpipe/rotbase/modified_model \
    THINK_ROW_SCALE=1.30 DRY_MULT=0.8 DRY_BASE=1.75 DRY_ALLOWED=2 \
    N_PROB=48 MAXNEW=2048 THINK=1 TEMP=0.6 BATCH=8 SEED=0 \
    OUT="$O" ROWS=output_sweep/hg_${TAG}_rows.json \
    ./.venv/bin/python src/math_correct.py > output_sweep/hg_${TAG}.log 2>&1
echo "  [$(date +%F' '%H:%M:%S)] END hg_$TAG rc=$?" >> output_sweep/head_results.txt
./.venv/bin/python - "$O" "$TAG" <<'PY' >> output_sweep/head_results.txt
import json,sys
d=json.load(open(sys.argv[1]))
print(f"  RESULT {sys.argv[2]:8s} acc={d['accuracy']:.4f} ({d['n_correct']}/{d['n']}) "
      f"closed={d['closed_rate']:.4f} think_len={d['mean_think_len']:.0f}")
PY
touch output_sweep/.hg_${TAG}_done
