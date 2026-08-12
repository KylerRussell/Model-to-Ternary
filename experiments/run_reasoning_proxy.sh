#!/bin/bash
# run_reasoning_proxy.sh — build the reasoning-trace proxy (researcher §H) then score the ternhead anchor.
# (1) generate long-CoT thinking-mode traces from the FP teacher; (2) teacher-force FP-vs-ternary over them,
# segmented by phase/position/block. The KEY numbers vs eval2k's forgiving 79.31% generic agreement:
# late-generation agreement + answer agreement (where sub-4-bit collapse compounds and generic eval is blind).
set -u; cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
ORIG=output_4b/untied_4b; ROT=output_4b/rot/modified_model; PY=./.venv/bin/python
TRACES=output_4b/reasoning_traces.json
ANCHOR=output_4b/ternhead/e2e/modified_model
export ORIG_MODEL="$ORIG"
rm -f output_4b/_traces_smoke.json
MG="systemd-run --user --scope --quiet -p MemoryHigh=38G -p MemoryMax=42G -p MemorySwapMax=0"
echo "########## REASONING-TRACE PROXY $(date) ##########"

# ---- (1) generate traces from the FP teacher ----
if [ ! -f "$TRACES" ]; then
  echo "--- generating reasoning traces (FP teacher, thinking mode) ---"
  $MG env CUDA_VISIBLE_DEVICES=0 $PY src/gen_reasoning_traces.py --model $ORIG --out "$TRACES" \
    --n 64 --max-new 1792 --seq-cap 2048 --temp 0.6 --top-p 0.95 > logs/reasoning_gen.log 2>&1
  [ -f "$TRACES" ] || { echo "FATAL gen (see logs/reasoning_gen.log)"; exit 1; }
  tail -2 logs/reasoning_gen.log
fi

# ---- (2) eval the ternhead anchor on the traces ----
echo "--- reasoning-trace eval: ternhead anchor ---"
$MG env CUDA_VISIBLE_DEVICES=0,1 ORIG=$ORIG FP_DIR=$ROT E2E_MODEL="$ANCHOR" TRACES="$TRACES" SEQ_CAP=2048 \
  RT_OUT=logs/reasoning_ternhead.json $PY src/reasoning_trace_eval.py 2>&1 | tee logs/reasoning_eval_ternhead.log
echo "########## DONE $(date) ##########"
