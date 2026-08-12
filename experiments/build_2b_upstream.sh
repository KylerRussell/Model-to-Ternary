#!/usr/bin/env bash
# build_2b_upstream.sh — rebuild ONLY what distillation-QAT needs on the 2B testbed:
#   untied FP 2B  +  QuaRot rotation (latent-QAT inits from these FP rotated weights)
#   +  diverse calib (+ held-out eval split)  +  FP teacher top-64 cache (for distillation).
# NO Block-AP (QAT is FP-init, not post-hoc-grid init). Writes under output_2b_ab/.
set -u
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=.venv/bin/python
WORK=output_2b_ab
TIED_2B=$(printf '%s' "$HOME"/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/*/)
CTM_DATA=${CTM_DATA:-$HOME/Documents/CTM-Transformer/data_cache}
ORIG_MODEL=$WORK/untied_2b; export ORIG_MODEL
mkdir -p "$WORK"

if [ ! -f "$ORIG_MODEL/config.json" ]; then
  echo "=== untie 2B ==="; $PY src/untie_embeddings.py --src "$TIED_2B" --out "$ORIG_MODEL" || exit 1; fi
if [ ! -d "$WORK/rot/modified_model" ]; then
  echo "=== rotation ==="; $PY src/convert.py --model-path "$ORIG_MODEL" --output-dir "$WORK/rot" --rotation-only --skip-gguf || exit 1; fi
if [ ! -f "$WORK/calibration_data.json" ]; then
  echo "=== calib (0.5M + eval split) ==="; $PY src/build_diverse_calib.py --orig-model "$ORIG_MODEL" \
      --ctm-data "$CTM_DATA" --tokens 500000 --out "$WORK/calibration_data.json" --eval-out "$WORK/calib_eval.json" || exit 1; fi
NSAMP=$($PY -c "import json;print(len(json.load(open('$WORK/calibration_data.json'))))")
if [ ! -f "$WORK/teacher_topk.pt" ]; then
  echo "=== FP teacher top-64 cache ($NSAMP seqs) ==="; $PY src/e2e_qp_distill.py --precompute-teacher \
      --teacher-path "$ORIG_MODEL" --calib "$WORK/calibration_data.json" --teacher-cache "$WORK/teacher_topk.pt" \
      --seq 1024 --topk 64 --teacher-dp 0 --gpu-mem 20GiB --cpu-mem 30GiB --max-samples "$NSAMP" || exit 1; fi
echo "QAT_UPSTREAM_DONE  (untied_2b, rot, calib, teacher_topk all present)"
