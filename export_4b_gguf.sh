#!/bin/bash
# export_4b_gguf.sh — build the GGUFs for the 4B same-memory demo:
#   4B-ternary (A6 winner) -> TQ2_0 (~1.0GB, embed q4_K / head q6_K, matching the pipeline export)
#   Qwen3.5-0.8B / 2B / 4B(FP) -> Q8_0  (the "fp8 at same/larger size" competitors + FP-4B ceiling)
# CPU-only (CUDA_VISIBLE_DEVICES="") so it never touches a running GPU job. Idempotent (skips existing).
set -u; cd "$(dirname "$0")"
PY=${VENV_PY:-./.venv/bin/python}; LQ=${LLAMA_QUANTIZE:-${LLAMA_DIR:-$HOME/llama.cpp}/build/bin/llama-quantize}
HUB=${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}; G=eval_gguf; mkdir -p $G
MG="systemd-run --user --scope --quiet -p MemoryHigh=28G -p MemoryMax=32G -p MemorySwapMax=0"
snap () { ls -d $HUB/models--Qwen--$1/snapshots/*/ 2>/dev/null | head -1 | sed 's:/*$::'; }

conv () {  # src_dir  f16_out
  [ -f "$2" ] && { echo "  [skip] f16 $2"; return 0; }
  $MG env CUDA_VISIBLE_DEVICES="" $PY src/convert_hf_to_gguf_patched.py "$1" --outfile "$2" --outtype f16 \
    || { echo "FATAL convert $1"; return 1; }
}

echo "########## 4B GGUF export $(date) ##########"

# --- 1) 4B ternary (A6) -> TQ2_0 ---
A6=output_4b/strengthen/a1_lr1e-4_ep4_x2_E2E/modified_model
if [ ! -f $G/tern4b_a6-TQ2_0.gguf ]; then
  conv "$A6" $G/tern4b_a6-f16.gguf || exit 1
  $MG env CUDA_VISIBLE_DEVICES="" $LQ --token-embedding-type q4_K --output-tensor-type q6_K \
    $G/tern4b_a6-f16.gguf $G/tern4b_a6-TQ2_0.gguf TQ2_0 || { echo "FATAL quantize ternary"; exit 1; }
  rm -f $G/tern4b_a6-f16.gguf
else echo "  [skip] tern4b_a6-TQ2_0"; fi

# --- 2) competitors -> Q8_0 ---
for spec in "Qwen3.5-0.8B:qwen0.8b" "Qwen3.5-2B:qwen2b" "Qwen3.5-4B:qwen4b"; do
  m=${spec%%:*}; tag=${spec##*:}
  if [ -f $G/${tag}-Q8_0.gguf ]; then echo "  [skip] ${tag}-Q8_0"; continue; fi
  S=$(snap "$m"); [ -n "$S" ] || { echo "FATAL: no snapshot for $m"; exit 1; }
  conv "$S" $G/${tag}-f16.gguf || exit 1
  $MG env CUDA_VISIBLE_DEVICES="" $LQ $G/${tag}-f16.gguf $G/${tag}-Q8_0.gguf Q8_0 || { echo "FATAL quantize $tag"; exit 1; }
  rm -f $G/${tag}-f16.gguf
done

echo "=== sizes (memory footprint = the same-size comparison axis) ==="
ls -la $G/tern4b_a6-TQ2_0.gguf $G/qwen0.8b-Q8_0.gguf $G/qwen2b-Q8_0.gguf $G/qwen4b-Q8_0.gguf 2>/dev/null | awk '{print $5"\t"$NF}'
echo "########## DONE $(date) ##########"
