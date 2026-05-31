#!/bin/bash
set -e
rm -rf output/modified_model

echo "=== Phase 1A: Rotation-Only (pure FP16 rotated model) ==="
./.venv/bin/python convert.py --download --output-dir ./output --rotation-only --skip-gguf

echo "=== Converting rotated model to F16 GGUF ==="
./.venv/bin/python convert_hf_to_gguf_patched.py output/modified_model --outfile output/Qwen3.6-27B-rotated-f16.gguf --outtype f16

echo "=== Testing with llama-cli ==="
llama-cli -m output/Qwen3.6-27B-rotated-f16.gguf -c 2048 -p "<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
" -n 64

echo "Done!"
