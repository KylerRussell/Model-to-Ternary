(
echo "=== Start Pipeline: $(date) ==="

echo "=== 1) Clean rotated FP16 model (Phase 1) ==="
rm -rf output/modified_model
./.venv/bin/python convert.py --download --output-dir ./output --rotation-only --skip-gguf

echo "=== 2) Calibration data (ungated wikitext/c4 to validate the path) ==="
./.venv/bin/python calibration.py --fallback --samples 64 \
    --output ./output_calib/calibration_data.json

echo "=== 3) GPTQ calibration ON THE ROTATED MODEL → new output dir (no collision) ==="
./.venv/bin/python calibrate_and_quantize.py \
    --model-path ./output/modified_model \
    --orig-config-path /home/kyler/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9 \
    --output-dir ./output_calib

echo "=== 4) GGUF + generation test ==="
./.venv/bin/python convert_hf_to_gguf_patched.py output_calib/modified_model \
    --outfile output/Qwen3.6-27B-ternary-gptq-f16.gguf --outtype f16
llama-completion -m output/Qwen3.6-27B-ternary-gptq-f16.gguf -c 2048 \
    -p "Once upon a time, there was a" -n 64 --temp 0 -no-cnv
) > pipeline.log 2>&1 &
