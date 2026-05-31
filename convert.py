#!/usr/bin/env python3
"""
Main conversion pipeline: Qwen3.6-27B → Ternary (1.58-bit).

Phase 1 (RTN):
  1. Download / locate the model
  2. Stream layers from safetensors (never loads full model into VRAM)
  3. For each targeted linear layer:
     a. Optionally apply Hadamard rotation
     b. Quantize to ternary {-1, 0, 1} with per-block scaling
  4. Save rotated+quantized weights alongside untouched FP16 layers
  5. Convert to GGUF using llama.cpp's convert_hf_to_gguf.py + llama-quantize

The key insight for VRAM management: we NEVER load the full 55GB model.
Instead, we process 1-2 layers at a time, keeping peak VRAM under ~4GB.

Usage:
    python convert.py --model-path /path/to/Qwen3.6-27B --output-dir ./output
    python convert.py --download --output-dir ./output
"""

import argparse
import gc
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# Defer torch import to check availability
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from safetensors import safe_open
    from safetensors.torch import save_file as save_safetensors
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

from config import (
    MODEL_ID, NUM_HIDDEN_LAYERS, LAYER_TYPES,
    HIDDEN_SIZE, INTERMEDIATE_SIZE, VOCAB_SIZE,
    BLOCK_SIZE, should_quantize, ConversionConfig,
    KEEP_FP16_PATTERNS, is_rotatable_projection,
)


def check_dependencies():
    """Verify all required packages are installed."""
    missing = []
    if not HAS_TORCH:
        missing.append("torch (pip install torch)")
    if not HAS_SAFETENSORS:
        missing.append("safetensors (pip install safetensors)")
    if missing:
        print("❌ Missing dependencies:")
        for m in missing:
            print(f"   - {m}")
        sys.exit(1)


def load_weight_index(model_path: Path) -> dict:
    """Load the safetensors weight map."""
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            data = json.load(f)
        return data["weight_map"]

    # Single shard fallback
    single = model_path / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt") as f:
            return {name: "model.safetensors" for name in f.keys()}

    raise FileNotFoundError(f"No safetensors found in {model_path}")


def group_tensors_by_shard(weight_map: dict) -> dict:
    """Group tensor names by their shard file for efficient I/O."""
    shards = {}
    for tensor_name, shard_file in weight_map.items():
        shards.setdefault(shard_file, []).append(tensor_name)
    return shards


def load_tensor(model_path: Path, shard_file: str, tensor_name: str,
                device: str = "cuda:0") -> torch.Tensor:
    """Load a single tensor from a shard file."""
    shard_path = model_path / shard_file
    with safe_open(str(shard_path), framework="pt", device=device) as f:
        return f.get_tensor(tensor_name)


class TernaryConverter:
    """
    Orchestrates the full conversion pipeline.

    Strategy: Process the model shard-by-shard, tensor-by-tensor.
    For each tensor:
      - If it's a quantization target: rotate (optional) → quantize → save
      - If it's FP16-preserved: copy as-is
    """

    def __init__(self, config: ConversionConfig):
        self.config = config
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.stats = {
            "quantized_tensors": 0,
            "fp16_tensors": 0,
            "total_params_quantized": 0,
            "total_params_fp16": 0,
            "errors": {},
        }

    def convert(self, model_path: Path) -> Path:
        """
        Run the full conversion.

        Returns the path to the modified model directory (ready for GGUF conversion).
        """
        from hadamard import (
            rotate_weight_matrix,
            analyze_quantization_friendliness,
            compute_rotation_error,
        )
        from quantizer import quantize_rtn, quantize_absmean, dequantize, compute_quantization_error

        print("\n" + "=" * 70)
        print("TERNARY CONVERSION PIPELINE — Phase 1 (RTN)")
        print("=" * 70)
        print(f"  Source model:  {model_path}")
        print(f"  Output dir:    {self.config.output_dir}")
        print(f"  Block size:    {self.config.block_size}")
        print(f"  Hadamard:      {'ON' if self.config.apply_hadamard else 'OFF'}")
        print(f"  Device:        {self.device}")
        print()

        # Load weight index
        weight_map = load_weight_index(model_path)
        shards = group_tensors_by_shard(weight_map)
        total_tensors = len(weight_map)
        print(f"  Total tensors: {total_tensors}")
        print(f"  Shard files:   {len(shards)}")

        # Pre-load LayerNorm weights for offline rotation
        self.norm_weights = {}
        if self.config.apply_hadamard:
            print(f"\n  Pre-loading LayerNorm weights for offline QuaRot rotation...")
            for tensor_name, shard_file in tqdm(weight_map.items(), desc="  Loading norms"):
                if "norm.weight" in tensor_name or "layernorm.weight" in tensor_name:
                    # Load to CPU to save VRAM, we'll move to GPU when needed
                    self.norm_weights[tensor_name] = load_tensor(model_path, shard_file, tensor_name, "cpu")
            print(f"  Loaded {len(self.norm_weights)} LayerNorm weights.")

        # Create output directory for modified model
        modified_model_dir = self.config.output_dir / "modified_model"
        modified_model_dir.mkdir(parents=True, exist_ok=True)

        # Copy non-weight files (config, tokenizer, etc.)
        self._copy_model_metadata(model_path, modified_model_dir)

        # Process each shard
        processed = 0
        new_weight_map = {}
        shard_list = sorted(shards.items())

        for shard_idx, (shard_file, tensor_names) in enumerate(shard_list):
            print(f"\n📦 Processing shard {shard_idx + 1}/{len(shard_list)}: {shard_file}")
            print(f"   Contains {len(tensor_names)} tensors")

            # We'll accumulate modified tensors for this shard
            modified_tensors = {}

            for tensor_name in tqdm(sorted(tensor_names), desc=f"   Shard {shard_idx+1}"):
                # Load tensor to GPU
                tensor = load_tensor(model_path, shard_file, tensor_name, self.device)

                if is_rotatable_projection(tensor_name):
                    # ── ROTATION (always) + QUANTIZATION (only if should_quantize) ──
                    # Every residual-stream projection is rotated + norm-absorbed here,
                    # whether or not it will be ternarized. This keeps the norm-zeroing in
                    # the FP16 branch valid for EVERY layer (each input_layernorm /
                    # post_attention_layernorm is genuinely absorbed). Quantization is a
                    # separate decision below, so protected projections become rotated FP16
                    # (the transparent rotation-only treatment) instead of being dumped to
                    # the FP16 branch un-rotated with their norm stripped.
                    original_dtype = tensor.dtype
                    weight = tensor.float()

                    if self.config.apply_hadamard and weight.ndim == 2:
                        # Apply QuaRot Offline Hadamard Rotation
                        weight_device = weight.to(self.device)
                        
                        # 1. Attention/MLP Outputs: dimension "output" rotation
                        if ".o_proj" in tensor_name or ".out_proj" in tensor_name or ".down_proj" in tensor_name:
                            rotated = rotate_weight_matrix(weight_device, dim="output")
                        
                        # 2. Attention/MLP Inputs: absorb norm, then dimension "input" rotation
                        else:
                            norm_w = None
                            if ".self_attn." in tensor_name or ".linear_attn." in tensor_name:
                                _parts = tensor_name.split(".")
                                layer_idx = _parts[_parts.index("layers") + 1] if "layers" in _parts else None
                                norm_name = f"model.language_model.layers.{layer_idx}.input_layernorm.weight"
                                if norm_name not in self.norm_weights:
                                    norm_name = f"model.layers.{layer_idx}.input_layernorm.weight"
                                if norm_name in self.norm_weights:
                                    norm_w = self.norm_weights[norm_name].to(self.device).float()
                                else:
                                    raise KeyError(
                                        f"input_layernorm not found for {tensor_name} "
                                        f"(layer_idx={layer_idx!r}); refusing to rotate without "
                                        f"norm absorption — this would silently corrupt the model")
                            elif ".mlp.gate_proj" in tensor_name or ".mlp.up_proj" in tensor_name:
                                _parts = tensor_name.split(".")
                                layer_idx = _parts[_parts.index("layers") + 1] if "layers" in _parts else None
                                norm_name = f"model.language_model.layers.{layer_idx}.post_attention_layernorm.weight"
                                if norm_name not in self.norm_weights:
                                    norm_name = f"model.layers.{layer_idx}.post_attention_layernorm.weight"
                                if norm_name in self.norm_weights:
                                    norm_w = self.norm_weights[norm_name].to(self.device).float()
                                else:
                                    raise KeyError(
                                        f"post_attention_layernorm not found for {tensor_name} "
                                        f"(layer_idx={layer_idx!r}); refusing to rotate without "
                                        f"norm absorption — this would silently corrupt the model")
                            
                            if norm_w is not None:
                                # Check if this is an offset-1 norm (Qwen3/Gemma style)
                                if not norm_name.endswith("linear_attn.norm.weight"):
                                    norm_w = norm_w + 1.0
                                weight_device = weight_device * norm_w  # Absorb diagonal norm
                                
                            rotated = rotate_weight_matrix(weight_device, dim="input")

                        weight_to_save = rotated
                    else:
                        weight_to_save = weight

                    if self.config.rotation_only or not should_quantize(tensor_name):
                        # ── ROTATED-FP16 MODE (rotation-only, or protected from quant) ──
                        # Save pure rotated FP16 weights (no ternary quantization).
                        modified_tensors[tensor_name] = weight_to_save.to(original_dtype).cpu()
                        self.stats["fp16_tensors"] += 1
                        self.stats["total_params_fp16"] += tensor.numel()
                    else:
                        # ── TERNARY QUANTIZATION ──
                        qt = quantize_absmean(weight_to_save, self.config.block_size)
                        reconstructed = dequantize(qt)

                        err = compute_quantization_error(weight_to_save, qt)
                        self.stats["errors"][tensor_name] = {
                            "cosine_sim": err["cosine_similarity"],
                            "sqnr_db": err["sqnr_db"],
                            "sparsity": err["sparsity"],
                            "distribution": err["ternary_distribution"],
                        }

                        modified_tensors[tensor_name] = reconstructed.to(original_dtype).cpu()
                        self.stats["quantized_tensors"] += 1
                        self.stats["total_params_quantized"] += tensor.numel()

                else:
                    # ── FP16 PRESERVATION PATH ──
                    if self.config.apply_hadamard:
                        original_dtype = tensor.dtype
                        
                        # Overwrite ONLY the specific LayerNorm weights we absorbed with 1.0s
                        is_absorbed_norm = False
                        if "model.language_model.norm.weight" in tensor_name or "model.norm.weight" in tensor_name:
                            is_absorbed_norm = True
                        elif tensor_name.startswith("model.") and ("input_layernorm.weight" in tensor_name or "post_attention_layernorm.weight" in tensor_name):
                            is_absorbed_norm = True

                        if is_absorbed_norm:
                            # Set to ZERO because convert_hf_to_gguf_patched.py adds +1
                            # to all Qwen3.6 norm weights (offset-1 format).
                            # 0.0 + 1.0 = 1.0 (identity) in the GGUF.
                            tensor = torch.zeros_like(tensor)
                            
                        # Rotate embeddings (dim="input" because W_emb_new = W_emb * H)
                        elif "embed_tokens.weight" in tensor_name and ("model.embed_tokens" in tensor_name or "model.language_model.embed_tokens" in tensor_name):
                            print(f"   Applying batched rotation for {tensor_name}...")
                            weight_device = tensor.float().to(self.device)
                            rotated_chunks = []
                            chunk_size = 4096
                            for i in range(0, weight_device.shape[0], chunk_size):
                                chunk = weight_device[i:i+chunk_size]
                                rotated_chunks.append(rotate_weight_matrix(chunk, dim="input"))
                            tensor = torch.cat(rotated_chunks, dim=0).to(original_dtype).cpu()
                            del weight_device, rotated_chunks
                            torch.cuda.empty_cache()
                            
                        # Absorb final norm and rotate LM Head (dim="input")
                        elif "lm_head.weight" in tensor_name:
                            print(f"   Applying batched rotation for {tensor_name}...")
                            weight_device = tensor.float().to(self.device)
                            norm_name = "model.language_model.norm.weight"
                            if norm_name not in self.norm_weights:
                                norm_name = "model.norm.weight"
                            if norm_name in self.norm_weights:
                                norm_w = self.norm_weights[norm_name].to(self.device).float()
                                if not norm_name.endswith("linear_attn.norm.weight"):
                                    norm_w = norm_w + 1.0
                                weight_device = weight_device * norm_w
                            rotated_chunks = []
                            chunk_size = 4096
                            for i in range(0, weight_device.shape[0], chunk_size):
                                chunk = weight_device[i:i+chunk_size]
                                rotated_chunks.append(rotate_weight_matrix(chunk, dim="input"))
                            tensor = torch.cat(rotated_chunks, dim=0).to(original_dtype).cpu()
                            del weight_device, rotated_chunks
                            torch.cuda.empty_cache()

                    modified_tensors[tensor_name] = tensor.cpu()
                    self.stats["fp16_tensors"] += 1
                    self.stats["total_params_fp16"] += tensor.numel()

                processed += 1

                # Free GPU memory after each tensor
                del tensor
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Save modified shard
            output_shard = modified_model_dir / shard_file
            print(f"   💾 Saving modified shard: {output_shard}")
            save_safetensors(modified_tensors, str(output_shard))

            for tn in tensor_names:
                new_weight_map[tn] = shard_file

            # Free memory
            del modified_tensors
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save new weight index
        self._save_weight_index(modified_model_dir, new_weight_map)

        # Print summary
        self._print_summary()

        # Save error report
        self._save_error_report()

        return modified_model_dir

    def _copy_model_metadata(self, src: Path, dst: Path):
        """Copy config, tokenizer, and other non-weight files."""
        metadata_files = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "generation_config.json",
            "merges.txt",
            "vocab.json",
            "preprocessor_config.json",
            "chat_template.json",
        ]
        for fname in metadata_files:
            src_file = src / fname
            if src_file.exists():
                shutil.copy2(str(src_file), str(dst / fname))

        # Also copy any .model files (sentencepiece)
        for f in src.glob("*.model"):
            shutil.copy2(str(f), str(dst / f.name))

    def _save_weight_index(self, model_dir: Path, weight_map: dict):
        """Save the safetensors index JSON."""
        index = {
            "metadata": {"total_size": 0},  # Will be recalculated
            "weight_map": weight_map,
        }
        with open(model_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)

    def _print_summary(self):
        """Print conversion summary."""
        print("\n" + "=" * 70)
        print("CONVERSION SUMMARY")
        print("=" * 70)
        print(f"  Quantized tensors:  {self.stats['quantized_tensors']}")
        print(f"  FP16 tensors:       {self.stats['fp16_tensors']}")
        print(f"  Params quantized:   {self.stats['total_params_quantized']:,}")
        print(f"  Params FP16:        {self.stats['total_params_fp16']:,}")

        # Average quality metrics
        if self.stats["errors"]:
            avg_cos = np.mean([e["cosine_sim"] for e in self.stats["errors"].values()])
            avg_sqnr = np.mean([e["sqnr_db"] for e in self.stats["errors"].values()])
            avg_sparsity = np.mean([e["sparsity"] for e in self.stats["errors"].values()])
            print(f"\n  Average cosine similarity: {avg_cos:.6f}")
            print(f"  Average SQNR:              {avg_sqnr:.1f} dB")
            print(f"  Average sparsity:          {avg_sparsity:.1%}")

            # Worst layers
            worst = sorted(
                self.stats["errors"].items(),
                key=lambda x: x[1]["cosine_sim"]
            )[:5]
            print(f"\n  Worst 5 layers by cosine similarity:")
            for name, err in worst:
                short = name.split("model.")[-1] if "model." in name else name
                print(f"    {short:<50s} cos={err['cosine_sim']:.6f}  "
                      f"sqnr={err['sqnr_db']:.1f}dB")

    def _save_error_report(self):
        """Save detailed error report to JSON."""
        report_path = self.config.output_dir / "quantization_report.json"
        with open(report_path, "w") as f:
            json.dump(self.stats, f, indent=2)
        print(f"\n  📊 Error report saved to: {report_path}")


def run_gguf_conversion(modified_model_dir: Path, output_dir: Path,
                        gguf_format: str = "tq2_0") -> Path:
    """
    Convert the modified HF model to GGUF using llama.cpp tools.

    Steps:
      1. convert_hf_to_gguf.py → F16 GGUF
      2. llama-quantize → TQ2_0 GGUF
    """
    print("\n" + "=" * 70)
    print("GGUF CONVERSION")
    print("=" * 70)

    # Find convert script — prefer local patched version (has Qwen3.6 tokenizer hash)
    script_dir = Path(__file__).parent
    local_patched = script_dir / "convert_hf_to_gguf_patched.py"
    convert_script = None
    if local_patched.exists():
        convert_script = str(local_patched)
    else:
        convert_script = shutil.which("convert_hf_to_gguf.py")
        if not convert_script:
            for path in ["/usr/bin/convert_hf_to_gguf.py",
                         "/usr/local/bin/convert_hf_to_gguf.py"]:
                if os.path.exists(path):
                    convert_script = path
                    break

    if not convert_script:
        print("❌ convert_hf_to_gguf.py not found!")
        print("   Install llama.cpp or specify the path manually.")
        return None

    # Step 1: Convert to F16 GGUF
    f16_gguf = output_dir / "Qwen3.6-27B-ternary-f16.gguf"
    print(f"\n  Step 1: Converting to F16 GGUF...")
    print(f"    Script: {convert_script}")
    print(f"    Input:  {modified_model_dir}")
    print(f"    Output: {f16_gguf}")

    cmd = [
        sys.executable, convert_script,
        str(modified_model_dir),
        "--outfile", str(f16_gguf),
        "--outtype", "f16",
    ]
    print(f"    Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ❌ F16 conversion failed!")
        print(f"  STDOUT: {result.stdout[-2000:]}")
        print(f"  STDERR: {result.stderr[-2000:]}")
        return None
    print(f"  ✅ F16 GGUF created: {f16_gguf}")

    # Step 2: Quantize to TQ2_0
    quantize_bin = shutil.which("llama-quantize")
    if not quantize_bin:
        print("❌ llama-quantize not found!")
        return f16_gguf

    ternary_gguf = output_dir / f"Qwen3.6-27B-ternary-{gguf_format.upper()}.gguf"
    print(f"\n  Step 2: Quantizing to {gguf_format.upper()}...")
    print(f"    Input:  {f16_gguf}")
    print(f"    Output: {ternary_gguf}")

    cmd = [quantize_bin, str(f16_gguf), str(ternary_gguf), gguf_format]
    print(f"    Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ❌ Quantization failed!")
        print(f"  STDOUT: {result.stdout[-2000:]}")
        print(f"  STDERR: {result.stderr[-2000:]}")
        return f16_gguf

    print(f"  ✅ Ternary GGUF created: {ternary_gguf}")

    # Report sizes
    f16_size = f16_gguf.stat().st_size / 1e9
    ternary_size = ternary_gguf.stat().st_size / 1e9
    print(f"\n  F16 GGUF size:     {f16_size:.2f} GB")
    print(f"  Ternary GGUF size: {ternary_size:.2f} GB")
    print(f"  Compression:       {f16_size / ternary_size:.1f}x")

    # Optionally clean up the F16 intermediate
    # (keeping it for now since it's useful for validation)

    return ternary_gguf


def main():
    parser = argparse.ArgumentParser(
        description="Convert Qwen3.6-27B to ternary (1.58-bit)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert from local path
  python convert.py --model-path /path/to/Qwen3.6-27B --output-dir ./output

  # Download and convert
  python convert.py --download --output-dir ./output

  # Skip Hadamard rotation (faster, lower quality)
  python convert.py --model-path /path/to/model --no-hadamard

  # Use different block size
  python convert.py --model-path /path/to/model --block-size 64
        """,
    )
    parser.add_argument("--model-path", type=str, help="Local path to model directory")
    parser.add_argument("--download", action="store_true", help="Download model from HuggingFace")
    parser.add_argument("--cache-dir", type=str, default=None, help="HF cache directory")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--block-size", type=int, default=128, help="Quantization block size")
    parser.add_argument("--no-hadamard", action="store_true", help="Skip Hadamard rotation")
    parser.add_argument("--rotation-only", action="store_true",
                        help="Apply rotation+norm absorption only (no ternary quantization). "
                             "Produces a pure FP16 rotated model for validation.")
    parser.add_argument("--skip-gguf", action="store_true", help="Skip GGUF conversion step")
    parser.add_argument("--gguf-format", type=str, default="tq2_0",
                        choices=["tq2_0", "tq1_0"], help="GGUF quantization format")
    args = parser.parse_args()

    check_dependencies()

    # Resolve model path
    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"❌ Model path does not exist: {model_path}")
            sys.exit(1)
    elif args.download:
        from huggingface_hub import snapshot_download
        print(f"📥 Downloading {MODEL_ID}...")
        model_path = Path(snapshot_download(
            MODEL_ID,
            cache_dir=args.cache_dir,
            ignore_patterns=["*.bin", "*.msgpack"],
        ))
    else:
        print("❌ Must specify --model-path or --download")
        sys.exit(1)

    # Build config
    config = ConversionConfig(
        output_dir=Path(args.output_dir),
        block_size=args.block_size,
        apply_hadamard=not args.no_hadamard,
        rotation_only=args.rotation_only,
        gguf_format=args.gguf_format,
    )

    # Run conversion
    converter = TernaryConverter(config)
    start = time.time()
    modified_dir = converter.convert(model_path)
    elapsed = time.time() - start
    print(f"\n⏱️  Conversion took {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # GGUF conversion
    if not args.skip_gguf:
        gguf_path = run_gguf_conversion(
            modified_dir, config.output_dir, args.gguf_format
        )
        if gguf_path:
            print(f"\n🎉 Done! Ternary GGUF: {gguf_path}")
            print(f"\n   To run with llama.cpp:")
            print(f"   llama-cli -m {gguf_path} -p 'Hello!' -n 256")
    else:
        print(f"\n   Modified model saved to: {modified_dir}")
        print(f"   Run GGUF conversion manually with:")
        print(f"   convert_hf_to_gguf.py {modified_dir} --outfile output.gguf --outtype f16")
        print(f"   llama-quantize output.gguf output-tq2_0.gguf tq2_0")


if __name__ == "__main__":
    main()