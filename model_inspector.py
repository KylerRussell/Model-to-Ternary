#!/usr/bin/env python3
"""
Model Inspector — Structural analysis of Qwen3.6-27B for ternary conversion.

Parses the model's safetensors index to enumerate every tensor, classify it
as a quantization target or FP16-preserved layer, and output a detailed
manifest for the conversion pipeline.

Usage:
    python model_inspector.py [--model-path /path/to/Qwen3.6-27B]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import snapshot_download
    HAS_HF_HUB = True
except ImportError:
    HAS_HF_HUB = False

from config import (
    MODEL_ID,
    LAYER_TYPES,
    NUM_HIDDEN_LAYERS,
    should_quantize,
    KEEP_FP16_PATTERNS,
    BLOCK_SIZE,
)


def load_tensor_index(model_path: Path) -> dict:
    """Load the safetensors weight index to get tensor→shard mapping."""
    index_path = model_path / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)

    # Single-shard model
    single_shard = model_path / "model.safetensors"
    if single_shard.exists():
        from safetensors import safe_open
        with safe_open(str(single_shard), framework="numpy") as f:
            return {
                "metadata": {"total_size": 0},
                "weight_map": {name: "model.safetensors" for name in f.keys()}
            }

    raise FileNotFoundError(
        f"No safetensors index found in {model_path}. "
        f"Expected model.safetensors.index.json or model.safetensors"
    )


def get_tensor_shapes(model_path: Path, weight_map: dict) -> dict:
    """Extract shapes for all tensors without loading full weights."""
    from safetensors import safe_open

    shapes = {}
    seen_shards = set()

    for tensor_name, shard_file in weight_map.items():
        if shard_file not in seen_shards:
            seen_shards.add(shard_file)
            shard_path = model_path / shard_file
            with safe_open(str(shard_path), framework="numpy") as f:
                for name in f.keys():
                    tensor = f.get_slice(name)
                    shapes[name] = list(tensor.get_shape())

    return shapes


def classify_tensor(name: str) -> str:
    """Classify a tensor as 'quantize', 'fp16', or 'skip'."""
    if should_quantize(name):
        return "quantize"
    return "fp16"


def extract_layer_idx(name: str) -> Optional[int]:
    """Extract the layer index from a tensor name like 'model.layers.42.xxx'."""
    parts = name.split(".")
    for i, part in enumerate(parts):
        if part == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                pass
    return None


def inspect_model(model_path: Path, output_path: Optional[Path] = None):
    """Full model inspection — enumerate, classify, and report."""
    print(f"🔍 Inspecting model at: {model_path}")

    # Load index
    index = load_tensor_index(model_path)
    weight_map = index["weight_map"]
    print(f"   Found {len(weight_map)} tensors across "
          f"{len(set(weight_map.values()))} shard files")

    # Get shapes
    print("   Loading tensor shapes...")
    shapes = get_tensor_shapes(model_path, weight_map)

    # Classify all tensors
    manifest = {
        "model_id": MODEL_ID,
        "model_path": str(model_path),
        "num_layers": NUM_HIDDEN_LAYERS,
        "layer_types": LAYER_TYPES,
        "block_size": BLOCK_SIZE,
        "tensors": {},
        "summary": {},
    }

    stats = defaultdict(lambda: {"count": 0, "params": 0, "bytes_fp16": 0})

    for name in sorted(shapes.keys()):
        shape = shapes[name]
        classification = classify_tensor(name)
        layer_idx = extract_layer_idx(name)
        layer_type = LAYER_TYPES[layer_idx] if layer_idx is not None else None

        num_params = 1
        for s in shape:
            num_params *= s

        # Number of quantization blocks (for quantized tensors)
        num_blocks = None
        if classification == "quantize":
            total_weights = num_params
            num_blocks = (total_weights + BLOCK_SIZE - 1) // BLOCK_SIZE

        entry = {
            "shape": shape,
            "params": num_params,
            "bytes_fp16": num_params * 2,
            "classification": classification,
            "layer_idx": layer_idx,
            "layer_type": layer_type,
            "shard_file": weight_map.get(name, "unknown"),
        }
        if num_blocks is not None:
            entry["num_blocks"] = num_blocks
            entry["bytes_ternary_tq2_0"] = num_blocks * 66  # 64 data + 2 scale per 256-block
            # For g128 Q2_0: num_blocks * 34 bytes (32 data + 2 scale)
            entry["bytes_ternary_q2_0_g128"] = (
                (num_params + BLOCK_SIZE - 1) // BLOCK_SIZE
            ) * 34

        manifest["tensors"][name] = entry

        stats[classification]["count"] += 1
        stats[classification]["params"] += num_params
        stats[classification]["bytes_fp16"] += num_params * 2

    # Compute summary
    total_params = sum(s["params"] for s in stats.values())
    quantize_params = stats["quantize"]["params"]
    fp16_params = stats["fp16"]["params"]

    # Estimate ternary size (TQ2_0: 2.0625 bpw for quantized, FP16 for rest)
    ternary_bytes = (quantize_params * 2.0625 / 8) + (fp16_params * 2)

    manifest["summary"] = {
        "total_params": total_params,
        "total_bytes_fp16": total_params * 2,
        "quantize_tensors": stats["quantize"]["count"],
        "quantize_params": quantize_params,
        "quantize_pct": round(quantize_params / total_params * 100, 1),
        "fp16_tensors": stats["fp16"]["count"],
        "fp16_params": fp16_params,
        "fp16_pct": round(fp16_params / total_params * 100, 1),
        "estimated_ternary_bytes": int(ternary_bytes),
        "estimated_ternary_gb": round(ternary_bytes / 1e9, 2),
        "compression_ratio": round((total_params * 2) / ternary_bytes, 1),
    }

    # Print report
    print("\n" + "=" * 70)
    print("MODEL INSPECTION REPORT")
    print("=" * 70)
    print(f"  Model:           {MODEL_ID}")
    print(f"  Total params:    {total_params:,}")
    print(f"  FP16 size:       {total_params * 2 / 1e9:.2f} GB")
    print()
    print(f"  QUANTIZE targets: {stats['quantize']['count']} tensors, "
          f"{quantize_params:,} params ({manifest['summary']['quantize_pct']}%)")
    print(f"  FP16 preserved:   {stats['fp16']['count']} tensors, "
          f"{fp16_params:,} params ({manifest['summary']['fp16_pct']}%)")
    print()
    print(f"  Estimated ternary size: {ternary_bytes / 1e9:.2f} GB")
    print(f"  Compression ratio:      {manifest['summary']['compression_ratio']}x")
    print()

    # Per-layer type breakdown
    print("  Per-layer-type quantized params:")
    for lt in ["full_attention", "linear_attention"]:
        lt_params = sum(
            e["params"] for e in manifest["tensors"].values()
            if e["classification"] == "quantize" and e["layer_type"] == lt
        )
        lt_count = len([
            e for e in manifest["tensors"].values()
            if e["classification"] == "quantize" and e["layer_type"] == lt
        ])
        print(f"    {lt:20s}: {lt_count:3d} tensors, {lt_params:>15,} params")

    # List all tensor classifications
    print("\n  Detailed tensor list:")
    print(f"  {'Tensor Name':<60s} {'Shape':<25s} {'Class':<10s} {'LayerType':<18s}")
    print("  " + "-" * 115)
    for name, entry in sorted(manifest["tensors"].items()):
        shape_str = str(entry["shape"])
        cls = entry["classification"]
        lt = entry["layer_type"] or "-"
        marker = "⚡" if cls == "quantize" else "🔒"
        print(f"  {marker} {name:<58s} {shape_str:<25s} {cls:<10s} {lt:<18s}")

    # Save manifest
    if output_path is None:
        output_path = Path("./output/manifest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest saved to: {output_path}")

    return manifest


def download_model_if_needed(model_id: str, cache_dir: Optional[str] = None) -> Path:
    """Download the model from HuggingFace if not already cached."""
    if not HAS_HF_HUB:
        raise ImportError("huggingface_hub is required. pip install huggingface_hub")

    print(f"📥 Ensuring {model_id} is downloaded...")
    local_path = snapshot_download(
        model_id,
        cache_dir=cache_dir,
        ignore_patterns=["*.bin", "*.msgpack"],  # only safetensors
    )
    return Path(local_path)


def main():
    parser = argparse.ArgumentParser(description="Inspect Qwen3.6-27B model structure")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Local path to the model directory. If not provided, downloads from HF.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="HuggingFace cache directory for model download.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output/manifest.json",
        help="Path to save the inspection manifest JSON.",
    )
    args = parser.parse_args()

    if args.model_path:
        model_path = Path(args.model_path)
        if not model_path.exists():
            print(f"❌ Model path does not exist: {model_path}")
            sys.exit(1)
    else:
        model_path = download_model_if_needed(MODEL_ID, args.cache_dir)

    inspect_model(model_path, Path(args.output))


if __name__ == "__main__":
    main()
