#!/usr/bin/env python3
"""
Phase 2 Calibration-Aware Ternary Quantization (GPTQ-style) for Qwen3.6-27B.
Safe zero-RAM edition: accumulates Hessians on-the-fly on GPU, using 0 bytes of CPU memory.
"""

import argparse
import gc
import json
import math
import os
import sys
import shutil
import inspect
import time
from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import save_file as save_safetensors

# Ensure local directories are in path
sys.path.append(str(Path(__file__).parent))

from config import (
    MODEL_ID, NUM_HIDDEN_LAYERS, BLOCK_SIZE,
    should_quantize, ConversionConfig
)
from quantizer import quantize_gptq_ternary, dequantize, compute_quantization_error


def load_calibration_samples(cache_path: Path) -> list:
    """Load tokenized calibration data."""
    if not cache_path.exists():
        print(f"❌ Calibration cache not found at {cache_path}!")
        print("   Please run: python build_diverse_calib.py --orig-model <snap> --ctm-data <dir> --out <path>")
        sys.exit(1)
    with open(cache_path) as f:
        samples = json.load(f)
    print(f"📚 Loaded {len(samples)} calibration samples from {cache_path}")
    return samples


def load_tensor_from_shards(model_path: Path, weight_map: dict, tensor_name: str, device: str = "cpu") -> torch.Tensor:
    """Slice load a single tensor from safetensors shards without loading other weights."""
    shard_file = weight_map.get(tensor_name)
    if not shard_file:
        # Fallback to language_model naming variations
        alt_name = tensor_name
        if "model.layers" in tensor_name:
            alt_name = tensor_name.replace("model.layers", "model.language_model.layers")
        elif "model.embed_tokens" in tensor_name:
            alt_name = tensor_name.replace("model.embed_tokens", "model.language_model.embed_tokens")
        elif "model.norm" in tensor_name:
            alt_name = tensor_name.replace("model.norm", "model.language_model.norm")
        
        shard_file = weight_map.get(alt_name)
        if not shard_file:
            raise KeyError(f"Tensor {tensor_name} not found in weight map index. Tried alt: {alt_name}")
        tensor_name = alt_name

    shard_path = model_path / shard_file
    with safe_open(str(shard_path), framework="pt", device=device) as f:
        return f.get_tensor(tensor_name)

def assign_tensor_to_module(module: nn.Module, param_name: str, tensor: torch.Tensor, device: str):
    """Safely delete existing meta parameter and register new active parameter on device."""
    parts = param_name.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    attr = parts[-1]
    
    val = tensor.to(device=device, dtype=torch.bfloat16)
    if hasattr(parent, attr):
        delattr(parent, attr)
    parent.register_parameter(attr, nn.Parameter(val))


def revert_module_param_to_meta(module: nn.Module, param_name: str, shape: torch.Size):
    """Delete GPU parameter and restore fresh empty meta parameter to guarantee VRAM cleanup."""
    parts = param_name.split(".")
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    attr = parts[-1]
    
    if hasattr(parent, attr):
        delattr(parent, attr)
    parent.register_parameter(attr, nn.Parameter(torch.empty(shape, device="meta")))
def main():
    parser = argparse.ArgumentParser(description="Phase 2 Safe Calibration-Aware Ternary Quantization")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the ROTATED model from Phase 1 (output/modified_model). "
                             "Weights, index, and save all come from here.")
    parser.add_argument("--orig-config-path", type=str, default=None,
                        help="Path to the ORIGINAL HF snapshot, used only for config.json + "
                             "the trust_remote_code modeling .py (which convert.py does not copy "
                             "into modified_model). Defaults to --model-path.")
    parser.add_argument("--output-dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--block-size", type=int, default=128, help="Quantization block size")
    parser.add_argument("--samples", type=int, default=64, help="Number of calibration samples to use")
    parser.add_argument("--batch-size", type=int, default=4, help="Inference batch size")
    parser.add_argument("--damp", type=float, default=0.01, help="Hessian diagonal dampening factor")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    # Config + modeling code come from the original snapshot; weights come from model_path
    # (the rotated Phase-1 output). They differ because convert.py doesn't copy the
    # trust_remote_code .py into modified_model.
    orig_config_path = Path(args.orig_config_path) if args.orig_config_path else model_path
    if args.orig_config_path is None:
        print("⚠️  --orig-config-path not set; loading config/modeling from --model-path. "
              "If that dir lacks the modeling .py, pass the original snapshot here.")
    output_dir = Path(args.output_dir)
    modified_model_dir = output_dir / "modified_model"
    modified_model_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load calibration data
    calib_cache = output_dir / "calibration_data.json"
    samples = load_calibration_samples(calib_cache)[:args.samples]
    
    # 2. Load model config
    from transformers import AutoConfig, AutoModelForCausalLM
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device

    print("⚙️ Loading model config...")
    config = AutoConfig.from_pretrained(str(orig_config_path), trust_remote_code=True)

    # Load original weight index to locate parameters
    index_path = model_path / "model.safetensors.index.json"
    with open(index_path) as f:
        weight_index = json.load(f)
    weight_map = weight_index["weight_map"]

    # 3. Instantiate model skeleton on the meta device (0 bytes of RAM)
    print("📦 Instantiating model shell on PyTorch META device (Zero RAM)...")
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ Using active device: {device}")

    # Prepare calibration batches
    input_ids_list = [torch.tensor(s, dtype=torch.long) for s in samples]
    batches = []
    for i in range(0, len(input_ids_list), args.batch_size):
        batch = input_ids_list[i:i + args.batch_size]
        batches.append(torch.stack(batch))

    num_batches = len(batches)
    print(f"🧪 Created {num_batches} batches of size {args.batch_size}")

    # 4. Slice load Embeddings & Rotary modules onto GPU to capture hidden states
    print("\n🚀 Step 1: Capture initial activations from Layer 0 inputs...")
    layer_inputs = []
    layer_kwargs = []

    # Get embed tokens weights directly from shard
    print("   Loading embed_tokens...")
    embed_weight = load_tensor_from_shards(model_path, weight_map, "model.embed_tokens.weight")
    vocab_size, embedding_dim = embed_weight.shape
    embed_tokens = nn.Embedding(vocab_size, embedding_dim).to(device=device, dtype=torch.bfloat16)
    embed_tokens.weight.data.copy_(embed_weight.to(device=device, dtype=torch.bfloat16))
    del embed_weight

    # We temporarily materialize layer 0 on CPU/GPU to capture forward kwargs
    print("   Materializing temporary Layer 0 to capture args/kwargs...")
    layer0 = model.model.layers[0]
    
    # Load weights for layer 0 only using helper function
    for param_name, param in list(layer0.named_parameters()):
        full_tensor_name = f"model.layers.0.{param_name}"
        w = load_tensor_from_shards(model_path, weight_map, full_tensor_name)
        assign_tensor_to_module(layer0, param_name, w, device)

    captured_args_kwargs = []
    original_l0_forward = layer0.forward

    def l0_wrapper(hidden_states, *l_args, **l_kwargs):
        args_cpu = [a.cpu() if isinstance(a, torch.Tensor) else a for a in l_args]
        kwargs_cpu = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in l_kwargs.items()}
        if "position_embeddings" in kwargs_cpu and isinstance(kwargs_cpu["position_embeddings"], tuple):
            kwargs_cpu["position_embeddings"] = tuple(t.cpu() for t in kwargs_cpu["position_embeddings"])
        
        # Explicitly strip custom class caches (past_key_values) to prevent VRAM accumulation
        if "past_key_value" in kwargs_cpu:
            kwargs_cpu["past_key_value"] = None
        if "past_key_values" in kwargs_cpu:
            kwargs_cpu["past_key_values"] = None
            
        captured_args_kwargs.append({
            "args": args_cpu,
            "kwargs": kwargs_cpu
        })
        return original_l0_forward(hidden_states, *l_args, **l_kwargs)

    layer0.forward = l0_wrapper

    # Temporarily bind GPU embed_tokens to model
    model.model.embed_tokens = embed_tokens

    with torch.no_grad():
        for batch in tqdm(batches, desc="Running embed_tokens pass"):
            batch_gpu = batch.to(device)
            try:
                model(batch_gpu)
            except Exception:
                pass

    # Clean up captured Layer 0 resources
    layer0.forward = original_l0_forward
    # Move Layer 0 back to meta device using helper function to free memory completely
    for param_name, param in list(layer0.named_parameters()):
        revert_module_param_to_meta(layer0, param_name, param.shape)

    # Extract initial hidden states on CPU to save memory
    print("   Extracting embedding hidden states...")
    with torch.no_grad():
        for batch in batches:
            batch_gpu = batch.to(device)
            h = embed_tokens(batch_gpu)
            layer_inputs.append(h.cpu())

    del embed_tokens
    torch.cuda.empty_cache()

    layer_kwargs = captured_args_kwargs
    print(f"   Collected {len(layer_inputs)} input activation batches.")

    # 5. Layer-by-layer calibration loop
    print("\n🚀 Step 2: Running layer-by-layer GPTQ ternary calibration...")
    stats = {"errors": {}, "quantized_tensors": 0}

    # Stage reconstructed weights to DISK, one file per layer, instead of holding the
    # whole dequantized model in a CPU dict (the previous code's `reconstructed_weights`
    # grew ~0.75 GB/layer with no trim → ~38 GB by layer 51 → system-RAM OOM / swap
    # thrash). This bounds resident CPU RAM to roughly a single layer.
    staging_dir = output_dir / "_calib_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_index = {}  # de-prefixed tensor name -> staging .safetensors path

    # ── Crash-resume ──────────────────────────────────────────────────────────
    # The per-layer forward pass below runs on the ORIGINAL (rotated) weights; the
    # quantization happens AFTER it and the layer is then reverted to meta. So the
    # activation stream feeding each layer is a pure function of the original weights
    # and is safe to checkpoint and resume. We checkpoint the activations after each
    # layer; on restart we resume from the highest complete checkpoint.
    present_acts = [l for l in range(NUM_HIDDEN_LAYERS)
                    if (staging_dir / f"inputs_after_{l}.pt").exists()]
    start_layer = 0
    if present_acts:
        A = max(present_acts)
        weights_ok = all((staging_dir / f"layer_{l}.safetensors").exists()
                         for l in range(A + 1))
        if weights_ok:
            start_layer = A + 1
            print(f"⏩ Resume: layers 0..{A} already staged; "
                  f"loading activation checkpoint inputs_after_{A}.pt")
            layer_inputs = torch.load(str(staging_dir / f"inputs_after_{A}.pt"))
            for l in range(A + 1):
                p = staging_dir / f"layer_{l}.safetensors"
                with safe_open(str(p), framework="pt") as f:
                    for k in f.keys():
                        staged_index[k] = p
        else:
            print("⚠️ Activation checkpoint present but some staged layer weights are "
                  "missing — restarting the layer loop from 0 to rebuild activations.")
    if start_layer >= NUM_HIDDEN_LAYERS:
        print("✅ All layers already staged; proceeding directly to the save step.")

    for l in range(start_layer, NUM_HIDDEN_LAYERS):
        print(f"\n⚡ Quantizing layer {l + 1}/{NUM_HIDDEN_LAYERS}...")
        layer = model.model.layers[l]
        
        # Load weights for layer l from shards onto CPU/GPU safely using helper function
        for param_name, param in list(layer.named_parameters()):
            full_tensor_name = f"model.layers.{l}.{param_name}"
            w = load_tensor_from_shards(model_path, weight_map, full_tensor_name)
            assign_tensor_to_module(layer, param_name, w, device)

        # Setup containers for ON-THE-FLY Hessian accumulation on GPU (0 bytes CPU RAM storage!)
        H_matrices = {}
        H_counts = {}
        hooks = []

        def get_hook(name, in_features):
            def hook(module, input_tensor, output_tensor):
                # input_tensor[0] has shape [batch_size, seq_len, in_features]
                # Flatten batch and sequence dims, and convert to float32 on GPU
                inp = input_tensor[0].detach().to(device=device, dtype=torch.float32).reshape(-1, in_features)
                
                # Incrementally accumulate Hessian H = X^T X directly on GPU
                H_chunk = inp.T @ inp
                
                if name not in H_matrices:
                    H_matrices[name] = torch.zeros((in_features, in_features), dtype=torch.float32, device=device)
                    H_counts[name] = 0
                
                H_matrices[name] += H_chunk
                H_counts[name] += inp.shape[0]
            return hook

        target_modules = {}
        for name, module in layer.named_modules():
            full_name = f"model.layers.{l}.{name}.weight"
            if isinstance(module, nn.Linear) and should_quantize(full_name):
                target_modules[name] = module
                # Register hook with explicit input dimension to ensure exact shape compatibility
                hooks.append(module.register_forward_hook(get_hook(name, module.in_features)))

        # Run forward pass of current layer for all calibration batches
        next_layer_inputs = []
        with torch.no_grad():
            for idx, h in enumerate(tqdm(layer_inputs, desc=f"   Layer {l} forward pass")):
                h_gpu = h.to(device)
                
                l_args = [a.to(device) if isinstance(a, torch.Tensor) else a for a in layer_kwargs[idx]["args"]]
                kwargs = {}
                for k, v in layer_kwargs[idx]["kwargs"].items():
                    if isinstance(v, torch.Tensor):
                        kwargs[k] = v.to(device)
                    elif isinstance(v, tuple):
                        kwargs[k] = tuple(t.to(device) for t in v)
                    else:
                        kwargs[k] = v

                kwargs["use_cache"] = False
                
                # Fetch valid parameter names for the current layer
                valid_params = inspect.signature(layer.forward).parameters
                has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in valid_params.values())
                
                if has_var_keyword:
                    filtered_kwargs = kwargs
                else:
                    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
                
                out = layer(h_gpu, *l_args, **filtered_kwargs)
                if isinstance(out, tuple):
                    next_layer_inputs.append(out[0].cpu())
                else:
                    next_layer_inputs.append(out.cpu())

        # Remove hooks immediately after the forward pass is complete
        for h in hooks:
            h.remove()

        # Perform GPTQ quantization per target module
        for name, module in target_modules.items():
            full_name = f"model.layers.{l}.{name}.weight"
            
            # WORKLOAD BALANCING: Offload MLP projections (80% of computation) to cuda:1 (GPU 1)
            # and run Attention projections on cuda:0 (GPU 0)
            target_device = device
            if "mlp" in name and torch.cuda.device_count() > 1:
                target_device = "cuda:1"

            print(f"   Quantizing: {full_name} ({module.weight.shape}) on device {target_device}")

            # Fetch the accumulated Hessian and normalize it
            H = H_matrices[name].to(target_device)
            count = H_counts[name]
            H /= count

            W = module.weight.data.to(target_device)
            
            # Quantize using GPTQ ternary algorithm on target device
            qt = quantize_gptq_ternary(W, H, block_size=args.block_size, damp_pct=args.damp)
            reconstructed = dequantize(qt).to(W.dtype)

            # Compute error metrics on target device
            err = compute_quantization_error(W, qt)
            stats["errors"][full_name] = {
                "cosine_sim": err["cosine_similarity"],
                "sqnr_db": err["sqnr_db"],
                "sparsity": err["sparsity"],
                "distribution": err["ternary_distribution"],
            }
            stats["quantized_tensors"] += 1

            # Update weights in-place back to module device (CPU/GPU 0)
            module.weight.data.copy_(reconstructed.to(module.weight.device))

            # Free GPU memory immediately
            del H, W, qt, reconstructed
            torch.cuda.empty_cache()
            if target_device == "cuda:1":
                torch.cuda.empty_cache()

        # Clear layer Hessian references
        H_matrices.clear()
        H_counts.clear()

        # Stage this layer's weights to DISK (atomic write), then revert params to meta
        # to free VRAM. This is what keeps CPU RAM flat across the 64-layer loop.
        layer_sd = {}
        for param_name, param in list(layer.named_parameters()):
            full_tensor_name = f"model.layers.{l}.{param_name}"
            layer_sd[full_tensor_name] = param.data.cpu().contiguous()
            revert_module_param_to_meta(layer, param_name, param.shape)
        layer_file = staging_dir / f"layer_{l}.safetensors"
        tmp = staging_dir / f"layer_{l}.safetensors.tmp"
        save_safetensors(layer_sd, str(tmp))
        os.replace(str(tmp), str(layer_file))
        for k in layer_sd:
            staged_index[k] = layer_file
        del layer_sd

        # Free memory
        del layer
        torch.cuda.empty_cache()
        gc.collect()

        # Advance activations to the next layer
        layer_inputs = next_layer_inputs
        torch.cuda.empty_cache()
        gc.collect()

        # Checkpoint activations for crash-resume (atomic), keeping only the latest.
        act_file = staging_dir / f"inputs_after_{l}.pt"
        tmp_act = staging_dir / f"inputs_after_{l}.pt.tmp"
        torch.save(layer_inputs, str(tmp_act))
        os.replace(str(tmp_act), str(act_file))
        prev_act = staging_dir / f"inputs_after_{l - 1}.pt"
        if prev_act.exists():
            prev_act.unlink()

    # 6. Save modified weights back to safetensors shards
    print("\n💾 Step 3: Saving modified calibrated weights...")
    
    # Copy non-weight metadata files
    metadata_files = [
        "config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "generation_config.json", "merges.txt",
        "vocab.json", "preprocessor_config.json", "chat_template.json",
    ]
    for fname in metadata_files:
        src_file = model_path / fname
        if src_file.exists() and src_file.resolve() != (modified_model_dir / fname).resolve():
            shutil.copy2(str(src_file), str(modified_model_dir / fname))

    for f in model_path.glob("*.model"):
        if f.resolve() != (modified_model_dir / f.name).resolve():
            shutil.copy2(str(f), str(modified_model_dir / f.name))

    # Group original tensors by shard
    shards = {}
    for name, shard_file in weight_map.items():
        shards.setdefault(shard_file, []).append(name)

    # Save state_dict back to safetensors shards one by one
    # (save_safetensors is imported at module level — do NOT re-import here, or it
    #  becomes a function-local name and shadows the staging write above.)
    for shard_file, tensor_names in sorted(shards.items()):
        shard_path = modified_model_dir / shard_file
        print(f"   💾 Saving shard: {shard_path}")
        shard_dict = {}
        for name in tensor_names:
            modified_key = name
            if "model.language_model.layers" in name:
                modified_key = name.replace("model.language_model.layers", "model.layers")

            if modified_key in staged_index:
                with safe_open(str(staged_index[modified_key]), framework="pt") as f:
                    shard_dict[name] = f.get_tensor(modified_key)
            else:
                shard_dict[name] = load_tensor_from_shards(model_path, weight_map, name, device="cpu")

        save_safetensors(shard_dict, str(shard_path))

    # Write weight map index
    with open(modified_model_dir / "model.safetensors.index.json", "w") as f:
        json.dump(weight_index, f, indent=2)

    # Final shards are written — remove the per-layer staging scratch.
    for p in staging_dir.glob("layer_*.safetensors"):
        p.unlink()
    for p in staging_dir.glob("inputs_after_*.pt"):
        p.unlink()
    try:
        staging_dir.rmdir()
    except OSError:
        pass

    # Save report
    report_path = output_dir / "calibration_report.json"
    with open(report_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n📊 Calibration complete! Report saved to: {report_path}")


if __name__ == "__main__":
    main()