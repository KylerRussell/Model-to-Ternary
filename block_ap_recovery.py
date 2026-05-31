#!/usr/bin/env python3
"""
block_ap_recovery.py — Phase-2B compute-efficient recovery for ternary Qwen3.6-27B.

This is the EfficientQAT "Block-AP" idea adapted to a hybrid Gated-DeltaNet model and
to a 48 GB (2x 3090) budget. Instead of one-shot GPTQ rounding, it RECONSTRUCTS each
quantized projection by short gradient descent on its OUTPUT error, then propagates the
quantized activations forward so the next layer is calibrated on the realistically
degraded input (the error-compensation that makes recovery work).

Why per-LINEAR (not full per-block backprop):
  The fused Gated-DeltaNet kernel is not reliably differentiable, so we never backprop
  THROUGH a block. For each quantized nn.Linear we minimise
        || X @ Wq(W,s)^T  -  X @ W_fp^T ||^2
  where X is the real (propagated) input captured by a forward hook, W_fp is the frozen
  rotated FP weight (the target), and Wq is the straight-through ternary of a trainable
  latent weight W and a trainable per-g128 scale s. Backprop flows only through a matmul.
  Inter-layer error propagation is handled by a separate INFERENCE forward (no grad),
  which the fused kernel runs fine. Intra-block input shift (o_proj/down_proj inputs move
  slightly once upstream projections are quantized) is a second-order effect we accept.

Input model MUST be the rotation-only Phase-1 model (convert.py --rotation-only): every
projection is already Hadamard-rotated and the norms are absorbed/zeroed. This script only
replaces the should_quantize() projections with recovered-ternary weights; everything else
(rotated embed/lm_head, zeroed norms, conv1d, and any projection you keep FP16 via the
config toggles) passes through untouched. It does NO rotation and touches NO norms.

By construction the per-linear objective at init (epoch 0) equals RTN-absmean, so training
can only reduce reconstruction error — recovery is >= RTN. Whether that clears the 64-layer
coherence cliff is the open question; if not, a short global self-distillation pass is the
documented follow-on (phase 2C).

Usage:
    python block_ap_recovery.py \
        --model-path ./output/modified_model \
        --orig-config-path /path/to/Qwen--Qwen3.6-27B/snapshots/<hash> \
        --output-dir ./output_recovery \
        --samples 32 --epochs 20 --lr 1e-3
"""

import argparse
import gc
import json
import os
import sys
import shutil
import inspect
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import save_file as save_safetensors

sys.path.append(str(Path(__file__).parent))
from config import NUM_HIDDEN_LAYERS, BLOCK_SIZE, should_quantize


# ───────────────────────── shared infrastructure (from calibrate_and_quantize.py) ──

def load_calibration_samples(cache_path: Path, n: int) -> list:
    if not cache_path.exists():
        print(f"❌ Calibration cache not found at {cache_path}!")
        print("   Run: python calibration.py --fallback --samples 64 --output "
              f"{cache_path}")
        sys.exit(1)
    with open(cache_path) as f:
        samples = json.load(f)
    print(f"📚 Loaded {len(samples)} calibration samples from {cache_path}")
    return samples[:n]


def load_tensor_from_shards(model_path: Path, weight_map: dict, tensor_name: str,
                            device: str = "cpu") -> torch.Tensor:
    shard_file = weight_map.get(tensor_name)
    if not shard_file:
        alt = tensor_name
        if "model.layers" in tensor_name:
            alt = tensor_name.replace("model.layers", "model.language_model.layers")
        elif "model.embed_tokens" in tensor_name:
            alt = tensor_name.replace("model.embed_tokens", "model.language_model.embed_tokens")
        elif "model.norm" in tensor_name:
            alt = tensor_name.replace("model.norm", "model.language_model.norm")
        shard_file = weight_map.get(alt)
        if not shard_file:
            raise KeyError(f"Tensor {tensor_name} not in index (tried alt: {alt})")
        tensor_name = alt
    with safe_open(str(model_path / shard_file), framework="pt", device=device) as f:
        return f.get_tensor(tensor_name)


def assign_tensor_to_module(module, param_name, tensor, device):
    parts = param_name.split(".")
    parent = module
    for p in parts[:-1]:
        parent = getattr(parent, p)
    attr = parts[-1]
    val = tensor.to(device=device, dtype=torch.bfloat16)
    if hasattr(parent, attr):
        delattr(parent, attr)
    parent.register_parameter(attr, nn.Parameter(val))


def revert_module_param_to_meta(module, param_name, shape):
    parts = param_name.split(".")
    parent = module
    for p in parts[:-1]:
        parent = getattr(parent, p)
    attr = parts[-1]
    if hasattr(parent, attr):
        delattr(parent, attr)
    parent.register_parameter(attr, nn.Parameter(torch.empty(shape, device="meta")))


# ───────────────────────── ternary STE + per-linear reconstruction ─────────────────

def _blocks(W: torch.Tensor, block_size: int):
    """Row-major [n_blocks, block_size] view, matching quantizer.quantize_absmean.
    Requires in_features % block_size == 0 (true for every projection here)."""
    out, inp = W.shape
    assert inp % block_size == 0, f"in_features {inp} not divisible by block_size {block_size}"
    return W.reshape(out * (inp // block_size), block_size), (out, inp)


def ste_ternary(W: torch.Tensor, s: torch.Tensor, block_size: int) -> torch.Tensor:
    """Differentiable ternary dequant: forward = s*round(clamp(W/s,-1,1)) in {-s,0,+s};
    backward is straight-through on round (gradient flows to W within the clamp and to s)."""
    flat, (out, inp) = _blocks(W, block_size)
    ws = flat / s.unsqueeze(1)
    wc = ws.clamp(-1, 1)
    wr = torch.round(wc)
    wq = wc + (wr - wc).detach()           # STE
    return (wq * s.unsqueeze(1)).reshape(out, inp)


def _deploy_ternary(W: torch.Tensor, s: torch.Tensor, block_size: int):
    """Detached final weight (true ternary*scale with the learned scale) + sparsity."""
    flat, (out, inp) = _blocks(W, block_size)
    q = torch.round((flat / s.unsqueeze(1)).clamp(-1, 1))
    deq = (q * s.unsqueeze(1)).reshape(out, inp)
    sparsity = (q == 0).float().mean().item()
    return deq, sparsity


@torch.no_grad()
def _out_mse(X_list, W_fp, W_use, device, inp):
    """Mean output MSE of (X @ W_use^T) vs the FP target (X @ W_fp^T) over batches."""
    tot, cnt = 0.0, 0
    for Xb in X_list:
        X = Xb.to(device, torch.float32).reshape(-1, inp)
        tot += F.mse_loss(X @ W_use.t(), X @ W_fp.t(), reduction="sum").item()
        cnt += X.shape[0] * W_fp.shape[0]
    return tot / max(cnt, 1)


def reconstruct_linear(module, input_batches, block_size, epochs, lr, device):
    """Short SGD reconstruction of one nn.Linear to ternary. Sets module.weight to the
    deployed ternary*scale weight and returns per-linear stats. Keep-best + grad-clip make
    it monotonic: the deployed weight is never worse (in output MSE) than the RTN init."""
    W_fp = module.weight.data.detach().to(device, torch.float32)     # frozen target
    out, inp = W_fp.shape
    flat, _ = _blocks(W_fp, block_size)
    s_init = flat.abs().mean(dim=1).clamp_min(1e-8)                    # abs-mean init == RTN

    # eval on a couple of batches; epoch-0 (RTN) is always a candidate
    eval_batches = input_batches[:min(2, len(input_batches))]
    best_deploy, best_sparsity = _deploy_ternary(W_fp, s_init, block_size)
    init_mse = _out_mse(eval_batches, W_fp, best_deploy, device, inp)
    best_mse = init_mse

    W = W_fp.clone().requires_grad_(True)
    s = s_init.clone().requires_grad_(True)
    opt = torch.optim.Adam([{"params": [W], "lr": lr},
                            {"params": [s], "lr": lr * 0.1}])

    n = len(input_batches)
    for _ in range(epochs):
        for bi in torch.randperm(n).tolist():
            X = input_batches[bi].to(device, torch.float32).reshape(-1, inp)
            with torch.no_grad():
                target = X @ W_fp.t()
            Wq = ste_ternary(W, s, block_size)
            loss = F.mse_loss(X @ Wq.t(), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([W, s], 1.0)               # stability
            opt.step()
        # keep-best on the DEPLOYED (rounded) weight — guards against divergence
        with torch.no_grad():
            dep, sp = _deploy_ternary(W, s, block_size)
            m = _out_mse(eval_batches, W_fp, dep, device, inp)
            if m < best_mse:
                best_mse, best_deploy, best_sparsity = m, dep.clone(), sp

    deq, sparsity, final_mse = best_deploy, best_sparsity, best_mse
    cos = F.cosine_similarity(deq.reshape(1, -1), W_fp.reshape(1, -1)).item()

    module.weight.data.copy_(deq.to(module.weight.dtype))
    del W_fp, W, s, opt, deq, best_deploy
    torch.cuda.empty_cache()
    return {"cosine": cos, "sparsity": sparsity,
            "init_mse": init_mse, "final_mse": final_mse}


# ───────────────────────────────────────── main ────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Block-AP per-linear ternary recovery")
    ap.add_argument("--model-path", required=True,
                    help="ROTATION-ONLY Phase-1 model (convert.py --rotation-only output).")
    ap.add_argument("--orig-config-path", default=None,
                    help="Original HF snapshot for config.json + trust_remote_code .py.")
    ap.add_argument("--output-dir", default="./output_recovery")
    ap.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    ap.add_argument("--samples", type=int, default=32,
                    help="Calibration samples (trades CPU activation cache + quality).")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=20, help="SGD passes per linear.")
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    model_path = Path(args.model_path)
    orig_config_path = Path(args.orig_config_path) if args.orig_config_path else model_path
    if args.orig_config_path is None:
        print("⚠️  --orig-config-path not set; config/modeling loaded from --model-path.")
    output_dir = Path(args.output_dir)
    modified_model_dir = output_dir / "modified_model"
    modified_model_dir.mkdir(parents=True, exist_ok=True)

    calib_cache = output_dir / "calibration_data.json"
    if not calib_cache.exists():
        # fall back to a sibling calibration file produced earlier
        alt = Path("./output_calib/calibration_data.json")
        if alt.exists():
            calib_cache = alt
    samples = load_calibration_samples(calib_cache, args.samples)

    from transformers import AutoConfig, AutoModelForCausalLM
    from accelerate import init_empty_weights

    print("⚙️ Loading config...")
    config = AutoConfig.from_pretrained(str(orig_config_path), trust_remote_code=True)
    with open(model_path / "model.safetensors.index.json") as f:
        weight_index = json.load(f)
    weight_map = weight_index["weight_map"]

    print("📦 Instantiating model shell on META device...")
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"🖥️ device: {device}")

    input_ids = [torch.tensor(s, dtype=torch.long) for s in samples]
    batches = [torch.stack(input_ids[i:i + args.batch_size])
               for i in range(0, len(input_ids), args.batch_size)]
    print(f"🧪 {len(batches)} batches of size {args.batch_size}")

    # ── capture layer-0 inputs + forward kwargs (identical to calibrate path) ───────
    print("\n🚀 Step 1: capture initial activations + forward kwargs...")
    embed_weight = load_tensor_from_shards(model_path, weight_map, "model.embed_tokens.weight")
    vocab, dim = embed_weight.shape
    embed_tokens = nn.Embedding(vocab, dim).to(device=device, dtype=torch.bfloat16)
    embed_tokens.weight.data.copy_(embed_weight.to(device=device, dtype=torch.bfloat16))
    del embed_weight

    layer0 = model.model.layers[0]
    for pname, _ in list(layer0.named_parameters()):
        w = load_tensor_from_shards(model_path, weight_map, f"model.layers.0.{pname}")
        assign_tensor_to_module(layer0, pname, w, device)

    captured = []
    orig_fwd = layer0.forward

    def l0_wrapper(hidden_states, *a, **k):
        a_cpu = [x.cpu() if isinstance(x, torch.Tensor) else x for x in a]
        k_cpu = {kk: (v.cpu() if isinstance(v, torch.Tensor) else v) for kk, v in k.items()}
        if isinstance(k_cpu.get("position_embeddings"), tuple):
            k_cpu["position_embeddings"] = tuple(t.cpu() for t in k_cpu["position_embeddings"])
        k_cpu["past_key_value"] = None
        k_cpu["past_key_values"] = None
        captured.append({"args": a_cpu, "kwargs": k_cpu})
        return orig_fwd(hidden_states, *a, **k)

    layer0.forward = l0_wrapper
    model.model.embed_tokens = embed_tokens
    with torch.no_grad():
        for b in tqdm(batches, desc="   embed pass"):
            try:
                model(b.to(device))
            except Exception:
                pass
    layer0.forward = orig_fwd
    for pname, param in list(layer0.named_parameters()):
        revert_module_param_to_meta(layer0, pname, param.shape)

    layer_inputs = []
    with torch.no_grad():
        for b in batches:
            layer_inputs.append(embed_tokens(b.to(device)).cpu())
    del embed_tokens
    torch.cuda.empty_cache()
    layer_kwargs = captured
    print(f"   captured {len(layer_inputs)} activation batches")

    # ── staging + crash-resume ──────────────────────────────────────────────────────
    stats = {"layers": {}}
    staging_dir = output_dir / "_recovery_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_index = {}

    # The propagated activation stream is deterministic given the (quantized) weights of
    # earlier layers, so checkpointing it lets us resume mid-model after a crash.
    present = [l for l in range(NUM_HIDDEN_LAYERS)
               if (staging_dir / f"inputs_after_{l}.pt").exists()]
    start_layer = 0
    if present:
        A = max(present)
        if all((staging_dir / f"layer_{l}.safetensors").exists() for l in range(A + 1)):
            start_layer = A + 1
            print(f"⏩ Resume: layers 0..{A} done; loading inputs_after_{A}.pt")
            layer_inputs = torch.load(str(staging_dir / f"inputs_after_{A}.pt"))
            for l in range(A + 1):
                p = staging_dir / f"layer_{l}.safetensors"
                with safe_open(str(p), framework="pt") as f:
                    for k in f.keys():
                        staged_index[k] = p
        else:
            print("⚠️ Activation checkpoint present but staged weights missing — restart at 0.")
    if start_layer >= NUM_HIDDEN_LAYERS:
        print("✅ All layers already recovered; jumping to save.")

    def filter_kwargs(layer, kwargs):
        params = inspect.signature(layer.forward).parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return kwargs
        return {k: v for k, v in kwargs.items() if k in params}

    def run_layer_forward(layer, desc):
        """Inference forward over all batches → next-layer inputs (CPU). No grad."""
        outs = []
        with torch.no_grad():
            for idx, h in enumerate(tqdm(layer_inputs, desc=desc, leave=False)):
                a = [x.to(device) if isinstance(x, torch.Tensor) else x
                     for x in layer_kwargs[idx]["args"]]
                kw = {}
                for k, v in layer_kwargs[idx]["kwargs"].items():
                    if isinstance(v, torch.Tensor):
                        kw[k] = v.to(device)
                    elif isinstance(v, tuple):
                        kw[k] = tuple(t.to(device) for t in v)
                    else:
                        kw[k] = v
                kw["use_cache"] = False
                out = layer(h.to(device), *a, **filter_kwargs(layer, kw))
                outs.append((out[0] if isinstance(out, tuple) else out).cpu())
        return outs

    # ── per-layer recovery loop ───────────────────────────────────────────────────
    print("\n🚀 Step 2: per-linear ternary recovery (rotation-only model in, recovered out)")
    for l in range(start_layer, NUM_HIDDEN_LAYERS):
        print(f"\n⚡ Recovering layer {l + 1}/{NUM_HIDDEN_LAYERS}...")
        layer = model.model.layers[l]
        for pname, _ in list(layer.named_parameters()):
            w = load_tensor_from_shards(model_path, weight_map, f"model.layers.{l}.{pname}")
            assign_tensor_to_module(layer, pname, w, device)

        # which Linears in this layer are quantization targets
        targets = {}
        for name, module in layer.named_modules():
            full = f"model.layers.{l}.{name}.weight"
            if isinstance(module, nn.Linear) and should_quantize(full):
                targets[name] = module

        if not targets:
            print("   (no quantization targets in this layer — passing through FP16)")
        else:
            # Pass A: capture each target Linear's real input (FP forward), cache on CPU
            cached = {name: [] for name in targets}
            hooks = []

            def mk_hook(name):
                def hook(mod, inp, out):
                    cached[name].append(inp[0].detach().to("cpu", torch.bfloat16))
                return hook

            for name, module in targets.items():
                hooks.append(module.register_forward_hook(mk_hook(name)))
            _ = run_layer_forward(layer, f"   L{l} capture (FP)")
            for h in hooks:
                h.remove()

            # Reconstruct each target Linear, then free its cached inputs
            stats["layers"][str(l)] = {}
            for name, module in targets.items():
                r = reconstruct_linear(module, cached[name], args.block_size,
                                       args.epochs, args.lr, device)
                stats["layers"][str(l)][name] = r
                print(f"   {name:28s} cos={r['cosine']:.4f}  spars={r['sparsity']:.3f}  "
                      f"mse {r['init_mse']:.3e} → {r['final_mse']:.3e}")
                cached[name] = None
            del cached

        # Pass B: propagate QUANTIZED activations to the next layer (error compensation)
        next_layer_inputs = run_layer_forward(layer, f"   L{l} propagate (quantized)")

        # stage weights to disk, revert to meta
        layer_sd = {}
        for pname, param in list(layer.named_parameters()):
            full = f"model.layers.{l}.{pname}"
            layer_sd[full] = param.data.cpu().contiguous()
            revert_module_param_to_meta(layer, pname, param.shape)
        lf = staging_dir / f"layer_{l}.safetensors"
        tmp = staging_dir / f"layer_{l}.safetensors.tmp"
        save_safetensors(layer_sd, str(tmp))
        os.replace(str(tmp), str(lf))
        for k in layer_sd:
            staged_index[k] = lf
        del layer_sd, layer
        torch.cuda.empty_cache()
        gc.collect()

        layer_inputs = next_layer_inputs
        af = staging_dir / f"inputs_after_{l}.pt"
        tmpa = staging_dir / f"inputs_after_{l}.pt.tmp"
        torch.save(layer_inputs, str(tmpa))
        os.replace(str(tmpa), str(af))
        prev = staging_dir / f"inputs_after_{l - 1}.pt"
        if prev.exists():
            prev.unlink()

    # ── save (assemble shards from staging + passthrough originals) ─────────────────
    print("\n💾 Step 3: saving recovered model...")
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "generation_config.json", "merges.txt",
                  "vocab.json", "preprocessor_config.json", "chat_template.json"]:
        src = model_path / fname
        if src.exists() and src.resolve() != (modified_model_dir / fname).resolve():
            shutil.copy2(str(src), str(modified_model_dir / fname))
    for f in model_path.glob("*.model"):
        if f.resolve() != (modified_model_dir / f.name).resolve():
            shutil.copy2(str(f), str(modified_model_dir / f.name))

    shards = {}
    for name, shard_file in weight_map.items():
        shards.setdefault(shard_file, []).append(name)

    for shard_file, names in sorted(shards.items()):
        print(f"   💾 {modified_model_dir / shard_file}")
        shard_dict = {}
        for name in names:
            key = name
            if "model.language_model.layers" in name:
                key = name.replace("model.language_model.layers", "model.layers")
            if key in staged_index:
                with safe_open(str(staged_index[key]), framework="pt") as f:
                    shard_dict[name] = f.get_tensor(key)
            else:
                shard_dict[name] = load_tensor_from_shards(model_path, weight_map, name, "cpu")
        save_safetensors(shard_dict, str(modified_model_dir / shard_file))

    with open(modified_model_dir / "model.safetensors.index.json", "w") as f:
        json.dump(weight_index, f, indent=2)

    for p in staging_dir.glob("layer_*.safetensors"):
        p.unlink()
    for p in staging_dir.glob("inputs_after_*.pt"):
        p.unlink()
    try:
        staging_dir.rmdir()
    except OSError:
        pass

    with open(output_dir / "recovery_report.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n📊 Recovery complete. Report: {output_dir / 'recovery_report.json'}")


if __name__ == "__main__":
    main()