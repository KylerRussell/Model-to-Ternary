#!/usr/bin/env python3
"""
deltanet_grad_probe.py — Does backprop flow through the Gated-DeltaNet recurrence?

Any global pass (end-to-end distillation / EfficientQAT E2E-QP) must backprop the
end-to-end loss through ALL 64 layers, including the 48 fused Gated-DeltaNet recurrences.
If that fused kernel has no differentiable backward, global training is impossible no
matter how the weights are stored. This probe materialises ONE DeltaNet layer, runs a
forward + backward on a random hidden state with REAL captured forward kwargs, and reports
whether a finite gradient reaches one of its in_proj weights.

  PASS    -> global E2E training is viable; the E2E-QP loop can be built.
  FAIL    -> the recurrence is non-differentiable as configured. Re-run with
             --attn-impl eager (and/or check the model's linear-attention "mode"/kernel
             flags). If still FAIL, the global path needs an eager DeltaNet reimpl, else
             fall back to mixed precision or a smaller base model.

Run on the GPU box (needs the trust_remote_code modeling files from the snapshot):
    python deltanet_grad_probe.py \
        --model-path ./output/modified_model \
        --orig-config-path /home/kyler/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/<hash>
"""
import argparse
import inspect
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open

sys.path.append(str(Path(__file__).parent))
from config import get_layer_type, NUM_HIDDEN_LAYERS


def load_tensor(model_path, wmap, name, device="cpu"):
    shard = wmap.get(name)
    if shard is None:
        for a, b in (("model.layers", "model.language_model.layers"),
                     ("model.embed_tokens", "model.language_model.embed_tokens")):
            if a in name and name.replace(a, b) in wmap:
                name = name.replace(a, b)
                shard = wmap[name]
                break
    if shard is None:
        raise KeyError(name)
    with safe_open(str(model_path / shard), framework="pt", device=device) as f:
        return f.get_tensor(name)


def assign(module, pname, tensor, device, dtype):
    parts = pname.split(".")
    parent = module
    for p in parts[:-1]:
        parent = getattr(parent, p)
    if hasattr(parent, parts[-1]):
        delattr(parent, parts[-1])
    parent.register_parameter(parts[-1], nn.Parameter(tensor.to(device=device, dtype=dtype)))


def filt(layer, kw):
    params = inspect.signature(layer.forward).parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kw
    return {k: v for k, v in kw.items() if k in params}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--orig-config-path", default=None)
    ap.add_argument("--seq", type=int, default=16)
    ap.add_argument("--dtype", default="bfloat16", help="bfloat16 (kernel-native) or float32")
    ap.add_argument("--attn-impl", default=None,
                    help="Force config attn implementation, e.g. 'eager' (default: as-is).")
    args = ap.parse_args()

    model_path = Path(args.model_path)
    cfg_path = Path(args.orig_config_path) if args.orig_config_path else model_path
    dtype = getattr(torch, args.dtype)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device={device} dtype={dtype} attn_impl={args.attn_impl or 'as-is'}")

    from transformers import AutoConfig, AutoModelForCausalLM
    from accelerate import init_empty_weights

    config = AutoConfig.from_pretrained(str(cfg_path), trust_remote_code=True)
    if args.attn_impl:
        for attr in ("_attn_implementation", "attn_implementation"):
            try:
                setattr(config, attr, args.attn_impl)
            except Exception:
                pass
    with open(model_path / "model.safetensors.index.json") as f:
        wmap = json.load(f)["weight_map"]
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    k = next((i for i in range(NUM_HIDDEN_LAYERS)
              if get_layer_type(i) == "linear_attention"), None)
    if k is None:
        print("No linear_attention layer found; nothing to probe.")
        return
    print(f"Probing DeltaNet (linear_attention) layer index {k}")

    # materialise embed + layer 0 to capture real forward kwargs
    emb_w = load_tensor(model_path, wmap, "model.embed_tokens.weight")
    vocab, hidden = emb_w.shape
    emb = nn.Embedding(vocab, hidden).to(device, dtype)
    emb.weight.data.copy_(emb_w.to(device, dtype))
    del emb_w
    l0 = model.model.layers[0]
    for pn, _ in list(l0.named_parameters()):
        assign(l0, pn, load_tensor(model_path, wmap, f"model.layers.0.{pn}"), device, dtype)

    cap = {}
    ofwd = l0.forward

    def wrap(hs, *a, **kw):
        kw2 = dict(kw)
        kw2["past_key_value"] = None
        kw2["past_key_values"] = None
        cap["a"], cap["kw"] = a, kw2
        return ofwd(hs, *a, **kw)

    l0.forward = wrap
    model.model.embed_tokens = emb
    with torch.no_grad():
        try:
            model(torch.zeros(1, args.seq, dtype=torch.long, device=device))
        except Exception:
            pass
    l0.forward = ofwd
    if "kw" not in cap:
        print("Could not capture forward kwargs; aborting.")
        return

    # materialise the DeltaNet layer under test
    lk = model.model.layers[k]
    for pn, _ in list(lk.named_parameters()):
        assign(lk, pn, load_tensor(model_path, wmap, f"model.layers.{k}.{pn}"), device, dtype)

    mods = dict(lk.named_modules())
    target = None
    for cand in ("linear_attn.in_proj_qkv", "linear_attn.in_proj_qkvz",
                 "linear_attn.in_proj_a", "linear_attn.in_proj_b", "linear_attn.in_proj_z"):
        m = mods.get(cand)
        if isinstance(m, nn.Linear):
            target = cand
            break
    if target is None:
        for n, m in lk.named_modules():
            if isinstance(m, nn.Linear):
                target = n
                break
    leaf = mods[target]
    leaf.weight.requires_grad_(True)
    print(f"Differentiable leaf: {target}.weight  shape={tuple(leaf.weight.shape)}")

    hs = torch.randn(1, args.seq, hidden, device=device, dtype=dtype)
    kw = {}
    for kk, v in cap["kw"].items():
        if isinstance(v, torch.Tensor):
            kw[kk] = v.to(device)
        elif isinstance(v, tuple):
            kw[kk] = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in v)
        else:
            kw[kk] = v
    kw["use_cache"] = False

    try:
        with torch.enable_grad():
            out = lk(hs, *cap["a"], **filt(lk, kw))
            out = out[0] if isinstance(out, tuple) else out
            loss = out.float().pow(2).mean()
            loss.backward()
    except Exception as e:
        print(f"\n❌ FAIL: forward/backward raised {type(e).__name__}: {e}")
        print("   DeltaNet is not differentiable as configured.")
        print("   Try: --attn-impl eager, or inspect the linear-attention kernel/mode flags.")
        return

    g = leaf.weight.grad
    if g is None:
        print("\n❌ FAIL: gradient is None — backward did not reach the DeltaNet weights.")
        print("   The fused recurrence has no differentiable path as configured.")
    elif not torch.isfinite(g).all():
        print("\n⚠️ PARTIAL: gradient is non-finite (NaN/Inf) — differentiable but unstable.")
        print("   Try --dtype float32, or gradient clipping in the training loop.")
    else:
        print(f"\n✅ PASS: finite gradient reached {target}.weight "
              f"(grad norm {g.norm().item():.3e}).")
        print("   Global end-to-end training is viable — the E2E-QP loop can be built.")


if __name__ == "__main__":
    main()
