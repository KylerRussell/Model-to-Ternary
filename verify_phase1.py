#!/usr/bin/env python3
"""
verify_phase1.py — Prove (or disprove) that the Phase-1 QuaRot rotation is a
mathematically transparent transform of the REAL Qwen3.6-27B weights, entirely
in PyTorch/fp32, with ZERO involvement from convert_hf_to_gguf_patched.py or
llama.cpp.

It replicates convert.py's exact per-tensor logic (offset-norm +1, absorption,
input/output rotation) using the project's own hadamard.py, then checks the
QuaRot cancellation identity on real tensors:

    input  proj:  rmsnorm(X) ⊙ w  @ Wᵀ   ==   rmsnorm(X·H) @ W_newᵀ
    output proj:  (Y @ Woᵀ) · H          ==   Y @ Wo_newᵀ
    embed      :  E_new                  ==   E · H
    lm_head    :  rmsnorm(X) ⊙ w_f @ Lᵀ  ==   rmsnorm(X·H) @ L_newᵀ

If every check passes (max|Δ| ~1e-5 at fp32), the rotation is transparent and
the autoregressive failure is NOT produced by convert.py — it lives in the
converter or in llama.cpp's QWEN35 runtime. The script also flags the single
realistic transparency-breaker convert.py does NOT handle: a BIAS on any
residual-writing projection (o_proj / out_proj / down_proj), which would need
to be rotated but is currently passed through untouched.

Usage:
    python verify_phase1.py --model-path /path/to/Qwen3.6-27B
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from hadamard import rotate_weight_matrix, hadamard_transform

EPS = 1e-6
TOL = 2e-3  # generous fp32 tolerance; transparent layers come in far under this


def rms(x):
    return x / torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + EPS)


def load_index(model_path: Path) -> dict:
    idx = model_path / "model.safetensors.index.json"
    if idx.exists():
        return json.load(open(idx))["weight_map"]
    single = model_path / "model.safetensors"
    if single.exists():
        with safe_open(str(single), framework="pt") as f:
            return {n: "model.safetensors" for n in f.keys()}
    raise FileNotFoundError(f"No safetensors in {model_path}")


class Loader:
    def __init__(self, model_path: Path, wmap: dict):
        self.model_path, self.wmap = model_path, wmap

    def has(self, name): return name in self.wmap

    def get(self, name, rows=None):
        shard = self.wmap[name]
        with safe_open(str(self.model_path / shard), framework="pt") as f:
            t = f.get_slice(name)[:rows] if rows else f.get_tensor(name)
        return t.float()


def find(loader, *needles):
    """Return the first tensor name containing all needles, else None."""
    for n in loader.wmap:
        if all(s in n for s in needles):
            return n
    return None


def norm_for(loader, layer_prefix, which):
    """Find input_layernorm / post_attention_layernorm weight for a layer prefix."""
    for cand in (f"{layer_prefix}.{which}.weight",):
        if loader.has(cand):
            return cand
    return None


def report(label, lhs, rhs):
    d = (lhs - rhs).abs().max().item()
    cos = F.cosine_similarity(lhs.flatten(), rhs.flatten(), dim=0).item()
    ok = d < TOL
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<46s} max|Δ|={d:.3e}  cos={cos:.8f}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--full-attn-layer", type=int, default=3)
    ap.add_argument("--linear-attn-layer", type=int, default=0)
    args = ap.parse_args()

    mp = Path(args.model_path)
    wmap = load_index(mp)
    L = Loader(mp, wmap)
    print(f"Loaded index: {len(wmap)} tensors\n")

    D = 5120
    H = hadamard_transform(torch.eye(D))
    orth = (H @ H.T - torch.eye(D)).abs().max().item()
    print(f"H(5120) orthogonality ||HHᵀ-I||={orth:.2e}  symmetric={(H-H.T).abs().max().item():.2e}\n")

    # locate the language-model prefix
    emb = find(L, "embed_tokens.weight")
    lp = emb.rsplit(".embed_tokens", 1)[0] if emb else "model"
    print(f"LM prefix: {lp}\n")

    all_ok = True
    bias_problems = []

    # ---- per-layer linear checks -------------------------------------------
    def check_layer(idx, attn_kind):
        nonlocal all_ok
        lpref = f"{lp}.layers.{idx}"
        print(f"── layer {idx} ({attn_kind}) ──")
        in_ln = norm_for(L, lpref, "input_layernorm")
        post_ln = norm_for(L, lpref, "post_attention_layernorm")

        # input-side projections (read residual): absorb input_layernorm + rotate input
        if attn_kind == "full_attention":
            in_projs = [f"{lpref}.self_attn.{p}_proj.weight" for p in ("q", "k", "v")]
            out_projs = [f"{lpref}.self_attn.o_proj.weight"]
        else:
            in_projs = [n for n in (
                f"{lpref}.linear_attn.in_proj_qkv.weight",
                f"{lpref}.linear_attn.in_proj_qkvz.weight",
                f"{lpref}.linear_attn.in_proj_z.weight",
                f"{lpref}.linear_attn.in_proj_a.weight",
                f"{lpref}.linear_attn.in_proj_b.weight",
            ) if L.has(n)]
            out_projs = [f"{lpref}.linear_attn.out_proj.weight"]
        in_projs += [f"{lpref}.mlp.gate_proj.weight", f"{lpref}.mlp.up_proj.weight"]
        out_projs += [f"{lpref}.mlp.down_proj.weight"]

        w_in = (L.get(in_ln) + 1.0) if in_ln else None          # offset norm +1
        w_post = (L.get(post_ln) + 1.0) if post_ln else None

        for name in in_projs:
            if not L.has(name):
                continue
            W = L.get(name)
            w = w_post if ".mlp." in name else w_in
            if w is None:
                print(f"  [SKIP] {name}: no norm found"); continue
            X = torch.randn(4, W.shape[1])
            W_new = rotate_weight_matrix(W * w, dim="input")
            y_orig = (rms(X) * w) @ W.T
            y_rot = rms(X @ H) @ W_new.T
            all_ok &= report(name.replace(lp + ".", ""), y_orig, y_rot)

        for name in out_projs:
            if not L.has(name):
                continue
            W = L.get(name)
            Y = torch.randn(4, W.shape[1])
            W_new = rotate_weight_matrix(W, dim="output")
            lhs = (Y @ W.T) @ H
            rhs = Y @ W_new.T
            all_ok &= report(name.replace(lp + ".", ""), lhs, rhs)
            bname = name.replace(".weight", ".bias")
            if L.has(bname):
                bias_problems.append(bname)
        print()

    check_layer(args.linear_attn_layer, "linear_attention")
    check_layer(args.full_attn_layer, "full_attention")

    # ---- embedding ----------------------------------------------------------
    print("── embedding / lm_head ──")
    E = L.get(emb, rows=512)
    E_new = rotate_weight_matrix(E, dim="input")
    all_ok &= report("embed_tokens (E·H)", E @ H, E_new)

    # ---- lm_head + final norm ----------------------------------------------
    fnorm = find(L, lp.split(".")[0], "norm.weight")
    # prefer the true final norm (…model.norm.weight / …language_model.norm.weight)
    for cand in (f"{lp}.norm.weight", "model.norm.weight"):
        if L.has(cand):
            fnorm = cand; break
    lmh = find(L, "lm_head.weight") or find(L, "lm_head")
    if lmh and fnorm:
        Lw = L.get(lmh, rows=512)
        wf = L.get(fnorm) + 1.0
        X = torch.randn(4, Lw.shape[1])
        Lw_new = rotate_weight_matrix(Lw * wf, dim="input")
        lo = (rms(X) * wf) @ Lw.T
        lr = rms(X @ H) @ Lw_new.T
        all_ok &= report("lm_head + final norm", lo, lr)
    else:
        print(f"  [SKIP] lm_head/final-norm not located (lmh={lmh}, fnorm={fnorm})")

    # ---- verdict ------------------------------------------------------------
    print("\n" + "=" * 64)
    if bias_problems:
        print("⚠️  RESIDUAL-WRITE BIASES FOUND (convert.py does NOT rotate these,")
        print("    which BREAKS transparency — each must be rotated by H):")
        for b in bias_problems:
            print(f"      {b}")
        print()
    if all_ok and not bias_problems:
        print("✅ ROTATION IS TRANSPARENT on the real weights.")
        print("   convert.py is NOT the cause. The bug is downstream:")
        print("   → run the bisection (unmodified model through the SAME patched")
        print("     converter + SAME llama.cpp build). If that ALSO garbles, the")
        print("     fault is in the converter or llama.cpp's QWEN35 runtime.")
    elif all_ok and bias_problems:
        print("◐  Linear transparency holds, but residual-write biases above are")
        print("   unrotated — fix those in convert.py first, then re-test.")
    else:
        print("❌ A layer FAILED transparency — the rotation/absorption for that")
        print("   tensor is wrong. Inspect the failing line above.")
    print("=" * 64)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
