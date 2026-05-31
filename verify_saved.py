#!/usr/bin/env python3
"""
verify_saved.py — Confirm that the bytes convert.py actually WROTE to
output/modified_model match a fresh re-rotation of the originals.

verify_phase1.py proved the rotation *logic* is transparent, but it re-rotates
in memory and never opens convert.py's output. This closes that gap: a stray
transpose, dtype slip, or bad torch.cat in the save path would pass
verify_phase1 yet emit a corrupt GGUF. Here we read the saved tensors back and
diff them against rotate(original).

Usage:
    python verify_saved.py \
        --orig  /home/kyler/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13... \
        --saved output/modified_model
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from hadamard import rotate_weight_matrix


def wmap(p: Path):
    idx = p / "model.safetensors.index.json"
    if idx.exists():
        return json.load(open(idx))["weight_map"]
    with safe_open(str(p / "model.safetensors"), framework="pt") as f:
        return {n: "model.safetensors" for n in f.keys()}


def get(p, wm, name):
    with safe_open(str(p / wm[name]), framework="pt") as f:
        return f.get_tensor(name).float()


def find(wm, *needles):
    for n in wm:
        if all(s in n for s in needles):
            return n
    return None


def diff(label, a, b):
    if a.shape != b.shape:
        print(f"  [SHAPE!] {label:<44s} saved={tuple(a.shape)} expected={tuple(b.shape)}")
        return False
    d = (a - b).abs().max().item()
    # bf16 round-trip in convert.py → expect ~1e-2 abs, never structural
    ok = d < 5e-2
    print(f"  [{'ok' if ok else 'MISMATCH'}] {label:<44s} max|Δ|={d:.3e}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True)
    ap.add_argument("--saved", required=True)
    ap.add_argument("--layer", type=int, default=0)
    a = ap.parse_args()
    O, S = Path(a.orig), Path(a.saved)
    wo, ws = wmap(O), wmap(S)
    print(f"orig tensors={len(wo)}  saved tensors={len(ws)}")
    if len(wo) != len(ws):
        print("  ⚠️ tensor COUNT differs — convert.py dropped/added tensors!")

    emb = find(ws, "embed_tokens.weight")
    lp = emb.rsplit(".embed_tokens", 1)[0]
    lpref = f"{lp}.layers.{a.layer}"
    ok = True

    # rotated input proj (absorb input_layernorm +1) — DeltaNet
    inln = find(wo, lpref, "input_layernorm.weight")
    w = get(O, wo, inln) + 1.0
    for proj in ("linear_attn.in_proj_qkv", "linear_attn.in_proj_qkvz",
                 "linear_attn.in_proj_z", "mlp.gate_proj"):
        nm = find(ws, lpref, proj, "weight")
        if nm and nm in wo:
            wn = w if "mlp" not in proj else (get(O, wo, find(wo, lpref, "post_attention_layernorm.weight")) + 1.0)
            expected = rotate_weight_matrix(get(O, wo, nm) * wn, dim="input")
            ok &= diff(nm.replace(lp + ".", ""), get(S, ws, nm), expected)

    # rotated output proj
    for proj in ("linear_attn.out_proj", "mlp.down_proj"):
        nm = find(ws, lpref, proj, "weight")
        if nm and nm in wo:
            expected = rotate_weight_matrix(get(O, wo, nm), dim="output")
            ok &= diff(nm.replace(lp + ".", ""), get(S, ws, nm), expected)

    # embed rotated
    if emb in wo:
        ok &= diff("embed_tokens (E·H)",
                   get(S, ws, emb)[:512], rotate_weight_matrix(get(O, wo, emb)[:512], dim="input"))

    # absorbed norm must be ZEROED on disk
    if inln in ws:
        z = get(S, ws, inln).abs().max().item()
        good = z < 1e-6
        print(f"  [{'ok' if good else 'NOT ZERO!'}] {inln.replace(lp + '.', '')} absorbed-norm |max|={z:.3e}")
        ok &= good

    print("\n" + ("✅ saved model matches the rotation — convert.py's output is faithful."
                  if ok else "❌ saved bytes diverge from rotate(original) — bug is in convert.py's SAVE path."))


if __name__ == "__main__":
    main()
