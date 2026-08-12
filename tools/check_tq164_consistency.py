#!/usr/bin/env python
"""check_tq164_consistency.py — guard the TQ1_64 format against implementation drift.

Three implementations must agree, or the C++ kernels will be validated against the wrong thing:
  1. src/tq164.py       — the REFERENCE spec (bit-level pack/unpack). The ggml/CUDA kernels must match this.
  2. src/sim_tq164.py   — the fast vectorised path used to evaluate whole models (eval2k / loop_gate).
  3. (later) the C round-trip test.

Run after touching either file:  ./.venv/bin/python tools/check_tq164_consistency.py
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tq164 import encode_row, decode_row, QK_K, GROUP, BLOCK_BYTES, bpw   # noqa: E402
from sim_tq164 import encode_decode, SUPERBLOCK                            # noqa: E402

MODEL = "output_4b/final_g64q8/e2e/modified_model/model.safetensors"
TENSORS = ["model.language_model.layers.0.mlp.down_proj.weight", "lm_head.weight"]
fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print(f"TQ1_64: {QK_K}w / {BLOCK_BYTES}B = {bpw():.4f} bpw, group {GROUP}")
check("sim SUPERBLOCK matches reference QK_K", SUPERBLOCK == QK_K, f"{SUPERBLOCK} vs {QK_K}")

# --- synthetic: trits must survive pack/unpack bit-exactly ---
rng = np.random.default_rng(0)
for cols in (512, 2560, 9216):
    s = rng.uniform(1e-3, 5e-2, size=cols // GROUP).astype(np.float32)
    t = rng.integers(-1, 2, size=cols).astype(np.float32)
    w = (t.reshape(-1, GROUP) * s[:, None]).reshape(-1)
    rec = decode_row(encode_row(w), cols)
    g = rec.reshape(-1, GROUP)
    rs = np.abs(g).max(axis=1)
    rt = np.where(rs[:, None] > 0, g / np.where(rs[:, None] > 0, rs[:, None], 1), 0).round()
    check(f"trits bit-exact (cols={cols})", np.array_equal(rt.reshape(-1), t))

# --- all-zero group must reconstruct as exact zeros ---
w0 = np.zeros(QK_K, dtype=np.float32)
check("all-zero superblock -> exact zeros", np.all(decode_row(encode_row(w0), QK_K) == 0))

# --- real weights: reference vs vectorised sim must be bit-identical ---
try:
    from safetensors import safe_open
    with safe_open(MODEL, "pt") as f:
        for n in TENSORS:
            W = f.get_tensor(n)[:8].float()
            sim, _, _ = encode_decode(W, GROUP)
            ref = np.stack([decode_row(encode_row(W[i].numpy()), W.shape[1]) for i in range(W.shape[0])])
            d = np.abs(ref - sim.numpy()).max()
            check(f"reference == vectorised sim  [{n.split('.')[-2]}]", d == 0.0, f"max|diff|={d:.3e}")
            # the fast whole-tensor packer must produce the SAME BYTES as row-by-row encode_row
            from tq164 import encode_tensor, QK_K as _QK
            bulk = encode_tensor(W.numpy())
            per_row = np.concatenate([encode_row(W[i].numpy()) for i in range(W.shape[0])], axis=0)
            check(f"encode_tensor == encode_row bytes [{n.split('.')[-2]}]",
                  np.array_equal(bulk, per_row), f"{bulk.shape} blocks")
            err = np.abs(ref - W.numpy()) / np.abs(W.numpy()).max()
            print(f"         vs original weights: max {100*err.max():.3f}%  rel-RMS {100*np.sqrt((err**2).mean()):.4f}%")
except FileNotFoundError:
    print(f"  [SKIP] real-weight check ({MODEL} not present)")

print(f"\n{'ALL CHECKS PASSED' if fail == 0 else f'{fail} CHECK(S) FAILED'}")
sys.exit(1 if fail else 0)
