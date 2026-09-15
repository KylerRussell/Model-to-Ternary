"""run_family_sweep.py — the family sweep on STOCK weights. See paper/format_taxonomy.md."""
import os, sys, json, glob, numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
from e2e_qp_distill import _shard_map, _get_tensor
from family_sweep import (LOG2_3, bpw_sym, bpw_asym, bpw_vq, bpw_microfloat, bpw_multiplane,
                          bpw_lowrank, bpw_sparse_hybrid, bpw_trellis, q_sym, q_asym, q_vq,
                          q_nonuniform, q_microfloat, q_lattice_e8, q_multiplane, q_lowrank,
                          q_sparse_hybrid, q_trellis, nf_levels, entropy_bpw, rel_err)

SRC = os.environ.get("SRC") or glob.glob(
    "/home/kasm-user/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*")[0]
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
wm = _shard_map(SRC)
# STRATIFIED by tensor KIND, not by alphabetical position. A strided slice of sorted names picked
# three (32, 2560) `in_proj_a` tensors -- identical tiny shapes -- which made the low-rank arm
# degenerate (rank 32 on min_dim 32 is an identity, not a compression) and gave every group-size
# comparison only ten g256 groups per row to work with. The families must be scored on the tensors
# that carry the model's parameters.
KINDS = os.environ.get("KINDS", "down_proj,gate_proj,up_proj,in_proj_qkv,out_proj").split(",")
names = []
for kind in KINDS:
    c = sorted(k for k in wm if k.endswith(f"{kind}.weight"))
    if c:
        names.append(c[len(c) // 2])            # a mid-depth layer of each kind
names = names[:int(os.environ.get("N_T", "5"))]
print(f"STOCK model: {SRC}\ntensors: {len(names)}\n", flush=True)

ARMS = []
for g in (32, 64, 128, 256):
    for b, nm in ((1.0, "binary"), (LOG2_3, "ternary"), (3.0, "int3"), (4.0, "int4")):
        ARMS.append((f"A sym {nm} g{g}", bpw_sym(b, g), lambda W, b=b, g=g: q_sym(W, b, g)))
for g in (32, 64, 128):
    for b in (2, 3, 4):
        ARMS.append((f"B asym int{b} g{g}", bpw_asym(b, g), lambda W, b=b, g=g: q_asym(W, b, g)))
for g in (32, 64, 128):
    ARMS.append((f"C nonunif IQ4NL g{g}", 4 + 16 / g, lambda W, g=g: q_nonuniform(W, g)))
    ARMS.append((f"C nonunif NF4 g{g}", 4 + 16 / g,
                 lambda W, g=g: q_nonuniform(W, g, levels=nf_levels(4, DEV))))
for g in (32, 64):
    ARMS.append((f"D microfloat E2M1 g{g}", bpw_microfloat(2, 1, g),
                 lambda W, g=g: q_microfloat(W, g, 2, 1)))
for k, d, g in ((256, 8, 256), (2048, 8, 256), (16384, 8, 256), (65536, 8, 256),
                (2048, 8, 64), (256, 4, 256), (65536, 16, 256)):
    ARMS.append((f"E vq k{k} d{d} g{g}", bpw_vq(k, d, g),
                 lambda W, k=k, d=d, g=g: q_vq(W, k, d, g, device=DEV)))

# H, I, J, L are MODIFIERS on a base format, not standalone formats -- see paper/format_taxonomy.md.
for g in (64, 128):
    for pl, b, nm in ((2, 1.0, "binary"), (3, 1.0, "binary"), (2, LOG2_3, "ternary")):
        ARMS.append((f"H plane x{pl} {nm} g{g}", bpw_multiplane(g, pl, b),
                     lambda W, g=g, pl=pl, b=b: q_multiplane(W, g, pl, b)))
for fr in (0.005, 0.01, 0.02, 0.05):
    ARMS.append((f"J sparse {fr*100:g}% tern g64", bpw_sparse_hybrid(fr, 64),
                 lambda W, fr=fr: q_sparse_hybrid(W, fr, 64)))
for kb, L in ((1, 10), (2, 10), (2, 12), (3, 10)):
    ARMS.append((f"G trellis k{kb} L{L} g256", bpw_trellis(kb, 256),
                 lambda W, kb=kb, L=L: q_trellis(W, kb, L, 256, device=DEV)))

rows = []
for n in names:
    W = _get_tensor(SRC, wm, n).float().to(DEV)
    if W.ndim != 2 or W.shape[1] % 256 or W.numel() > 40 * 1024 * 1024 or min(W.shape) < 256:
        del W
        continue
    print(f"--- {n}  {tuple(W.shape)}", flush=True)
    for label, bpw, fn in ARMS:
        try:
            e = rel_err(fn(W), W)
            rows.append({"tensor": n, "arm": label, "bpw": float(bpw), "err": e})
            print(f"   {label:26s} {bpw:6.3f} bpw   {e:.5f}", flush=True)
        except Exception as ex:
            print(f"   {label:26s} SKIP ({type(ex).__name__})", flush=True)
    for r in (16, 32, 64):                       # I: low-rank -- bpw depends on tensor shape
        if r >= min(W.shape) // 2:               # rank near full rank is an identity, not compression
            print(f"   I lowrank r{r}: SKIP (rank {r} >= half of min_dim {min(W.shape)})", flush=True)
            continue
        try:
            e = rel_err(q_lowrank(W, r, 64), W)
            bp = bpw_lowrank(W.shape, r, 64)
            rows.append({"tensor": n, "arm": f"I lowrank r{r} tern g64", "bpw": bp, "err": e})
            print(f"   {'I lowrank r'+str(r)+' tern g64':26s} {bp:6.3f} bpw   {e:.5f}", flush=True)
        except Exception as ex:
            print(f"   I lowrank r{r}: SKIP ({type(ex).__name__})", flush=True)
    for g in (64, 128, 256):                     # F: lattice, rate measured not assumed
        r, bp = q_lattice_e8(W, g, return_bpw=True)
        rows.append({"tensor": n, "arm": f"F lattice E8 g{g}", "bpw": bp, "err": rel_err(r, W)})
        print(f"   {'F lattice E8 g'+str(g):26s} {bp:6.3f} bpw   {rows[-1]['err']:.5f}", flush=True)
    del W
    torch.cuda.empty_cache()

json.dump(rows, open(os.environ.get("OUT", "output_sweep/family_sweep.json"), "w"), indent=1)
print(f"\nwrote {os.environ.get('OUT','output_sweep/family_sweep.json')}  ({len(rows)} rows)")
