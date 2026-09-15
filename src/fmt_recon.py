"""fmt_recon.py — stage 1 of the format head-to-head: reconstruction error on real weights.

Cheap signal before committing to a full end-to-end evaluation. Every arm is RTN over the SAME
weights with the SAME objective, so the only thing that varies is the format's representational
capacity. Arms:

    TQ1_0   g256 ternary, fp16 scale                 1.6875 bpw
    TQ1_64  g64  ternary, 8-bit sub-scale + super    1.7812 bpw
    IQ1_S   g32  3-bit sub-scale, 2048-entry 8-D grid, +/-0.125 delta   1.5625 bpw

Reconstruction error is a SCREEN, not a verdict -- this project has documented repeatedly that
block-MSE endorses broken models (it improved 99.8% on a collapsed residual stream). A format that
loses badly here is not worth an end-to-end run; a format that wins here still has to prove it
end-to-end.
"""
import os, sys, json, torch
sys.path.insert(0, os.path.dirname(__file__))
from e2e_qp_distill import _shard_map, _get_tensor
from format_sim import quant_ternary, quant_iq1s, quant_iq1m, rel_err, verify_iq1s

SRC = os.environ.get("FMT_SRC", "output_4bpipe/rotbase/modified_model")
N_T = int(os.environ.get("N_TENSORS", "12"))
# Cap tensor size: the IQ1_S search is 192 (d x s x delta) passes over a [N,2048] score matrix, so
# lm_head (2.5M blocks) dominates wall time without adding signal about LINEAR-PROJECTION format
# capacity. The embedding/head question is separate (they are 29.1% of the 4B) and is measured on
# its own, not folded in here.
MAXEL = int(os.environ.get("MAX_ELEMS", str(40 * 1024 * 1024)))
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"

verify_iq1s(n=32, device="cpu")          # refuse to run on an unfaithful simulator

wm = _shard_map(SRC)
# An EXPLICIT tensor list, so the rotated and unrotated arms are scored on the SAME tensors.
# Sampling each model independently produced non-comparable sets (the unrotated draw was 3/4 tiny
# (32,2560) projections), and a format comparison across different tensors measures the tensors.
names = [n for n in (os.environ.get("TENSORS", "").split(",")) if n]
if not names:
    names = [k for k in wm if k.endswith(".weight") and "proj" in k]
    names.sort()
    step = max(1, len(names) // N_T)
    names = names[::step][:N_T]
missing = [n for n in names if n not in wm]
assert not missing, f"tensors absent from {SRC}: {missing[:3]}"
print(f"\n{len(names)} tensors from {SRC}\n")

rows = []
print(f"{'tensor':46s} {'shape':>16s} {'TQ1_0':>9s} {'TQ1_64':>9s} {'IQ1_S':>9s} {'IQ1_M':>9s}  winner")
for ti, n in enumerate(names):
    W = _get_tensor(SRC, wm, n).float().to(DEV)
    if W.ndim != 2 or W.shape[1] % 256 or W.numel() > MAXEL:
        del W; continue
    e = {}
    e["TQ1_0"] = rel_err(quant_ternary(W, group=256), W)
    e["TQ1_64"] = rel_err(quant_ternary(W, group=64), W)
    e["IQ1_S"] = rel_err(quant_iq1s(W, device=DEV), W)
    e["IQ1_M"] = rel_err(quant_iq1m(W, device=DEV), W)
    win = min(e, key=e.get)
    rows.append({"tensor": n, "shape": list(W.shape), **e, "winner": win})
    print(f"[{ti+1}/{len(names)}] {n[-40:]:40s} {str(tuple(W.shape)):>16s} "
          f"{e['TQ1_0']:9.5f} {e['TQ1_64']:9.5f} {e['IQ1_S']:9.5f} {e['IQ1_M']:9.5f}  {win}", flush=True)
    del W
    torch.cuda.empty_cache()

print("\n" + "=" * 78)
for k in ("IQ1_S", "TQ1_0", "IQ1_M", "TQ1_64"):
    m = sum(r[k] for r in rows) / max(len(rows), 1)
    print(f"  mean rel reconstruction error  {k:7s} {m:.5f}   "
          f"wins {sum(1 for r in rows if r['winner']==k):2d}/{len(rows)}")
print("  bpw:  IQ1_S 1.5625 < TQ1_0 1.6875 < IQ1_M 1.7500 < TQ1_64 1.7812")
print("  IQ1_M vs TQ1_64 is the MATCHED-RATE pair (1.75 vs 1.7812, within 1.8%)")
print("=" * 78)
if os.environ.get("OUT"):
    json.dump(rows, open(os.environ["OUT"], "w"), indent=1)
