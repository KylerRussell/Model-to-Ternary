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
from format_sim import quant_ternary, quant_iq1s, rel_err, verify_iq1s

SRC = os.environ.get("FMT_SRC", "output_4bpipe/rotbase/modified_model")
N_T = int(os.environ.get("N_TENSORS", "12"))
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"

verify_iq1s(n=32, device="cpu")          # refuse to run on an unfaithful simulator

wm = _shard_map(SRC)
names = [k for k in wm if k.endswith(".weight") and ("proj" in k or "lm_head" in k)]
names.sort()
step = max(1, len(names) // N_T)
names = names[::step][:N_T]
print(f"\n{len(names)} tensors from {SRC}\n")

rows = []
print(f"{'tensor':46s} {'shape':>16s} {'TQ1_0':>9s} {'TQ1_64':>9s} {'IQ1_S':>9s}  winner")
for n in names:
    W = _get_tensor(SRC, wm, n).float().to(DEV)
    if W.ndim != 2 or W.shape[1] % 256:
        continue
    e = {}
    e["TQ1_0"] = rel_err(quant_ternary(W, group=256), W)
    e["TQ1_64"] = rel_err(quant_ternary(W, group=64), W)
    e["IQ1_S"] = rel_err(quant_iq1s(W, device=DEV), W)
    win = min(e, key=e.get)
    rows.append({"tensor": n, "shape": list(W.shape), **e, "winner": win})
    print(f"{n[-46:]:46s} {str(tuple(W.shape)):>16s} "
          f"{e['TQ1_0']:9.5f} {e['TQ1_64']:9.5f} {e['IQ1_S']:9.5f}  {win}")
    del W
    torch.cuda.empty_cache()

print("\n" + "=" * 78)
for k in ("TQ1_0", "TQ1_64", "IQ1_S"):
    m = sum(r[k] for r in rows) / max(len(rows), 1)
    print(f"  mean rel reconstruction error  {k:7s} {m:.5f}   "
          f"wins {sum(1 for r in rows if r['winner']==k):2d}/{len(rows)}")
print("  bpw:  TQ1_0 1.6875   TQ1_64 1.7812   IQ1_S 1.5625")
print("=" * 78)
if os.environ.get("OUT"):
    json.dump(rows, open(os.environ["OUT"], "w"), indent=1)
