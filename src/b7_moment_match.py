#!/usr/bin/env python
"""B7 kill-check (axes battery, stage-2 report): per-256-block moment match.

For every quantized linear, compare the block-AP skeleton's dequantized weights (t*s, fp16) against the
rotated-FP weights it quantized: per 256-input-element block, Sigma(eps) = Sigma(w_tern) - Sigma(w_fp).
If per-block sums already match FP within noise (median |Sigma eps| / Sigma|w| < ~1%), a deterministic
anti-bias / moment-matching assignment has nothing to correct -> B7 dead, and A6/B2's additive-moment
branch is pre-killed (DFQ arXiv:1906.04721-adjacent).

Env: FP_DIR (rotated FP), TERN_DIR (block-AP skeleton). CPU-only, streams shard-by-shard.
"""
import os, json, re
import torch
from pathlib import Path
from safetensors import safe_open

FP_DIR = os.environ.get("FP_DIR", "output_4b/rot/modified_model")
TERN_DIR = os.environ.get("TERN_DIR", "output_4b/strengthen/a1_lr1e-4_ep4/modified_model")
BLOCK = 256
QPAT = re.compile(r"(mlp\.(gate|up|down)_proj|self_attn\.(q|k|v|o)_proj|linear_attn\..*proj.*)\.weight$")

def wmap(d):
    p = Path(d) / "model.safetensors.index.json"
    if p.exists():
        return json.load(open(p))["weight_map"]
    return {None: "model.safetensors"}

def get(d, m, name):
    sf = m[name] if name in m else m[None]
    with safe_open(str(Path(d) / sf), framework="pt", device="cpu") as f:
        return f.get_tensor(name).float()

mf, mt = wmap(FP_DIR), wmap(TERN_DIR)
names = [n for n in (mt.keys() if None not in mt else mt) if n and QPAT.search(n)]
print(f"B7 moment-match: {len(names)} quantized linears | FP={FP_DIR} vs TERN={TERN_DIR}")

by_type = {}
for i, n in enumerate(sorted(names)):
    wf, wt = get(FP_DIR, mf, n), get(TERN_DIR, mt, n)
    if wf.shape != wt.shape:
        print(f"  SKIP {n}: shape {wf.shape} vs {wt.shape}"); continue
    o, d = wf.shape
    nb = d // BLOCK
    if nb == 0:
        continue
    ef = (wt[:, :nb*BLOCK] - wf[:, :nb*BLOCK]).view(o, nb, BLOCK).sum(-1)      # per-block Sigma(eps)
    aw = wf[:, :nb*BLOCK].abs().view(o, nb, BLOCK).sum(-1).clamp_min(1e-12)    # per-block Sigma|w|
    rel = (ef.abs() / aw).flatten()
    t = "mlp" if ".mlp." in n else ("attn" if "self_attn" in n else "deltanet")
    by_type.setdefault(t, []).append(rel)
    if (i + 1) % 64 == 0:
        print(f"  {i+1}/{len(names)}...", flush=True)

print("\n== per-block |Sigma eps| / Sigma|w| (the moment-match statistic) ==")
allr = []
for t, rs in sorted(by_type.items()):
    r = torch.cat(rs); allr.append(r)
    q = lambda p: r.quantile(p).item()
    print(f"  {t:9s}: median={q(0.5)*100:.3f}%  p90={q(0.9)*100:.3f}%  p99={q(0.99)*100:.3f}%  n_blocks={r.numel()}")
r = torch.cat(allr)
med = r.quantile(0.5).item()
print(f"  ALL      : median={med*100:.3f}%  p90={r.quantile(0.9).item()*100:.3f}%")
print(f"\nVERDICT: {'B7 DEAD (median < 1% — nothing for moment-matching to correct)' if med < 0.01 else 'B7 ALIVE (median >= 1% — systematic per-block bias exists)'}")
