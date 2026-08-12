#!/usr/bin/env python3
"""
C4 super-outlier diagnostic (from research_reports/PRIORITIZED_MECHANISM_CHECKLIST.md).

Decisive question: has the folded QuaRot rotation already SPREAD the super-outliers?
If yes across the board, the whole outlier family (C4 quarantine/bias-refund, C13 log-lattice)
is a near no-op and effort should go to the recurrence clusters (C2/C3).

Method: for every ternarized 2D projection, reshape into TQ2_0 blocks = [out_row, 256 input-cols]
(one fp16 scale per block) and measure per-block OUTLIER DOMINATION, scale-agnostic:
  - excess kurtosis            (Gaussian=0; heavy tail = outlier-dominated)
  - top-1 energy fraction      max(w^2)/sum(w^2) over the 256 (1/256=0.0039 uniform; ->1 = one weight owns the block)
  - absmax-RTN sparsity        frac |w| < 0.5*absmax  (Gaussian-256 ~= 0.866; matches the code's "~84% zeros")
  - domination ratio           absmax / mean|w|        (Gaussian-256 ~= 3.7)
plus a per-matrix super-weight proxy  max|w| / matrix-std  (isolated super weight => large).

Splits by ROTATED-INPUT {q,k,v,gate,up,in_proj_*} vs UN-ROTATED-INPUT {o_proj,down_proj,out_proj}
(QuaRot rotates the input side, so un-rotated-input matrices — esp. down_proj, where Yu et al.
find super-weights — are where outliers can survive), and by DeltaNet vs full-attn layer.
Compares PRE-rotation (untied_4b) vs POST-rotation (rot/modified_model).
"""
import re, sys, json, math
import torch
from safetensors import safe_open

BLK = 256
MODELS = {
    "PRE-rot  (untied_4b)": "output_4b/untied_4b/model.safetensors",
    "POST-rot (rot)":       "output_4b/rot/modified_model/model.safetensors",
}
# projection family -> input-rotation group under QuaRot
ROTATED   = {"q_proj","k_proj","v_proj","gate_proj","up_proj",
             "in_proj_a","in_proj_b","in_proj_qkv","in_proj_qkvz","in_proj_z"}
UNROTATED = {"o_proj","down_proj","out_proj"}
FAM_RE = re.compile(r'\.(self_attn|linear_attn|mlp)\.([a-z0-9_]+)\.weight$')
LAYER_RE = re.compile(r'\.(\d+)\.')

def block_stats(W):
    """W [out,in] float32 -> flat 1D tensors of per-block metrics."""
    out, inp = W.shape
    nb = inp // BLK
    W3 = W[:, :nb*BLK].reshape(out, nb, BLK)
    mu = W3.mean(-1, keepdim=True)
    d = W3 - mu
    var = (d*d).mean(-1)
    m4 = (d.pow(4)).mean(-1)
    exkurt = m4 / (var*var + 1e-20) - 3.0
    absW = W3.abs()
    absmax = absW.amax(-1)
    meanabs = absW.mean(-1)
    dom = absmax / (meanabs + 1e-20)
    sparsity = (absW < 0.5*absmax.unsqueeze(-1)).float().mean(-1)
    sq = W3*W3
    t1e = sq.amax(-1) / (sq.sum(-1) + 1e-20)
    return (exkurt.reshape(-1), dom.reshape(-1), sparsity.reshape(-1), t1e.reshape(-1))

class Acc:
    __slots__=("n","s_k","s_d","s_sp","s_t","mx_k","mx_d","mx_t",
               "c_t25","c_t50","c_k10","c_k50","c_sp95")
    def __init__(s):
        s.n=0; s.s_k=s.s_d=s.s_sp=s.s_t=0.0
        s.mx_k=s.mx_d=s.mx_t=0.0
        s.c_t25=s.c_t50=s.c_k10=s.c_k50=s.c_sp95=0
    def add(s,k,d,sp,t):
        s.n+=k.numel()
        s.s_k+=k.sum().item(); s.s_d+=d.sum().item(); s.s_sp+=sp.sum().item(); s.s_t+=t.sum().item()
        s.mx_k=max(s.mx_k,k.max().item()); s.mx_d=max(s.mx_d,d.max().item()); s.mx_t=max(s.mx_t,t.max().item())
        s.c_t25+=(t>0.25).sum().item(); s.c_t50+=(t>0.50).sum().item()
        s.c_k10+=(k>10).sum().item(); s.c_k50+=(k>50).sum().item()
        s.c_sp95+=(sp>0.95).sum().item()
    def row(s):
        n=max(s.n,1)
        return (f"{s.n:>10,d} {s.s_k/n:>8.2f} {s.mx_k:>9.0f} "
                f"{s.s_d/n:>7.2f} {s.mx_d:>8.1f} {s.s_sp/n:>7.3f} "
                f"{s.s_t/n:>7.4f} {s.mx_t:>7.3f} "
                f"{100*s.c_t25/n:>6.2f} {100*s.c_t50/n:>6.2f} {100*s.c_k10/n:>6.2f} {100*s.c_sp95/n:>6.2f}")

HDR = (f"{'group':<34}{'#blocks':>11}{'exk_avg':>9}{'exk_max':>10}"
       f"{'dom':>8}{'dom_mx':>9}{'spars':>8}{'t1e_avg':>8}{'t1e_mx':>8}"
       f"{'%t1>.25':>7}{'%t1>.5':>7}{'%k>10':>7}{'%sp>.95':>8}")

def is_target(blk, fam):
    if fam in ROTATED or fam in UNROTATED: return True
    return False

for tag, path in MODELS.items():
    print("\n" + "="*len(HDR)); print(f"MODEL: {tag}   ({path})"); print("="*len(HDR)); print(HDR)
    groups = {}                      # keyed by descriptive group name -> Acc
    superweights = []                # (ratio, name)
    def G(name): return groups.setdefault(name, Acc())
    with safe_open(path, framework="pt", device="cpu") as f:
        for k in f.keys():
            m = FAM_RE.search(k)
            if not m: continue
            blk, fam = m.group(1), m.group(2)
            if not is_target(blk, fam): continue
            W = f.get_tensor(k)
            if W.dim()!=2 or W.shape[1] < BLK: continue
            W = W.float()
            exk, dom, sp, t1e = block_stats(W)
            rot = "ROT-in " if fam in ROTATED else "UNROT-in"
            lt  = "delta" if blk=="linear_attn" else ("fullattn" if blk=="self_attn" else "mlp")
            G(f"[{rot}] {lt:8s} {fam}").add(exk,dom,sp,t1e)
            G(f"ALL  {rot}").add(exk,dom,sp,t1e)
            G("ALL  (everything)").add(exk,dom,sp,t1e)
            mstd = W.std().item()
            ratio = W.abs().max().item()/(mstd+1e-20)
            superweights.append((ratio, k))
            del W
    # per-family detail
    for name in sorted(groups):
        if name.startswith("[") : print(f"{name:<34}{groups[name].row()}")
    print("-"*len(HDR))
    for name in ["ALL  ROT-in ","ALL  UNROT-in","ALL  (everything)"]:
        if name in groups: print(f"{name:<34}{groups[name].row()}")
    superweights.sort(reverse=True)
    print("\n  top-10 per-matrix super-weight proxy (max|w| / matrix-std):")
    for r,k in superweights[:10]:
        print(f"    {r:7.1f}x   {k}")

print("\nReference: iid Gaussian-256 block => exkurt~0, dom~3.7, sparsity~0.866, t1e~0.03.")
print("Verdict rule: if POST-rot ROT-in ~ Gaussian AND << PRE-rot => rotation spread outliers there (C4 no-op).")
print("Watch UNROT-in (o_proj/down_proj): if it stays heavy-tailed POST-rot, C4/A5 still alive THERE.")
