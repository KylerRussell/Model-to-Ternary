"""family_sweep.py — what does each FAMILY of weight formats buy per bit?

WHY THIS AND NOT A LIST OF NAMED FORMATS. Measuring TQ1_0 / IQ1_S / IQ1_M / Q2_K samples the space
wherever llama.cpp happened to define a type. Those are convenience points, not a design space, and
a comparison built from them inherits whatever the ggml authors chose to implement. Worse, our own
earlier comparison anchored everything at ~1.78 bpw -- a number that came from matching a released
ternary model's operating point, not from any constraint -- so it asked "which format is best at the
bpw we already picked" rather than "what should the bpw be".

This sweeps the three structural families with their knobs exposed, so the named formats fall out as
points on the curves:

  SYMMETRIC SCALAR   b bits/weight, one positive scale per group g.
                     b=1 binary (Q1_0-like) | b=log2(3) ternary (TQ1_0/TQ2_0/TQ1_64-like)
                     | b=2,3,4,8 the Q*_0 family.       bpw = b + 16/g
  ASYMMETRIC SCALAR  adds a per-group zero-point: the k-quant / GPTQ / EXL2 / MLX family.
                     Q2_K-like, Q4_K-like.              bpw = b + 32/g   (fp16 scale + fp16 zero)
  VECTOR / CODEBOOK  k centroids over d-dim sub-vectors, plus a per-group scale: the IQ / AQLM /
                     QuIP# / VPTQ family.               bpw = log2(k)/d + 16/g

Each family is given its BEST ACHIEVABLE form, not llama.cpp's particular instance: the codebook is
fitted to the actual weight distribution by k-means rather than using a fixed published grid. That
is deliberate -- we are measuring what a family can do, so that a method which needs that family can
be screened against its ceiling rather than against one implementation of it.

MODEL. Stock weights only. Our own checkpoints are untied and QuaRot-rotated, and rotation in
particular reshapes the distribution that fine-grained scales exist to capture, so measuring format
families on them would let our pipeline pick the winner.
"""
import numpy as np
import torch

LOG2_3 = float(np.log2(3))


def bpw_sym(bits, group, scale_bits=16):
    return bits + scale_bits / group


def bpw_asym(bits, group, scale_bits=16, zero_bits=16):
    return bits + (scale_bits + zero_bits) / group


def bpw_vq(k, dim, group, scale_bits=16):
    return float(np.log2(k)) / dim + scale_bits / group


@torch.no_grad()
def q_sym(W, bits, group, n_scan=32):
    """Symmetric scalar: levels are {-L..L} with L = 2^(b-1)-1, or {-1,0,1} for ternary, {-1,1} binary."""
    out, inp = W.shape
    w = W.reshape(-1, group).float()
    if abs(bits - 1.0) < 1e-6:                       # binary: no zero state
        amax = w.abs().amax(-1, keepdim=True).clamp_min(1e-12)
        best = None
        for f in torch.linspace(0.3, 1.0, n_scan, device=w.device):
            s = (amax * f).half().float()
            r = torch.where(w >= 0, s, -s)
            e = ((r - w) ** 2).sum(-1, keepdim=True)
            best = (e, r) if best is None else (
                torch.where(e < best[0], e, best[0]), torch.where(e < best[0], r, best[1]))
        return best[1].reshape(out, inp).to(W.dtype)
    # NOTE: symmetric int-b has levels {-L..L}, L = 2^(b-1)-1. At b=2 that is {-1,0,1} -- symmetric
    # 2-bit IS ternary with one of its four codes unused. The TQ formats exist precisely to reclaim
    # that waste: ternary needs only log2(3) = 1.585 bits, so packing recovers 0.415 bpw for free.
    L = 1 if abs(bits - LOG2_3) < 1e-6 else (2 ** (int(round(bits)) - 1) - 1)
    amax = w.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    best = None
    for f in torch.linspace(0.3, 1.0, n_scan, device=w.device):
        s = (amax * f / L).half().float().clamp_min(1e-12)
        r = torch.round(w / s).clamp(-L, L) * s
        e = ((r - w) ** 2).sum(-1, keepdim=True)
        best = (e, r) if best is None else (
            torch.where(e < best[0], e, best[0]), torch.where(e < best[0], r, best[1]))
    return best[1].reshape(out, inp).to(W.dtype)


@torch.no_grad()
def q_asym(W, bits, group, n_scan=16):
    """Asymmetric scalar: q in [0, 2^b-1], x = s*q + z. The zero-point is the capability at issue."""
    out, inp = W.shape
    w = W.reshape(-1, group).float()
    n = 2 ** int(round(bits)) - 1
    lo0, hi0 = w.amin(-1, keepdim=True), w.amax(-1, keepdim=True)
    best = None
    # shrink the [lo,hi] range inward: the min/max fit is not the MSE optimum under rounding
    for f in torch.linspace(0.6, 1.0, n_scan, device=w.device):
        mid = (lo0 + hi0) / 2
        half = (hi0 - lo0) / 2 * f
        lo, hi = mid - half, mid + half
        s = ((hi - lo) / n).half().float().clamp_min(1e-12)
        z = lo.half().float()
        r = torch.round((w - z) / s).clamp(0, n) * s + z
        e = ((r - w) ** 2).sum(-1, keepdim=True)
        best = (e, r) if best is None else (
            torch.where(e < best[0], e, best[0]), torch.where(e < best[0], r, best[1]))
    return best[1].reshape(out, inp).to(W.dtype)


@torch.no_grad()
def q_vq(W, k, dim, group, iters=12, sample=200_000, seed=0, device=None):
    """Vector quantization: k centroids over dim-dimensional sub-vectors, per-group scale.

    The codebook is FITTED to this tensor by k-means rather than taken from a published grid, so the
    family is measured at its ceiling. Sub-vectors are scale-normalised first, which is what lets one
    codebook serve a whole tensor -- the same trick the IQ formats use with their super-scale.
    """
    dev = device or W.device
    out, inp = W.shape
    assert inp % group == 0 and group % dim == 0
    w = W.reshape(-1, group).float().to(dev)
    s = w.abs().amax(-1, keepdim=True).clamp_min(1e-12).half().float()   # per-group scale
    v = (w / s).reshape(-1, dim)                                          # normalised sub-vectors
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(v.shape[0], generator=g)[:min(sample, v.shape[0])].to(dev)
    S = v[idx]
    if S.shape[0] < k:                       # a small tensor cannot support a large codebook; say so
        raise ValueError(f"k={k} exceeds {S.shape[0]} available sub-vectors -- the codebook would be "
                         f"larger than the data, which measures nothing")
    C = S[torch.randperm(S.shape[0], generator=g)[:k].to(dev)].clone()     # k-means++ is overkill here
    for _ in range(iters):
        a = (S @ C.T * 2 - (C * C).sum(-1)).argmax(-1)
        for _pass in range(1):
            Cn = torch.zeros_like(C)
            cnt = torch.zeros(k, device=dev)
            Cn.index_add_(0, a, S)
            cnt.index_add_(0, a, torch.ones_like(a, dtype=torch.float32))
            m = cnt > 0
            C[m] = Cn[m] / cnt[m].unsqueeze(-1)
    out_chunks = []
    CB = 1 << 20
    for i in range(0, v.shape[0], CB):
        vv = v[i:i + CB]
        j = (vv @ C.T * 2 - (C * C).sum(-1)).argmax(-1)
        out_chunks.append(C[j])
    rec = torch.cat(out_chunks, 0).reshape(-1, group) * s
    return rec.reshape(out, inp).to(W.dtype).to(W.device)


def rel_err(a, b):
    return ((a.float() - b.float()) ** 2).sum().div(
        (b.float() ** 2).sum().clamp_min(1e-12)).sqrt().item()


# ═══════════════ C. NON-UNIFORM SCALAR (companded level sets) ═══════════════
# Omitted from the first pass entirely, despite IQ4_NL and NF4 both shipping. A uniform grid is only
# MSE-optimal for a uniform source; LLM weights are approximately Gaussian, so equal-probability
# (quantile) levels beat equal-spaced ones at the same bit count for free.

def nf_levels(bits, device="cpu"):
    """NormalFloat-style levels: quantiles of a standard normal, normalised to [-1,1] (QLoRA NF4)."""
    n = 2 ** int(round(bits))
    from math import erf, sqrt
    # inverse CDF by bisection -- avoids a scipy dependency
    def ppf(p):
        lo, hi = -8.0, 8.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if 0.5 * (1 + erf(mid / sqrt(2))) < p:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    qs = [(i + 0.5) / n for i in range(n)]
    v = torch.tensor([ppf(p) for p in qs], device=device)
    return v / v.abs().max()


IQ4NL_LEVELS = torch.tensor(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=torch.float32) / 127.0


@torch.no_grad()
def q_nonuniform(W, group, levels=None, bits=4, n_scan=24):
    """Fixed non-uniform codebook shared by all groups, one scale per group. bpw = log2(len)+16/g."""
    dev = W.device
    lv = (IQ4NL_LEVELS.to(dev) if levels is None else levels.to(dev))
    out, inp = W.shape
    w = W.reshape(-1, group).float()
    amax = w.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    best = None
    for f in torch.linspace(0.5, 1.2, n_scan, device=dev):
        s = (amax * f).half().float()
        q = (w / s).unsqueeze(-1)
        j = (q - lv.view(1, 1, -1)).abs().argmin(-1)
        r = lv[j] * s
        e = ((r - w) ** 2).sum(-1, keepdim=True)
        best = (e, r) if best is None else (
            torch.where(e < best[0], e, best[0]), torch.where(e < best[0], r, best[1]))
    return best[1].reshape(out, inp).to(W.dtype)


# ═══════════════ D. MICRO-FLOAT / BLOCK FLOATING POINT ═══════════════
# MXFP4 / NVFP4 / OCP-MX. A per-weight (sign, exponent, mantissa) with one shared block exponent.
# Deployed at scale on current NVIDIA silicon, and absent from the first pass.

def fp_levels(e_bits, m_bits, device="cpu"):
    """All representable magnitudes of a tiny float, normalised to max=1. E2M1 -> FP4."""
    vals = {0.0}
    bias = 2 ** (e_bits - 1) - 1
    for e in range(0, 2 ** e_bits):
        for m in range(0, 2 ** m_bits):
            if e == 0:
                v = (m / 2 ** m_bits) * 2.0 ** (1 - bias)        # subnormal
            else:
                v = (1 + m / 2 ** m_bits) * 2.0 ** (e - bias)
            vals.add(v)
    v = sorted(vals)
    t = torch.tensor(v + [-x for x in v if x > 0], device=device).unique()
    return t / t.abs().max()


@torch.no_grad()
def q_microfloat(W, group=32, e_bits=2, m_bits=1, n_scan=24):
    """Block floating point: FP(e,m) per weight + one shared scale per block. bpw = 1+e+m + 8/g."""
    return q_nonuniform(W, group, levels=fp_levels(e_bits, m_bits, W.device), n_scan=n_scan)


def bpw_microfloat(e_bits, m_bits, group, scale_bits=8):
    return (1 + e_bits + m_bits) + scale_bits / group


# ═══════════════ F. LATTICE (structured VQ, no codebook table) ═══════════════

@torch.no_grad()
def q_lattice_e8(W, group=256, scale_bits=16, n_scan=16, return_bpw=False):
    """E8 lattice quantization, as used by QuIP#.

    E8 = D8 union (D8 + 1/2), where D8 is the set of integer 8-vectors with even coordinate sum.
    Decoding is a closed form -- round to D8, round to D8+1/2, keep the nearer -- so unlike family E
    it needs no codebook table at all. That is the structural distinction, and it is why a lattice
    can be fast where an arbitrary codebook is a memory-indirection problem.
    """
    dev = W.device
    out, inp = W.shape
    assert inp % group == 0 and group % 8 == 0
    w = W.reshape(-1, group).float()
    s0 = w.abs().amax(-1, keepdim=True).clamp_min(1e-12)

    def round_d8(x):
        r = torch.round(x)
        # enforce even coordinate sum: flip the coordinate with the largest rounding residual
        bad = (r.sum(-1) % 2 != 0)
        if bad.any():
            d = (x - r).abs()
            j = d.argmax(-1)
            adj = torch.where((x - r).gather(-1, j.unsqueeze(-1)) > 0,
                              torch.ones_like(j, dtype=x.dtype).unsqueeze(-1),
                              -torch.ones_like(j, dtype=x.dtype).unsqueeze(-1))
            r = r.clone()
            r[bad] = r[bad].scatter_add(-1, j[bad].unsqueeze(-1), adj[bad])
        return r

    best = None
    for f in torch.linspace(0.4, 1.6, n_scan, device=dev):
        s = (s0 * f).half().float()
        v = (w / s).reshape(-1, 8)
        a = round_d8(v)
        b = round_d8(v - 0.5) + 0.5
        pick = ((v - a) ** 2).sum(-1) <= ((v - b) ** 2).sum(-1)
        q = torch.where(pick.unsqueeze(-1), a, b)
        r = (q.reshape(-1, group) * s)
        e = ((r - w) ** 2).sum(-1, keepdim=True)
        if best is None or True:
            if best is None:
                best, bq = (e, r), q
            else:
                m = e < best[0]
                best = (torch.where(m, e, best[0]), torch.where(m, r, best[1]))
                bq = torch.where(m.reshape(-1, 1).expand(-1, group).reshape(-1, 8), q, bq)
    rec = best[1].reshape(out, inp).to(W.dtype)
    if not return_bpw:
        return rec
    # HONEST RATE. Rounding to the infinite E8 lattice has no bounded index, so quoting it at a
    # nominal bpw would credit the family with an error achievable only at unbounded rate. Count the
    # distinct lattice points actually used and charge log2 of that per 8 weights -- the smallest
    # codebook that could have produced this reconstruction.
    pts = torch.unique(bq * 2, dim=0)                 # x2 makes the half-integer coset integral
    idx_bits = float(np.log2(max(pts.shape[0], 2))) / 8.0
    return rec, idx_bits + scale_bits / group


# ═══════════════ K. ENTROPY CODING — a RATE transform, not a distortion one ═══════════════

def entropy_bpw(codes, n_levels):
    """Empirical symbol entropy in bits/weight.

    Entropy coding leaves the reconstruction UNCHANGED and only shrinks the rate, so on a
    rate-distortion plot it moves a family horizontally left at identical error. Every family
    therefore has an entropy-coded twin that dominates it on pure R-D and is excluded by
    deployability alone -- which is why the R-D frontier and the DEPLOYABLE frontier are two
    different frontiers, and why the gap between them is the real price of the deployment constraint.
    """
    c = codes.flatten().long()
    cnt = torch.bincount(c - c.min(), minlength=n_levels).float()
    p = cnt / cnt.sum().clamp_min(1)
    p = p[p > 0]
    return float(-(p * p.log2()).sum())
