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


# ═══════════════ H. ADDITIVE MULTI-PLANE  (PTQTP, DB-LLM, BiLLM, E2M-ATQ) ═══════════════

@torch.no_grad()
def q_multiplane(W, group=64, planes=2, bits=LOG2_3, iters=3, n_scan=24):
    """W ~ sum_p alpha_p * T_p, each T_p a discrete plane with its own per-group scale.

    Fitted by residual refinement then alternating re-fit, which is the fair version: a single greedy
    pass understates the family. bpw = planes*(bits + 16/group) -- BOTH planes and BOTH scale sets are
    charged, because the accounting failure this paper documents is exactly hiding a second plane.
    """
    out, inp = W.shape
    w = W.reshape(-1, group).float()
    Ts, As = [], []
    R = w.clone()
    binary = abs(bits - 1.0) < 1e-6          # L = 2^(b-1)-1 is 0 at b=1; binary is {-1,+1}, not a
    L = 1 if (binary or abs(bits - LOG2_3) < 1e-6) else (2 ** (int(round(bits)) - 1) - 1)
    for _ in range(planes):
        amax = R.abs().amax(-1, keepdim=True).clamp_min(1e-12)
        best = None
        for f in torch.linspace(0.3, 1.0, n_scan, device=w.device):
            s = (amax * f / L).half().float().clamp_min(1e-12)
            t = (torch.where(R >= 0, 1.0, -1.0) if binary
                 else torch.round(R / s).clamp(-L, L))
            e = ((t * s - R) ** 2).sum(-1, keepdim=True)
            best = (e, t, s) if best is None else (
                torch.where(e < best[0], e, best[0]),
                torch.where(e < best[0], t, best[1]), torch.where(e < best[0], s, best[2]))
        Ts.append(best[1]); As.append(best[2])
        R = R - best[1] * best[2]
    for _ in range(iters):                      # alternating least squares on the scales
        for p in range(planes):
            Rp = w - sum(Ts[q] * As[q] for q in range(planes) if q != p)
            num = (Ts[p] * Rp).sum(-1, keepdim=True)
            den = (Ts[p] * Ts[p]).sum(-1, keepdim=True).clamp_min(1e-12)
            As[p] = (num / den).half().float()
    rec = sum(Ts[p] * As[p] for p in range(planes))
    return rec.reshape(out, inp).to(W.dtype)


def bpw_multiplane(group, planes=2, bits=LOG2_3, scale_bits=16):
    return planes * (bits + scale_bits / group)


# ═══════════════ I. LOW-RANK + QUANTIZED RESIDUAL  (OneBit SVID, ExTernD) ═══════════════

@torch.no_grad()
def q_lowrank(W, rank=32, group=64, bits=LOG2_3, base_fp16=True):
    """W ~ A@B + Q(W - A@B). The low-rank term stays fp16; only the residual is quantized."""
    out, inp = W.shape
    Wf = W.float()
    U, S, Vh = torch.linalg.svd(Wf, full_matrices=False)
    A = U[:, :rank] * S[:rank]
    B = Vh[:rank]
    if base_fp16:
        A, B = A.half().float(), B.half().float()
    R = Wf - A @ B
    return (A @ B + q_sym(R, bits, group)).to(W.dtype)


def bpw_lowrank(shape, rank, group, bits=LOG2_3, scale_bits=16, lr_bits=16):
    out, inp = shape
    lr = rank * (out + inp) * lr_bits / (out * inp)      # the factors are NOT free
    return bits + scale_bits / group + lr


# ═══════════════ J. SPARSE-DENSE HYBRID  (SpQR, SqueezeLLM, PB-LLM, FlashQuant) ═══════════════

@torch.no_grad()
def q_sparse_hybrid(W, frac=0.01, group=64, bits=LOG2_3):
    """Top-|w| fraction kept in fp16, remainder quantized. Outliers are EXCLUDED from the scale fit,
    which is the point of the family -- one outlier otherwise dominates its group's range."""
    out, inp = W.shape
    Wf = W.float()
    k = max(int(frac * Wf.numel()), 1)
    thr = Wf.abs().flatten().kthvalue(Wf.numel() - k + 1).values
    mask = Wf.abs() >= thr
    dense = torch.where(mask, torch.zeros_like(Wf), Wf)
    rec = q_sym(dense, bits, group).float()
    return torch.where(mask, Wf, rec).to(W.dtype)


def bpw_sparse_hybrid(frac, group, bits=LOG2_3, scale_bits=16, val_bits=16, idx_bits=16):
    """Charged CSR-style: every retained outlier costs a value AND an index. Storing the fp16 value
    and forgetting the index is the accounting error that makes this family look cheap."""
    return bits + scale_bits / group + frac * (val_bits + idx_bits)


# ═══════════════ G. TRELLIS-CODED QUANTIZATION  (QTIP) ═══════════════

@torch.no_grad()
def q_trellis(W, k_bits=2, L=10, group=256, device=None, chunk=4096, n_scan=9):
    """QTIP-style bitshift trellis with exact Viterbi decoding.

    State is an L-bit window. Emitting a symbol c in [0,2^k) shifts it: s' = ((s<<k)|c) & (2^L-1).
    The reconstruction value of a state is a deterministic pseudorandom Gaussian of that state, so the
    effective codebook is continuous and COSTS NOTHING TO STORE -- that is the structural trick, and
    why a trellis is not merely a codebook with extra steps.

    Because consecutive weights share overlapping state, the code is a PATH, not per-weight symbols;
    the optimal path is found by Viterbi, not by rounding. Rate is exactly k_bits/weight + scale.
    """
    dev = device or W.device
    S = 1 << L
    # value table: deterministic hash of state -> standard normal quantile
    st = torch.arange(S, device=dev, dtype=torch.int64)
    h = (st * 2654435761) % (1 << 31)
    u = (h.float() + 0.5) / float(1 << 31)
    vals = torch.erfinv(2 * u - 1) * float(np.sqrt(2))          # ~N(0,1), fixed, not stored
    nsym = 1 << k_bits
    # predecessors: s' has low k bits = symbol, high L-k bits = low L-k bits of s
    sp = torch.arange(S, device=dev, dtype=torch.int64)
    pred = ((torch.arange(nsym, device=dev, dtype=torch.int64).view(-1, 1) << (L - k_bits))
            | (sp >> k_bits).view(1, -1))                        # [nsym, S]

    out, inp = W.shape
    w_all = W.reshape(-1, group).float()
    recs = []
    for a in range(0, w_all.shape[0], chunk):
        w = w_all[a:b] if False else w_all[a:min(a + chunk, w_all.shape[0])].to(dev)
        n = w.shape[0]
        sd = w.std(-1, keepdim=True).clamp_min(1e-12)
        best_rec, best_err = None, None
        for f in torch.linspace(0.6, 1.4, n_scan, device=dev):
            s = (sd * f).half().float()
            x = w / s
            cost = torch.zeros(n, S, device=dev)
            bp = torch.zeros(n, group, S, dtype=torch.int16, device=dev)
            for t in range(group):
                c = cost[:, pred.reshape(-1)].reshape(n, nsym, S)   # [n, nsym, S]
                mv, mi = c.min(1)
                cost = mv + (x[:, t:t + 1] - vals.view(1, S)) ** 2
                bp[:, t] = mi.to(torch.int16)
            end = cost.argmin(-1)
            rec = torch.zeros(n, group, device=dev)
            cur = end
            for t in range(group - 1, -1, -1):
                rec[:, t] = vals[cur]
                j = bp[:, t].gather(1, cur.unsqueeze(-1)).squeeze(-1).long()
                cur = ((j << (L - k_bits)) | (cur >> k_bits))
            rec = rec * s
            e = ((rec - w) ** 2).sum(-1, keepdim=True)
            if best_err is None:
                best_rec, best_err = rec, e
            else:
                m = e < best_err
                best_rec = torch.where(m, rec, best_rec)
                best_err = torch.where(m, e, best_err)
            del cost, bp
        recs.append(best_rec.cpu())
        del w
        if dev != "cpu":
            torch.cuda.empty_cache()
    return torch.cat(recs, 0).reshape(out, inp).to(W.dtype).to(W.device)


def bpw_trellis(k_bits, group, scale_bits=16):
    return k_bits + scale_bits / group


# ═══════════════ L. MIXED PRECISION / per-channel bit allocation  (EXL2, PTQ1.61, AWSRC) ═══════════

@torch.no_grad()
def q_mixed(W, b_lo=1.0, b_hi=LOG2_3, frac_hi=0.25, group=64, axis=0):
    """Allocate b_hi to the `frac_hi` most error-prone channels and b_lo to the rest.

    Sensitivity is measured, not guessed: quantize everything at b_lo, rank channels by the error
    they actually incur, and promote the worst. That is greedy-optimal for this objective and is what
    EXL2 does in spirit. Using a heuristic proxy (row norm, kurtosis) would understate the family,
    since the family's whole claim is that it spends bits where they are needed.
    """
    Wf = W.float()
    lo = q_sym(Wf, b_lo, group).float()
    err = ((lo - Wf) ** 2).sum(dim=1 - axis)                      # per-channel error at the low rate
    k = max(int(round(frac_hi * err.numel())), 1)
    idx = err.topk(k).indices
    hi = q_sym(Wf, b_hi, group).float()
    out = lo.clone()
    if axis == 0:
        out[idx, :] = hi[idx, :]
    else:
        out[:, idx] = hi[:, idx]
    return out.to(W.dtype)


def bpw_mixed(shape, b_lo, b_hi, frac_hi, group, axis=0, scale_bits=16):
    """Average rate, plus the per-channel allocation MAP, which is not free."""
    n_ch = shape[0] if axis == 0 else shape[1]
    per_ch = shape[1] if axis == 0 else shape[0]
    avg = frac_hi * b_hi + (1 - frac_hi) * b_lo
    map_bits = 1.0 / per_ch                                       # 1 bit per channel, amortised
    return avg + scale_bits / group + map_bits
