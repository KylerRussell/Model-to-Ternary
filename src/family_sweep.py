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
