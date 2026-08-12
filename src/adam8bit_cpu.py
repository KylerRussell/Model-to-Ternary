"""Block-wise 8-bit Adam state for CPU-resident assignment latents.

WHY THIS EXISTS. Our STE assignment latents live in CPU RAM (`--latent-offload`), so bitsandbytes'
8-bit optimizers -- which are CUDA kernels -- cannot be dropped in. This is the same idea implemented
for the CPU path: store Adam's two moments as int8 codes + one fp32 absmax per block, cutting them from
8 bytes/latent to ~2.03 (int8 m + int8 v + 2 x fp32 per 4096-element block).

WHAT IT DOES **NOT** TOUCH. Only the OPTIMIZER STATE is quantized. The latent itself stays fp32, because
the STE gate `|L| < 1.5*s` decides flips from a latent's *distance to a bin boundary*, and bf16 latents
were measured to destroy that (held-out KL 0.9495 vs 0.6624) -- a floating exponent puts resolution at a
fixed FRACTION of |L| rather than a uniform step. The moments carry no boundary information, which is why
quantizing them is the low-risk lever and quantizing the latent is not.

MEMORY (per latent, replacing 4B exp_avg + 4B exp_avg_sq):
    int8 m + int8 v                     2.000 B
    2 x fp32 absmax per 4096-elem block 0.002 B
    ------------------------------------------
                                        2.002 B   => saves ~6 B/latent

The update runs CHUNKED so dequantisation never materialises a full-size fp32 temporary: a chunk of
CHUNK elements costs ~3*CHUNK*4 bytes of transient, independent of tensor size.
"""
from __future__ import annotations

import math
import torch

QBLOCK = 4096          # elements per absmax block; matches bitsandbytes' default
CHUNK = 1 << 22        # 4M elements/chunk => ~48MB transient regardless of latent size
_EPS = 1e-12


def _nblocks(n: int, block: int = QBLOCK) -> int:
    return (n + block - 1) // block


def quantize_blockwise(x: torch.Tensor, codes: torch.Tensor, absmax: torch.Tensor,
                       block: int = QBLOCK) -> None:
    """Signed linear absmax quantisation of `x` (1-D fp32) into int8 `codes` + per-block `absmax`.

    In-place into the provided buffers so the caller controls allocation. Symmetric/signed because
    exp_avg takes both signs; exp_avg_sq is non-negative but shares the path for simplicity (it costs
    one code point of resolution, far below the noise the servo already tolerates).
    """
    n = x.numel()
    nb = _nblocks(n, block)
    pad = nb * block - n
    v = torch.nn.functional.pad(x, (0, pad)) if pad else x
    v = v.view(nb, block)
    am = v.abs().amax(dim=1)
    absmax[:nb].copy_(am)
    scale = (am / 127.0).clamp_min(_EPS).unsqueeze(1)
    q = torch.round(v / scale).clamp_(-127, 127).to(torch.int8)
    codes[:n].copy_(q.view(-1)[:n])


def dequantize_blockwise(codes: torch.Tensor, absmax: torch.Tensor, out: torch.Tensor,
                         block: int = QBLOCK) -> None:
    """Inverse of `quantize_blockwise`, writing fp32 into `out` (same length as `codes`)."""
    n = out.numel()
    nb = _nblocks(n, block)
    pad = nb * block - n
    c = codes[:n]
    if pad:
        c = torch.nn.functional.pad(c, (0, pad))
    cf = c.view(nb, block).to(torch.float32)
    cf.mul_((absmax[:nb] / 127.0).clamp_min(_EPS).unsqueeze(1))
    out.copy_(cf.view(-1)[:n])


class Adam8bitState:
    """Quantised (exp_avg, exp_avg_sq) for ONE parameter tensor, plus the fp32 step counter."""

    __slots__ = ("n", "m_codes", "m_absmax", "v_codes", "v_absmax", "step")

    def __init__(self, numel: int):
        nb = _nblocks(numel)
        self.n = numel
        self.m_codes = torch.zeros(numel, dtype=torch.int8)
        self.m_absmax = torch.zeros(nb, dtype=torch.float32)
        self.v_codes = torch.zeros(numel, dtype=torch.int8)
        self.v_absmax = torch.zeros(nb, dtype=torch.float32)
        self.step = 0.0

    def bytes(self) -> int:
        return (self.m_codes.numel() + self.v_codes.numel()
                + 4 * (self.m_absmax.numel() + self.v_absmax.numel()))


def adam_step_8bit(param: torch.Tensor, grad: torch.Tensor, st: Adam8bitState,
                   lr: float, betas=(0.9, 0.999), eps: float = 1e-8,
                   scratch: dict | None = None) -> None:
    """Exact-Adam update with block-wise 8-bit moments, applied chunk-by-chunk in place.

    `scratch` is an optional dict reused across calls to avoid per-step allocation (the same pattern the
    fp32 path uses -- a full-size temporary per latent per step ratcheted RSS ~170MB/step and wedged the
    run in D state around step 100).
    """
    b1, b2 = betas
    st.step += 1.0
    t = st.step
    bc1 = 1.0 - b1 ** t
    bc2 = 1.0 - b2 ** t
    p_flat = param.view(-1)
    g_flat = grad.view(-1)
    if g_flat.dtype != torch.float32:
        g_flat = g_flat.float()

    if scratch is None:
        scratch = {}
    need = min(CHUNK, st.n)
    buf = scratch.get("buf")
    if buf is None or buf.shape[0] < 3 or buf.shape[1] < need:
        buf = torch.empty(3, need, dtype=torch.float32)
        scratch["buf"] = buf

    for lo in range(0, st.n, CHUNK):
        hi = min(lo + CHUNK, st.n)
        ln = hi - lo
        # chunk boundaries are multiples of CHUNK, and CHUNK % QBLOCK == 0, so blocks never straddle
        blo, bhi = lo // QBLOCK, _nblocks(hi)
        m = buf[0, :ln]
        v = buf[1, :ln]
        d = buf[2, :ln]
        dequantize_blockwise(st.m_codes[lo:hi], st.m_absmax[blo:bhi], m)
        dequantize_blockwise(st.v_codes[lo:hi], st.v_absmax[blo:bhi], v)

        g = g_flat[lo:hi]
        m.mul_(b1).add_(g, alpha=1.0 - b1)
        v.mul_(b2).addcmul_(g, g, value=1.0 - b2)

        d.copy_(v).sqrt_().div_(math.sqrt(bc2)).add_(eps)
        p_flat[lo:hi].addcdiv_(m, d, value=-lr / bc1)

        quantize_blockwise(m, st.m_codes[lo:hi], st.m_absmax[blo:bhi])
        quantize_blockwise(v, st.v_codes[lo:hi], st.v_absmax[blo:bhi])
