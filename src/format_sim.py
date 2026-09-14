"""format_sim.py — head-to-head simulation of TQ1_64 / TQ1_0 / IQ1_S on the same weights.

THE QUESTION. `IQ1_S` is **1.5625 bpw** against TQ1_64's **1.7812** -- 12.3% smaller -- and carries
**g32** sub-block scales, FINER than our g64. Our entire published justification for TQ1_64 is that
scale granularity matters (79.31% / 80.62% / 82.06% eval2k agreement at g256 / g128 / g64). IQ1_S
goes further in exactly the direction we argued for, at lower cost. It pays for that by discarding
68.8% of the 8-dimensional ternary alphabet: it keeps 2048 of the 6561 possible vectors and
compensates with a one-bit +/-0.125 grid displacement.

    Does finer scaling on a RESTRICTED alphabet beat coarser scaling on the FULL alphabet?

We have argued one half of that and never tested the other. The format census asserts IQ1_S wins but
attaches no measurement, so it settles nothing.

FAIRNESS IS THE WHOLE EXPERIMENT. Every arm here gets:
  * the same weights, from the same rotated checkpoint;
  * the same optional importance weighting (`imp`), so no arm is scored under a different objective;
  * a genuine search over its own format's free parameters -- a scale sweep for the ternary formats,
    a joint (super-scale, 3-bit sub-scale, delta sign, grid index) search for IQ1_S.
Handicapping IQ1_S -- by skipping its delta, its sub-scales, or its search -- would produce a
flattering number and destroy the only reason to run this.

SCOPE. This is round-to-nearest post-training quantization for ALL arms. It measures FORMAT CAPACITY,
not our trained pipeline: our deployed TQ1_64 model is assignment-trained, and IQ1_S has no trained
counterpart here. Comparing a trained arm against an RTN arm would be the mismatched-baseline error
we criticise in others. A format that wins at RTN is the one worth building a trainer for.

The IQ1_S grid and dequantization come from the installed `gguf` package -- the REAL `iq1s_grid` and
the real formula `x = d * (2*s + 1) * (grid + delta)` -- not a reimplementation from the paper.
`verify_iq1s()` packs our chosen parameters into real 50-byte blocks and checks our reconstruction
against `gguf`'s own dequantizer, so a divergence between spec and simulation fails loudly.
"""
import numpy as np
import torch

from gguf.quants import IQ1_S as _IQ1S

QK_K = 256
_GRID = None


def iq1s_grid(device="cpu"):
    """The real 2048 x 8 ternary grid from llama.cpp, via gguf."""
    global _GRID
    if _GRID is None:
        _IQ1S.init_grid()
        # gguf stores it as (1, 1, 2048, 8); we want a plain [2048, 8] codebook
        _GRID = torch.tensor(np.asarray(_IQ1S.grid, dtype=np.float32)).reshape(-1, 8)
    return _GRID.to(device)


# ─────────────────────────── ternary formats ───────────────────────────

def _ternary_best_scale(w, imp, n_scan=24):
    """Per-block ternary RTN with a scale SEARCH, not a closed-form heuristic.

    w, imp: [n_blocks, block]. Returns the dequantized weights.

    absmean (the BitNet rule) and absmax are both defensible closed forms, and neither is optimal --
    the ternary optimum sits near 0.5-0.7 x absmax and moves with the weight distribution. QUASAR
    failed in this project for exactly this reason (a clip grid parameterised as f*amax with f<=1
    can never reach it). Scanning gives every ternary arm the same quality of search that the IQ1_S
    arm gets over its own parameters.
    """
    amax = w.abs().amax(-1, keepdim=True).clamp_min(1e-12)
    best_err = None
    best = None
    for f in torch.linspace(0.3, 1.0, n_scan, device=w.device):
        s = amax * f
        q = torch.round(w / s).clamp_(-1, 1)
        r = q * s
        err = (imp * (r - w) ** 2).sum(-1, keepdim=True)
        if best_err is None:
            best_err, best = err, r
        else:
            m = err < best_err
            best_err = torch.where(m, err, best_err)
            best = torch.where(m, r, best)
    return best


def quant_ternary(W, imp=None, group=64):
    """TQ1_64 (group=64) or TQ1_0 (group=256): symmetric ternary, one positive scale per group."""
    out, inp = W.shape
    assert inp % group == 0, f"{inp} not divisible by group {group}"
    w = W.reshape(-1, group).float()
    i = (torch.ones_like(w) if imp is None
         else imp.reshape(1, -1).expand(out, -1).reshape(-1, group).float())
    return _ternary_best_scale(w, i).reshape(out, inp).to(W.dtype)


# ─────────────────────────── IQ1_S ───────────────────────────

@torch.no_grad()
def _iq1s_chunk(w, i_full, grid, gsq, delta_vals, d_scan):
    """Search one chunk of 256-weight blocks. w, i_full: [nb, 256]."""
    dev = w.device
    nb = w.shape[0]
    ws = w.reshape(nb, 8, 4, 8)
    isb = i_full.reshape(nb, 8, 4, 8)
    wg = ws.reshape(-1, 8)
    ig = isb.reshape(-1, 8)
    iwg = ig * wg
    # j-dependence of the weighted error is two GEMMs, reusable across every (d, s):
    lin0 = iwg @ grid.T                                   # [N, 2048]
    rs = iwg.sum(-1, keepdim=True)                        # [N, 1]
    quad = {dv: ig @ gsq[dv].T for dv in delta_vals}      # [N, 2048]

    amax_sb = ws.abs().amax(-1).amax(-1)
    d0 = (amax_sb.amax(-1, keepdim=True) / 15.0).clamp_min(1e-12)
    best_total = torch.full((nb,), float("inf"), device=dev)
    best_rec = torch.zeros_like(w)
    best_p = {"d": torch.zeros(nb, device=dev), "s": torch.zeros(nb, 8, dtype=torch.long, device=dev),
              "dsign": torch.zeros(nb, 8, dtype=torch.long, device=dev),
              "idx": torch.zeros(nb, 8, 4, dtype=torch.long, device=dev)}
    for fi in range(d_scan):
        f = 0.45 + 0.75 * fi / max(d_scan - 1, 1)
        # the block stores d as fp16, so the SIMULATION must round it too -- otherwise we
        # credit IQ1_S with super-scale precision the format does not have.
        d = (d0 * f).half().float()
        sb_err = torch.full((nb, 8), float("inf"), device=dev)
        sb_rec = torch.zeros(nb, 8, 4, 8, device=dev)
        sb_s = torch.zeros(nb, 8, dtype=torch.long, device=dev)
        sb_ds = torch.zeros(nb, 8, dtype=torch.long, device=dev)
        sb_idx = torch.zeros(nb, 8, 4, dtype=torch.long, device=dev)
        for s in range(8):
            dl = (d * (2 * s + 1))                                  # [nb,1]
            dlf = dl.reshape(nb, 1, 1, 1).expand(nb, 8, 4, 1).reshape(-1, 1)
            for di, dv in enumerate(delta_vals):
                score = 2 * dlf * (lin0 + dv * rs) - (dlf ** 2) * quad[dv]
                jbest = score.argmax(-1)
                rec = dlf * (grid[jbest] + dv)
                err = (ig * (rec - wg) ** 2).sum(-1).reshape(nb, 8, 4).sum(-1)
                m = err < sb_err
                sb_err = torch.where(m, err, sb_err)
                sb_rec = torch.where(m.unsqueeze(-1).unsqueeze(-1), rec.reshape(nb, 8, 4, 8), sb_rec)
                sb_s = torch.where(m, torch.full_like(sb_s, s), sb_s)
                sb_ds = torch.where(m, torch.full_like(sb_ds, di), sb_ds)
                sb_idx = torch.where(m.unsqueeze(-1), jbest.reshape(nb, 8, 4), sb_idx)
                del score, jbest, rec, err
        tot = sb_err.sum(-1)
        m = tot < best_total
        best_total = torch.where(m, tot, best_total)
        best_rec = torch.where(m.unsqueeze(-1), sb_rec.reshape(nb, QK_K), best_rec)
        best_p["d"] = torch.where(m, d.squeeze(-1), best_p["d"])
        best_p["s"] = torch.where(m.unsqueeze(-1), sb_s, best_p["s"])
        best_p["dsign"] = torch.where(m.unsqueeze(-1), sb_ds, best_p["dsign"])
        best_p["idx"] = torch.where(m.unsqueeze(-1).unsqueeze(-1), sb_idx, best_p["idx"])
    return best_rec, best_p


@torch.no_grad()
def quant_iq1s(W, imp=None, d_scan=12, chunk=4096, device=None, return_params=False):
    """IQ1_S: 2048-entry 8-D ternary grid, g32 3-bit sub-scales, fp16 super-scale, +/-0.125 delta.

    Reconstruction (llama.cpp / gguf):   x = d * (2*s + 1) * (grid[j] + delta)
    with s in [0,7] and delta in {+0.125, -0.125} per 32-weight sub-block, and j per 8 weights.

    Search: for each candidate super-scale d and each (s, delta), the best grid index per group is an
    exact weighted nearest neighbour. Expanding the weighted error

        sum_k imp_k (w_k - dl*(g_k + delta))^2
          = sum_k imp_k w_k^2 - 2*dl*sum_k imp_k w_k (g_k+delta) + dl^2 * sum_k imp_k (g_k+delta)^2

    makes the j-dependence two matmuls, so the 2048-way argmin is GEMMs rather than a codebook loop.

    CHUNKED over blocks: the [N, 2048] score matrix is 606 GiB for lm_head taken whole.
    """
    dev = device or W.device
    out, inp = W.shape
    assert inp % QK_K == 0, f"{inp} not divisible by {QK_K}"
    grid = iq1s_grid(dev)
    delta_vals = (float(_IQ1S.delta), -float(_IQ1S.delta))
    gsq = {dv: ((grid + dv) ** 2) for dv in delta_vals}

    w_all = W.reshape(-1, QK_K).float()
    nb_total = w_all.shape[0]
    recs, ps = [], []
    for a in range(0, nb_total, chunk):
        b = min(a + chunk, nb_total)
        wc = w_all[a:b].to(dev)
        if imp is None:
            ic = torch.ones_like(wc)
        else:
            rows = torch.arange(a, b, device="cpu")
            ic = imp.reshape(1, -1).expand(out, -1).reshape(-1, QK_K)[rows].to(dev).float()
        r, p = _iq1s_chunk(wc, ic, grid, gsq, delta_vals, d_scan)
        recs.append(r.cpu())
        if return_params:
            ps.append(p)
        del wc, ic, r
        if dev != "cpu":
            torch.cuda.empty_cache()
    rec = torch.cat(recs, 0).reshape(out, inp).to(W.dtype).to(W.device)
    if not return_params:
        return rec
    merged = {k: torch.cat([p[k].cpu() for p in ps], 0) for k in ps[0]}
    return rec, merged


def pack_iq1s(d, s, dsign, idx):
    """Pack chosen parameters into real 50-byte IQ1_S blocks, so gguf can dequantize them.

    Layout: 2 B fp16 super-scale | 32 B low byte of each 11-bit grid index | 16 B of eight uint16,
    each holding four 3-bit index-high fields, a 3-bit sub-scale at bits 12-14, and the delta sign
    at bit 15.
    """
    nb = d.shape[0]
    dh = d.cpu().numpy().astype(np.float16).view(np.uint8).reshape(nb, 2)
    idx_np = idx.cpu().numpy().astype(np.uint16)                     # [nb, 8, 4]
    qs = (idx_np & 0xFF).astype(np.uint8).reshape(nb, 32)
    hi = (idx_np >> 8) & 0x7                                         # [nb,8,4]
    qh = (hi[..., 0] | (hi[..., 1] << 3) | (hi[..., 2] << 6) | (hi[..., 3] << 9)).astype(np.uint16)
    qh |= (s.cpu().numpy().astype(np.uint16) & 0x7) << 12
    qh |= (dsign.cpu().numpy().astype(np.uint16) & 1) << 15
    return np.concatenate([dh, qs, qh.view(np.uint8).reshape(nb, 16)], axis=1)


def verify_iq1s(n=64, seed=0, device="cpu"):
    """Our reconstruction must equal gguf's own dequantizer on the same packed bytes.

    This is the load-bearing check: it proves the simulation implements IQ1_S as llama.cpp defines
    it, rather than something that merely resembles the paper description.
    """
    torch.manual_seed(seed)
    W = (torch.randn(n, QK_K, device=device) * 0.02)
    rec, p = quant_iq1s(W, return_params=True, device=device)
    blocks = pack_iq1s(p["d"], p["s"], p["dsign"], p["idx"])
    ref = _IQ1S.dequantize_blocks(blocks).reshape(n, QK_K)
    ours = rec.cpu().numpy().astype(np.float32)
    err = np.abs(ref - ours).max()
    assert err == 0.0, f"IQ1_S simulation does NOT match gguf dequantize: max|delta|={err:.3e}"
    print(f"verify_iq1s: reconstruction matches gguf.dequantize_blocks exactly "
          f"({n} blocks, max|delta|=0)")
    return True


def rel_err(a, b, imp=None):
    """Weighted relative reconstruction error, the metric all arms are scored on."""
    d = (a.float() - b.float()) ** 2
    r = b.float() ** 2
    if imp is not None:
        w = imp.reshape(1, -1).to(d.device)
        d, r = d * w, r * w
    return (d.sum() / r.sum().clamp_min(1e-12)).sqrt().item()
