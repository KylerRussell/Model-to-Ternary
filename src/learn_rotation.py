#!/usr/bin/env python
"""Learn a foldable residual-stream rotation R1 (SpinQuant-style) that beats the fixed Hadamard at
ternary quantization, by minimizing the Hessian-weighted ternary OUTPUT error under rotation.

Why this and not the assignment levers (SSR/CDQuant/act-order, all washes): the QuaRot rotation already
homogenizes per-channel salience, so it neutralizes assignment-order tricks. A LEARNED rotation instead
attacks the irreducible DIRECTIONAL residual error — it rotates the basis so the ternary error lands in
less-sensitive directions of the activation Hessian. R1 stays exactly orthogonal (foldable, zero runtime
cost) via Cayley SGD on the Stiefel manifold; we init from the Hadamard so the learned R1 is ≥ Hadamard
on the objective by construction (start there, descend).

Objective (one shared R1 over the residual stream):
  reading projections (q/k/v/gate/up/in_proj_*, input-rotated W'=W·Rᵀ, Hessian over the rotated input):
       ‖X·Rᵀ·(W·Rᵀ − tern(W·Rᵀ))ᵀ‖²  =  tr( D · (R·H·Rᵀ) · Dᵀ ),  D = W·Rᵀ − tern(W·Rᵀ)
  writing projections (o_proj/down_proj/out_proj, output-rotated W'=R·W, input Hessian un-rotated):
       ‖X·(R·W − tern(R·W))ᵀ‖²        =  tr( D · H · Dᵀ ),         D = R·W − tern(R·W)
Ternary is via STE (per-256-block MSE-optimal scale), so gradients flow to R; Cayley keeps R orthogonal.

  python learn_rotation.py --self-test       # validate the optimizer on synthetic data
"""
import argparse, math, torch
import torch.nn.functional as F


def _block_scale(W1):
    """Per-row MSE-optimal ternary scale for a [out, block] tile (matches block_ap_recovery)."""
    m = W1.abs().mean(1, keepdim=True).clamp_min(1e-8)
    cands = torch.linspace(0.6, 2.4, 19, device=W1.device).view(1, -1, 1)
    s = m.unsqueeze(1) * cands
    q = torch.round((W1.unsqueeze(1) / s).clamp(-1, 1)) * s
    mse = ((W1.unsqueeze(1) - q) ** 2).mean(-1)
    return s[torch.arange(W1.shape[0]), mse.argmin(1), 0]


def ternary_ste(W, block_size):
    """Differentiable ternary dequant with per-256-block MSE-optimal scale; STE through round. The
    scale SEARCH is detached (computed without grad) — it keeps the autograd graph tiny (no [out,19,256]
    candidate tensors retained) and the dominant gradient still flows to R through the round-STE."""
    out, inp = W.shape
    flat = W.reshape(out * (inp // block_size), block_size)
    with torch.no_grad():
        s = _block_scale(flat).clamp_min(1e-8).unsqueeze(1)        # [out*nb, 1] (detached)
    ws = (flat / s).clamp(-1, 1)
    q = ws + (torch.round(ws) - ws).detach()                      # STE
    return (q * s).reshape(out, inp)


def _term(R, W, H, kind, block_size, objective="hessian"):
    """Ternary error of one linear under rotation R (differentiable in R). objective='hessian' =
    activation-weighted OUTPUT error (A2 arm); objective='trimodal' = UNWEIGHTED weight-space ternary
    distortion (B4 arm — minimizing distance to {−s,0,+s} IS maximizing histogram tri-modality)."""
    Wp = (W @ R.t()) if kind == "read" else (R @ W)
    D = Wp - ternary_ste(Wp, block_size)
    if objective == "trimodal":
        return (D * D).sum()
    if kind == "read":                                            # input rotation: Hessian rotates too
        HR = R @ H @ R.t()
        return ((D @ HR) * D).sum()
    return ((D @ H) * D).sum()                                    # write: input Hessian un-rotated


def optimize_rotation(linears, dim, R_init, block_size=256, steps=200, lr=1e-3, log=print,
                      objective="hessian"):
    """Cayley SGD on the Stiefel manifold. `linears` = list of (W[out,inp], H[inp,inp], kind∈{read,write}).
    Returns the learned orthogonal R [dim,dim] (and the objective trace). Starts at R_init (Hadamard)."""
    dev = R_init.device
    R = R_init.clone().float()
    I = torch.eye(dim, device=dev)

    def obj_grad(Rc, want_grad):
        """Sum the per-linear terms; if want_grad, accumulate dObj/dRc ONE term at a time (peak memory
        = a single term's graph, not all ~186 at once). (W,H) may live on CPU (the 9216² write-side
        Hessians don't all fit on 24GB) — each term is STREAMED to the GPU and freed. Returns
        (objective_value, grad_or_None)."""
        if not want_grad:
            with torch.no_grad():
                tot = 0.0
                for (W, H, k) in linears:
                    tot += _term(Rc, W.to(dev, torch.float32), H.to(dev, torch.float32),
                                 k, block_size, objective).item()
                return tot, None
        Rg = Rc.detach().clone().requires_grad_(True)
        tot = 0.0
        for (W, H, k) in linears:
            term = _term(Rg, W.to(dev, torch.float32), H.to(dev, torch.float32), k, block_size, objective)
            term.backward()                                       # accumulates into Rg.grad, frees graph
            tot += term.item()
        return tot, Rg.grad

    base, _ = obj_grad(R, False)
    best, bestR = base, R.clone()
    for t in range(steps):
        _, g = obj_grad(R, True)
        with torch.no_grad():
            A = g @ R.t() - R @ g.t()                             # skew-symmetric tangent direction
            A = A / (A.norm() + 1e-12) * min(A.norm(), 1.0)       # clip step magnitude for stability
            # Cayley retraction: R ← (I + lr/2·A)⁻¹ (I − lr/2·A) R   (orthogonality-preserving)
            half = (lr / 2) * A
            R = torch.linalg.solve(I + half, (I - half) @ R)
            fv, _ = obj_grad(R, False)
            if fv < best:
                best, bestR = fv, R.clone()
            if t % 20 == 0 or t == steps - 1:
                ortho = (R @ R.t() - I).abs().max().item()
                log(f"   step {t:4d}  obj {fv:.6e}  (init {base:.6e}, {100*(1-fv/base):+.2f}%)  ‖RRᵀ−I‖={ortho:.1e}")
    log(f"   best obj {best:.6e}  vs Hadamard-init {base:.6e}  → {100*(1-best/base):+.2f}%")
    return bestR, base, best


def _self_test():
    torch.manual_seed(0)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    dim, out, bs = 512, 256, 256
    # synthetic residual-reading linears sharing one rotation; anisotropic Hessian (where rotation matters)
    from hadamard import hadamard_transform
    H_init = hadamard_transform(torch.eye(dim, device=dev))        # the fixed-Hadamard init (orthogonal)
    lins = []
    for _ in range(6):
        T = 2048
        X = torch.randn(T, dim, device=dev)
        X[:, :20] *= 6.0                                          # outlier channels (anisotropy)
        H = X.t() @ X
        W = torch.randn(out, dim, device=dev) * 0.05
        lins.append((W, H, "read"))
    R, base, best = optimize_rotation(lins, dim, H_init, block_size=bs, steps=120, lr=2e-3)
    ortho = (R @ R.t() - torch.eye(dim, device=dev)).abs().max().item()
    print(f"\n  orthogonal: ‖RRᵀ−I‖ = {ortho:.2e}  ({'OK' if ortho < 1e-3 else 'FAIL'})")
    print(f"  learned R beats Hadamard init on objective: {best < base}  ({100*(1-best/base):+.2f}%)")


_READ = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj",
         "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "in_proj_qkvz")
_WRITE = ("o_proj", "out_proj", "down_proj")


def _kind(name):
    if any(name.endswith(p) or name.endswith(p + ".weight") or f".{p}" in name for p in _WRITE):
        return "write"
    if any(name.endswith(p) or name.endswith(p + ".weight") or f".{p}" in name for p in _READ):
        return "read"
    return None


def capture_and_learn(args):
    import json
    from transformers import AutoModelForCausalLM
    from hadamard import hadamard_transform
    dev = "cuda:0"
    print(f"loading ORIGINAL (un-rotated) model {args.model} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
                                                 dtype=torch.bfloat16, low_cpu_mem_usage=True).to(dev).eval()
    # the rotation acts on the residual/hidden dim
    dim = model.config.text_config.hidden_size if hasattr(model.config, "text_config") else model.config.hidden_size
    import torch.nn as nn
    targets = {}                                                   # name -> (module, kind)
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            k = _kind(name)
            # only the residual-coupled projections (input or output == hidden dim)
            if k == "read" and mod.in_features == dim:
                targets[name] = (mod, "read")
            elif k == "write" and mod.out_features == dim:
                targets[name] = (mod, "write")
    print(f"  rotation dim {dim}; {len(targets)} residual-coupled linears", flush=True)

    # CHUNKED Gram capture: the write-side Hessians (down_proj input = interm² ≈ 340MB fp32 each) do NOT
    # all fit on 24GB next to the model — capture ~8 layers' targets per pass and offload to CPU.
    import re as _re
    seqs = json.load(open(args.calib))[:args.nsamp]
    def layer_of(n):
        m = _re.search(r"layers\.(\d+)\.", n)
        return int(m.group(1)) if m else -1
    layer_ids = sorted({layer_of(n) for n in targets})
    CHUNK_LAYERS = 8
    grams_cpu = {}
    for c0 in range(0, len(layer_ids), CHUNK_LAYERS):
        chunk = set(layer_ids[c0:c0 + CHUNK_LAYERS])
        names = [n for n in targets if layer_of(n) in chunk]
        grams = {n: None for n in names}
        def mk(n):
            def hook(m, inp, out):
                X = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                g = grams[n]
                if g is None:
                    grams[n] = [torch.zeros(X.shape[1], X.shape[1], device=dev), 0]
                    g = grams[n]
                g[0].addmm_(X.t(), X); g[1] += X.shape[0]
            return hook
        hooks = [targets[n][0].register_forward_hook(mk(n)) for n in names]
        with torch.no_grad():
            for i, s in enumerate(seqs):
                model(torch.tensor(s[:args.seq], device=dev).unsqueeze(0), use_cache=False)
        for h in hooks:
            h.remove()
        for n in names:
            if grams[n] is not None and grams[n][1] > 0:
                grams_cpu[n] = grams[n][0].cpu()
        del grams
        torch.cuda.empty_cache()
        print(f"   grams: layers chunk {c0//CHUNK_LAYERS + 1}/{-(-len(layer_ids)//CHUNK_LAYERS)} done "
              f"({len(grams_cpu)} tensors on CPU)", flush=True)

    linears = []
    for n, (mod, kind) in targets.items():
        if n in grams_cpu:
            linears.append((mod.weight.data.float().cpu(), grams_cpu[n], kind))
    del model; torch.cuda.empty_cache()                            # free the model — only (W,H) needed now
    print(f"  built {len(linears)} (W,H,kind) terms; optimizing R (Hadamard init)...", flush=True)

    R_had = hadamard_transform(torch.eye(dim, device=dev))         # the current fixed-Hadamard rotation
    R, base, best = optimize_rotation(linears, dim, R_had, block_size=args.block_size,
                                      steps=args.steps, lr=args.lr, objective=args.objective)

    def stats(Rc):
        """Both fork statistics for the {μ,T}→KL regression, evaluated over all residual-coupled linears
        (CPU-stored (W,H) streamed to the GPU per term)."""
        with torch.no_grad():
            hess = trim = 0.0
            mus = []
            for (W, H, k) in linears:
                Wd, Hd = W.to(dev, torch.float32), H.to(dev, torch.float32)
                hess += float(_term(Rc, Wd, Hd, k, args.block_size, "hessian").item())
                trim += float(_term(Rc, Wd, Hd, k, args.block_size, "trimodal").item())
                Wp = (Wd @ Rc.t()) if k == "read" else (Rc @ Wd)
                n_, m_ = Wp.shape
                mus.append(float(Wp.abs().max() * (n_ * m_) ** 0.5 / Wp.norm().clamp_min(1e-12)))
                del Wd, Hd, Wp
            return {"obj_hessian": hess, "obj_trimodal": trim, "coherence_mu_mean": sum(mus) / len(mus)}
    st_had, st_R = stats(R_had), stats(R)
    print(f"  stats Hadamard : {st_had}")
    print(f"  stats learned  : {st_R}")
    if args.out:
        torch.save({"R": R.cpu(), "dim": dim, "objective": args.objective, "obj_init": base,
                    "obj_best": best, "stats_hadamard": st_had, "stats_learned": st_R}, args.out)
        print(f"saved learned rotation -> {args.out}  ({args.objective} objective {100*(1-best/base):+.2f}% vs Hadamard)")
    return R, base, best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model", help="ORIGINAL (un-rotated) model snapshot to learn the rotation for")
    ap.add_argument("--calib", help="calibration JSON (list of token-id lists)")
    ap.add_argument("--out", help="output .pt for the learned rotation")
    ap.add_argument("--nsamp", type=int, default=128, help="calib seqs for the Hessian (saturates ~128)")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--objective", choices=["hessian", "trimodal"], default="hessian",
                    help="A2 arm = hessian (activation-weighted ternary output error); "
                         "B4 arm = trimodal (unweighted weight-space ternary distortion).")
    args = ap.parse_args()
    if args.self_test:
        _self_test()
    elif args.model:
        capture_and_learn(args)
