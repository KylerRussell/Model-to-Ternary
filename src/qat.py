#!/usr/bin/env python
"""Assignment-QAT (redesigned per the post-mortem): after Block-AP, retrain the ternary ASSIGNMENTS
with FROZEN per-256 scales, then hand off to the existing E2E-QP for the scale refit.

Why this differs from the QAT that failed before (and should not repeat its regressions):
  • FROZEN SCALES — the prior QAT's scales drifted up to 13% and carried ~82% of the damage (swap
    decomp: QAT-assignments + clean-scales = 3.7037 BEAT the baseline; the scales were the problem).
    Here scales are fixed buffers, get no gradient; only the latent weights (assignments) move.
  • PERPLEXITY-ALIGNED OBJECTIVE — the prior QAT used top-64 KL, which is anti-correlated with
    perplexity at 1.58-bit (tail collapse). Options here:
      --objective ce   : cross-entropy on the TRUE next-token targets = the actual perplexity metric
                         (global; can't tail-collapse; can move assignments across boundaries).
      --objective mse  : MSE of the student's final hidden vs the FP teacher's (feature matching;
                         needs a teacher cache built with --cache-hidden).
  • STE latent init = Block-AP's deployed ternary weight, so QAT STARTS exactly at Block-AP's good
    assignments (round(latent/scale) reproduces them) and can only search outward from there.

Deploys a standard HF model dir (hard ternary × frozen scale) → feed to E2E-QP exactly like Block-AP.
Gate on deployable held-out ppl, NOT the training loss.
"""
import argparse, gc, json, math, os
import torch
import torch.nn as nn
import torch.nn.functional as F

from e2e_qp_distill import (build_student, load_calib_batches, TernaryScaleLinear,
                            _shard_map, _get_tensor, extract_ternary_scale)
from config import BLOCK_SIZE


def gumbel_sigmoid(logit, tau, train=True):
    """Binary Gumbel-Softmax (concrete) sigmoid: differentiable relaxation of a Bernoulli. Adds
    logistic (Gumbel-difference) noise during training so marginal logits (≈0) flip stochastically
    while confident ones (|logit|≫0) stay; τ→0 hardens to the step function."""
    if train:
        u = torch.rand_like(logit).clamp_(1e-6, 1 - 1e-6)
        logit = logit + (torch.log(u) - torch.log1p(-u))          # + logistic noise
    return torch.sigmoid(logit / tau)


class GumbelTernaryLinear(nn.Module):
    """GSQ-style JOINT ternary via mask+sign Gumbel-Softmax (arXiv:2604.18556). Ternary {−1,0,+1} is
    factorized into two binary decisions: a NONZERO mask m and a SIGN — each a Gumbel-sigmoid over a
    learned logit (bf16, ~2 logits/weight). The logits init from the E2E assignment + noise, so the
    ARGMAX exactly preserves E2E's good assignments while the soft Gumbel lets MARGINAL weights (logit
    ≈0) flip and CONFIDENT ones (|logit|≫0) stay — the selectivity STE lacks, AND it dodges the
    bin-center/FP-init dilemma. The per-256 scale is jointly trained with LSQ-scaled gradient (×1/√256).
    τ anneals → near-0; deploy = hard argmax → exact ternary × fp16 scale (foldable, on-grid)."""
    tau = 1.0
    kappa = 1.0                                              # forward logit multiplier, annealed UP

    def __init__(self, tern, scale, block_size, alpha=3.0, sigma=0.1):
        super().__init__()
        nz = (tern != 0).float()
        pos = (tern > 0).float()
        # SMALL trainable logits (sign = E2E assignment, magnitude ~σ·α) so Lion's sign-based ~lr/step
        # updates can actually FLIP them in a few thousand steps; κ in the forward (not stored) sharpens.
        mask_logit = sigma * (alpha * (2 * nz - 1) + torch.randn_like(nz))     # argmax>0 ⇒ nonzero
        sign_logit = sigma * (alpha * (2 * pos - 1) + torch.randn_like(pos))   # argmax>0 ⇒ +1
        self.mask_logit = nn.Parameter(mask_logit.to(torch.bfloat16))
        self.sign_logit = nn.Parameter(sign_logit.to(torch.bfloat16))
        self.s = nn.Parameter(scale.float())
        self.block_size = block_size
        self.lsq_c = 1.0 / math.sqrt(block_size)
        self.out_features, self.in_features = tern.shape
        self.register_buffer("_init_tern", self._hard_tern().clone())

    def _s_full(self, lsq=True):
        out, inp = self.out_features, self.in_features
        s = _grad_scale(self.s, self.lsq_c) if lsq else self.s
        s = s.clamp_min(1e-8)
        return s.reshape(out, inp // self.block_size).repeat_interleave(self.block_size, dim=1)

    def forward(self, x):
        t, k = GumbelTernaryLinear.tau, GumbelTernaryLinear.kappa
        m = gumbel_sigmoid(k * self.mask_logit.float(), t, train=self.training)   # P(nonzero)
        sg = gumbel_sigmoid(k * self.sign_logit.float(), t, train=self.training)  # P(+1)
        soft = m * (2 * sg - 1)                                                   # soft ternary ∈ (−1,1)
        return F.linear(x, (soft * self._s_full(True)).to(x.dtype), None)

    @torch.no_grad()
    def _hard_tern(self):
        return (self.mask_logit > 0).float() * torch.where(self.sign_logit > 0, 1.0, -1.0)

    @torch.no_grad()
    def deploy(self):
        return self._hard_tern() * self._s_full(lsq=False)

    @torch.no_grad()
    def flip_frac(self):
        return (self._hard_tern() != self._init_tern).float().mean().item()


class STELatentLinear(nn.Module):
    """Trainable latent FP weight + FROZEN per-256 scale; forward = STE-ternary(latent, scale)·scale.
    Latent inits to Block-AP's deployed weight so round(latent/scale) == Block-AP's assignments."""

    def __init__(self, latent, scale, block_size):
        super().__init__()
        self.weight = nn.Parameter(latent.float())              # trainable assignments (via STE)
        self.register_buffer("scale", scale.float())            # FROZEN [n_blocks]
        self.block_size = block_size
        self.out_features, self.in_features = latent.shape
        self.register_buffer("_init_tern", self._hard_tern().clone())   # for the flip diagnostic

    def _hard_tern(self):
        out, inp = self.weight.shape
        flat = self.weight.detach().reshape(-1, self.block_size)
        return torch.round((flat / self.scale.unsqueeze(1)).clamp(-1, 1)).reshape(out, inp)

    def forward(self, x):
        out, inp = self.weight.shape
        flat = self.weight.reshape(-1, self.block_size)
        s = self.scale.unsqueeze(1)
        ws = (flat / s).clamp(-1, 1)
        q = ws + (torch.round(ws) - ws).detach()               # STE
        return F.linear(x, (q * s).reshape(out, inp).to(x.dtype), None)

    @torch.no_grad()
    def deploy(self):
        out, inp = self.weight.shape
        flat = self.weight.reshape(-1, self.block_size)
        s = self.scale.unsqueeze(1)
        return (torch.round((flat / s).clamp(-1, 1)) * s).reshape(out, inp)

    @torch.no_grad()
    def flip_frac(self):
        return (self._hard_tern() != self._init_tern).float().mean().item()


def _grad_scale(x, c):
    """LSQ gradient scaling: forward = x, backward gradient ×c. Used to down-scale the per-256 SCALE
    gradient by 1/√256 so a single scalar over 256 weights doesn't overdose (the prior QAT's 82% damage)."""
    return x * c + (x - x * c).detach()


class SoftTernaryLinear(nn.Module):
    """JOINT assignment+scale, soft-to-hard. Latent (init = ROTATED FP WEIGHT, not bin centers — so
    confident weights stay and marginal ones are a small step from flipping) AND per-256 scale are BOTH
    trainable. Forward uses a DISTANCE-SENSITIVE soft ternary σ((z−.5)/τ)−σ((−z−.5)/τ) (z=w/s) — its
    gradient peaks at the ±0.5 boundaries, so marginal weights move first and confident ones don't (the
    selectivity vanilla STE lacks). τ anneals → 0 so the soft collapses to exact ternary; deploy is the
    hard round. Scale gradient is LSQ-scaled by 1/√256. Never freezes either lever → avoids both our
    failure modes (E2E disrupting frozen-scale assignments; frozen scale leaving no gradient headroom)."""
    tau = 0.5                                                # class-level temperature, set during annealing

    def __init__(self, latent, scale, block_size):
        super().__init__()
        self.weight = nn.Parameter(latent.float())          # trainable latent (rotated FP init)
        self.s = nn.Parameter(scale.float())                # trainable per-256 scale (JOINT)
        self.block_size = block_size
        self.lsq_c = 1.0 / math.sqrt(block_size)
        self.out_features, self.in_features = latent.shape
        self.register_buffer("_init_tern", self._hard_tern().clone())

    def _s_full(self, lsq=True):
        out, inp = self.weight.shape
        s = self.s
        if lsq:
            s = _grad_scale(s, self.lsq_c)
        s = s.clamp_min(1e-8)
        return s.reshape(out, inp // self.block_size).repeat_interleave(self.block_size, dim=1)

    def forward(self, x):
        sf = self._s_full(lsq=True)
        z = self.weight / sf
        t = SoftTernaryLinear.tau
        soft = torch.sigmoid((z - 0.5) / t) - torch.sigmoid((-z - 0.5) / t)    # soft ternary ∈ (−1,1)
        return F.linear(x, (soft * sf).to(x.dtype), None)

    @torch.no_grad()
    def _hard_tern(self):
        sf = self._s_full(lsq=False)
        return torch.round((self.weight.detach() / sf).clamp(-1, 1))

    @torch.no_grad()
    def deploy(self):
        sf = self._s_full(lsq=False)
        return torch.round((self.weight.detach() / sf).clamp(-1, 1)) * sf

    @torch.no_grad()
    def flip_frac(self):
        return (self._hard_tern() != self._init_tern).float().mean().item()


def build_joint_student(rot_path, e2e_path, orig_config_path, block_size, device):
    """Joint soft-to-hard student: scales (and frozen non-body) from the E2E-refined model; the LATENTS
    init from the ROTATED FP weights (rot_path) — natural positions near boundaries, NOT bin centers."""
    model, config = build_student(e2e_path, orig_config_path, block_size, device)   # TSL: E2E scales
    rot_wmap = _shard_map(rot_path)
    n = 0
    for name, mod in list(model.named_modules()):
        if isinstance(mod, TernaryScaleLinear):
            latent = _get_tensor(rot_path, rot_wmap, f"{name}.weight").to(device, torch.float32)
            soft = SoftTernaryLinear(latent, mod.scale.detach(), block_size).to(device)
            parent = model
            *ps, leaf = name.split(".")
            for p in ps:
                parent = getattr(parent, p)
            setattr(parent, leaf, soft)
            n += 1
            del mod
    print(f"   replaced {n} linears with SoftTernaryLinear (joint latent+scale, rotated-FP init)", flush=True)
    return model, config


def save_joint(model, e2e_path, out_dir, block_size):
    """Save the joint result as hard ternary × trained scale (same format E2E consumes / deploys)."""
    from pathlib import Path
    from safetensors.torch import save_file
    import shutil
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    smap = {f"{n}.weight": m for n, m in model.named_modules() if isinstance(m, SoftTernaryLinear)}
    wmap = _shard_map(e2e_path)
    shards = {}
    for name, sf in wmap.items():
        shards.setdefault(sf, []).append(name)
    for sf, names in shards.items():
        d = {}
        for name in names:
            key = name.replace("model.language_model.layers", "model.layers")
            mod = smap.get(key) or smap.get(name)
            d[name] = (mod.deploy().to(torch.float16).cpu() if mod is not None
                       else _get_tensor(e2e_path, wmap, name))
        save_file(d, str(out / sf)); del d; gc.collect()
    for fn in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
               "generation_config.json", "merges.txt", "vocab.json", "model.safetensors.index.json"]:
        src = Path(e2e_path) / fn
        if src.exists():
            shutil.copy2(src, out / fn)
    print(f"saved joint model -> {out}", flush=True)


def train_joint(args):
    dev = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = True
    model, config = build_joint_student(args.rot_path, args.student_path, args.orig_config_path,
                                        BLOCK_SIZE, dev)
    model.train()
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    model.config.use_cache = False

    softs = [m for m in model.modules() if isinstance(m, SoftTernaryLinear)]
    for n, p in model.named_parameters():
        p.requires_grad_(False)
    params = []
    for m in softs:
        m.weight.requires_grad_(True); m.s.requires_grad_(True)
        params += [m.weight, m.s]
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=args.lr); print("   optimizer: AdamW8bit", flush=True)
    except Exception as e:
        opt = torch.optim.AdamW(params, lr=args.lr); print(f"   optimizer: AdamW fp32 ({e})", flush=True)

    batches = load_calib_batches(args.calib, 1, args.seq, "cpu")
    if args.max_samples:
        batches = batches[:args.max_samples]
    N = len(batches)
    accum = max(1, args.accum)
    print(f"   {N} seqs; JOINT soft-to-hard; lr={args.lr}; steps={args.steps}; accum={accum}; "
          f"τ {args.tau_start}→{args.tau_end}; warmup={args.warmup}", flush=True)

    order = list(range(N)); ptr = [0]
    def next_bi():
        if ptr[0] >= N:
            import random; random.shuffle(order); ptr[0] = 0
        bi = order[ptr[0]]; ptr[0] += 1; return bi

    step, lossema = 0, None
    while step < args.steps:
        frac = step / max(1, args.steps - 1)
        SoftTernaryLinear.tau = args.tau_start + (args.tau_end - args.tau_start) * frac   # linear anneal
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(accum):
            ids = batches[next_bi()].to(dev)
            logits = model(ids).logits
            l = F.cross_entropy(logits[0, :-1].float(), ids[0, 1:]) / accum
            l.backward()
            acc += l.item() * accum
        acc /= accum
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        if step < args.warmup:
            for g in opt.param_groups:
                g["lr"] = args.lr * (step + 1) / max(1, args.warmup)
        opt.step()
        lossema = acc if lossema is None else 0.9 * lossema + 0.1 * acc
        if step % 25 == 0:
            fl = sum(m.flip_frac() for m in softs) / len(softs)
            print(f"   step {step:4d}/{args.steps}  ce={acc:.4f}  ema={lossema:.4f}  τ={SoftTernaryLinear.tau:.3f}  "
                  f"flips={100*fl:.3f}%", flush=True)
        step += 1

    SoftTernaryLinear.tau = args.tau_end
    fl = sum(m.flip_frac() for m in softs) / len(softs)
    print(f"   final assignment-flip vs E2E: {100*fl:.3f}%", flush=True)
    save_joint(model, args.student_path, args.out, BLOCK_SIZE)


def build_gumbel_student(e2e_path, orig_config_path, block_size, device, alpha, sigma):
    """GSQ student from the E2E-refined model: extract each linear's E2E ternary assignment + scale,
    init the mask/sign logits to preserve that assignment (+noise), scale init = E2E scale."""
    model, config = build_student(e2e_path, orig_config_path, block_size, device)
    n = 0
    for name, mod in list(model.named_modules()):
        if isinstance(mod, TernaryScaleLinear):
            tern, _ = extract_ternary_scale(mod.dequant().detach().float(), block_size)
            g = GumbelTernaryLinear(tern.to(device).float(), mod.scale.detach(), block_size, alpha, sigma).to(device)
            parent = model
            *ps, leaf = name.split(".")
            for p in ps:
                parent = getattr(parent, p)
            setattr(parent, leaf, g)
            n += 1
            del mod
    print(f"   replaced {n} linears with GumbelTernaryLinear (mask+sign, E2E-assignment init)", flush=True)
    return model, config


def save_gumbel(model, e2e_path, out_dir, block_size):
    from pathlib import Path
    from safetensors.torch import save_file
    import shutil
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    smap = {f"{n}.weight": m for n, m in model.named_modules() if isinstance(m, GumbelTernaryLinear)}
    wmap = _shard_map(e2e_path)
    shards = {}
    for name, sf in wmap.items():
        shards.setdefault(sf, []).append(name)
    for sf, names in shards.items():
        d = {}
        for name in names:
            key = name.replace("model.language_model.layers", "model.layers")
            mod = smap.get(key) or smap.get(name)
            d[name] = (mod.deploy().to(torch.float16).cpu() if mod is not None
                       else _get_tensor(e2e_path, wmap, name))
        save_file(d, str(out / sf)); del d; gc.collect()
    for fn in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
               "generation_config.json", "merges.txt", "vocab.json", "model.safetensors.index.json"]:
        src = Path(e2e_path) / fn
        if src.exists():
            shutil.copy2(src, out / fn)
    print(f"saved gumbel model -> {out}", flush=True)


def train_gumbel(args):
    dev = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = True
    model, config = build_gumbel_student(args.student_path, args.orig_config_path, BLOCK_SIZE, dev,
                                         args.alpha, args.sigma_init)
    model.train()
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    model.config.use_cache = False

    gms = [m for m in model.modules() if isinstance(m, GumbelTernaryLinear)]
    for n, p in model.named_parameters():
        p.requires_grad_(False)
    params = []
    for m in gms:
        m.mask_logit.requires_grad_(True); m.sign_logit.requires_grad_(True); m.s.requires_grad_(True)
        params += [m.mask_logit, m.sign_logit, m.s]
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.Lion8bit(params, lr=args.lr); print("   optimizer: Lion8bit (GSQ-recommended)", flush=True)
    except Exception as e:
        opt = torch.optim.AdamW(params, lr=args.lr); print(f"   optimizer: AdamW fp32 ({e})", flush=True)

    batches = load_calib_batches(args.calib, 1, args.seq, "cpu")
    if args.max_samples:
        batches = batches[:args.max_samples]
    N = len(batches)
    accum = max(1, args.accum)
    print(f"   {N} seqs; GUMBEL mask+sign; lr={args.lr}; steps={args.steps}; accum={accum}; "
          f"τ {args.tau_start}→{args.tau_end}; α={args.alpha}; σ={args.sigma_init}; warmup={args.warmup}", flush=True)

    order = list(range(N)); ptr = [0]
    def next_bi():
        if ptr[0] >= N:
            import random; random.shuffle(order); ptr[0] = 0
        bi = order[ptr[0]]; ptr[0] += 1; return bi

    step, lossema = 0, None
    while step < args.steps:
        frac = step / max(1, args.steps - 1)
        GumbelTernaryLinear.tau = args.tau_start + (args.tau_end - args.tau_start) * frac
        GumbelTernaryLinear.kappa = args.kappa_start + (args.kappa_end - args.kappa_start) * frac  # ↑ sharpen
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(accum):
            ids = batches[next_bi()].to(dev)
            logits = model(ids).logits
            l = F.cross_entropy(logits[0, :-1].float(), ids[0, 1:]) / accum
            l.backward()
            acc += l.item() * accum
        acc /= accum
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        if step < args.warmup:
            for g in opt.param_groups:
                g["lr"] = args.lr * (step + 1) / max(1, args.warmup)
        opt.step()
        lossema = acc if lossema is None else 0.9 * lossema + 0.1 * acc
        if step % 25 == 0:
            fl = sum(m.flip_frac() for m in gms) / len(gms)
            print(f"   step {step:4d}/{args.steps}  ce={acc:.4f}  ema={lossema:.4f}  τ={GumbelTernaryLinear.tau:.2f} "
                  f"κ={GumbelTernaryLinear.kappa:.1f}  flips={100*fl:.3f}%", flush=True)
        step += 1

    GumbelTernaryLinear.tau = args.tau_end
    GumbelTernaryLinear.kappa = args.kappa_end
    fl = sum(m.flip_frac() for m in gms) / len(gms)
    print(f"   final assignment-flip vs E2E: {100*fl:.3f}%", flush=True)
    save_gumbel(model, args.student_path, args.out, BLOCK_SIZE)


def build_qat_student(student_path, orig_config_path, block_size, device):
    """Block-AP recovered model → swap each packed TernaryScaleLinear for an STELatentLinear
    (latent = its dequantized ternary weight, scale = its per-block scale, frozen)."""
    model, config = build_student(student_path, orig_config_path, block_size, device)
    n = 0
    for name, mod in list(model.named_modules()):
        if isinstance(mod, TernaryScaleLinear):
            latent = mod.dequant().detach()                    # = ternary × scale (Block-AP's Q)
            scale = mod.scale.detach()
            ste = STELatentLinear(latent, scale, block_size).to(device)
            parent = model
            *ps, leaf = name.split(".")
            for p in ps:
                parent = getattr(parent, p)
            setattr(parent, leaf, ste)
            n += 1
            del mod
    print(f"   replaced {n} linears with STELatentLinear (frozen scales)", flush=True)
    return model, config


def save_qat(model, student_path, out_dir, block_size):
    """Write the QAT result as dequantized fp16 (hard ternary × frozen scale) — same format Block-AP
    emits, so E2E-QP consumes it unchanged. Streams shard-by-shard."""
    from pathlib import Path
    from safetensors.torch import save_file
    import shutil
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    ste_map = {f"{n}.weight": m for n, m in model.named_modules() if isinstance(m, STELatentLinear)}
    wmap = _shard_map(student_path)
    shards = {}
    for name, sf in wmap.items():
        shards.setdefault(sf, []).append(name)
    for sf, names in shards.items():
        d = {}
        for name in names:
            key = name.replace("model.language_model.layers", "model.layers")
            mod = ste_map.get(key) or ste_map.get(name)
            d[name] = (mod.deploy().to(torch.float16).cpu() if mod is not None
                       else _get_tensor(student_path, wmap, name))
        save_file(d, str(out / sf)); del d; gc.collect()
    for fn in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
               "generation_config.json", "merges.txt", "vocab.json", "model.safetensors.index.json"]:
        src = Path(student_path) / fn
        if src.exists():
            shutil.copy2(src, out / fn)
    print(f"saved QAT model -> {out}", flush=True)


def train(args):
    from contextlib import nullcontext
    # DDP when launched via torchrun (LOCAL_RANK set) — both GPUs, 2× throughput, effective batch
    # = world × accum. Per-GPU memory is unchanged (full model copy each) so the weight-EMA lives on
    # CPU. Plain single-GPU otherwise.
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world = int(os.environ.get("WORLD_SIZE", 1))
    ddp = local_rank >= 0 and world > 1
    if ddp:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        dev = f"cuda:{local_rank}"
        rank = dist.get_rank()
    else:
        dev = "cuda:0"
        rank = 0
    is_main = (rank == 0)
    def log(*a):
        if is_main:
            print(*a, flush=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    core, config = build_qat_student(args.student_path, args.orig_config_path, BLOCK_SIZE, dev)
    core.train()
    try:
        core.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception:
        try:
            core.gradient_checkpointing_enable()
        except Exception:
            pass
    core.config.use_cache = False

    stes = [m for m in core.modules() if isinstance(m, STELatentLinear)]
    # freeze everything except the STE latents (so DDP only syncs the latent grads)
    for n, p in core.named_parameters():
        p.requires_grad_(False)
    latents = [m.weight for m in stes]
    for w in latents:
        w.requires_grad_(True)

    model = core
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(core, device_ids=[local_rank], output_device=local_rank,
                    broadcast_buffers=False, find_unused_parameters=False)
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(latents, lr=args.lr); log("   optimizer: AdamW8bit (bitsandbytes)")
    except Exception as e:
        opt = torch.optim.AdamW(latents, lr=args.lr); log(f"   optimizer: AdamW fp32 ({e})")

    batches = load_calib_batches(args.calib, 1, args.seq, "cpu")
    if args.max_samples:
        batches = batches[:args.max_samples]
    cache = None
    if args.objective == "mse":
        cache = torch.load(args.teacher_cache, map_location="cpu")
        assert "hidden" in cache, "--objective mse needs a teacher cache built with --cache-hidden"
    N = min(len(batches), len(cache["hidden"]) if cache else len(batches))
    accum = max(1, args.accum)
    ema_decay, warmup = args.ema_decay, args.warmup
    log(f"   {N} seqs; obj={args.objective}; lr={args.lr}; steps={args.steps}; accum={accum}; "
        f"world={world} → EFF BATCH {accum*world}; ema={ema_decay}; warmup={warmup}")

    # weight-EMA on CPU (Polyak) — deploys averaged latents so residual update noise cancels; CPU so it
    # never competes with the ~23.6 GB/GPU training footprint.
    ema_w = [m.weight.detach().float().cpu().clone() for m in stes] if (ema_decay > 0 and is_main) else None

    def loss_for(bi):
        ids = batches[bi].to(dev)
        if args.objective == "ce":
            logits = model(ids).logits
            return F.cross_entropy(logits[0, :-1].float(), ids[0, 1:])
        out = model(ids, output_hidden_states=True)
        return F.mse_loss(out.hidden_states[-1][0].float(), cache["hidden"][bi].to(dev).float())

    shard = list(range(rank, N, world)) if ddp else list(range(N))      # each rank a disjoint slice
    ptr = [0]
    def next_bi():
        if ptr[0] >= len(shard):
            import random; random.shuffle(shard); ptr[0] = 0
        bi = shard[ptr[0]]; ptr[0] += 1; return bi

    step, lossema = 0, None
    while step < args.steps:
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for i in range(accum):                                          # accumulate; sync only last
            sync_ctx = model.no_sync() if (ddp and i < accum - 1) else nullcontext()
            with sync_ctx:
                l = loss_for(next_bi()) / accum
                l.backward()
            acc += l.item() * accum
        acc /= accum
        torch.nn.utils.clip_grad_norm_(latents, 1.0)
        if step < warmup:
            for g in opt.param_groups:
                g["lr"] = args.lr * (step + 1) / max(1, warmup)
        opt.step()
        if ema_w is not None:
            with torch.no_grad():
                for e, m in zip(ema_w, stes):
                    e.mul_(ema_decay).add_(m.weight.detach().float().cpu(), alpha=1 - ema_decay)
        lossema = acc if lossema is None else 0.9 * lossema + 0.1 * acc
        if step % 25 == 0:
            fl = sum(m.flip_frac() for m in stes) / len(stes)
            log(f"   step {step:4d}/{args.steps}  {args.objective}={acc:.4f}  ema={lossema:.4f}  "
                f"assign-flips={100*fl:.3f}%")
        step += 1

    if ema_w is not None:
        with torch.no_grad():
            for e, m in zip(ema_w, stes):
                m.weight.copy_(e.to(m.weight.device, m.weight.dtype))
        log("   deployed EMA-averaged latents")
    if is_main:
        fl = sum(m.flip_frac() for m in stes) / len(stes)
        log(f"   final mean assignment-flip vs Block-AP: {100*fl:.3f}%")
        save_qat(core, args.student_path, args.out, BLOCK_SIZE)
    if ddp:
        import torch.distributed as dist
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--student-path", required=True, help="Block-AP recovered model dir")
    ap.add_argument("--orig-config-path", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--objective", choices=["ce", "mse"], default="ce")
    ap.add_argument("--teacher-cache", default=None, help="needed for --objective mse (--cache-hidden)")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=2000, help="OPTIMIZER updates (each sees --accum seqs).")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--accum", type=int, default=1,
                    help="Gradient accumulation: seqs averaged per optimizer step (effective batch). "
                         "The main stability lever — batch=1 is the dominant gradient-variance source.")
    ap.add_argument("--ema-decay", type=float, default=0.0,
                    help="Polyak weight-EMA decay (e.g. 0.999); deploys the AVERAGED latents so residual "
                         "update noise cancels. 0 = off.")
    ap.add_argument("--warmup", type=int, default=0, help="Linear LR warmup steps.")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--mode", choices=["ste", "joint", "gumbel"], default="ste",
                    help="ste = frozen-scale STE (failed); joint = single-latent soft-to-hard; "
                         "gumbel = GSQ mask+sign Gumbel-Softmax joint assign+scale (the report's recipe).")
    ap.add_argument("--rot-path", default=None, help="[joint] rotated-FP model dir for the latent init.")
    ap.add_argument("--tau-start", type=float, default=2.0, help="Gumbel/soft temperature start.")
    ap.add_argument("--tau-end", type=float, default=0.1, help="temperature end (→ near-hard).")
    ap.add_argument("--alpha", type=float, default=3.0, help="[gumbel] assignment strength in the small-logit init.")
    ap.add_argument("--sigma-init", type=float, default=0.1, help="[gumbel] init logit scale (SMALL so Lion can flip).")
    ap.add_argument("--kappa-start", type=float, default=2.0, help="[gumbel] forward logit multiplier start (soft).")
    ap.add_argument("--kappa-end", type=float, default=20.0, help="[gumbel] forward multiplier end (sharp/commit).")
    args = ap.parse_args()
    if args.train:
        if args.mode == "gumbel":
            train_gumbel(args)
        elif args.mode == "joint":
            assert args.rot_path, "--mode joint needs --rot-path (rotated-FP model for latent init)"
            train_joint(args)
        else:
            train(args)
