"""Does a bf16 latent GRADIENT change which assignments flip?

Quality can only change through the ternary assignment q, because the forward value is q*s. So the
question is not "how big is the gradient error" but "does it move a different SET of latents across a
bin boundary". Simulated in L/s units, where --latent-init center puts every latent exactly on an
integer and the decision boundary is 0.5 away, using the REAL adam-blockv kernel (lerp_ + block-256
vector_norm second moment). Both arms see the IDENTICAL gradient sequence; the only difference is
that arm 2 rounds each gradient to bf16 first, exactly as --latent-bf16-compute delivers it.
"""
import torch
torch.manual_seed(0); torch.set_num_threads(8)
BLKV = 256
NB, T = 1500, 200
N = NB * BLKV
B1, B2, EPS = 0.9, 0.999, 1e-8

def trits(L):
    return torch.round(L).clamp_(-1, 1)

def run(mode, lr, sig, noi, seed=1234):
    g_gen = torch.Generator().manual_seed(seed)
    # SEPARATE generator for the control perturbation. Drawing it from g_gen would advance the
    # gradient stream and make the control a DIFFERENT RUN rather than the same run perturbed --
    # which is exactly the confound that made the first version of this control meaningless.
    e_gen = torch.Generator().manual_seed(seed + 99991)
    L = torch.randint(-1, 2, (N,), generator=g_gen).float()   # center init: exactly on a bin centre
    signal = torch.randn(N, generator=g_gen) * sig            # persistent per-latent drift
    m = torch.zeros(N); v = torch.zeros(NB)
    for t in range(1, T + 1):
        g = signal + torch.randn(N, generator=g_gen) * noi
        if mode == "bf16":
            g = g.to(torch.bfloat16).float()                  # what --latent-bf16-compute delivers
        elif mode == "eps":
            # CONTROL: fp32 reduction-order nondeterminism. A GPU reduction summing in a different
            # order perturbs each element at the fp32 rounding level. This is noise the pipeline
            # ALREADY has run-to-run (measured: identical configs gave KL 9.4018/9.4285/9.4200),
            # so it is the right yardstick for whether the bf16 effect matters.
            g = g * (1.0 + torch.randn(N, generator=e_gen) * 1.2e-7)
        m.lerp_(g, 1 - B1)
        sq = torch.linalg.vector_norm(g.view(-1, BLKV), dim=1); sq.pow_(2).div_(BLKV)
        v.mul_(B2).add_(sq, alpha=1 - B2)
        bc1, bc2 = 1 - B1 ** t, 1 - B2 ** t
        d = (v / bc2).sqrt().add_(EPS).unsqueeze(1)
        L.view(-1, BLKV).addcdiv_(m.view(-1, BLKV), d.expand(-1, BLKV), value=-lr / bc1)
    return L

import sys
print("%22s %8s %11s %9s %8s %9s %8s" % ("regime","lr","moved fp32","bf16 dif","of movd","eps dif","of movd"), flush=True)
for name, sig, noi in [("signal-dominated", 1.0, 0.3), ("balanced", 1.0, 1.0), ("noise-dominated", 0.2, 1.0)]:
    for lr in (5e-3,):
        L0 = torch.randint(-1, 2, (N,), generator=torch.Generator().manual_seed(1234)).float()
        q0 = trits(L0.clone())
        a = trits(run("fp32", lr, sig, noi))
        b = trits(run("bf16", lr, sig, noi))
        c = trits(run("eps",  lr, sig, noi))
        ma = (a != q0).float().mean().item() * 100
        mb = (b != q0).float().mean().item() * 100
        diff = (a != b).float().mean().item() * 100
        dctl = (a != c).float().mean().item() * 100
        rel = 100.0 * diff / max(ma, 1e-9)
        rctl = 100.0 * dctl / max(ma, 1e-9)
        print(f"{name:>22} {lr:>8.0e} {ma:>10.3f}% {diff:>8.4f}% {rel:>7.2f}% "
              f"{dctl:>8.4f}% {rctl:>7.2f}%", flush=True)
