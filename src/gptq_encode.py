"""gptq_encode.py — Hessian-aware encoding with a PLUGGABLE format quantizer.

THE QUESTION. The cross-product report claims second-order PTQ moves scalar ternary from 64.8% to
only ~71.5% of the Shannon bound "before diverging", and gives no citation. That number decides
deployment bands B and C: if a good encoder closes most of scalar ternary's structural gap, the band
picks should be ternary; if it closes little, they should stay non-scalar. We measure it here rather
than cite it.

WHAT GPTQ ACTUALLY OPTIMISES, and why that matters for the comparison. GPTQ does not minimise
||W - W_hat||_F. It minimises the PROXY LOSS

    Tr( dW H dW^T ),      H = E[x x^T]  (uncentered input activation covariance)

which is the second-order Taylor term of the task loss. So a GPTQ-encoded weight can have HIGHER
plain reconstruction error than RTN while being strictly better for the model. Reporting only
reconstruction error would therefore make GPTQ look worse than it is, and reporting only proxy loss
would break comparability with our Shannon analysis. **Both are reported, and the Shannon efficiency
figure is computed on the reconstruction metric it is defined for, with the mismatch stated.**

GENERALISATION TO BLOCK FORMATS. Textbook GPTQ is column-sequential: quantize column j, push its
error into columns > j. Trellis and VQ do not quantize single columns -- they encode a block jointly
(256 weights for trellis, 8-dim sub-vectors for VQ). We therefore quantize a COLUMN BLOCK with the
format's own encoder, then propagate the block's aggregate error to all later columns. With a block
of one this reduces exactly to GPTQ, so scalar formats are unaffected by the generalisation.
"""
import os, sys, json, torch
sys.path.insert(0, os.path.dirname(__file__))


@torch.no_grad()
def capture_hessian(model, layer_name, batches, device, damp=0.01):
    """H = sum_x x x^T over calibration tokens, for the input of one named Linear."""
    mod = model.get_submodule(layer_name)
    d_in = mod.in_features
    H = torch.zeros(d_in, d_in, device=device, dtype=torch.float32)
    n = 0

    def hook(_m, inp, _out):
        nonlocal H, n
        x = inp[0].detach().reshape(-1, d_in).float()
        H += x.T @ x
        n += x.shape[0]

    h = mod.register_forward_hook(hook)
    for b in batches:
        model(b.to(device))
    h.remove()
    H /= max(n, 1)
    # damping is not cosmetic: H is rank-deficient whenever tokens < d_in, and the Cholesky below
    # fails outright without it. GPTQ's own implementation does the same.
    d = torch.diag(H).mean().clamp_min(1e-8)
    H += torch.eye(d_in, device=device) * (damp * d)
    return H


@torch.no_grad()
def gptq_quantize(W, H, block_quant, group, blocksize=128):
    """Block-wise GPTQ. `block_quant(Wblk) -> Wblk_hat` is the format's own encoder.

    W: [out, in]  H: [in, in]  group: columns quantized jointly by the format.
    """
    dev = W.device
    Wq = W.clone().float()
    d_in = W.shape[1]
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    Wq[:, dead] = 0
    # inverse-Hessian Cholesky, upper triangular, as in the GPTQ formulation
    Hinv = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(Hinv)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    Q = torch.zeros_like(Wq)
    step = max(group, 1)
    for a in range(0, d_in, step):
        b = min(a + step, d_in)
        Wblk = Wq[:, a:b].clone()
        Qblk = block_quant(Wblk).float()
        Q[:, a:b] = Qblk
        if b >= d_in:
            break
        # aggregate error of the block, whitened by the block's own Cholesky diagonal, pushed into
        # every later column -- the block generalisation of GPTQ's single-column update
        Eblk = (Wblk - Qblk) / torch.diag(Hinv)[a:b].unsqueeze(0).clamp_min(1e-8)
        Wq[:, b:] -= Eblk @ Hinv[a:b, b:]
    return Q


def proxy_loss(W, Q, H):
    """Tr(dW H dW^T), normalised by the same quantity for W itself -- the objective GPTQ minimises."""
    d = (Q.float() - W.float())
    num = torch.einsum("oi,ij,oj->", d, H, d)
    den = torch.einsum("oi,ij,oj->", W.float(), H, W.float()).clamp_min(1e-12)
    return (num / den).sqrt().item()
