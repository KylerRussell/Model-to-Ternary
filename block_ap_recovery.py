#!/usr/bin/env python3
"""
block_ap_recovery.py — Phase-2B compute-efficient recovery for ternary Qwen3.6-27B.

This is the EfficientQAT "Block-AP" idea adapted to a hybrid Gated-DeltaNet model and
to a 48 GB (2x 3090) budget. Instead of one-shot GPTQ rounding, it RECONSTRUCTS each
quantized projection by short gradient descent on its OUTPUT error, then propagates the
quantized activations forward so the next layer is calibrated on the realistically
degraded input (the error-compensation that makes recovery work).

Why per-LINEAR (not full per-block backprop):
  The fused Gated-DeltaNet kernel is not reliably differentiable, so we never backprop
  THROUGH a block. For each quantized nn.Linear we minimise
        || X @ Wq(W,s)^T  -  X @ W_fp^T ||^2
  where X is the real (propagated) input captured by a forward hook, W_fp is the frozen
  rotated FP weight (the target), and Wq is the straight-through ternary of a trainable
  latent weight W and a trainable per-g128 scale s. Backprop flows only through a matmul.
  Inter-layer error propagation is handled by a separate INFERENCE forward (no grad),
  which the fused kernel runs fine. Intra-block input shift (o_proj/down_proj inputs move
  slightly once upstream projections are quantized) is a second-order effect we accept.

Input model MUST be the rotation-only Phase-1 model (convert.py --rotation-only): every
projection is already Hadamard-rotated and the norms are absorbed/zeroed. This script only
replaces the should_quantize() projections with recovered-ternary weights; everything else
(rotated embed/lm_head, zeroed norms, conv1d, and any projection you keep FP16 via the
config toggles) passes through untouched. It does NO rotation and touches NO norms.

By construction the per-linear objective at init (epoch 0) equals RTN-absmean, so training
can only reduce reconstruction error — recovery is >= RTN. Whether that clears the 64-layer
coherence cliff is the open question; if not, a short global self-distillation pass is the
documented follow-on (phase 2C).

Usage:
    python block_ap_recovery.py \
        --model-path ./output/modified_model \
        --orig-config-path /path/to/Qwen--Qwen3.6-27B/snapshots/<hash> \
        --output-dir ./output_recovery \
        --samples 32 --epochs 20 --lr 1e-3
"""

import argparse
import gc
import json
import os
import sys
import shutil
import inspect
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from safetensors import safe_open
from safetensors.torch import save_file as save_safetensors

sys.path.append(str(Path(__file__).parent))
from config import NUM_HIDDEN_LAYERS, BLOCK_SIZE, should_quantize


# ───────────────────────── shared infrastructure (from calibrate_and_quantize.py) ──

def load_calibration_samples(cache_path: Path, n: int) -> list:
    if not cache_path.exists():
        print(f"❌ Calibration cache not found at {cache_path}!")
        print("   Run: python build_diverse_calib.py --orig-model <snap> --ctm-data <dir> "
              f"--out {cache_path}")
        sys.exit(1)
    with open(cache_path) as f:
        samples = json.load(f)
    print(f"📚 Loaded {len(samples)} calibration samples from {cache_path}")
    return samples[:n]


def load_tensor_from_shards(model_path: Path, weight_map: dict, tensor_name: str,
                            device: str = "cpu") -> torch.Tensor:
    shard_file = weight_map.get(tensor_name)
    if not shard_file:
        alt = tensor_name
        if "model.layers" in tensor_name:
            alt = tensor_name.replace("model.layers", "model.language_model.layers")
        elif "model.embed_tokens" in tensor_name:
            alt = tensor_name.replace("model.embed_tokens", "model.language_model.embed_tokens")
        elif "model.norm" in tensor_name:
            alt = tensor_name.replace("model.norm", "model.language_model.norm")
        shard_file = weight_map.get(alt)
        if not shard_file:
            raise KeyError(f"Tensor {tensor_name} not in index (tried alt: {alt})")
        tensor_name = alt
    with safe_open(str(model_path / shard_file), framework="pt", device=device) as f:
        return f.get_tensor(tensor_name)


def assign_tensor_to_module(module, param_name, tensor, device):
    parts = param_name.split(".")
    parent = module
    for p in parts[:-1]:
        parent = getattr(parent, p)
    attr = parts[-1]
    val = tensor.to(device=device, dtype=torch.bfloat16)
    if hasattr(parent, attr):
        delattr(parent, attr)
    parent.register_parameter(attr, nn.Parameter(val))


def revert_module_param_to_meta(module, param_name, shape):
    parts = param_name.split(".")
    parent = module
    for p in parts[:-1]:
        parent = getattr(parent, p)
    attr = parts[-1]
    if hasattr(parent, attr):
        delattr(parent, attr)
    parent.register_parameter(attr, nn.Parameter(torch.empty(shape, device="meta")))


# ───────────────────────── ternary STE + per-linear reconstruction ─────────────────

def _blocks(W: torch.Tensor, block_size: int):
    """Row-major [n_blocks, block_size] view, matching quantizer.quantize_absmean.
    Requires in_features % block_size == 0 (true for every projection here)."""
    out, inp = W.shape
    assert inp % block_size == 0, f"in_features {inp} not divisible by block_size {block_size}"
    return W.reshape(out * (inp // block_size), block_size), (out, inp)


def ste_ternary(W: torch.Tensor, s: torch.Tensor, block_size: int) -> torch.Tensor:
    """Differentiable ternary dequant: forward = s*round(clamp(W/s,-1,1)) in {-s,0,+s};
    backward is straight-through on round (gradient flows to W within the clamp and to s)."""
    flat, (out, inp) = _blocks(W, block_size)
    ws = flat / s.unsqueeze(1)
    wc = ws.clamp(-1, 1)
    wr = torch.round(wc)
    wq = wc + (wr - wc).detach()           # STE
    return (wq * s.unsqueeze(1)).reshape(out, inp)


def _deploy_ternary(W: torch.Tensor, s: torch.Tensor, block_size: int):
    """Detached final weight (true ternary*scale with the learned scale) + sparsity."""
    flat, (out, inp) = _blocks(W, block_size)
    q = torch.round((flat / s.unsqueeze(1)).clamp(-1, 1))
    deq = (q * s.unsqueeze(1)).reshape(out, inp)
    sparsity = (q == 0).float().mean().item()
    return deq, sparsity


def _block_scale(W1):
    """Per-row MSE-optimal ternary scale for one block: search multipliers of mean|w| and pick the
    one minimising local ternary MSE. The single per-block scale sets BOTH the round threshold
    (0.5·s) and the level (±s), so absmax over-sparsifies and plain mean is not optimal — a small
    search wins (GPTQ's `find_params`). Returns s [out] (one scale per output row for this block)."""
    m = W1.abs().mean(1, keepdim=True).clamp_min(1e-8)               # [out,1]
    cands = torch.linspace(0.6, 2.4, 19, device=W1.device).view(1, -1, 1)
    s = m.unsqueeze(1) * cands                                       # [out,C,1]
    q = torch.round((W1.unsqueeze(1) / s).clamp(-1, 1)) * s          # [out,C,block]
    mse = ((W1.unsqueeze(1) - q) ** 2).mean(-1)                      # [out,C]
    return s[torch.arange(W1.shape[0]), mse.argmin(1), 0]            # [out]


def _gptq_ternary(W_fp, H, block_size, percdamp=0.01):
    """One-shot GPTQ/OBC error-feedback ternary fit. Quantises input columns left→right; each
    column's rounding residual is pushed into the not-yet-quantised columns through the inverse
    Hessian (H = XᵀX), so the OUTPUT error ‖X(W−Q)ᵀ‖² is *compensated*, not just locally minimised
    per weight — the inter-weight coupling our previous Adam-on-Gram+STE fit ignored. A group =
    `block_size` consecutive input columns = exactly one TQ2_0 block per output row (with its own
    absmax scale), so Q ∈ {−s,0,+s} is exactly representable on the per-256 ternary grid. Returns
    the deployed Q [out,inp]."""
    out, inp = W_fp.shape
    dev = W_fp.device
    W = W_fp.clone().float()
    H = H.clone().float()
    dead = torch.diag(H) == 0                        # input channels with no activation variance
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    damp = percdamp * torch.diag(H).mean()           # Hessian damping for a stable inverse
    di = torch.arange(inp, device=dev)
    H[di, di] += damp
    Hinv = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(Hinv)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)   # upper-triangular factor of H⁻¹ (GPTQ)
    Q = torch.zeros_like(W)
    for i1 in range(0, inp, block_size):
        i2 = min(i1 + block_size, inp)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        s = _block_scale(W1).clamp_min(1e-8)         # [out] MSE-optimal per-row scale for this block
        for i in range(i2 - i1):
            w = W1[:, i]
            d = Hinv1[i, i]
            q = torch.round((w / s).clamp(-1, 1)) * s            # ternary {−s,0,+s}
            Q1[:, i] = q
            e = (w - q) / d
            W1[:, i:] -= e.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)   # push residual within block
            E1[:, i] = e
        Q[:, i1:i2] = Q1
        if i2 < inp:
            W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]       # push block residual to later blocks
    return Q


def _cdquant_refine(Q, W_fp, H, block_size, sweeps=4):
    """CDQuant (greedy/Jacobi coordinate descent): refine the ternary ASSIGNMENTS to further reduce the
    SAME on-grid output error (Q−W_fp)H(Q−W_fp)ᵀ that GPTQ targets — a stronger local search than GPTQ's
    one-shot left→right pass. Per-256 scales are FIXED (recovered from Q); only the {−1,0,+1} choices
    move, so the result stays exactly on the TQ2_0 grid. Each sweep flips every coordinate to its locally
    best level using the closed-form Δf = Δ²·H_ii + 2Δ·(Hd)_i (Jacobi, all rows/cols vectorized). QuaRot
    decorrelates H (small off-diagonals) so simultaneous flips are near-independent and descend; keep-best
    on the true objective guarantees the result is ≥ the GPTQ input (never worse)."""
    out, inp = W_fp.shape
    flat, _ = _blocks(Q, block_size)
    s_block = flat.abs().amax(dim=1)                                    # [n_blocks] recover per-block scale
    s = s_block.reshape(out, inp // block_size).repeat_interleave(block_size, dim=1)   # [out,inp]
    Hd = torch.diag(H).clamp_min(0)                                     # [inp]
    Q = Q.clone(); D = Q - W_fp
    best_Q, best_f = Q.clone(), ((D @ H) * D).sum()
    for _ in range(sweeps):
        grad = D @ H                                                   # [out,inp] = (H·d) per row
        best_df = torch.zeros_like(Q)                                  # 0 = keep current assignment
        best_v = Q.clone()
        for lvl in (-1.0, 0.0, 1.0):
            v = lvl * s
            dl = v - Q
            df = dl * dl * Hd.unsqueeze(0) + 2.0 * dl * grad
            upd = df < best_df
            best_df = torch.where(upd, df, best_df)
            best_v = torch.where(upd, v, best_v)
        Q = best_v
        D = Q - W_fp
        f = ((D @ H) * D).sum()
        if f < best_f - 1e-9:                                          # accept only real improvement
            best_f, best_Q = f, Q.clone()
        else:                                                          # converged / Jacobi overshoot → stop
            break
    return best_Q


def reconstruct_linear(module, gram, block_size, iters, lr, device, sensitivity=None,
                       cdquant=False, cd_sweeps=4):
    """Ternary reconstruction of one nn.Linear via GPTQ/OBC error feedback, floored by RTN keep-best.
    Minimises the OUTPUT error ‖X(Q−W_fp)ᵀ‖²_F through the precomputed Gram H = XᵀX (exact, data-size
    -independent). Error feedback compensates each column's rounding residual into the remaining
    columns — the inter-weight coupling the prior Adam-on-Gram+STE fit ignored — which QuaRot's own
    2-bit ablation shows survives Hadamard rotation (a different axis from the closed cross-linear
    rewrite). We keep whichever of {GPTQ, absmean-RTN} has lower Gram-MSE, so the result is
    guaranteed ≥ RTN per linear.

    `iters`/`lr` are accepted for caller-compat and unused (no gradient loop now). `sensitivity`
    is ignored — the diagonal-√S reserve was tested and discarded (−0.040 nats). `gram` is the
    prebuilt (H, N) pair accumulated incrementally in the capture hook (never holds raw inputs)."""
    W_fp = module.weight.data.detach().to(device, torch.float32)     # frozen target [out,inp]
    out, inp = W_fp.shape
    flat, _ = _blocks(W_fp, block_size)
    s_init = flat.abs().mean(dim=1).clamp_min(1e-8)                   # abs-mean init == RTN

    H, N = gram                                                      # prebuilt Gram (no raw inputs)
    H = H.to(device, torch.float32)
    denom = float(N * out)

    def gram_mse(W_use):                                            # mean output MSE via H
        D = W_use - W_fp
        return ((D @ H) * D).sum() / denom

    rtn_deploy, _ = _deploy_ternary(W_fp, s_init, block_size)
    init_mse = gram_mse(rtn_deploy).item()                          # RTN floor / keep-best candidate

    try:                                                           # GPTQ can throw on a non-PD Hessian
        gptq_deploy = _gptq_ternary(W_fp, H, block_size)
        if cdquant:                                                # refine assignments on-grid (≥ GPTQ)
            gptq_deploy = _cdquant_refine(gptq_deploy, W_fp, H, block_size, sweeps=cd_sweeps)
        gptq_mse = gram_mse(gptq_deploy).item()
    except Exception as e:                                          # degenerate layer → RTN fallback
        print(f"   ⚠️ GPTQ failed ({type(e).__name__}: {e}); keeping RTN", flush=True)
        gptq_deploy, gptq_mse = rtn_deploy, float("inf")

    if gptq_mse < init_mse:                                         # GPTQ won → take it
        deq, final_mse = gptq_deploy, gptq_mse
    else:                                                           # safety floor → keep RTN
        deq, final_mse = rtn_deploy, init_mse
    sparsity = (deq == 0).float().mean().item()
    cos = F.cosine_similarity(deq.reshape(1, -1), W_fp.reshape(1, -1)).item()

    module.weight.data.copy_(deq.to(module.weight.dtype))
    del W_fp, H, deq, rtn_deploy, gptq_deploy
    return {"cosine": cos, "sparsity": sparsity,
            "init_mse": init_mse, "final_mse": final_mse}


def _qep_correct(W, M, H, alpha, percdamp=0.01):
    """QEP inter-layer (cross-depth) error compensation (Arai & Ichikawa, arXiv:2504.09629).
    Each layer's input is corrupted by the accumulated upstream quantization error. Before
    quantizing this linear, pre-compensate its weight so it reproduces the CLEAN output on the
    CORRUPTED input: min_{W*} ‖W*·X̂ − W·X‖²  ⇒  W* = W·M·Ĥ⁻¹, with M = Σ Xᵀ X̂ (clean×quantised
    cross-Gram) and Ĥ = Σ X̂ᵀ X̂ (quantised-input Gram, the same H GPTQ uses). α∈[0,1] ridges
    between the full correction (α=1) and the uncorrected weight (α=0):
        W* = (1−α)·W + α·W·(M Ĥ⁻¹).
    This is a one-shot closed-form weight delta folded in BEFORE quantization — not assignment
    retraining, so it sidesteps the QAT objective-misalignment trap. Returns W* (fp32)."""
    Wf = W.to(torch.float32)
    Hf, Mf = H.to(torch.float32), M.to(torch.float32)
    n = Hf.shape[0]
    damp = percdamp * torch.diag(Hf).mean()
    Hd = Hf + damp * torch.eye(n, device=Hf.device, dtype=Hf.dtype)
    G = torch.linalg.solve(Hd, Mf.t()).t()                 # M Ĥ⁻¹  [inp,inp]
    return (1.0 - alpha) * Wf + alpha * (Wf @ G)


def _ssr_permute_mlp(layer, device):
    """SSR (PT²-LLM): structural-similarity channel reordering for the MLP. Group similar-magnitude
    INTERMEDIATE channels into contiguous 256-blocks so down_proj's per-block scale fits each block
    (instead of a lone outlier channel inflating the scale and dead-zoning the other 255). We sort the
    intermediate channels by down_proj input-column magnitude (outliers cluster into a few blocks),
    permute down_proj's input columns, and fold the SAME permutation into gate_proj/up_proj OUTPUT rows
    — SwiGLU has no norm between them and the intermediate is NOT rotated, so the output is unchanged,
    the fold is offline/local, and the result stays strictly on the per-256 ternary grid (no side tensor)."""
    mlp = getattr(layer, "mlp", None)
    need = ("gate_proj", "up_proj", "down_proj")
    if mlp is None or not all(hasattr(mlp, n) for n in need):
        return False
    dw = mlp.down_proj.weight.data
    P = torch.argsort(dw.float().norm(dim=0))              # [intermediate] small→large; outliers to the tail
    mlp.down_proj.weight.data = dw.index_select(1, P).contiguous()
    mlp.gate_proj.weight.data = mlp.gate_proj.weight.data.index_select(0, P).contiguous()
    mlp.up_proj.weight.data = mlp.up_proj.weight.data.index_select(0, P).contiguous()
    return True


# ───────────────────────────────────────── main ────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Block-AP per-linear ternary recovery")
    ap.add_argument("--model-path", required=True,
                    help="ROTATION-ONLY Phase-1 model (convert.py --rotation-only output).")
    ap.add_argument("--orig-config-path", default=None,
                    help="Original HF snapshot for config.json + trust_remote_code .py.")
    ap.add_argument("--output-dir", default="./output_recovery")
    ap.add_argument("--block-size", type=int, default=BLOCK_SIZE)
    ap.add_argument("--sensitivity", default=None,
                    help="Optional per-linear output-channel sensitivity .pt (compute_sensitivity.py). "
                         "Reweights the reconstruction objective by √S (reserve hedge). Default off.")
    ap.add_argument("--samples", type=int, default=32,
                    help="Calibration samples (trades CPU activation cache + quality).")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--iters", type=int, default=200,
                    help="Max full-batch reconstruction steps per linear "
                         "(early-stops after 40 steps without improvement).")
    ap.add_argument("--epochs", type=int, default=None,
                    help="(Legacy, ignored — superseded by --iters.)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--qep", action="store_true",
                    help="Enable QEP inter-layer error compensation: propagate a 2nd CLEAN-FP "
                         "activation stream and pre-correct each weight for accumulated upstream "
                         "quantization error before the GPTQ fit (W*=W[(1−α)I+αMĤ⁻¹]).")
    ap.add_argument("--qep-alpha", type=float, default=0.5,
                    help="QEP ridge coefficient α∈[0,1]: 1=full least-squares correction, "
                         "0=off. Default 0.5 (robust to distribution shift; the paper's default).")
    ap.add_argument("--ssr", action="store_true",
                    help="SSR (PT²-LLM): reorder MLP intermediate channels (foldable into gate/up/down) "
                         "so down_proj's per-256 scale fits each block — outliers clustered, not scattered.")
    ap.add_argument("--cdquant", action="store_true",
                    help="CDQuant: greedy/Jacobi coordinate-descent refinement of the ternary assignments "
                         "after GPTQ (same on-grid objective, stronger local search; keep-best, ≥ GPTQ).")
    ap.add_argument("--cd-sweeps", type=int, default=4, help="CDQuant max coordinate-descent sweeps per linear.")
    args = ap.parse_args()
    if args.epochs is not None:
        print(f"⚠️  --epochs is ignored in the Gram-matrix reconstruction; using --iters {args.iters}")

    SENS = None
    if args.sensitivity:
        SENS = torch.load(args.sensitivity)
        print(f"🎯 sensitivity reweighting ON: √S from {args.sensitivity} ({len(SENS)} linears)")

    model_path = Path(args.model_path)
    orig_config_path = Path(args.orig_config_path) if args.orig_config_path else model_path
    if args.orig_config_path is None:
        print("⚠️  --orig-config-path not set; config/modeling loaded from --model-path.")
    output_dir = Path(args.output_dir)
    modified_model_dir = output_dir / "modified_model"
    modified_model_dir.mkdir(parents=True, exist_ok=True)

    calib_cache = output_dir / "calibration_data.json"
    if not calib_cache.exists():
        # fall back to a sibling calibration file produced earlier
        alt = Path("./output_calib/calibration_data.json")
        if alt.exists():
            calib_cache = alt
    samples = load_calibration_samples(calib_cache, args.samples)

    from transformers import AutoConfig, AutoModelForCausalLM
    from accelerate import init_empty_weights

    print("⚙️ Loading config...")
    config = AutoConfig.from_pretrained(str(orig_config_path), trust_remote_code=True)
    with open(model_path / "model.safetensors.index.json") as f:
        weight_index = json.load(f)
    weight_map = weight_index["weight_map"]

    print("📦 Instantiating model shell on META device...")
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device != "cpu":
        torch.backends.cuda.matmul.allow_tf32 = True   # tensor-core fp32 matmul (~2x on Ampere)
        torch.backends.cudnn.allow_tf32 = True
    print(f"🖥️ device: {device}")

    input_ids = [torch.tensor(s, dtype=torch.long) for s in samples]
    batches = [torch.stack(input_ids[i:i + args.batch_size])
               for i in range(0, len(input_ids), args.batch_size)]
    print(f"🧪 {len(batches)} batches of size {args.batch_size}")

    # ── capture layer-0 inputs + forward kwargs (identical to calibrate path) ───────
    print("\n🚀 Step 1: capture initial activations + forward kwargs...")
    embed_weight = load_tensor_from_shards(model_path, weight_map, "model.embed_tokens.weight")
    vocab, dim = embed_weight.shape
    embed_tokens = nn.Embedding(vocab, dim).to(device=device, dtype=torch.bfloat16)
    embed_tokens.weight.data.copy_(embed_weight.to(device=device, dtype=torch.bfloat16))
    del embed_weight

    layer0 = model.model.layers[0]
    for pname, _ in list(layer0.named_parameters()):
        w = load_tensor_from_shards(model_path, weight_map, f"model.layers.0.{pname}")
        assign_tensor_to_module(layer0, pname, w, device)

    captured = []
    orig_fwd = layer0.forward

    def l0_wrapper(hidden_states, *a, **k):
        a_cpu = [x.cpu() if isinstance(x, torch.Tensor) else x for x in a]
        k_cpu = {kk: (v.cpu() if isinstance(v, torch.Tensor) else v) for kk, v in k.items()}
        if isinstance(k_cpu.get("position_embeddings"), tuple):
            k_cpu["position_embeddings"] = tuple(t.cpu() for t in k_cpu["position_embeddings"])
        k_cpu["past_key_value"] = None
        k_cpu["past_key_values"] = None
        captured.append({"args": a_cpu, "kwargs": k_cpu})
        return orig_fwd(hidden_states, *a, **k)

    layer0.forward = l0_wrapper
    model.model.embed_tokens = embed_tokens
    with torch.no_grad():
        for b in tqdm(batches, desc="   embed pass"):
            try:
                model(b.to(device))
            except Exception:
                pass
    layer0.forward = orig_fwd
    for pname, param in list(layer0.named_parameters()):
        revert_module_param_to_meta(layer0, pname, param.shape)

    layer_inputs = []
    with torch.no_grad():
        for b in batches:
            layer_inputs.append(embed_tokens(b.to(device)).cpu())
    del embed_tokens
    torch.cuda.empty_cache()
    layer_kwargs = captured
    print(f"   captured {len(layer_inputs)} activation batches")

    # QEP keeps a 2nd CLEAN-FP activation stream alongside the quantized one; at layer 0 they are
    # identical (no upstream error yet), then they diverge as layers get quantized.
    QEP = args.qep
    SSR = args.ssr
    CDQ = args.cdquant
    clean_inputs = [t.clone() for t in layer_inputs] if QEP else None
    if QEP:
        print(f"🔗 QEP inter-layer error compensation ON (α={args.qep_alpha}); dual clean+quant stream")
    if SSR:
        print("🔀 SSR ON: MLP intermediate channel reordering (down_proj per-256 scale fit, foldable)")
    if CDQ:
        print(f"🪛 CDQuant ON: {args.cd_sweeps}-sweep coordinate-descent assignment refinement after GPTQ")

    # ── staging + crash-resume ──────────────────────────────────────────────────────
    stats = {"layers": {}}
    staging_dir = output_dir / "_recovery_staging"
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_index = {}

    # The propagated activation stream is deterministic given the (quantized) weights of
    # earlier layers, so checkpointing it lets us resume mid-model after a crash.
    present = [l for l in range(NUM_HIDDEN_LAYERS)
               if (staging_dir / f"inputs_after_{l}.pt").exists()]
    start_layer = 0
    if present:
        A = max(present)
        if all((staging_dir / f"layer_{l}.safetensors").exists() for l in range(A + 1)):
            qep_ok = (not QEP) or (staging_dir / f"clean_after_{A}.pt").exists()
            if not qep_ok:
                print("⚠️ QEP clean-stream checkpoint missing — restart at 0.")
            else:
                start_layer = A + 1
                print(f"⏩ Resume: layers 0..{A} done; loading inputs_after_{A}.pt")
                layer_inputs = torch.load(str(staging_dir / f"inputs_after_{A}.pt"))
                if QEP:
                    clean_inputs = torch.load(str(staging_dir / f"clean_after_{A}.pt"))
                for l in range(A + 1):
                    p = staging_dir / f"layer_{l}.safetensors"
                    with safe_open(str(p), framework="pt") as f:
                        for k in f.keys():
                            staged_index[k] = p
        else:
            print("⚠️ Activation checkpoint present but staged weights missing — restart at 0.")
    if start_layer >= NUM_HIDDEN_LAYERS:
        print("✅ All layers already recovered; jumping to save.")

    def filter_kwargs(layer, kwargs):
        params = inspect.signature(layer.forward).parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return kwargs
        return {k: v for k, v in kwargs.items() if k in params}

    def run_layer_forward(layer, desc, collect_outputs=True, inputs=None):
        """Inference forward over all batches → next-layer inputs (CPU). No grad.
        With collect_outputs=False (the Pass-A capture, whose outputs are discarded) we
        never build the full CPU output list — the hooks have already harvested what we
        need, so holding ~all-batches of outputs would just be wasted host RAM.
        `inputs` defaults to the quantized stream (layer_inputs); pass the clean stream to
        propagate it through an FP layer (QEP)."""
        src = layer_inputs if inputs is None else inputs
        outs = [] if collect_outputs else None
        with torch.no_grad():
            for idx, h in enumerate(tqdm(src, desc=desc, leave=False)):
                a = [x.to(device) if isinstance(x, torch.Tensor) else x
                     for x in layer_kwargs[idx]["args"]]
                kw = {}
                for k, v in layer_kwargs[idx]["kwargs"].items():
                    if isinstance(v, torch.Tensor):
                        kw[k] = v.to(device)
                    elif isinstance(v, tuple):
                        kw[k] = tuple(t.to(device) for t in v)
                    else:
                        kw[k] = v
                kw["use_cache"] = False
                out = layer(h.to(device), *a, **filter_kwargs(layer, kw))
                o = out[0] if isinstance(out, tuple) else out
                if collect_outputs:
                    outs.append(o.cpu())
                del out, o
        return outs

    def qep_capture(layer, targets, clean_src):
        """Dual-stream capture for QEP. Per batch, run the FP layer on BOTH the quantized-propagated
        input (X̂, from layer_inputs) and the clean-propagated input (X, from clean_src), reading
        each target Linear's input under each via hooks. Accumulate, per linear, Ĥ = Σ X̂ᵀX̂ (quant
        Gram, = the H GPTQ uses) and M = Σ XᵀX̂ (clean×quant cross-Gram). Also collect the clean
        FP-layer outputs = the next clean stream. Only one batch's inputs are live at a time."""
        grams = {n: None for n in targets}            # name -> [Ĥ, N]
        cross = {n: None for n in targets}            # name -> M
        cur = {}
        def mk(name):
            def hook(mod, inp, out):
                cur[name] = inp[0].detach().reshape(-1, inp[0].shape[-1]).to(torch.float32)
            return hook
        hooks = [m.register_forward_hook(mk(n)) for n, m in targets.items()]
        clean_outs = []
        with torch.no_grad():
            for idx in tqdm(range(len(layer_inputs)), desc=f"   L{l} QEP capture (dual)", leave=False):
                a = [x.to(device) if isinstance(x, torch.Tensor) else x
                     for x in layer_kwargs[idx]["args"]]
                kw = {}
                for k, v in layer_kwargs[idx]["kwargs"].items():
                    if isinstance(v, torch.Tensor):
                        kw[k] = v.to(device)
                    elif isinstance(v, tuple):
                        kw[k] = tuple(t.to(device) for t in v)
                    else:
                        kw[k] = v
                kw["use_cache"] = False
                fkw = filter_kwargs(layer, kw)
                cur.clear()
                layer(layer_inputs[idx].to(device), *a, **fkw)        # quantized-path input → Â
                ahat = {n: cur[n] for n in targets}
                cur.clear()
                oc = layer(clean_src[idx].to(device), *a, **fkw)      # clean-path input → A (+ output)
                oc = oc[0] if isinstance(oc, tuple) else oc
                clean_outs.append(oc.cpu())
                for n in targets:
                    ah, ac = ahat[n], cur[n]
                    if grams[n] is None:
                        d = ah.shape[1]
                        grams[n] = [torch.zeros(d, d, device=device, dtype=torch.float32), 0]
                        cross[n] = torch.zeros(d, d, device=device, dtype=torch.float32)
                    grams[n][0].addmm_(ah.t(), ah); grams[n][1] += ah.shape[0]
                    cross[n].addmm_(ac.t(), ah)                       # M += Aᵀ Â
                del ahat, oc
        for h in hooks:
            h.remove()
        return grams, cross, clean_outs

    # ── per-layer recovery loop ───────────────────────────────────────────────────
    print("\n🚀 Step 2: per-linear ternary recovery (rotation-only model in, recovered out)")
    for l in range(start_layer, NUM_HIDDEN_LAYERS):
        print(f"\n⚡ Recovering layer {l + 1}/{NUM_HIDDEN_LAYERS}...")
        layer = model.model.layers[l]
        for pname, _ in list(layer.named_parameters()):
            w = load_tensor_from_shards(model_path, weight_map, f"model.layers.{l}.{pname}")
            assign_tensor_to_module(layer, pname, w, device)

        if SSR and _ssr_permute_mlp(layer, device):        # group MLP intermediate channels (foldable, on-grid)
            if l == start_layer:
                print("   [SSR] MLP intermediate channels reordered (down_proj cols ↔ gate/up rows)")

        # which Linears in this layer are quantization targets
        targets = {}
        for name, module in layer.named_modules():
            full = f"model.layers.{l}.{name}.weight"
            if isinstance(module, nn.Linear) and should_quantize(full):
                targets[name] = module

        qep_clean_outs = None
        if not targets:
            print("   (no quantization targets in this layer — passing through FP16)")
        else:
            stats["layers"][str(l)] = {}
            cross = None
            if QEP:
                # Dual-stream capture: Ĥ (quant Gram, for GPTQ) + M (clean×quant cross-Gram, for QEP).
                grams, cross, qep_clean_outs = qep_capture(layer, targets, clean_inputs)
            else:
                # Pass A: accumulate each target Linear's Gram H=ΣXᵀX during the FP forward, collapsed
                # on the fly in the hook (never caches raw per-batch inputs — that froze the box).
                grams = {name: None for name in targets}       # name -> [H (device fp32), N]
                hooks = []

                def mk_hook(name):
                    def hook(mod, inp, out):
                        X = inp[0].detach().reshape(-1, inp[0].shape[-1]).to(torch.float32)
                        g = grams[name]
                        if g is None:
                            H0 = torch.zeros(X.shape[1], X.shape[1],
                                             device=X.device, dtype=torch.float32)
                            grams[name] = g = [H0, 0]
                        g[0].addmm_(X.t(), X)                  # H += Xᵀ X
                        g[1] += X.shape[0]
                        del X
                    return hook

                for name, module in targets.items():
                    hooks.append(module.register_forward_hook(mk_hook(name)))
                run_layer_forward(layer, f"   L{l} capture (FP)", collect_outputs=False)
                for h in hooks:
                    h.remove()

            # Reconstruct each target Linear (independent given its prebuilt Gram).
            for name, module in targets.items():
                g = grams[name]
                if g is None or g[1] == 0:
                    print(f"   {name:28s} ⚠️ no activations captured — left FP16", flush=True)
                    continue
                if QEP:                                        # pre-correct W for upstream error
                    try:
                        Wc = _qep_correct(module.weight.data.to(device, torch.float32),
                                          cross[name], g[0], args.qep_alpha)
                        module.weight.data.copy_(Wc.to(module.weight.dtype))
                        del Wc
                    except Exception as e:
                        print(f"   {name:28s} ⚠️ QEP correct failed ({type(e).__name__}); raw W", flush=True)
                    cross[name] = None
                r = reconstruct_linear(module, g, args.block_size,
                                       args.iters, args.lr, device,
                                       sensitivity=(SENS.get(f"model.layers.{l}.{name}.weight")
                                                    if SENS is not None else None),
                                       cdquant=CDQ, cd_sweeps=args.cd_sweeps)
                stats["layers"][str(l)][name] = r
                grams[name] = None                             # free this linear's Gram
                print(f"   {name:28s} cos={r['cosine']:.4f}  spars={r['sparsity']:.3f}  "
                      f"mse {r['init_mse']:.3e} → {r['final_mse']:.3e}", flush=True)
            del grams
            cross = None                                    # free the QEP cross-Grams (M), if any
            if device != "cpu":
                torch.cuda.empty_cache()                    # once per layer, not per linear

        # Advance the CLEAN-FP stream FIRST (for QEP) so the *previous* clean stream is released
        # before Pass B allocates the next quantized stream. This caps peak host RAM at ~2 activation
        # streams instead of 3-4 — the transient stacking that drove the 2026-06-18 host freeze.
        if QEP:
            if targets:
                clean_next, qep_clean_outs = qep_clean_outs, None
            else:
                clean_next = run_layer_forward(layer, f"   L{l} propagate (clean)", inputs=clean_inputs)
            clean_inputs = clean_next            # previous clean stream dropped here
            del clean_next
            gc.collect()

        # Pass B: propagate QUANTIZED activations to the next layer (error compensation)
        next_layer_inputs = run_layer_forward(layer, f"   L{l} propagate (quantized)")

        # stage weights to disk, revert to meta
        layer_sd = {}
        for pname, param in list(layer.named_parameters()):
            full = f"model.layers.{l}.{pname}"
            layer_sd[full] = param.data.cpu().contiguous()
            revert_module_param_to_meta(layer, pname, param.shape)
        lf = staging_dir / f"layer_{l}.safetensors"
        tmp = staging_dir / f"layer_{l}.safetensors.tmp"
        save_safetensors(layer_sd, str(tmp))
        os.replace(str(tmp), str(lf))
        for k in layer_sd:
            staged_index[k] = lf
        del layer_sd, layer
        torch.cuda.empty_cache()
        gc.collect()

        layer_inputs = next_layer_inputs
        af = staging_dir / f"inputs_after_{l}.pt"
        tmpa = staging_dir / f"inputs_after_{l}.pt.tmp"
        torch.save(layer_inputs, str(tmpa))
        os.replace(str(tmpa), str(af))
        prev = staging_dir / f"inputs_after_{l - 1}.pt"
        if prev.exists():
            prev.unlink()
        if QEP:                                            # checkpoint the clean stream in lockstep
            cf = staging_dir / f"clean_after_{l}.pt"
            tmpc = staging_dir / f"clean_after_{l}.pt.tmp"
            torch.save(clean_inputs, str(tmpc))
            os.replace(str(tmpc), str(cf))
            pcf = staging_dir / f"clean_after_{l - 1}.pt"
            if pcf.exists():
                pcf.unlink()

    # ── save (assemble shards from staging + passthrough originals) ─────────────────
    print("\n💾 Step 3: saving recovered model...")
    for fname in ["config.json", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "generation_config.json", "merges.txt",
                  "vocab.json", "preprocessor_config.json", "chat_template.json"]:
        src = model_path / fname
        if src.exists() and src.resolve() != (modified_model_dir / fname).resolve():
            shutil.copy2(str(src), str(modified_model_dir / fname))
    for f in model_path.glob("*.model"):
        if f.resolve() != (modified_model_dir / f.name).resolve():
            shutil.copy2(str(f), str(modified_model_dir / f.name))

    shards = {}
    for name, shard_file in weight_map.items():
        shards.setdefault(shard_file, []).append(name)

    for shard_file, names in sorted(shards.items()):
        print(f"   💾 {modified_model_dir / shard_file}")
        shard_dict = {}
        for name in names:
            key = name
            if "model.language_model.layers" in name:
                key = name.replace("model.language_model.layers", "model.layers")
            if key in staged_index:
                with safe_open(str(staged_index[key]), framework="pt") as f:
                    shard_dict[name] = f.get_tensor(key)
            else:
                shard_dict[name] = load_tensor_from_shards(model_path, weight_map, name, "cpu")
        save_safetensors(shard_dict, str(modified_model_dir / shard_file))

    with open(modified_model_dir / "model.safetensors.index.json", "w") as f:
        json.dump(weight_index, f, indent=2)

    for p in staging_dir.glob("layer_*.safetensors"):
        p.unlink()
    for p in staging_dir.glob("inputs_after_*.pt"):
        p.unlink()
    try:
        staging_dir.rmdir()
    except OSError:
        pass

    with open(output_dir / "recovery_report.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n📊 Recovery complete. Report: {output_dir / 'recovery_report.json'}")


if __name__ == "__main__":
    main()