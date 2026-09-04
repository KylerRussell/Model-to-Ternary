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
import math
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


# ───────────────────────── NVMe-backed activation streams (RAM cap) ─────────────────
# Block-AP holds the whole propagated activation stream (and, transiently, the FP-target
# and next-layer streams) in CPU RAM. That RAM is LINEAR in calibration tokens, so it caps
# how far QAT can scale: on a 60 GB box the fixed overhead (model + framework ≈ 24 GB)
# plus 2-3 concurrent streams OOM-kills the host around 512 samples @ seq 2560 — the exact
# crash that motivated this. GPTQ Hessians (d×d) do NOT scale with tokens, so QAT is the
# only data-hungry consumer and the only thing this needs to fix.
#
# ACT_SPILL_GB (env) > 0 routes every activation stream to NVMe and keeps only a shared,
# byte-bounded LRU window resident. Total activation RAM is then CONSTANT in token count
# (= the budget), no matter how many samples — so QAT can no longer OOM the host. Default
# 0 = OFF ⇒ behaviour is byte-identical to before (plain python lists, everything in RAM).
#
# The budget is GLOBAL across all live streams (layer_inputs + fp_outs + next_inputs), so
# the cap is honoured regardless of how many streams coexist. When a whole stream fits in
# the budget it stays fully cached (zero disk reads after the first epoch); only larger
# streams spill and re-read from NVMe (the accepted cost of scaling past RAM).
_ACT_SPILL_GB = float(os.environ.get("ACT_SPILL_GB", "0"))


class _ActCache:
    """Process-global byte-bounded LRU over (stream_uid, idx) → CPU tensor."""
    def __init__(self, budget_bytes):
        from collections import OrderedDict
        self.budget = int(budget_bytes)
        self.od = OrderedDict()
        self.used = 0

    @staticmethod
    def _nb(obj):
        # Resident-byte estimate. Handles bare tensors AND the dict/list/tuple entries used for
        # layer_kwargs (position_embeddings, masks, …); non-tensor leaves count as 0.
        if torch.is_tensor(obj):
            return obj.element_size() * obj.nelement()
        if isinstance(obj, dict):
            return sum(_ActCache._nb(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(_ActCache._nb(v) for v in obj)
        return 0

    def get(self, key):
        t = self.od.get(key)
        if t is not None:
            self.od.move_to_end(key)
        return t

    def put(self, key, t):
        if key in self.od:
            self.used -= self._nb(self.od.pop(key))
        self.od[key] = t
        self.used += self._nb(t)
        # keep at least the just-inserted item even if it alone exceeds the budget
        while self.used > self.budget and len(self.od) > 1:
            _, old = self.od.popitem(last=False)
            self.used -= self._nb(old)

    def drop_stream(self, uid):
        for k in [k for k in self.od if k[0] == uid]:
            self.used -= self._nb(self.od.pop(k))


_ACT_CACHE = None   # lazily created (once) when spilling is enabled


class DiskActivationList:
    """List-like activation stream backed by NVMe with a shared bounded RAM cache.

    Supports exactly the operations block-AP uses on its streams: append, len, positional
    __getitem__ (the hot QAT read), and iteration. Each item is a .pt on disk; the shared
    _ACT_CACHE bounds resident bytes. Dropping the object (e.g. `layer_inputs = next_inputs`)
    unlinks that stream's files, so disk does not grow across layers.

    NOT picklable on purpose: torch.save(stream) would silently pull the whole stream back
    into RAM. The resume checkpoint (inputs_after_*.pt) is therefore skipped while spilling.
    """
    def __init__(self, root):
        import uuid
        self.uid = uuid.uuid4().hex
        self.dir = Path(root) / f"act_{self.uid}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.n = 0

    @staticmethod
    def _to_cpu(obj):
        # Move any tensors to CPU before spilling, including those nested in the layer_kwargs
        # dicts (position_embeddings tuple, masks). Non-tensor leaves pass through unchanged.
        if torch.is_tensor(obj):
            return obj.detach().contiguous().cpu()
        if isinstance(obj, dict):
            return {k: DiskActivationList._to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(DiskActivationList._to_cpu(v) for v in obj)
        return obj

    def append(self, t):
        t = self._to_cpu(t)
        i = self.n
        torch.save(t, str(self.dir / f"{i}.pt"))
        _ACT_CACHE.put((self.uid, i), t)
        self.n += 1

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if i < 0:
            i += self.n
        if i < 0 or i >= self.n:
            raise IndexError(i)
        t = _ACT_CACHE.get((self.uid, i))
        if t is None:
            t = torch.load(str(self.dir / f"{i}.pt"))
            _ACT_CACHE.put((self.uid, i), t)
        return t

    def __iter__(self):
        for i in range(self.n):
            yield self[i]

    def __del__(self):
        try:
            if _ACT_CACHE is not None:
                _ACT_CACHE.drop_stream(self.uid)
            shutil.rmtree(self.dir, ignore_errors=True)
        except Exception:
            pass


class _ConstList:
    """Read-only list that returns ONE shared object for every index. Used for layer_kwargs when
    the per-batch forward kwargs (position_embeddings/mask/cache_position) are identical across all
    batches — which they are for fixed-length packed calibration. Turns an O(samples) RAM structure
    into O(1) with no disk I/O in the hot QAT loop."""
    def __init__(self, item, n):
        self._item, self._n = item, n
    def __len__(self):
        return self._n
    def __getitem__(self, i):
        if i < 0:
            i += self._n
        if i < 0 or i >= self._n:
            raise IndexError(i)
        return self._item
    def __iter__(self):
        for _ in range(self._n):
            yield self._item


def _obj_equal(a, b):
    """Structural equality for the nested kwargs dicts (tensors compared by value)."""
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_obj_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_obj_equal(x, y) for x, y in zip(a, b))
    return a == b


def _bound_kwargs(captured, new_stream, spilling):
    """Bound the O(samples) layer_kwargs. When spilling: if every entry equals the first (the usual
    case — identical position/mask metadata for equal-length packed sequences), collapse to a single
    shared copy (O(1) RAM); otherwise spill the differing entries to NVMe. When not spilling, return
    the plain list unchanged (byte-identical legacy behaviour)."""
    if not spilling or len(captured) <= 1:
        return captured
    if all(_obj_equal(captured[0], e) for e in captured):
        print(f"   [act-spill] layer_kwargs identical across {len(captured)} batches → single shared copy",
              flush=True)
        return _ConstList(captured[0], len(captured))
    print(f"   [act-spill] layer_kwargs differ across batches → spilling {len(captured)} to NVMe", flush=True)
    ds = new_stream()
    for e in captured:
        ds.append(e)
    return ds


def _init_act_spill(spill_root):
    """Enable spilling if ACT_SPILL_GB>0. Returns (new_stream_factory, spilling_bool)."""
    global _ACT_CACHE
    if _ACT_SPILL_GB <= 0:
        return (lambda: []), False
    if _ACT_CACHE is None:
        _ACT_CACHE = _ActCache(_ACT_SPILL_GB * (1024 ** 3))
    root = Path(spill_root) / "act_spill"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    print(f"💽 ACT_SPILL ON: activation streams on NVMe ({root}), "
          f"shared RAM cap {_ACT_SPILL_GB:.1f} GB (constant in token count)", flush=True)
    return (lambda: DiskActivationList(root)), True


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
    wr = _tern_round(wc)
    wq = wc + (wr - wc).detach()           # STE
    return (wq * s.unsqueeze(1)).reshape(out, inp)


def _deploy_ternary(W: torch.Tensor, s: torch.Tensor, block_size: int):
    """Detached final weight (true ternary*scale with the learned scale) + sparsity."""
    flat, (out, inp) = _blocks(W, block_size)
    q = _tern_round((flat / s.unsqueeze(1)).clamp(-1, 1))
    deq = (q * s.unsqueeze(1)).reshape(out, inp)
    sparsity = (q == 0).float().mean().item()
    return deq, sparsity


def _block_scale(W1):
    """Per-row MSE-optimal ternary scale for one block: search multipliers of mean|w| and pick the
    one minimising local ternary MSE. The single per-block scale sets BOTH the round threshold
    (0.5·s) and the level (±s), so absmax over-sparsifies and plain mean is not optimal — a small
    search wins (GPTQ's `find_params`). Returns s [out] (one scale per output row for this block)."""
    m = W1.abs().mean(1, keepdim=True).clamp_min(1e-8)               # [N,1]
    best_s, best_mse = m.squeeze(1).clone(), None                    # loop candidates: peak [N,block], NOT
    for c in torch.linspace(0.6, 2.4, 19, device=W1.device).tolist():  # [N,19,block] (~6GB on a 27B MLP linear)
        s = m * c                                                   # [N,1]
        q = _tern_round((W1 / s).clamp(-1, 1)) * s                  # [N,block] (κ-aware)
        mse = ((W1 - q) ** 2).mean(1)                               # [N]
        if best_mse is None:
            best_mse, best_s = mse, s.squeeze(1).clone()
        else:
            better = mse < best_mse
            best_s = torch.where(better, s.squeeze(1), best_s)
            best_mse = torch.where(better, mse, best_mse)
    return best_s                                                    # [N]


def _mse_rtn_ternary(W, block_size, chunk=8192):
    """MSE-optimal RTN ternary for a top-level weight (lm_head / embed_tokens): per-(row, block) scale
    from the same search block-AP uses, then round. Returns the dequantized ON-GRID weight (in {-s,0,+s}
    per block) so extract_ternary_scale (amax) recovers it losslessly. Chunked over rows for the big
    [vocab, hidden] tensors. Block is over the input/hidden dim (columns)."""
    out, inp = W.shape
    assert inp % block_size == 0, f"{inp} not divisible by block {block_size}"
    nb = inp // block_size
    dev = W.device
    outq = torch.empty_like(W)
    for r0 in range(0, out, chunk):
        r1 = min(r0 + chunk, out)
        flat = W[r0:r1].reshape((r1 - r0) * nb, block_size).float()
        s = _block_scale(flat).clamp_min(1e-8)                       # MSE-optimal per (row,block) scale
        q = _tern_round((flat / s.unsqueeze(1)).clamp(-1, 1)) * s.unsqueeze(1)
        outq[r0:r1] = q.reshape(r1 - r0, inp).to(W.dtype)
    return outq


def _grad_scale(x, g):                       # LSQ: scale the gradient by g, value unchanged
    return (x - x * g).detach() + x * g


class _QATLinear(nn.Module):
    """Latent fp weight (FP-init) + learnable per-block scale, STE-ternary forward. Used for
    EfficientQAT-style block reconstruction: train these to match the FP block's OUTPUT, then
    deploy() to a true ternary weight. Folds to the SAME ternary+per-256-scale as post-hoc."""
    def __init__(self, weight, bias, block_size, init_deq=None):
        super().__init__()
        self.bs = block_size
        self.out, self.inp = weight.shape
        if init_deq is not None:
            # GPTQ warm-start: put the latent AT the GPTQ ternary solution (deq ∈ {-s,0,+s} per block)
            # and set scale = the per-block level |s|, so STE reproduces the GPTQ assignment EXACTLY at
            # step 0. A tiny-LR polish then only moves off GPTQ if the block objective says to.
            self.latent = nn.Parameter(init_deq.detach().to(torch.float32))
            flat, _ = _blocks(self.latent.data, block_size)
            self.scale = nn.Parameter(flat.abs().amax(dim=1).clamp_min(1e-8))  # |nonzero level| == GPTQ scale
        else:
            self.latent = nn.Parameter(weight.detach().to(torch.float32))
            flat, _ = _blocks(self.latent.data, block_size)
            self.scale = nn.Parameter(_block_scale(flat).clamp_min(1e-8))      # [out*nblocks] MSE-opt init
        self.register_buffer("bias_t", None if bias is None else bias.detach().clone())

    def forward(self, x):
        s = _grad_scale(self.scale.clamp_min(1e-8), 1.0 / math.sqrt(self.bs))
        w = ste_ternary(self.latent, s, self.bs).to(x.dtype)
        return F.linear(x, w, self.bias_t.to(x.dtype) if self.bias_t is not None else None)

    @torch.no_grad()
    def deploy(self):
        return _deploy_ternary(self.latent.data, self.scale.data.clamp_min(1e-8), self.bs)


_AR_GAMMA, _AR_ZETA = -0.1, 1.1                                       # AdaRound rectified-sigmoid stretch


class _AdaRoundLinear(nn.Module):
    """A2: AdaRound ternary learned rounding (arXiv:2004.10568). Fixed MSE-opt per-block scale s; a
    per-weight continuous var V (rectified sigmoid → h∈[0,1]) chooses round-UP vs round-DOWN within the
    ternary grid, trained on block-output MSE + an annealed regulariser that forces h→{0,1}. A principled,
    *guarded* flip-learner — only flips a weight when it lowers the objective — i.e. the continuous analogue
    of keep-best, without STE's grid oscillation. Used on the low-coupling attention/DeltaNet linears."""
    def __init__(self, weight, bias, block_size):
        super().__init__()
        self.bs = block_size; self.out, self.inp = weight.shape
        flat, _ = _blocks(weight.detach().float(), block_size)
        s = _block_scale(flat).clamp_min(1e-8)                        # [out*nblocks] fixed MSE-opt scale
        self.register_buffer("s", s)
        t = (flat / s.unsqueeze(1)).clamp(-1, 1)                      # continuous ternary coord ∈[-1,1]
        self.register_buffer("floor", torch.floor(t))                # base level ∈{-1,0}(,1 at t=1)
        alpha = (t - self.floor).clamp(1e-4, 1 - 1e-4)               # sub-grid residual ∈(0,1)
        self.V = nn.Parameter(-torch.log((_AR_ZETA - _AR_GAMMA) / (alpha - _AR_GAMMA) - 1))  # init soft==W/s
        self.register_buffer("bias_t", None if bias is None else bias.detach().clone())
        self.beta = 20.0                                             # rounding-reg temperature (annealed ↓)

    def _h(self):
        return (torch.sigmoid(self.V) * (_AR_ZETA - _AR_GAMMA) + _AR_GAMMA).clamp(0, 1)

    def _wq(self, hard):
        h = (self.V >= 0).float() if hard else self._h()             # hard: threshold at deploy
        q = (self.floor + h).clamp(-1, 1)                            # ternary ∈{-1,0,1}
        return (q * self.s.unsqueeze(1)).reshape(self.out, self.inp)

    def forward(self, x):
        w = self._wq(False).to(x.dtype)
        return F.linear(x, w, self.bias_t.to(x.dtype) if self.bias_t is not None else None)

    def reg(self):                                                   # Σ 1-|2h-1|^β → 0 as h→{0,1}
        h = self._h()
        return (1 - (2 * h - 1).abs().pow(self.beta)).sum()

    @torch.no_grad()
    def deploy(self):
        deq = self._wq(True)
        flat, _ = _blocks(deq, self.bs)
        return deq, (flat.abs() < 1e-12).float().mean().item()


def _gptq_actorder(W_fp, H, s_full, percdamp, block_size):
    """GPTQ with ACT-ORDER: quantise input columns in DESCENDING Hessian-diagonal order (most-salient
    first, so the remaining columns compensate the largest errors), then un-permute so the deployed
    weight is in natural column order. Each column carries its NATURAL contiguous-256-block scale
    (precomputed in `s_full` from the original weight), so after un-permuting, every contiguous block
    is uniform-scale → TQ2_0 grid-compliant with ZERO side tensor (no runtime permutation/index). The
    error-feedback compute uses a chunk of `block_size` purely for the lazy batched Hinv update — that
    is independent of the per-256 SCALE grid here (scale is per-column via s_full)."""
    out, inp = W_fp.shape
    dev = W_fp.device
    perm = torch.argsort(torch.diag(H), descending=True)
    invperm = torch.argsort(perm)
    W = W_fp[:, perm].clone()
    Hp = H[perm][:, perm].clone()
    s_p = s_full[:, perm]
    di = torch.arange(inp, device=dev)
    Hp[di, di] += percdamp * torch.diag(Hp).mean()
    Hinv = torch.linalg.cholesky(Hp)
    Hinv = torch.cholesky_inverse(Hinv)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    Q = torch.zeros_like(W)
    for i1 in range(0, inp, block_size):
        i2 = min(i1 + block_size, inp)
        W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); E1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]; s1 = s_p[:, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]; d = Hinv1[i, i]; sc = s1[:, i]
            q = _tern_round((w / sc).clamp(-1, 1)) * sc
            Q1[:, i] = q
            e = (w - q) / d
            W1[:, i:] -= e.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)
            E1[:, i] = e
        Q[:, i1:i2] = Q1
        if i2 < inp:
            W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q[:, invperm]                              # back to natural order (contiguous block scales)


_ZERO_KAPPA = 1.0     # A5 grid-geometry knob: zero-threshold multiplier. Rounding zeroes |w/s| < 0.5·κ.
                      # κ<1 → fewer zeros (finer ±1 resolution), κ>1 → more zeros. Set from --zero-thresh-kappa.


def _tern_round(x, kappa=None):
    """κ-aware ternary rounding of x∈[-1,1]: 0 iff |x| < 0.5·κ else sign(x). κ=1 ≡ torch.round."""
    k = _ZERO_KAPPA if kappa is None else kappa
    if k == 1.0:
        return torch.round(x)
    return torch.where(x.abs() < 0.5 * k, torch.zeros_like(x), torch.sign(x))


# SchurOpt grid refit, off by default so the validated recipe is unchanged. Env/CLI settable.
_GPTQ_REFIT_ITERS = int(os.environ.get("GPTQ_REFIT_ITERS", "0"))


def _schur_refit(W1, S, s, iters):
    """SchurOpt (arXiv 2608.15567) Prop. 2, specialised to SYMMETRIC ternary.

    The paper's two named gaps in GPTQ-family PTQ are (a) group decisions ignore what the continuous
    suffix can absorb and (b) "discrete refinements typically keep the affine quantization grid
    fixed". This is (b): our grid `s` came from _block_scale's UNWEIGHTED MSE search, chosen once
    BEFORE any code is picked, and was never revisited. Prop. 2 instead solves for the grid that is
    optimal GIVEN the codes, under the group curvature S:

        codes  z  = tern_round(W1 / s)
        scale  s* = diag(Z S W1^T) / diag(Z S Z^T)        (Eq. 16 with zero-point o = 0)

    Ternary is symmetric so the zero-point drops out and only the scale remains. Alternating the two
    is the single largest component in the paper's own ablation on Qwen3-4B at 2 bits -- grid refit
    alone moves the controlled loss 1.645% -> 0.872% and PPL 2344.82 -> 152.95, more than Schur
    conditioning alone (1.165% / 324.81). The two compose (0.550% / 86.64).

    S is the group curvature: the raw block Hessian here. Format-safe -- one positive scale per row
    per block, exactly what TQ1_64 already stores, so bpw is unchanged.
    """
    s = s.reshape(-1)                                     # [N]; _block_scale returns per-row, not [N,1]
    for _ in range(iters):
        z = _tern_round((W1 / s.unsqueeze(1)).clamp(-1, 1))
        ZS = z @ S                                        # [N, g]
        num = (ZS * W1).sum(1)                            # diag(Z S W1^T)
        den = (ZS * z).sum(1)                             # diag(Z S Z^T)
        s_new = num / den.clamp_min(1e-12)                # [N]
        # A row whose codes are all zero (or a degenerate solve) has no defined scale: keep the
        # previous one rather than emit a non-positive or non-finite grid.
        ok = torch.isfinite(s_new) & (s_new > 0)
        s = torch.where(ok, s_new, s)
    return s.clamp_min(1e-8)


def _gptq_ternary(W_fp, H, block_size, percdamp=0.01, act_order=False, refit_iters=None):
    """One-shot GPTQ/OBC error-feedback ternary fit. Quantises input columns left→right; each
    column's rounding residual is pushed into the not-yet-quantised columns through the inverse
    Hessian (H = XᵀX), so the OUTPUT error ‖X(W−Q)ᵀ‖² is *compensated*, not just locally minimised
    per weight — the inter-weight coupling our previous Adam-on-Gram+STE fit ignored. A group =
    `block_size` consecutive input columns = exactly one TQ2_0 block per output row (with its own
    absmax scale), so Q ∈ {−s,0,+s} is exactly representable on the per-256 ternary grid. With
    act_order=True, quantise high-Hessian columns first (un-permuted at the end; foldable). Returns
    the deployed Q [out,inp]."""
    if refit_iters is None:
        refit_iters = _GPTQ_REFIT_ITERS
    out, inp = W_fp.shape
    dev = W_fp.device
    W = W_fp.clone().float()
    H = H.clone().float()
    dead = torch.diag(H) == 0                        # input channels with no activation variance
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    if act_order:
        # per-column scale from the ORIGINAL natural contiguous blocks (so un-permuted deploy is grid-ok)
        s_full = torch.empty_like(W)
        for i1 in range(0, inp, block_size):
            i2 = min(i1 + block_size, inp)
            s_full[:, i1:i2] = _block_scale(W[:, i1:i2]).clamp_min(1e-8).unsqueeze(1)
        return _gptq_actorder(W, H, s_full, percdamp, block_size)
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
        if refit_iters > 0:                          # SchurOpt Eq. 16: curvature-aware grid refit
            s = _schur_refit(W1, H[i1:i2, i1:i2], s, refit_iters)
        for i in range(i2 - i1):
            w = W1[:, i]
            d = Hinv1[i, i]
            q = _tern_round((w / s).clamp(-1, 1)) * s            # ternary {−s,0,+s} (κ-aware zero threshold)
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
                       cdquant=False, cd_sweeps=4, act_order=False):
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
        gptq_deploy = _gptq_ternary(W_fp, H, block_size, act_order=act_order)
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


def _permute_mlp(layer, mode, perm_map=None, seed=0, layer_idx=0):
    """MLP-intermediate channel permutation (A1/B1 axis; generalizes the old SSR-only hook). Block
    membership of down_proj's per-256 scales is set purely by input-column ORDER, so a permutation of the
    intermediate channels re-groups which weights share a scale. Permute down_proj's input columns and fold
    the SAME permutation into gate_proj/up_proj OUTPUT rows — SwiGLU has no norm between them and the
    intermediate is NOT rotated, so the output is unchanged, the fold is offline/local, and the result
    stays strictly on the per-256 ternary grid (no side tensor). Modes:
      ssr       — PT²-LLM structural-similarity sort by down-column norm (outliers cluster into few blocks)
      random    — seeded random permutation (the kill-check CONTROL arm)
      optimized — per-layer permutation from --perm-file (B1 ternary-distortion local search)"""
    mlp = getattr(layer, "mlp", None)
    need = ("gate_proj", "up_proj", "down_proj")
    if mlp is None or not all(hasattr(mlp, n) for n in need):
        return False
    dw = mlp.down_proj.weight.data
    if mode == "ssr":
        P = torch.argsort(dw.float().norm(dim=0))          # [intermediate] small→large; outliers to the tail
    elif mode == "random":
        g = torch.Generator().manual_seed(int(seed) * 100003 + int(layer_idx))
        P = torch.randperm(dw.shape[1], generator=g)
    elif mode == "optimized":
        if perm_map is None or int(layer_idx) not in perm_map:
            return False
        P = perm_map[int(layer_idx)].to(torch.long)
    else:
        return False
    P = P.to(dw.device)
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
    ap.add_argument("--joint-mlp", action="store_true",
                    help="C6: joint MLP reconstruction. After gate/up are ternarized, refit down_proj to "
                         "map the TERNARY intermediate h_st -> the CLEAN FP MLP output (down absorbs the "
                         "gate/up quantization error), via the QEP correction on the intra-MLP clean×quant "
                         "cross-Gram. Foldable (still ternary gate/up/down + scales). 4B probe: -41pct MLP out-err.")
    ap.add_argument("--joint-mlp-alpha", type=float, default=1.0,
                    help="Correction strength for --joint-mlp (1.0 = full least-squares down correction).")
    ap.add_argument("--ssr", action="store_true",
                    help="SSR (PT²-LLM): reorder MLP intermediate channels (foldable into gate/up/down) "
                         "so down_proj's per-256 scale fits each block — outliers clustered, not scattered.")
    ap.add_argument("--perm-mode", choices=["none", "ssr", "random", "optimized"], default="none",
                    help="A1/B1 MLP-intermediate permutation mode (--ssr is an alias for 'ssr'). "
                         "'random' = kill-check control (see --perm-seed); 'optimized' needs --perm-file.")
    ap.add_argument("--perm-seed", type=int, default=0, help="Seed for --perm-mode random (per-layer derived).")
    ap.add_argument("--perm-file", default=None,
                    help="torch.save'd {layer_idx: LongTensor} permutations for --perm-mode optimized "
                         "(from src/perm_search_b1.py).")
    ap.add_argument("--zero-thresh-kappa", type=float, default=1.0,
                    help="A5 grid geometry: ternary zero-threshold multiplier (round zeroes |w/s|<0.5·κ). "
                         "κ<1 = fewer zeros, κ>1 = more zeros. 1.0 = standard rounding.")
    ap.add_argument("--quant-embed-head", action="store_true",
                    help="Bonsai-parity: ternarize embed_tokens (MSE-RTN) + lm_head (GPTQ-Hessian). lm_head also "
                         "needs 'lm_head.weight' in QUANTIZE_PATTERNS so build_student loads it as a trainable TSL.")
    ap.add_argument("--gptq-embed", action="store_true",
                    help="With --quant-embed-head, ternarize embed_tokens by GPTQ (embedding-output Gram) instead "
                         "of MSE-RTN — minimises downstream error rather than uniform weight error.")
    ap.add_argument("--cdquant", action="store_true",
                    help="CDQuant: greedy/Jacobi coordinate-descent refinement of the ternary assignments "
                         "after GPTQ (same on-grid objective, stronger local search; keep-best, ≥ GPTQ).")
    ap.add_argument("--cd-sweeps", type=int, default=4, help="CDQuant max coordinate-descent sweeps per linear.")
    ap.add_argument("--gptq-refit-iters", type=int, default=0,
                    help="SchurOpt (arXiv 2608.15567) Eq. 16 grid refit: alternate ternary codes with "
                         "the curvature-weighted closed-form per-row scale s* = diag(ZSW^T)/diag(ZSZ^T), "
                         "instead of keeping _block_scale's one-shot UNWEIGHTED MSE grid. 0 = off "
                         "(validated recipe). Format-safe: still one positive scale per row per block, "
                         "so bpw is unchanged. Converges by ~2 iters; 4 is ample. Gain scales with "
                         "Hessian anisotropy (measured on synthetic groups: 0.1%% isotropic, 16.9%% at "
                         "a realistic power-law condition ~4e3).")
    ap.add_argument("--act-order", action="store_true",
                    help="GPTQ act-order: quantise high-Hessian-diagonal columns first, un-permuted at the "
                         "end (each column keeps its natural contiguous-block scale → grid-compliant, no "
                         "side tensor). Off by default.")
    ap.add_argument("--qat", action="store_true",
                    help="EfficientQAT block reconstruction: per-LAYER latent-weight QAT (FP-init + STE "
                         "ternary + learnable per-256 scale) trained to match the FP block output, instead "
                         "of per-linear GPTQ. Data-efficient + overfit-resistant; folds to the same TQ2_0.")
    ap.add_argument("--qat-epochs", type=int, default=2, help="QAT passes over the calib per layer.")
    ap.add_argument("--qat-gptq-init", action="store_true",
                    help="Warm-start QAT latents at the per-linear GPTQ ternary solution instead of cold FP "
                         "(the report's Experiment-1 decider). STE reproduces GPTQ at step 0, so a tiny-LR "
                         "polish tests whether QAT is optimizer-bound (improves off GPTQ) or objective-bound "
                         "(drifts worse). Captures per-linear grams + runs _gptq_ternary before the QAT loop.")
    ap.add_argument("--qat-keep-best", action="store_true",
                    help="Per-linear floor: after the QAT polish, keep whichever of {QAT-deploy, GPTQ-init} "
                         "has lower per-linear Gram-MSE — guarantees the folded skeleton is ≥ GPTQ (mirrors "
                         "the RTN keep-best guard). Requires --qat-gptq-init (needs the GPTQ deq + gram).")
    ap.add_argument("--qat-attn-only", action="store_true",
                    help="A1 routing: polish ONLY the low-coupling attention/DeltaNet projections; FREEZE the "
                         "MLP linears at their GPTQ-init skeleton (excluded from the optimizer). MLP is "
                         "near-Babai-optimal (GPTQ≡CVP) so its polish always reverts — freezing it removes STE "
                         "thrash from the shared block-output loss and lets the attention LR be raised safely. "
                         "Requires --qat-gptq-init. MLP stays in the forward (frozen) so the loss is exact.")
    ap.add_argument("--qat-adaround", action="store_true",
                    help="A2: use AdaRound learned rounding (arXiv:2004.10568) instead of STE for the polished "
                         "linears — a guarded continuous flip-learner without STE grid oscillation. Fixed scale, "
                         "learns per-weight round up/down + annealed rounding reg. Requires --qat-gptq-init.")
    ap.add_argument("--qat-round-lambda", type=float, default=1e-3,
                    help="A2 AdaRound rounding-regulariser weight (Σ 1-|2h-1|^β term). Larger = harder/faster "
                         "commitment to the ternary grid.")
    ap.add_argument("--qat-route-coupling", action="store_true",
                    help="A3: route by MEASURED Hessian off-diagonal mass ‖H-diag(H)‖/‖H‖ instead of by name — "
                         "freeze (GPTQ) the high-coupling linears (near-Babai-optimal), polish the low-coupling "
                         "ones. Supersedes --qat-attn-only's name test. Requires --qat-gptq-init. Logs per-linear "
                         "off-diag mass + keep/revert so the coupling→keepable correlation can be read off.")
    ap.add_argument("--coupling-thresh", type=float, default=0.9,
                    help="A3 threshold on normalised off-diagonal mass ‖H-diag‖/‖H‖; linears ABOVE it are frozen "
                         "at GPTQ. Empirically these Grams are off-diag-dominated (mass≈0.83–1.0), so the useful "
                         "band is ~0.85–0.99, not 0.5. Sweep knob; the split it induces is the A3 result.")
    ap.add_argument("--qat-loss", choices=["mse", "saliency"], default="mse",
                    help="A4: block-output reconstruction objective. 'mse' = plain per-block output MSE; "
                         "'saliency' = Fisher/end-loss-saliency-weighted MSE (GuidedQuant arXiv:2505.07004), "
                         "weighting each output channel by the FP model's ∂(LM-loss)²/∂out — aligns the polish "
                         "with generalisation-relevant directions (the OOD channel), not raw variance.")
    ap.add_argument("--qat-saliency-cache", default=None,
                    help="A4: path to the {layer_idx: [hidden] saliency} cache from src/build_saliency.py "
                         "(required when --qat-loss saliency).")
    ap.add_argument("--qat-lr", type=float, default=1e-4, help="QAT latent-weight LR.")
    ap.add_argument("--qat-scale-lr", type=float, default=1e-5, help="QAT per-256 scale LR.")
    ap.add_argument("--qat-block-layers", type=int, default=1,
                    help="Multi-layer block QAT: reconstruct N decoder layers JOINTLY per block (local "
                         "co-adaptation across the DeltaNet recurrence) to the N-layer FP output, instead "
                         "of single-layer. N=1 = current behavior. Use 2-4 (memory grows with N).")
    ap.add_argument("--tequila", action="store_true",
                    help="Tequila deadzone->bias: fold the ternary quantization RESIDUAL's mean contribution "
                         "(R @ E[x], where R = latent - deployed_ternary, incl. dead weights rounded to 0) "
                         "into a per-output bias — recovers un-representable capacity at ~zero inference cost.")
    ap.add_argument("--qat-teacher-force", action="store_true",
                    help="DIAGNOSTIC (Teacher Intervention): propagate the CLEAN FP stream so every block "
                         "trains on clean input (input=FP-propagated, target=FP_layer(clean)). Isolates "
                         "ternary CAPACITY from drift-compounding: if per-layer MSE flattens, the problem is "
                         "fixable drift; if it still explodes with clean input, it is intrinsic (go 2-bit).")
    args = ap.parse_args()
    globals()['_GPTQ_REFIT_ITERS'] = int(getattr(args, 'gptq_refit_iters', 0) or 0)
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
    # NVMe-backed streams (RAM cap): BOTH the activation streams AND the per-batch forward-kwargs
    # (position_embeddings/masks) are O(calibration tokens); both must be bounded or host RAM still
    # grows with samples. new_stream() → DiskActivationList when ACT_SPILL_GB>0, else a plain list.
    new_stream, _SPILL = _init_act_spill(output_dir / "_recovery_staging")
    embed_weight = load_tensor_from_shards(model_path, weight_map, "model.embed_tokens.weight")
    vocab, dim = embed_weight.shape
    embed_tokens = nn.Embedding(vocab, dim).to(device=device, dtype=torch.bfloat16)
    embed_tokens.weight.data.copy_(embed_weight.to(device=device, dtype=torch.bfloat16))
    del embed_weight

    layer0 = model.model.layers[0]
    for pname, _ in list(layer0.named_parameters()):
        w = load_tensor_from_shards(model_path, weight_map, f"model.layers.0.{pname}")
        assign_tensor_to_module(layer0, pname, w, device)

    captured = []                    # deduped in-place during capture; bounded after (below)
    orig_fwd = layer0.forward

    def l0_wrapper(hidden_states, *a, **k):
        a_cpu = [x.cpu() if isinstance(x, torch.Tensor) else x for x in a]
        k_cpu = {kk: (v.cpu() if isinstance(v, torch.Tensor) else v) for kk, v in k.items()}
        if isinstance(k_cpu.get("position_embeddings"), tuple):
            k_cpu["position_embeddings"] = tuple(t.cpu() for t in k_cpu["position_embeddings"])
        k_cpu["past_key_value"] = None
        k_cpu["past_key_values"] = None
        entry = {"args": a_cpu, "kwargs": k_cpu}
        # On-the-fly dedup: for equal-length packed calib the kwargs (position_embeddings/mask/
        # cache_position) are identical every batch. Reference-share the first entry instead of
        # retaining N distinct copies, so capture RAM is O(1) even at very large token counts —
        # the new copies are dropped when this returns. (Differing entries are kept as-is.)
        if _SPILL and captured and _obj_equal(captured[0], entry):
            captured.append(captured[0])
        else:
            captured.append(entry)
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

    layer_inputs = new_stream()      # disk-backed when spilling (factory created before the capture above)
    with torch.no_grad():
        for b in batches:
            layer_inputs.append(embed_tokens(b.to(device)).cpu())
    embed_H = None                                            # GPTQ-embed: Gram of the embedding OUTPUTS (residual-stream
    if getattr(args, "gptq_embed", False):                   # input) → GPTQ minimises downstream error vs uniform MSE-RTN
        hid = layer_inputs[0].shape[-1]
        embed_H = torch.zeros(hid, hid, device=device, dtype=torch.float32); ne = 0
        for x in layer_inputs:
            xx = x.to(device).float().reshape(-1, hid)
            embed_H.addmm_(xx.t(), xx); ne += xx.shape[0]; del xx
        print(f"   [gptq-embed] embed Gram from {ne} embedding-output tokens")
    del embed_tokens
    torch.cuda.empty_cache()
    layer_kwargs = _bound_kwargs(captured, new_stream, _SPILL)   # collapse identical / spill differing
    del captured
    print(f"   captured {len(layer_inputs)} activation batches")

    # QEP keeps a 2nd CLEAN-FP activation stream alongside the quantized one; at layer 0 they are
    # identical (no upstream error yet), then they diverge as layers get quantized.
    QEP = args.qep
    global _ZERO_KAPPA
    _ZERO_KAPPA = float(args.zero_thresh_kappa)
    if _ZERO_KAPPA != 1.0:
        print(f"🎯 A5 zero-threshold κ = {_ZERO_KAPPA} (round zeroes |w/s| < {0.5*_ZERO_KAPPA:.3f})")
    PERM_MODE = args.perm_mode if args.perm_mode != "none" else ("ssr" if args.ssr else "none")
    PERM_MAP = None
    if args.perm_file:
        PERM_MAP = torch.load(args.perm_file, map_location="cpu")
        print(f"   [perm] loaded optimized permutations for {len(PERM_MAP)} layers from {args.perm_file}")
    CDQ = args.cdquant
    ACTORDER = args.act_order
    saliency_cache = None                                        # A4: {layer_idx: [hidden] channel weights}
    if args.qat_loss == "saliency":
        sc = getattr(args, "qat_saliency_cache", None)
        if not sc or not os.path.exists(sc):
            raise SystemExit(f"--qat-loss saliency needs --qat-saliency-cache (missing: {sc}). "
                             f"Build it with src/build_saliency.py.")
        raw = torch.load(sc)
        saliency_cache = {int(k): v.float() for k, v in raw.items()}
        print(f"🎯 A4 saliency-weighted objective: loaded {len(saliency_cache)} per-layer channel-weight vectors")
    clean_inputs = [t.clone() for t in layer_inputs] if QEP else None
    if QEP:
        print(f"🔗 QEP inter-layer error compensation ON (α={args.qep_alpha}); dual clean+quant stream")
    if PERM_MODE != "none":
        print(f"🔀 PERM ON ({PERM_MODE}): MLP intermediate channel reordering (down_proj per-256 scale fit, foldable)")
    if CDQ:
        print(f"🪛 CDQuant ON: {args.cd_sweeps}-sweep coordinate-descent assignment refinement after GPTQ")
    if ACTORDER:
        print("🔢 ACT-ORDER ON: GPTQ quantises high-Hessian columns first (un-permuted, grid-compliant)")

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
        outs = new_stream() if collect_outputs else None
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

    def joint_down_grams(layer, gate_fp, up_fp):
        """C6: with gate/up ALREADY ternarized in the layer, run one forward over the calib and, per
        batch, form the ternary intermediate h_st = act(gate_q x)·(up_q x) and the clean intermediate
        h_fp = act(gate_fp x)·(up_fp x). Accumulate Ĥ = Σ h_stᵀh_st (the Gram GPTQ will use for down)
        and M = Σ h_fpᵀh_st (clean×quant cross-Gram) so _qep_correct fits down_proj to reproduce the
        CLEAN MLP output from the CORRUPTED intermediate — absorbing the gate/up quantization error."""
        mlp = layer.mlp
        act = mlp.act_fn
        Wg_q = mlp.gate_proj.weight.data.to(device, torch.float32)   # already ternary
        Wu_q = mlp.up_proj.weight.data.to(device, torch.float32)
        Wg_f = gate_fp.to(device, torch.float32)
        Wu_f = up_fp.to(device, torch.float32)
        cap = {}
        def pre(mod, inp):
            cap["x"] = inp[0].detach().reshape(-1, inp[0].shape[-1]).to(device, torch.float32)
        hk = mlp.gate_proj.register_forward_pre_hook(pre)
        H = M = None
        N = 0
        with torch.no_grad():
            for idx in tqdm(range(len(layer_inputs)), desc=f"   L{l} joint-MLP capture", leave=False):
                a = [x.to(device) if isinstance(x, torch.Tensor) else x for x in layer_kwargs[idx]["args"]]
                kw = {}
                for k, v in layer_kwargs[idx]["kwargs"].items():
                    kw[k] = (v.to(device) if isinstance(v, torch.Tensor)
                             else tuple(t.to(device) for t in v) if isinstance(v, tuple) else v)
                kw["use_cache"] = False
                layer(layer_inputs[idx].to(device), *a, **filter_kwargs(layer, kw))
                x = cap["x"]
                h_fp = act(x @ Wg_f.T) * (x @ Wu_f.T)
                h_st = act(x @ Wg_q.T) * (x @ Wu_q.T)
                if H is None:
                    d = h_st.shape[1]
                    H = torch.zeros(d, d, device=device, dtype=torch.float32)
                    M = torch.zeros(d, d, device=device, dtype=torch.float32)
                H.addmm_(h_st.t(), h_st); M.addmm_(h_fp.t(), h_st); N += h_st.shape[0]
                del x, h_fp, h_st
        hk.remove()
        return H, N, M

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

    def run_multilayer_qat():
        """(a) Multi-layer block QAT: reconstruct N decoder layers JOINTLY to the N-layer FP output, so the
        layers CO-ADAPT to each other's ternary error (vs single-layer, which can't). Reuses the split +
        staging; folds to the same TQ2_0. Populates staged_index so Step 3 save works unchanged."""
        nonlocal layer_inputs
        from torch.utils.checkpoint import checkpoint as _ckpt
        def _sub(root, dotted, mod):
            *par, leaf = dotted.split(".");  p = root
            for q in par: p = getattr(p, q)
            setattr(p, leaf, mod)
        N = args.qat_block_layers; two_gpu = torch.cuda.device_count() >= 2; ntok = len(layer_inputs)
        print(f"\n🚀 Step 2 (multi-layer QAT): {N}-layer JOINT block reconstruction, {ntok} samples")
        l = start_layer
        while l < NUM_HIDDEN_LAYERS:
            nb = min(N, NUM_HIDDEN_LAYERS - l); block = list(range(l, l + nb))
            print(f"\n⚡ QAT block L{l}..L{l+nb-1} ({nb} layers jointly)")
            layers = []
            for li in block:
                lyr = model.model.layers[li]
                for pname, _ in list(lyr.named_parameters()):
                    assign_tensor_to_module(lyr, pname, load_tensor_from_shards(model_path, weight_map, f"model.layers.{li}.{pname}"), device)
                layers.append(lyr)
            h = layer_inputs                                   # FP block target: chain nb FP layers
            for li, lyr in zip(block, layers):
                h = run_layer_forward(lyr, f"   Lblk{li} FP target", inputs=h, collect_outputs=True)
            fp_outs = h
            qmods = {}; split_hooks = []
            for li, lyr in zip(block, layers):
                for p in lyr.parameters(): p.requires_grad_(False)
                tg = {nm: m for nm, m in lyr.named_modules()
                      if isinstance(m, nn.Linear) and should_quantize(f"model.layers.{li}.{nm}.weight")}
                for nm, m in tg.items():
                    dev_t = "cuda:1" if (two_gpu and "mlp" in nm) else device
                    qm = _QATLinear(m.weight.data.to(dev_t), m.bias.data.to(dev_t) if m.bias is not None else None, args.block_size)
                    _sub(lyr, nm, qm); qmods[(li, nm)] = qm
                if two_gpu and hasattr(lyr, "mlp"):
                    lyr.mlp.to("cuda:1")
                    def _pre(mod, a_, k_, _d="cuda:1"):
                        return (tuple(x.to(_d) if torch.is_tensor(x) else x for x in a_),
                                {k:(v.to(_d) if torch.is_tensor(v) else v) for k,v in k_.items()})
                    def _post(mod, a_, k_, o_, _d=device):
                        if torch.is_tensor(o_): return o_.to(_d)
                        if isinstance(o_, tuple): return tuple(x.to(_d) if torch.is_tensor(x) else x for x in o_)
                        return o_
                    split_hooks += [lyr.mlp.register_forward_pre_hook(_pre, with_kwargs=True),
                                    lyr.mlp.register_forward_hook(_post, with_kwargs=True)]
            allp = [q.latent for q in qmods.values()] + [q.scale for q in qmods.values()]
            opt = torch.optim.AdamW([{"params":[q.latent for q in qmods.values()],"lr":args.qat_lr},
                                     {"params":[q.scale for q in qmods.values()],"lr":args.qat_scale_lr}])
            nsteps = max(1, args.qat_epochs * ntok); step = 0
            for ep in range(args.qat_epochs):
                rm = 0.0
                for idx in torch.randperm(ntok).tolist():
                    a = [x.to(device) if isinstance(x, torch.Tensor) else x for x in layer_kwargs[idx]["args"]]
                    kw = {}
                    for k, v in layer_kwargs[idx]["kwargs"].items():
                        kw[k] = (v.to(device) if isinstance(v, torch.Tensor)
                                 else tuple(t.to(device) for t in v) if isinstance(v, tuple) else v)
                    kw["use_cache"] = False
                    h = layer_inputs[idx].to(device)
                    for lyr in layers:
                        fkw = filter_kwargs(lyr, kw)
                        out = _ckpt(lambda hh, _L=lyr, _a=a, _k=fkw: _L(hh, *_a, **_k), h, use_reentrant=False)
                        h = out[0] if isinstance(out, tuple) else out
                    loss = F.mse_loss(h.float(), fp_outs[idx].to(device).float())
                    opt.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(allp, 1.0)
                    mult = 0.05 + 0.95*0.5*(1+math.cos(math.pi*step/nsteps))
                    opt.param_groups[0]["lr"]=args.qat_lr*mult; opt.param_groups[1]["lr"]=args.qat_scale_lr*mult
                    opt.step(); step += 1; rm += loss.item(); del out, h, loss
                print(f"   blkL{l} QAT epoch {ep+1}/{args.qat_epochs} mean block-MSE {rm/ntok:.3e}", flush=True)
            for (li, nm), qm in qmods.items():                 # fold all layers -> ternary
                deq, spars = qm.deploy()
                lin = nn.Linear(qm.inp, qm.out, bias=qm.bias_t is not None).to(deq.device)
                lin.weight.data = deq.to(torch.bfloat16)
                if qm.bias_t is not None: lin.bias.data = qm.bias_t.to(deq.device, torch.bfloat16)
                _sub(model.model.layers[li], nm, lin)
            del qmods, opt, allp
            if device != "cpu": torch.cuda.empty_cache()
            h = layer_inputs                                   # propagate through the nb folded layers
            for li, lyr in zip(block, layers):
                h = run_layer_forward(lyr, f"   Lblk{li} propagate", inputs=h, collect_outputs=True)
            next_inputs = h; del fp_outs
            for li, lyr in zip(block, layers):                 # stage each layer, revert to meta
                sd = {}
                for pname, param in list(lyr.named_parameters()):
                    sd[f"model.layers.{li}.{pname}"] = param.data.cpu().contiguous()
                    revert_module_param_to_meta(lyr, pname, param.shape)
                lf = staging_dir / f"layer_{li}.safetensors"; tp = staging_dir / f"layer_{li}.safetensors.tmp"
                save_safetensors(sd, str(tp)); os.replace(str(tp), str(lf))
                for k in sd: staged_index[k] = lf
                del sd
            for hk in split_hooks: hk.remove()
            del layers; torch.cuda.empty_cache(); gc.collect()
            layer_inputs = next_inputs; l += nb

    # ── recovery: multi-layer JOINT QAT (a), else per-layer loop ─────────────────────
    _ml_qat = args.qat and args.qat_block_layers > 1
    if _ml_qat:
        run_multilayer_qat()
    print("\n🚀 Step 2: per-linear ternary recovery (rotation-only model in, recovered out)")
    for l in range(start_layer, NUM_HIDDEN_LAYERS):
        if _ml_qat:
            break
        print(f"\n⚡ Recovering layer {l + 1}/{NUM_HIDDEN_LAYERS}...")
        layer = model.model.layers[l]
        for pname, _ in list(layer.named_parameters()):
            w = load_tensor_from_shards(model_path, weight_map, f"model.layers.{l}.{pname}")
            assign_tensor_to_module(layer, pname, w, device)

        if PERM_MODE != "none" and _permute_mlp(layer, PERM_MODE, PERM_MAP, args.perm_seed, l):
            if l == start_layer:
                print(f"   [perm:{PERM_MODE}] MLP intermediate channels reordered (down_proj cols ↔ gate/up rows)")

        # which Linears in this layer are quantization targets
        targets = {}
        for name, module in layer.named_modules():
            full = f"model.layers.{l}.{name}.weight"
            if isinstance(module, nn.Linear) and should_quantize(full):
                targets[name] = module

        qep_clean_outs = None
        if not targets:
            print("   (no quantization targets in this layer — passing through FP16)")
        elif args.qat:
            # ── EfficientQAT block reconstruction: latent-weight QAT to match the FP block OUTPUT ──
            stats["layers"][str(l)] = {}
            fp_outs = run_layer_forward(layer, f"   L{l} FP target", collect_outputs=True)   # FP weights still loaded

            def _set_sub(root, dotted, mod):
                *par, leaf = dotted.split(".")
                p = root
                for q in par:
                    p = getattr(p, q)
                setattr(p, leaf, mod)

            for p in layer.parameters():
                p.requires_grad_(False)                          # freeze norms/conv/gates (strict: stay FP)
            qmods = {}
            two_gpu = torch.cuda.device_count() >= 2
            # GPTQ warm-start (report Experiment-1): capture each target's input Gram on the FP forward and
            # solve the per-linear GPTQ ternary fit, so the QAT latent starts AT GPTQ instead of cold FP.
            gptq_deq, gptq_gram, wfp, offdiag_mass = {}, {}, {}, {}
            if args.qat_gptq_init:
                gg = {name: None for name in targets}
                def _mk_g(nm):
                    def h(mod, inp, out):
                        X = inp[0].detach().reshape(-1, inp[0].shape[-1]).to(torch.float32)
                        g = gg[nm]
                        if g is None:
                            gg[nm] = g = [torch.zeros(X.shape[1], X.shape[1], device=X.device,
                                                      dtype=torch.float32), 0]
                        g[0].addmm_(X.t(), X); g[1] += X.shape[0]; del X
                    return h
                ghooks = [m.register_forward_hook(_mk_g(n)) for n, m in targets.items()]
                run_layer_forward(layer, f"   L{l} GPTQ-init gram (FP)", collect_outputs=False)
                for h in ghooks:
                    h.remove()
                for name, module in targets.items():
                    g = gg[name]
                    if g is None or g[1] == 0:
                        continue
                    try:
                        gptq_deq[name] = _gptq_ternary(module.weight.data.to(g[0].device, torch.float32),
                                                       g[0], args.block_size, act_order=ACTORDER)
                    except Exception as e:
                        print(f"   {name:28s} ⚠️ GPTQ-init failed ({type(e).__name__}); cold FP init", flush=True)
                        continue
                    if args.qat_keep_best:                       # stash H,N,W_fp for the ≥GPTQ floor compare
                        gptq_gram[name] = (g[0], g[1]); wfp[name] = module.weight.data.detach().clone()
                    if args.qat_route_coupling:                  # A3: normalised off-diag Gram mass ‖H-diag‖/‖H‖
                        H = g[0]; fro = H.norm()
                        offdiag_mass[name] = float((H - torch.diag(torch.diagonal(H))).norm() / fro.clamp_min(1e-12))
                del gg
            def _route_frozen(name):                              # True → freeze this linear at GPTQ (skip polish)
                if args.qat_route_coupling and name in offdiag_mass:   # A3: high-coupling → GPTQ near-optimal
                    return offdiag_mass[name] > args.coupling_thresh
                if args.qat_attn_only:                            # A1: freeze MLP by name
                    return "mlp" in name
                return False
            frozen_names = set()
            for name, module in targets.items():
                dev_t = "cuda:1" if (two_gpu and "mlp" in name) else device   # MLP→cuda:1, attn+norms→cuda:0
                w = module.weight.data.to(dev_t)
                b = module.bias.data.to(dev_t) if module.bias is not None else None
                _idq = gptq_deq.get(name)
                frz = _route_frozen(name)
                if args.qat_adaround and not frz and _idq is not None:   # A2: AdaRound on the POLISHED linears
                    qm = _AdaRoundLinear(w, b, args.block_size).to(dev_t)
                else:                                             # STE latent (or frozen GPTQ skeleton)
                    qm = _QATLinear(w, b, args.block_size, init_deq=(_idq.to(dev_t) if _idq is not None else None))
                if frz:                                           # A1/A3: freeze at GPTQ, exclude from optimiser
                    for p in qm.parameters():
                        p.requires_grad_(False)
                    frozen_names.add(name)
                _set_sub(layer, name, qm); qmods[name] = qm       # so cuda:0 never accumulates the whole layer
                if args.tequila:                                  # accumulate E[x] per linear during training
                    qm._xsum, qm._xn = None, 0
                    def _acc(mod, inp):
                        x = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                        mod._xsum = x.sum(0) if mod._xsum is None else mod._xsum + x.sum(0)
                        mod._xn += x.shape[0]
                    qm.register_forward_pre_hook(_acc)
            # One 27B layer's QAT (fp32 latents + AdamW states + the STE activation graph) overflows a
            # single 24GB GPU, and checkpointing can't help a SINGLE layer (the backward recompute still
            # peaks at one full layer fwd). So SPLIT the layer across both GPUs: MLP (the bulk) on cuda:1,
            # attention+norms on cuda:0, with align-hooks moving activations across the boundary.
            split_hooks = []
            if torch.cuda.device_count() >= 2 and hasattr(layer, "mlp"):
                d1 = "cuda:1"; layer.mlp.to(d1)
                def _pre(mod, a_, k_):
                    return (tuple(x.to(d1) if torch.is_tensor(x) else x for x in a_),
                            {k: (v.to(d1) if torch.is_tensor(v) else v) for k, v in k_.items()})
                def _post(mod, a_, k_, out_):
                    if torch.is_tensor(out_): return out_.to(device)
                    if isinstance(out_, tuple):
                        return tuple(o.to(device) if torch.is_tensor(o) else o for o in out_)
                    return out_
                split_hooks = [layer.mlp.register_forward_pre_hook(_pre, with_kwargs=True),
                               layer.mlp.register_forward_hook(_post, with_kwargs=True)]
                print(f"   L{l} QAT split across 2 GPUs (mlp→cuda:1, rest→cuda:0)", flush=True)
            # 32-bit AdamW (NOT 8-bit): the per-block scale params are tiny [N] vectors and bnb's blockwise
            # 8-bit quantization of their optimizer state under-trains them → bad assignments the E2E can't
            # fix. The 2-GPU split + chunked _block_scale leave ample room for fp32 states.
            trainable = {n: q for n, q in qmods.items() if n not in frozen_names}   # A1/A3: drop frozen linears
            if frozen_names:
                how = "coupling" if args.qat_route_coupling else "attn-only"
                print(f"   L{l} routing ({how}): polishing {len(trainable)}/{len(qmods)} linears, "
                      f"{len(frozen_names)} frozen at GPTQ", flush=True)
            if args.qat_route_coupling and offdiag_mass:         # A3: per-linear off-diag mass + route decision
                for nm in sorted(offdiag_mass, key=offdiag_mass.get, reverse=True):
                    print(f"      coupling {nm:32s} offdiag={offdiag_mass[nm]:.3f}  "
                          f"{'FROZEN(GPTQ)' if nm in frozen_names else 'polish'}", flush=True)
            latent_params, scale_params = [], []                 # AdaRound V shares the qat_lr group
            for q in trainable.values():
                if isinstance(q, _AdaRoundLinear):
                    latent_params.append(q.V)
                else:
                    latent_params.append(q.latent); scale_params.append(q.scale)
            adaround_mods = [q for q in trainable.values() if isinstance(q, _AdaRoundLinear)]
            sal_l = saliency_cache.get(l) if saliency_cache is not None else None    # A4: [hidden] channel weights
            if l == start_layer:
                print(f"   [mem] post-swap cuda:0={torch.cuda.memory_allocated(0)/1e9:.1f}GB"
                      + (f" cuda:1={torch.cuda.memory_allocated(1)/1e9:.1f}GB" if two_gpu else ""), flush=True)
            if not trainable:                                    # all linears frozen (routing) → deploy GPTQ skeleton
                print(f"   L{l} all {len(qmods)} linears frozen at GPTQ (no polish this layer)", flush=True)
            opt = torch.optim.AdamW([{"params": latent_params, "lr": args.qat_lr},
                                     {"params": scale_params, "lr": args.qat_scale_lr}]) if trainable else None
            allp = latent_params + scale_params
            ntok = len(layer_inputs); nsteps = max(1, args.qat_epochs * ntok); step = 0
            for ep in range(args.qat_epochs if trainable else 0):
                run_mse = 0.0
                for idx in torch.randperm(ntok).tolist():
                    a = [x.to(device) if isinstance(x, torch.Tensor) else x for x in layer_kwargs[idx]["args"]]
                    kw = {}
                    for k, v in layer_kwargs[idx]["kwargs"].items():
                        kw[k] = (v.to(device) if isinstance(v, torch.Tensor)
                                 else tuple(t.to(device) for t in v) if isinstance(v, tuple) else v)
                    kw["use_cache"] = False
                    out = layer(layer_inputs[idx].to(device), *a, **filter_kwargs(layer, kw))
                    o = out[0] if isinstance(out, tuple) else out
                    o_f, tgt = o.float(), fp_outs[idx].to(device).float()
                    if sal_l is not None:                        # A4: saliency-weighted block-output MSE
                        loss = (((o_f - tgt) ** 2) * sal_l.to(o_f.device)).mean()
                    else:
                        loss = F.mse_loss(o_f, tgt)
                    mse_item = loss.item()
                    if adaround_mods:                            # A2: annealed rounding regulariser
                        b_now = max(2.0, 20.0 - 18.0 * (step / nsteps))
                        reg = 0.0
                        for q in adaround_mods:                  # reg() lives on each module's device (2-GPU split)
                            q.beta = b_now
                            reg = reg + q.reg().to(o_f.device)   # reduce onto the loss device
                        loss = loss + args.qat_round_lambda * reg
                    opt.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(allp, 1.0)
                    mult = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * step / nsteps))
                    opt.param_groups[0]["lr"] = args.qat_lr * mult
                    opt.param_groups[1]["lr"] = args.qat_scale_lr * mult
                    opt.step(); step += 1; run_mse += mse_item; del out, o, loss
                print(f"   L{l} QAT epoch {ep+1}/{args.qat_epochs}  mean block-MSE {run_mse/ntok:.3e}", flush=True)
            for name, qm in qmods.items():                       # fold latent -> deployed ternary nn.Linear
                deq, spars = qm.deploy()
                if args.qat_keep_best and name in gptq_gram:      # ≥GPTQ floor: revert if the polish lost ground
                    H, N = gptq_gram[name]; Wf = wfp[name].to(H.device).float()
                    def _gmse(Wc):                               # per-linear output MSE tr((Wc-Wf)H(Wc-Wf)ᵀ)
                        D = Wc.to(H.device).float() - Wf
                        return ((D @ H) * D).sum().item() / max(N, 1)
                    if _gmse(gptq_deq[name]) < _gmse(deq):
                        deq = gptq_deq[name].to(deq.device)
                        if hasattr(qm, "latent"):                # STE only; AdaRound has no latent (tequila n/a)
                            qm.latent.data = deq.clone()         # keep latent≡deq so tequila R stays consistent
                        spars = (deq == 0).float().mean().item()
                        print(f"   {name:28s} keep-best → reverted to GPTQ floor", flush=True)
                bias_vec = None
                if args.tequila and getattr(qm, "_xn", 0) > 0:   # Tequila: fold residual mean-contribution -> bias
                    R = (qm.latent.data.float() - deq.float())    # [out,inp] quant residual (incl. dead weights)
                    Ex = (qm._xsum / qm._xn).to(R.device).float() # [inp] mean input over the (quantized) stream
                    bias_vec = R @ Ex                             # [out] mean output error recovered as a bias
                    if qm.bias_t is not None:
                        bias_vec = bias_vec + qm.bias_t.float()
                has_bias = bias_vec is not None or qm.bias_t is not None
                lin = nn.Linear(qm.inp, qm.out, bias=has_bias).to(deq.device)   # may be cuda:1 (split)
                lin.weight.data = deq.to(torch.bfloat16)         # match model/activation dtype (not fp16)
                if bias_vec is not None:
                    lin.bias.data = bias_vec.to(deq.device, torch.bfloat16)
                elif qm.bias_t is not None:
                    lin.bias.data = qm.bias_t.to(deq.device, torch.bfloat16)
                _set_sub(layer, name, lin)
                stats["layers"][str(l)][name] = {"sparsity": spars}
                print(f"   {name:28s} QAT spars={spars:.3f}{' +tequila-bias' if bias_vec is not None else ''}", flush=True)
            del qmods, opt, allp          # fp_outs kept for the epilogue (teacher-force clean propagation)
            if device != "cpu":
                torch.cuda.empty_cache()
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
            mlp_fp = {}                                        # C6: stash FP gate/up before they ternarize
            for name, module in targets.items():
                if args.joint_mlp and (name.endswith("gate_proj") or name.endswith("up_proj")):
                    mlp_fp[name] = module.weight.data.detach().clone()
                if args.joint_mlp and name.endswith("down_proj") and "mlp" in name:
                    # C6: refit down against the TERNARY intermediate + clean output (gate/up already done)
                    base = name.rsplit(".", 1)[0]
                    Hh, Nn, Mm = joint_down_grams(layer, mlp_fp[f"{base}.gate_proj"], mlp_fp[f"{base}.up_proj"])
                    try:
                        Wc = _qep_correct(module.weight.data.to(device, torch.float32),
                                          Mm, Hh, args.joint_mlp_alpha)
                        module.weight.data.copy_(Wc.to(module.weight.dtype)); del Wc
                    except Exception as e:
                        print(f"   {name:28s} ⚠️ joint-MLP correct failed ({type(e).__name__}); raw W", flush=True)
                    r = reconstruct_linear(module, (Hh, Nn), args.block_size, args.iters, args.lr, device,
                                           cdquant=CDQ, cd_sweeps=args.cd_sweeps, act_order=ACTORDER)
                    stats["layers"][str(l)][name] = r
                    del Hh, Mm, grams[name]
                    print(f"   {name:28s} [joint-MLP] cos={r['cosine']:.4f}  spars={r['sparsity']:.3f}  "
                          f"mse {r['init_mse']:.3e} → {r['final_mse']:.3e}", flush=True)
                    continue
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
                                       cdquant=CDQ, cd_sweeps=args.cd_sweeps, act_order=ACTORDER)
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

        # Pass B: propagate to the next layer. TEACHER-FORCE (diagnostic) propagates the CLEAN FP stream
        # (fp_outs = FP_layer(clean)) so every block trains on clean input; else the quantized stream.
        if args.qat and args.qat_teacher_force:
            next_layer_inputs = fp_outs
        else:
            next_layer_inputs = run_layer_forward(layer, f"   L{l} propagate (quantized)")
        if args.qat:
            del fp_outs

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
        # Resume-checkpoint the propagated stream — but ONLY in the legacy in-RAM mode. When spilling,
        # layer_inputs is a DiskActivationList (deliberately not picklable: torch.save would drag the whole
        # stream back into RAM, defeating the cap), so mid-model crash-resume is unavailable in spill mode.
        # Acceptable: spill mode targets long single-shot scaling runs, not resume.
        if not _SPILL:
            af = staging_dir / f"inputs_after_{l}.pt"
            tmpa = staging_dir / f"inputs_after_{l}.pt.tmp"
            torch.save(layer_inputs, str(tmpa))
            os.replace(str(tmpa), str(af))
            prev = staging_dir / f"inputs_after_{l - 1}.pt"
            if prev.exists():
                prev.unlink()
        if QEP and not _SPILL:                             # checkpoint the clean stream in lockstep
            cf = staging_dir / f"clean_after_{l}.pt"
            tmpc = staging_dir / f"clean_after_{l}.pt.tmp"
            torch.save(clean_inputs, str(tmpc))
            os.replace(str(tmpc), str(cf))
            pcf = staging_dir / f"clean_after_{l - 1}.pt"
            if pcf.exists():
                pcf.unlink()

    # ── Bonsai-parity: ternarize embed_tokens (MSE-RTN) + lm_head (GPTQ-Hessian) ─────
    # lm_head directly makes logits incl. EOS — MSE-RTN there miscalibrated the stop token and broke free-gen
    # (chat mode emitted EOS after ~4 tokens). GPTQ minimises the OUTPUT (logit) error using the final-hidden
    # Hessian, preserving EOS calibration far better. embed stays RTN (input lookup, low leverage, gen coherent).
    override_tensors = {}
    if getattr(args, "quant_embed_head", False):
        lmhead_H = None
        try:                                                    # final-hidden Hessian for GPTQ lm_head
            li = layer_inputs if ('layer_inputs' in dir() and layer_inputs) else None
        except Exception:
            li = None
        if li:
            nw = None
            for nn_ in ("model.norm.weight", "model.language_model.norm.weight"):
                if nn_ in weight_map:
                    nw = load_tensor_from_shards(model_path, weight_map, nn_, device).float(); break
            hid = li[0].shape[-1]
            H = torch.zeros(hid, hid, device=device, dtype=torch.float32); ntok = 0
            for h in li:
                hh = h.to(device).float().reshape(-1, hid)
                x = hh * hh.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()   # zero-centered RMSNorm
                if nw is not None: x = x * (1.0 + nw)
                H.addmm_(x.t(), x); ntok += x.shape[0]
                del hh, x
            lmhead_H = H
            print(f"   [quant-embed-head] lm_head GPTQ Hessian from {ntok} final-hidden tokens")
        for disk in list(weight_map.keys()):
            if disk.endswith("embed_tokens.weight") or disk.endswith("lm_head.weight"):
                W = load_tensor_from_shards(model_path, weight_map, disk, device).float()
                if W.shape[-1] % args.block_size != 0:
                    print(f"   [quant-embed-head] SKIP {disk}: hidden {W.shape[-1]} %{args.block_size}≠0")
                    continue
                is_head = disk.endswith("lm_head.weight")
                if is_head and lmhead_H is not None:
                    q = _gptq_ternary(W, lmhead_H, args.block_size); meth = "GPTQ"
                elif (not is_head) and ('embed_H' in dir()) and embed_H is not None:
                    q = _gptq_ternary(W, embed_H, args.block_size); meth = "GPTQ"
                else:
                    q = _mse_rtn_ternary(W, args.block_size); meth = "MSE-RTN"
                spars = float((q == 0).float().mean())
                override_tensors[disk] = q.to(torch.bfloat16).cpu()
                print(f"   [quant-embed-head] ternarized {disk} {tuple(W.shape)} "
                      f"({meth}, block {args.block_size}, {100*spars:.1f}% zeros)")
                del W, q
                torch.cuda.empty_cache()

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
            if name in override_tensors:                       # Bonsai-parity ternary embed/lm_head
                shard_dict[name] = override_tensors[name]
            elif key in staged_index:
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
    shutil.rmtree(staging_dir / "act_spill", ignore_errors=True)   # NVMe activation spill (belt-and-suspenders)
    try:
        staging_dir.rmdir()
    except OSError:
        pass

    with open(output_dir / "recovery_report.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n📊 Recovery complete. Report: {output_dir / 'recovery_report.json'}")


if __name__ == "__main__":
    main()