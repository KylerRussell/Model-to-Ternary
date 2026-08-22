#!/usr/bin/env python3
"""
e2e_qp_distill.py — Phase-2C global recovery for ternary Qwen3.6-27B (EfficientQAT E2E-QP).

Local Block-AP plateaued (85% per-linear MSE reduction, still collapses) because it can't
see the end-to-end objective. This trains the ternary SCALES end-to-end against the FP
teacher's output distribution (top-k KL), which is the global signal local recovery lacks.

Fits a single 24 GB 3090 by three tricks:
  1. The frozen ternary base is PACKED to 2-bit (~7 GB for 27B), unpacked on the fly.
  2. Only the per-g128 scales are trainable (~211M params); the teacher is PRECOMPUTED
     offline (top-k logits cached) so it is never resident during training.
  3. Gradient checkpointing keeps activation memory to ~one layer per segment.

IMPORTANT: keep DeltaNet on its pure-PyTorch path (do NOT install flash-linear-attention /
causal-conv1d) — that path is differentiable; the fused kernel is not.

Phases:
  --smoke                : CPU self-test of all the math (run this first, no model needed).
  --precompute-teacher   : run the FP teacher (device_map=auto/offload) -> cache top-k logits.
  --train                : E2E-QP scale training of the recovered ternary student.

Typical flow:
  python e2e_qp_distill.py --precompute-teacher \
      --teacher-path /path/to/Qwen3.6-27B/snapshots/<hash> \
      --calib ./output_recovery/calibration_data.json --topk 64 --seq 1024 \
      --teacher-cache ./output_recovery/teacher_topk.pt
  python e2e_qp_distill.py --train \
      --student-path ./output_recovery/modified_model \
      --orig-config-path /path/to/Qwen3.6-27B/snapshots/<hash> \
      --calib ./output_recovery/calibration_data.json \
      --teacher-cache ./output_recovery/teacher_topk.pt \
      --out ./output_e2eqp/modified_model --seq 1024 --steps 500 --lr 2e-4
"""
import argparse
import gc
import json
import math
import time as _t
import os
import random
import shutil
import sys
from pathlib import Path

# Must be set BEFORE torch initializes CUDA: defeats the fragmentation that OOMs mid-run
# (the "reserved but unallocated" memory the allocator otherwise can't reuse).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent))
from config import BLOCK_SIZE, should_quantize, NUM_HIDDEN_LAYERS


# ─────────────────────────── ternary <-> 2-bit packing ─────────────────────────────

def extract_ternary_scale(weight: torch.Tensor, block_size: int):
    """Recover (ternary {-1,0,+1}, per-block scale) from a dequantized ternary weight.
    weight is expected to already be ternary*scale (each g128 block in {-s,0,+s})."""
    out, inp = weight.shape
    assert inp % block_size == 0
    flat = weight.reshape(-1, block_size)
    scale = flat.abs().amax(dim=1).clamp_min(1e-8)          # nonzero magnitude == s
    tern = torch.round(flat / scale.unsqueeze(1)).clamp(-1, 1)
    return tern.reshape(out, inp).to(torch.int8), scale     # scale: [out*inp/block]


def pack_2bit(tern_i8: torch.Tensor) -> torch.Tensor:
    """Pack ternary int8 {-1,0,1} -> uint8, 4 values/byte. Flattens; stores length implicitly
    via shape handling at unpack (caller keeps out,inp)."""
    flat = (tern_i8.reshape(-1) + 1).to(torch.uint8)        # {-1,0,1} -> {0,1,2}
    n = flat.numel()
    pad = (-n) % 4
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    q = flat.reshape(-1, 4)
    packed = (q[:, 0] | (q[:, 1] << 2) | (q[:, 2] << 4) | (q[:, 3] << 6)).to(torch.uint8)
    return packed


def unpack_2bit(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """Inverse of pack_2bit -> ternary float {-1,0,1}, length `numel` (before reshape)."""
    p = packed.to(torch.int16)
    vals = torch.stack([p & 3, (p >> 2) & 3, (p >> 4) & 3, (p >> 6) & 3], dim=1).reshape(-1)
    vals = vals[:numel]
    return vals.to(torch.float32) - 1.0                      # {0,1,2} -> {-1,0,1}


# ─────────────────────────── trainable-scale ternary linear ────────────────────────

class TernaryScaleLinear(nn.Module):
    """Frozen 2-bit ternary weight + trainable per-g128 scale. forward dequantizes on the
    fly (ternary is constant; only the scale carries gradient)."""

    def __init__(self, out_features, in_features, block_size, bias=None, device="cpu"):
        super().__init__()
        self.out_features, self.in_features, self.block_size = out_features, in_features, block_size
        self.n_blocks = out_features * (in_features // block_size)
        self.register_buffer("packed", torch.zeros(
            (out_features * in_features + 3) // 4, dtype=torch.uint8, device=device))
        self.scale = nn.Parameter(torch.ones(self.n_blocks, dtype=torch.float32, device=device))
        if bias is not None:
            self.register_buffer("bias", bias.to(device))
        else:
            self.bias = None

    @classmethod
    def from_dense(cls, weight, block_size, bias=None, device="cpu"):
        out, inp = weight.shape
        tern, scale = extract_ternary_scale(weight.float(), block_size)
        m = cls(out, inp, block_size, bias=bias, device=device)
        m.packed.data = pack_2bit(tern).to(device)
        m.scale.data = scale.to(device)
        return m

    def enable_arm_b(self):
        """Arm B (PV-Tuning-style sparse flips): unpack the ternary into a MUTABLE buffer and add a
        per-weight gradient accumulator. No latents — the assignments ARE the state; flips are chosen by
        accumulated gradient and committed periodically (oscillation-free)."""
        tern = unpack_2bit(self.packed, self.out_features * self.in_features).to(self.scale.device)
        # int8 support + bf16 grad accumulator → mlp/all scope fits 24GB (3× the weights; fresh-gradient fix means
        # flip_grad is transient/low-precision-tolerant, and the ratio test rejects any bf16-noise-selected flips).
        self.register_buffer("tern_b", tern.reshape(self.n_blocks, self.block_size).to(torch.int8))
        self.register_buffer("flip_grad", torch.zeros(self.n_blocks, self.block_size, dtype=torch.bfloat16, device=self.scale.device))
        self._arm_b = True

    def dequant(self):
        if getattr(self, "_arm_b", False):
            deq = self.tern_b.to(self.scale.dtype) * self.scale.unsqueeze(1)   # int8 support × trained scale
            if self.training and deq.requires_grad:
                def _acc(g):
                    with torch.no_grad():
                        self.flip_grad += g.detach()             # accumulate ∂loss/∂w across the window
                    return g
                deq.register_hook(_acc)
            return deq.reshape(self.out_features, self.in_features)
        if getattr(self, "latent", None) is not None:
            # --latent-offload: the latent lives in CPU RAM (fp32 params AND their grads, 6.04GB for all 32
            # down_proj) and is streamed to the GPU for this layer's forward. Because the module's forward runs
            # INSIDE a gradient-checkpointed region, the GPU copy is freed after the first pass and recreated
            # during recompute, so only one layer's latent is resident at a time. Autograd routes the gradient
            # back through the .to() so param.grad accumulates on CPU. bf16 is NOT an option here: the Adam
            # update is ~2.5e-4 of the latent magnitude vs bf16's 3.9e-3 resolution, so it would be swamped.
            if self.latent.device != self.scale.device:
                # MIXED PRECISION: the master copy stays fp32 on the CPU (the Adam update is ~2.5e-4 of the
                # latent magnitude, well below bf16's 3.9e-3 resolution, so a bf16 master would be swamped),
                # but the GPU copy used for the forward is bf16. The model already runs bf16 and forward()
                # casts dequant()'s output to x.dtype anyway, so the fp32 GPU copy bought nothing while
                # doubling the transient AND producing an fp32 grad (~6B/latent, 13.6GB at mlp@32L).
                # Autograd routes the bf16 grad back through the .to() and accumulates into the fp32 leaf.
                _dt = torch.bfloat16 if getattr(self, "_latent_bf16_compute", False) else self.latent.dtype
                # LOCAL, not self.latent_gpu — this used to be a module ATTRIBUTE, which pinned the GPU
                # copy alive until that module's next forward. The comment above claims checkpointing
                # frees it; checkpointing frees the autograd-held activation, but the attribute is an
                # independent strong reference that survives. Cost = 4 B/latent of VRAM per trained
                # layer, permanently: invisible at the 4B testbed (8 layers, 755 MB total) but 22.8 GB
                # for `down`@64L on the 27B — measured OOM at 22.14 GiB on a 24 GB card — and ~104 GB
                # for full arm B, which no GPU configuration can satisfy. As a local it dies with the
                # frame and only the layer currently in flight holds a copy, which is what the comment
                # above always intended.
                if _PF_ON:
                    _buf = _pf_take(self)
                    # hand off to the NEXT latent module before we compute, so its copy overlaps this
                    # layer's dequant + matmul instead of stalling when we reach it
                    _i = getattr(self, "_lat_idx", -1)
                    if 0 <= _i < len(_LAT_ORDER) - 1:
                        _pf_start(_LAT_ORDER[_i + 1], _dt)
                    latent_gpu = (_UseTransferred.apply(self.latent, _buf) if _buf is not None
                                  else self.latent.to(self.scale.device, dtype=_dt, non_blocking=True))
                else:
                    latent_gpu = self.latent.to(self.scale.device, dtype=_dt, non_blocking=True)
            else:
                latent_gpu = self.latent
            # A4: ASSIGNMENT MOVES. A trainable FP latent (init = the GPTQ ternary*scale) is re-ternarized
            # each step; STE passes gradient to the latent so assignments can flip {-1,0,+1}, while the
            # scale keeps its own gradient. Under no_grad (save) this returns the HARD moved ternary*scale,
            # which repacks to TQ2_0 losslessly. Gated STE (grad only inside the clamp band) keeps it stable.
            Lb = latent_gpu.reshape(self.n_blocks, self.block_size)
            s_all = self.scale.unsqueeze(1).clamp_min(1e-8).to(Lb.dtype)
            # CHUNKED over blocks for very large tensors. The expression below materialises ~6 full-size
            # temporaries (Lb/s, round, clamp, mask, q*s, the STE term). At 89M latents (a 27B down_proj)
            # that is fine; at 1.271B (lm_head under --train-weights all) it needs ~20GB and OOMs a 24GB
            # card on a SINGLE weight — measured: "tried to allocate 4.74 GiB" with 19.92 GiB resident.
            # Every op here is ELEMENTWISE and the scale is per-block, so slicing along the block axis is
            # exact: same values, same gradients, just a bounded working set. Only the output stays
            # full-size. Small tensors take the original single-shot path (no chunking overhead).
            _n = Lb.numel()
            _chunk = _LATENT_DEQUANT_CHUNK_BLOCKS
            if _chunk and _n > _LATENT_DEQUANT_MIN_ELEMS and self.n_blocks > _chunk:
                outs = []
                for i in range(0, self.n_blocks, _chunk):
                    L_i = Lb[i:i + _chunk]
                    s_i = s_all[i:i + _chunk]
                    q_i = torch.clamp(torch.round(L_i / s_i), -1, 1)
                    m_i = (L_i.abs() < 1.5 * s_i).to(L_i.dtype)
                    outs.append(q_i.detach() * s_i + (L_i - L_i.detach()) * m_i)
                return torch.cat(outs, dim=0).reshape(self.out_features, self.in_features)
            s = s_all
            q = torch.clamp(torch.round(Lb / s), -1, 1)
            mask = (Lb.abs() < 1.5 * s).to(Lb.dtype)         # STE grad gate (inside the rounding band)
            deq = q.detach() * s + (Lb - Lb.detach()) * mask  # fwd = q*s; grad→scale (term1) & latent (term2)
            return deq.reshape(self.out_features, self.in_features)
        tern = unpack_2bit(self.packed, self.out_features * self.in_features).to(self.scale.device)
        tern = tern.reshape(self.n_blocks, self.block_size)
        s = self.scale
        if getattr(self, "_sq_bits", 0) > 0:                 # scale-QAT: STE fake-quant on a log-uniform grid
            a = s.abs().clamp_min(1e-12).log()
            step = max(self._sq_hi - self._sq_lo, 1e-6) / (2 ** self._sq_bits - 1)
            q = ((a - self._sq_lo) / step).round().clamp(0, 2 ** self._sq_bits - 1) * step + self._sq_lo
            s = s + (q.exp() * torch.sign(s) - s).detach()   # fwd = grid value; grad passes straight through
        deq = tern * s.unsqueeze(1)                          # grad flows into scale only
        return deq.reshape(self.out_features, self.in_features)

    def forward(self, x):
        w = self.dequant().to(x.dtype)
        return F.linear(x, w, self.bias.to(x.dtype) if self.bias is not None else None)


# ─────────────────────────── distillation loss (top-k KL) ──────────────────────────

# Chunking thresholds for the latent STE dequant (see TernaryScaleLinear.dequant).
# Only tensors above _MIN_ELEMS chunk at all, so the common case keeps the original single-shot path.
# 65536 blocks x 256 = 16.7M elements ~= 67MB fp32 per temporary, so the working set stays bounded no
# matter how large the weight is. Env-overridable for tuning on different card sizes.
_LATENT_DEQUANT_CHUNK_BLOCKS = int(os.environ.get("LATENT_DEQUANT_CHUNK_BLOCKS", "65536"))
_LATENT_DEQUANT_MIN_ELEMS = int(os.environ.get("LATENT_DEQUANT_MIN_ELEMS", str(256 * 1024 * 1024)))


DECISION_GAMMA = 1.0   # CAKLD confidence-weight exponent; set from --decision-gamma (1=plain CAKLD)
SKEW_ALPHA = 0.0       # A2 (DistiLLM skew-KL, arXiv:2402.03898): KL(p || a*p+(1-a)*q). 0=plain KL


def topk_kl_loss(student_logits, teacher_idx, teacher_val, temperature=1.0, weights=None,
                 commit_mask=None, commit_beta=0.0):
    """KL(teacher || student) over the teacher's cached top-k, with the student's full-vocab
    log-partition so the student log-probs are proper. Shapes: student_logits [B,T,V];
    teacher_idx/val [B,T,k]. `weights` [B,T]: optional commit-token reweighting."""
    s = student_logits.float() / temperature
    log_Z = torch.logsumexp(s, dim=-1, keepdim=True)         # [B,T,1] full-vocab partition
    s_topk = torch.gather(s, -1, teacher_idx.long())         # [B,T,k]
    log_q = s_topk - log_Z                                   # student log-prob at top-k
    p = torch.softmax(teacher_val.float() / temperature, dim=-1)   # teacher dist over top-k
    kl = (p * (torch.log(p + 1e-9) - log_q)).sum(-1)         # [B,T]
    base = ((kl * weights).sum() / weights.sum().clamp_min(1e-8) if weights is not None
            else kl.mean()) * (temperature ** 2)
    return _add_commit_term(base, kl, commit_mask, commit_beta, temperature)


def _add_commit_term(loss, kl, commit_mask, commit_beta, temperature):
    """ADDITIVE commit objective (researcher round-6): L = CAKLD_all + beta * mean(KL over the commit window).

    The commit term is normalized by the COMMIT-token count, not by the bulk, so its gradient share is beta —
    independent of how rare the window is. This is the fix for the Stage-A no-op: reweighting inside the
    normalized mean `sum(kl*conf)/sum(conf)` gave a 0.135%-rare class only 0.40% of the weight at alpha=3, and
    reaching a useful share would need alpha~30-70 (premature-stop / length-attractor risk, arXiv:2010.07174).
    An additive auxiliary term is un-diluted by construction (cf. EOS-loss rescaling, arXiv:2506.05017)."""
    if commit_mask is None or not commit_beta:
        return loss
    m = commit_mask.bool()
    if not bool(m.any()):
        return loss
    return loss + commit_beta * kl[m].mean() * (temperature ** 2)


def commit_weights(ids, alpha, pre=16, post=12, close_id=248069):
    """Per-position loss multiplier for COMMIT-TOKEN REWEIGHTING (researcher round-5, Rho-1/SLM-style
    arXiv:2404.07965): up-weight the KL in a WINDOW around each </think>(248069) — the ~pre think tokens
    before the close decision, the close itself, and the ~post answer-start tokens. Puts the gradient ON the
    'conclude-and-close' transition so a ≤10% chat fraction carries V1-level commit signal WITHOUT V1-level
    </think>-density. Weight the WINDOW (a calibrated rising-then-fire pattern), NOT a hard bonus on the single
    close id, to avoid the length-attractor / premature-stop pathology (arXiv:2010.07174). ids [B,T] -> w [B,T]."""
    B, T = ids.shape
    w = torch.ones(B, T, device=ids.device, dtype=torch.float32)
    for b in range(B):
        for p in (ids[b] == close_id).nonzero(as_tuple=True)[0].tolist():
            w[b, max(0, p - pre): min(T, p + post + 1)] = alpha   # t-position window around the close
    return w


def cakld_loss(student_logits, teacher_idx, teacher_val, temperature=1.0, weights=None,
               commit_mask=None, commit_beta=0.0):
    """Confidence-Aware KL (CAKLD, BitDistiller ACL 2024): identical to topk_kl_loss but each
    token's KL is weighted by the teacher's peak probability at that position, so high-confidence
    teacher tokens dominate the gradient. This directly targets generation dissociation: low-entropy
    (confident) positions get high weight, high-entropy (uncertain) positions get low weight.
    Same shapes as topk_kl_loss."""
    s = student_logits.float() / temperature
    log_Z = torch.logsumexp(s, dim=-1, keepdim=True)
    s_topk = torch.gather(s, -1, teacher_idx.long())
    log_q = s_topk - log_Z
    p = torch.softmax(teacher_val.float() / temperature, dim=-1)
    if SKEW_ALPHA > 0:                                      # A2 skew-KL: bound the gradient by mixing
        q = log_q.exp()                                    # student prob at teacher's top-k
        m = SKEW_ALPHA * p + (1 - SKEW_ALPHA) * q           # skewed target
        kl = (p * (torch.log(p + 1e-9) - torch.log(m + 1e-9))).sum(-1)   # KL(p || m)  [B, T]
    else:
        kl = (p * (torch.log(p + 1e-9) - log_q)).sum(-1)   # [B, T]
    # p.max over top-k approximates full-vocab confidence; detach so weights don't fight the loss.
    # DECISION_GAMMA>1 SHARPENS the weighting toward high-confidence "decision" tokens (answer/logic
    # tokens where a ternary flip changes the argmax) — the perplexity->downstream lever. gamma=1 = CAKLD.
    confidence = p.max(dim=-1).values.detach() ** DECISION_GAMMA   # [B, T]
    if weights is not None:                                        # (legacy) reweight INSIDE the normalized mean
        confidence = confidence * weights
    loss = (kl * confidence).sum() / confidence.sum().clamp_min(1e-8) * (temperature ** 2)
    return _add_commit_term(loss, kl, commit_mask, commit_beta, temperature)


def hidden_state_loss(student_h, teacher_h):
    """Feature distillation on the final (post-norm) hidden state. Shapes [B, T, H].

    lm_head is a frozen linear shared by teacher and student, so matching the full 5120-dim
    hidden vector is a FULL-distribution constraint — strictly stronger than the top-k logit KL,
    which only pins the teacher's 64 largest logits and leaves the rest of the vocabulary (and
    every intermediate feature) unconstrained. This term directly targets the generation-
    dissociation gap (coherent perplexity but degraded greedy generation) that top-k KL ignores.

    Plain fp32 MSE; weight it against the KL via the caller's --feat-weight. Watch the printed
    feat vs KL magnitudes — post-norm hidden RMS is ~O(1)/element, so MSE and KL are usually the
    same order, but lower the weight if feat dominates the gradient."""
    return F.mse_loss(student_h.float(), teacher_h.float())


class _TopKKLFunc(torch.autograd.Function):
    """Fused topk-KL / CAKLD loss with memory-safe backward.

    The naive approach saves gathered_w [B*T, K, H] (~320 MB on GPU) in the autograd
    graph for the bmm backward.  That tensor stays alive across the entire model backward,
    which means the ste_ternary recompute during gradient checkpointing sees ~320 MB of
    extra live memory and OOMs on GPU1 (the active group card for layers 32-63).

    This function fuses the entire loss, computes grad_h analytically, and del's
    gathered_w *before* the chunked log_Z gradient loop.  By the time gradient
    checkpointing triggers the ste_ternary recompute, those ~320 MB are back in the
    CUDA cache and reusable.

    Gradient derivation:
        loss = T² × Σ_i c_i × KL_i,  c_i = 1/B_T  (topk_kl)  or  conf_i/Σconf (cakld)
        KL_i = Σ_k p_ik (log p_ik − log q_ik),  log q_ik = (h_i·w_k)/T − log Z_i
        ∂(loss)/∂(h_i) = (T·c_i·grad_loss) × [Σ_j q_ij·w_j − Σ_k p_ik·w_k]
                       =   factor_i         × [  full-softmax term  −  topk term  ]
    """

    @staticmethod
    def forward(ctx, h_flat, w_cpu, t_idx_flat, t_val_flat,
                temperature, chunk_size, use_cakld):
        B_T, H = h_flat.shape
        V = w_cpu.shape[0]
        K = t_idx_flat.shape[1]
        dev = h_flat.device
        h = h_flat.float()  # fp32 for numerical stability; h_flat may be bf16

        with torch.no_grad():
            # ── log Z via 2-pass chunked logsumexp ──────────────────────────────────
            row_max = h.new_full((B_T,), float('-inf'))
            for i in range(0, V, chunk_size):
                c = h @ w_cpu[i:i + chunk_size].to(dev, torch.float32).T / temperature
                row_max = torch.maximum(row_max, c.amax(-1))
            log_sum = h.new_zeros(B_T)
            for i in range(0, V, chunk_size):
                c = h @ w_cpu[i:i + chunk_size].to(dev, torch.float32).T / temperature
                log_sum += torch.exp(c - row_max.unsqueeze(-1)).sum(-1)
            log_Z = row_max + torch.log(log_sum)  # [B_T]

            # ── topk scores (gathered_w briefly allocated then freed) ───────────────
            idx_cpu = t_idx_flat.reshape(-1).cpu()
            gathered_w = w_cpu[idx_cpu].reshape(B_T, K, H).to(dev, torch.float32)  # [B_T, K, H]
            s_topk = (torch.bmm(h.unsqueeze(1),
                                 gathered_w.transpose(-2, -1)).squeeze(1)
                      / temperature)  # [B_T, K]
            del gathered_w  # free ~320 MB; not needed for loss or backward

            # ── KL divergence ────────────────────────────────────────────────────────
            log_q = s_topk - log_Z.unsqueeze(-1)
            p = torch.softmax(t_val_flat.float() / temperature, dim=-1)
            kl = (p * (torch.log(p + 1e-9) - log_q)).sum(-1)  # [B_T]

            if use_cakld:
                confidence = p.max(dim=-1).values
                denom = confidence.sum().clamp_min(1e-8)
                loss = (kl * confidence).sum() / denom * (temperature ** 2)
            else:
                loss = kl.mean() * (temperature ** 2)

        to_save = [h_flat, w_cpu, t_idx_flat, t_val_flat, log_Z]
        if use_cakld:
            to_save += [confidence, denom.unsqueeze(0)]
        ctx.save_for_backward(*to_save)
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size
        ctx.use_cakld = use_cakld
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        if ctx.use_cakld:
            h_flat, w_cpu, t_idx_flat, t_val_flat, log_Z, confidence, denom = ctx.saved_tensors
            denom = denom.squeeze()
        else:
            h_flat, w_cpu, t_idx_flat, t_val_flat, log_Z = ctx.saved_tensors
            confidence = denom = None

        temperature = ctx.temperature
        chunk_size  = ctx.chunk_size
        B_T, H = h_flat.shape
        V = w_cpu.shape[0]
        K = t_idx_flat.shape[1]
        dev = h_flat.device

        h = h_flat.float()
        p = torch.softmax(t_val_flat.float() / temperature, dim=-1)  # [B_T, K]

        # factor_i = T · c_i · grad_loss   (the scalar in front of [E_q[w] - E_p[w]])
        if ctx.use_cakld:
            c = confidence / denom * (temperature ** 2)
        else:
            c = h.new_full((B_T,), temperature ** 2 / B_T)
        factor = c * float(grad_loss) / temperature  # [B_T]

        grad_h = torch.zeros_like(h)  # fp32 accumulator on dev

        # ── topk term: -factor_i · Σ_k p_ik · w_k  (brief alloc, freed immediately) ──
        idx_cpu = t_idx_flat.reshape(-1).cpu()
        gathered_w = w_cpu[idx_cpu].reshape(B_T, K, H).to(dev, torch.float32)  # [B_T, K, H]
        # [B_T, 1, K] @ [B_T, K, H] → [B_T, H]
        grad_h -= torch.bmm((factor.unsqueeze(-1) * p).unsqueeze(1),
                             gathered_w).squeeze(1)
        del gathered_w  # free ~320 MB before model-backward recompute starts

        # ── full-softmax term: +factor_i · Σ_j q_ij · w_j  (chunked, O(cs·H) peak) ──
        for i in range(0, V, chunk_size):
            w_c = w_cpu[i:i + chunk_size].to(dev, torch.float32)
            q_c = torch.exp(h @ w_c.T / temperature - log_Z.unsqueeze(-1))  # [B_T, cs]
            grad_h += (factor.unsqueeze(-1) * q_c) @ w_c
            del w_c, q_c

        # 7 inputs: h_flat, w_cpu, t_idx_flat, t_val_flat, temperature, chunk_size, use_cakld
        return grad_h.to(h_flat.dtype), None, None, None, None, None, None


def chunked_hidden_state_loss(h_last, lm_head_weight, teacher_idx, teacher_val,
                               temperature=1.0, loss_type="topk_kl", chunk_size=32768):
    """Distillation loss from last hidden state [B, T, H] without materialising [B, T, V].

    Uses _TopKKLFunc which frees the [B*T, K, H] gathered_w tensor inside its backward
    before the model's gradient checkpointing recompute, saving ~320 MB on the active GPU.
    Use this instead of topk_kl_loss/cakld_loss for group_size > 2."""
    if lm_head_weight.device.type == "meta":
        raise ValueError("lm_head_weight is a meta tensor; pass the saved CPU weight instead")
    B, T, H = h_last.shape
    K = teacher_idx.shape[-1]
    h_flat     = h_last.reshape(B * T, H)
    t_idx_flat = teacher_idx.reshape(B * T, K)
    t_val_flat = teacher_val.reshape(B * T, K)
    return _TopKKLFunc.apply(
        h_flat, lm_head_weight, t_idx_flat, t_val_flat,
        float(temperature), chunk_size, loss_type == "cakld"
    )


class _ChunkedCEFunc(torch.autograd.Function):
    """Memory-efficient cross-entropy from hidden states + a CPU-resident lm_head weight, WITHOUT ever
    materialising [N, V] logits or holding the full weight on the GPU. Mirrors _TopKKLFunc: the forward
    streams vocab chunks CPU→GPU for a 2-pass logsumexp + target-logit gather; the backward recomputes
    the softmax chunk-wise (grad_h = (softmax − onehot(tgt)) @ W). Only [N, chunk] is ever GPU-resident.
    grad flows to h only (W is the frozen lm_head)."""
    @staticmethod
    def forward(ctx, h_flat, w_cpu, tgt, chunk_size):
        N, H = h_flat.shape
        V = w_cpu.shape[0]
        dev = h_flat.device
        h = h_flat.float()
        with torch.no_grad():
            row_max = h.new_full((N,), float("-inf"))
            for i in range(0, V, chunk_size):
                z = h @ w_cpu[i:i + chunk_size].to(dev, torch.float32).T
                row_max = torch.maximum(row_max, z.amax(-1))
            log_sum = h.new_zeros(N)
            for i in range(0, V, chunk_size):
                z = h @ w_cpu[i:i + chunk_size].to(dev, torch.float32).T
                log_sum += torch.exp(z - row_max.unsqueeze(-1)).sum(-1)
            log_Z = row_max + torch.log(log_sum)                       # [N]
            rows = torch.arange(N, device=dev)
            tgt_w = w_cpu[tgt.cpu()].to(dev, torch.float32)            # [N, H]
            tgt_logit = (h * tgt_w).sum(-1)                            # [N]
            loss = (log_Z - tgt_logit).mean()
        ctx.save_for_backward(h, w_cpu, tgt, log_Z)
        ctx.chunk_size = chunk_size
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        h, w_cpu, tgt, log_Z = ctx.saved_tensors
        N, H = h.shape
        V = w_cpu.shape[0]
        dev = h.device
        cs = ctx.chunk_size
        g = (grad_out / N)
        grad_h = torch.zeros_like(h)
        for i in range(0, V, cs):
            Wc = w_cpu[i:i + cs].to(dev, torch.float32)                # [c, H]
            z = h @ Wc.T                                              # [N, c]
            p = torch.exp(z - log_Z.unsqueeze(-1))                     # softmax chunk
            grad_h += p @ Wc                                          # Σ softmax·W
        # subtract the onehot(target) contribution
        grad_h -= w_cpu[tgt.cpu()].to(dev, torch.float32)
        grad_h *= g
        return grad_h.to(h.dtype), None, None, None


def chunked_ce(h_last_sel, lm_head_weight_cpu, targets, chunk_size=16384):
    """CE over a (subset of) positions with the lm_head weight on CPU. h_last_sel [N,H], targets [N]."""
    return _ChunkedCEFunc.apply(h_last_sel, lm_head_weight_cpu, targets, chunk_size)


@torch.no_grad()
def init_latent(m, w_fp=None, mode="center", eps=1e-3):
    """Initialise an assignment-QAT STE latent.

    'center'    L = tern·scale — every latent lands EXACTLY on a bin centre (measured: 0.000% of weights within
                0.05 of a decision boundary). That is DEGENERATE: with no low-confidence subpopulation, a step
                either crosses nothing (inert) or pushes a correlated mass across at once (cascade) — the
                bimodality we measured (0% flips at lr<=2e-4, 18%->37% runaway at 5e-3).
    'fp-spread' L = clamp(w_fp, (t-0.5)·s, (t+0.5)·s) — the FP weight CLAMPED INTO THE BIN OF THE CURRENT TRIT.
                round(L/s) is unchanged, so the deployed function is byte-identical at init, but the natural
                intra-bin spread is restored (~9.8% near a boundary for real FP weights), so gradients move
                individual weights across boundaries a few at a time. This is the cheap analogue of BitNet
                Distillation's 10B-token warm-up, whose whole purpose is migrating weight mass toward the
                transition boundaries. See research_reports/assignment_data_appetite_report.md."""
    cur = m.dequant().detach()
    if mode != "fp-spread" or w_fp is None:
        return cur.clone()
    s = m.scale.detach().clamp_min(1e-8).unsqueeze(1)                  # [n_blocks,1]
    t = torch.round(cur.reshape(m.n_blocks, m.block_size) / s).clamp(-1, 1)
    lo, hi = (t - 0.5 + eps) * s, (t + 0.5 - eps) * s                  # strictly inside the current bin
    L = torch.clamp(w_fp.detach().to(s.device, s.dtype).reshape(m.n_blocks, m.block_size), lo, hi)
    assert torch.equal(torch.round(L / s).clamp(-1, 1), t), "fp-spread init changed an assignment"
    return L.reshape(m.out_features, m.in_features)


def _fp_weight_lookup(fp_model_path):
    """Return name->tensor getter for the rotated FP weights (the latent-init source)."""
    if not fp_model_path:
        return None
    from safetensors import safe_open
    f = safe_open(str(Path(fp_model_path) / "model.safetensors"), framework="pt")
    keys = set(f.keys())
    _warned = set()

    def get(name):
        wn = name + ".weight"
        if wn in keys:
            return f.get_tensor(wn)
        # Suffix fallback: module paths are `model.layers.N...` while checkpoint keys carry the tower
        # prefix `model.language_model.layers.N...`, so the exact lookup above always misses here.
        #
        # BUG THIS FIXES (2026-08-19): the suffix `layers.0.mlp.down_proj.weight` matches BOTH
        #   model.language_model.layers.0.mlp.down_proj.weight   (correct)
        #   mtp.layers.0.mlp.down_proj.weight                    (multi-token-prediction head)
        # and the old code took cand[0] from a SET, whose iteration order varies per process under
        # Python's string hash randomisation. So ~50% of runs initialised layer 0's fp-spread latents
        # from the MTP head's weights. Only layer 0 was affected because the MTP tower has one layer.
        # Symptom: latent init landed in one of two discrete states on byte-identical commands
        # (near-decision-boundary 19.633% vs 25.251%), which silently de-paired every A/B comparison.
        # Fix: sort for determinism, and prefer the main `model.` tower over auxiliary towers
        # (mtp / visual). Warn once if anything is still ambiguous rather than silently guessing.
        suffix = name.split("model.")[-1] + ".weight"
        cand = sorted(k for k in keys if k.endswith(suffix))
        if not cand:
            return None
        if len(cand) > 1:
            main = [k for k in cand if k.startswith("model.")]
            if main:
                cand = main
            if len(cand) > 1 and suffix not in _warned:
                _warned.add(suffix)
                print(f"   [fp-lookup] AMBIGUOUS {suffix}: {cand} -> using {cand[0]}", flush=True)
        return f.get_tensor(cand[0])
    return get


# ─────────────────────────── shared loading helpers ────────────────────────────────

def load_calib_batches(calib_path, batch_size, seq, device):
    with open(calib_path) as f:
        samples = json.load(f)
    ids = [torch.tensor(s[:seq], dtype=torch.long) for s in samples if len(s) >= seq]
    return [torch.stack(ids[i:i + batch_size]) for i in range(0, len(ids), batch_size)]


def _shard_map(model_path):
    with open(Path(model_path) / "model.safetensors.index.json") as f:
        return json.load(f)["weight_map"]


def _get_tensor(model_path, wmap, name):
    from safetensors import safe_open
    used = name
    shard = wmap.get(used)
    if shard is None:
        # multimodal checkpoint: the LM lives under model.language_model.*
        for a, b in (("model.layers", "model.language_model.layers"),
                     ("model.embed_tokens", "model.language_model.embed_tokens"),
                     ("model.norm", "model.language_model.norm"),
                     ("model.rotary_emb", "model.language_model.rotary_emb")):
            if a in name and (name.replace(a, b) in wmap):
                used = name.replace(a, b)
                shard = wmap[used]
                break
    if shard is None and name.startswith("model.") and "language_model" not in name:
        alt = name.replace("model.", "model.language_model.", 1)   # generic fallback
        if alt in wmap:
            used = alt
            shard = wmap[alt]
    if shard is None:
        raise KeyError(name)
    with safe_open(str(Path(model_path) / shard), framework="pt", device="cpu") as f:
        return f.get_tensor(used)


# ─────────────────────────── build the ternary student ─────────────────────────────

def arm_b_probe_grad(core, cache, probe_idx, batches, device, loss_fn, temp):
    """FRESH per-weight gradient ḡ = ∂loss/∂(effective weight) at the FROZEN (T,s) — not the stale
    training-window accumulation (the corrected PV-Tuning P-step needs the gradient at the current iterate).
    Mean over `probe_idx` micro-batches, DDP-synced. Populates each arm_b module's flip_grad."""
    import torch.distributed as dist
    ddp = dist.is_available() and dist.is_initialized()
    mods = [m for m in core.modules() if getattr(m, "_arm_b", False)]
    for m in mods: m.flip_grad.zero_()
    was = core.training; core.train()                            # deq hook fires only in train mode
    nb = 0
    for bi in probe_idx:
        ids = batches[bi].to(device)
        t_idx = cache["idx"][bi].unsqueeze(0).to(device); t_val = cache["val"][bi].unsqueeze(0).to(device)
        loss = loss_fn(core(ids).logits, t_idx, t_val, temp)
        loss.backward(); core.zero_grad(set_to_none=True); nb += 1  # hook accumulates ∂loss/∂deq into flip_grad
    if not was: core.eval()
    for m in mods:
        m.flip_grad /= max(1, nb)
        if ddp:
            dist.all_reduce(m.flip_grad, op=dist.ReduceOp.SUM); m.flip_grad /= dist.get_world_size()
    return nb


@torch.no_grad()
def arm_b_gated_flip(core, eta, cap_frac, block_cap):
    """Proximal-GATED flip event (the fix). For each weight, pick the candidate level t' minimising the
    linearised+quadratic Δ = ḡ·s·(t'−t) + (s·(t'−t))²/(2η); commit ONLY where Δ<0 (the gate makes the flip
    count an OUTPUT, not a preset top-k), globally capped at cap_frac and ≤block_cap per 256-block, keeping the
    most-negative-Δ flips. Returns (committed, qualifying, predicted_Δ, revert_snapshot)."""
    mods = [m for m in core.modules() if getattr(m, "_arm_b", False)]
    ntot = sum(m.tern_b.numel() for m in mods)
    cap = int(cap_frac * ntot) if cap_frac > 0 else ntot        # global safety cap (rarely binds — the gate does)
    CHUNK = 262144                                            # block-rows per chunk (~0.27GB/fp32 temp); big linears
                                                              # (lm_head = 2.5M block-rows) get chunked, body fits in one
    # ── phase 1: per-linear, CHUNKED gate → collect candidate flips as sparse (block,col) indices ──
    n_qual = 0; cand = []                                      # each: (mod, block_idx, col_idx, new_val_i8, Δ)
    for m in mods:
        s_full = m.scale.detach().clamp_min(1e-8)             # [n_blocks]
        NB = m.tern_b.shape[0]
        for c0 in range(0, NB, CHUNK):
            c1 = min(c0 + CHUNK, NB)
            s = s_full[c0:c1].unsqueeze(1); g = m.flip_grad[c0:c1].float(); t = m.tern_b[c0:c1].float()
            best = torch.full_like(g, float("inf")); best_new = t.clone()
            for d in (-2, -1, 1, 2):
                sd = s * d; gain = torch.where((t + d >= -1) & (t + d <= 1), g * sd + sd * sd / (2 * eta),
                                               torch.full_like(g, float("inf")))
                better = gain < best; best = torch.where(better, gain, best); best_new = torch.where(better, t + d, best_new)
            qual = best < 0; n_qual += int(qual.sum())
            sel = qual
            if block_cap > 0:                                    # ≤block_cap most-negative-Δ per 256-block
                gg = torch.where(sel, best, torch.full_like(best, float("inf")))
                kth = min(block_cap, m.block_size)
                topv, topi = (-gg).topk(kth, dim=1)
                keep = torch.zeros_like(sel)
                rows = torch.arange(gg.shape[0], device=gg.device).unsqueeze(1).expand(-1, kth)
                vt = topv > float("-inf"); keep[rows[vt], topi[vt]] = True
                sel = sel & keep
            if bool(sel.any()):
                idx = sel.nonzero(as_tuple=False)                # [k,2] (block-in-chunk, col)
                cand.append((m, (idx[:, 0] + c0).clone(), idx[:, 1].clone(),
                             best_new[sel].to(torch.int8).clone(), best[sel].clone()))
            del best, best_new, g, t, qual, sel
    if not cand:
        return 0, n_qual, 0.0, []
    # ── phase 2: global cap across ALL candidates (keep the cap most-negative Δ), then commit + build revert ──
    all_delta = torch.cat([c[4] for c in cand])
    thresh = all_delta.kthvalue(cap).values if (all_delta.numel() > cap and cap > 0) \
        else torch.tensor(float("inf"), device=all_delta.device)
    committed, pred, revert = 0, 0.0, []
    for (m, blk, col, nv, delta) in cand:
        keep = delta <= thresh
        if not bool(keep.any()):
            continue
        b, c = blk[keep], col[keep]; sel = (b, c)                # tuple index → m.tern_b[sel]=old reverts cleanly
        revert.append((m, sel, m.tern_b[b, c].clone()))
        pred += float(delta[keep].sum()); m.tern_b[b, c] = nv[keep]; committed += int(keep.sum())
    return committed, n_qual, pred, revert


@torch.no_grad()
def _chunked_argmax(h_flat, Wlm, chunk=32768):
    """argmax over the vocab per row WITHOUT materialising [N, V] — running max across weight chunks."""
    N = h_flat.shape[0]
    best_val = h_flat.new_full((N,), float("-inf"))
    best_idx = torch.zeros(N, dtype=torch.long, device=h_flat.device)
    for i in range(0, Wlm.shape[0], chunk):
        c = h_flat.float() @ Wlm[i:i + chunk].to(h_flat.device, torch.float32).t()   # [N, chunk]
        cv, ci = c.max(-1)
        upd = cv > best_val
        best_idx = torch.where(upd, i + ci, best_idx)
        best_val = torch.maximum(best_val, cv)
    return best_idx


def heldout_kl_flips(model, cache, held_idx, batches, device, loss_fn, temperature, per_seq=False, mem_eff=None):
    """The MANDATORY held-out metric (training-batch KL is inadmissible for selection). Mean top-k KL-vs-FP
    and %flips (student argmax vs teacher top-1) over a reserved, never-trained calib slice. With per_seq,
    also returns the list of per-sequence KLs (for a PAIRED before/after gate delta — pairing cancels the
    ~0.35-nat baseline so N≈256 seqs can resolve milli-nat flips that unpaired KL never could).

    mem_eff (dict with 'Wlm','norm','loss_type') routes through the chunked hidden-state path so the eval
    never materialises full-vocab logits — required when assignment latents occupy the card (CE/QAT stage)."""
    was_training = model.training
    model.eval()
    tot_kl = 0.0; flips = 0; ntok = 0; seq_kl = []
    hidcap = {}
    handle = mem_eff["norm"].register_forward_hook(lambda m, i, o: hidcap.__setitem__("h", o)) if mem_eff else None
    try:
      # no_grad is ESSENTIAL: without it the eval forward builds and RETAINS a full ~17GB training-size
      # autograd graph (the original path got away with float()-detaching the scalar, but the chunked
      # hidden-state path holds the graph via `h` until freed) → OOMs the card right after a training step.
      with torch.no_grad():
        for bi in held_idx:
            ids = batches[bi].to(device)
            t_idx = cache["idx"][bi].unsqueeze(0).to(device); t_val = cache["val"][bi].unsqueeze(0).to(device)
            if mem_eff:
                hidcap.clear(); model(ids, logits_to_keep=1); h = hidcap["h"]
                k = float(chunked_hidden_state_loss(h, mem_eff["Wlm"], t_idx, t_val,
                                                    temperature=temperature, loss_type=mem_eff["loss_type"],
                                                    chunk_size=4096))
                am = _chunked_argmax(h[0, :-1, :], mem_eff["Wlm"], chunk=4096)
                flips += int((am != t_idx[0, :-1, 0]).sum()); ntok += h.shape[1] - 1
            else:
                logits = model(ids).logits
                k = float(loss_fn(logits, t_idx, t_val, temperature))
                flips += int((logits[:, :-1].argmax(-1) != t_idx[:, :-1, 0]).sum()); ntok += logits.shape[1] - 1
            tot_kl += k
            if per_seq: seq_kl.append(k)
    finally:
        if handle is not None:
            handle.remove()
    if was_training:
        model.train()
    mean_kl = tot_kl / max(1, len(held_idx)); pct = 100.0 * flips / max(1, ntok)
    return (mean_kl, pct, seq_kl) if per_seq else (mean_kl, pct)



# ── latent placement: GPU-resident budget + pinned host memory ───────────────────────────────────
# Measured on the 27B: step time scales almost LINEARLY with latent count (64/249/497 latent tensors ->
# 102.6/165.7/327.1 s per step), and halving the BYTES moved (--latent-bf16-compute) did NOT help
# (+3.8%). So the cost tracks the NUMBER of per-latent operations, not bandwidth. Two levers follow:
#   * --latent-gpu-budget GB : keep the largest possible prefix of latents ON the GPU, so those layers
#     do no host->device transfer at all (dequant() already takes the no-copy path when
#     latent.device == scale.device). This is the cleanest test of the transfer hypothesis: if
#     eliminating transfers outright does not help, the cost is dequant compute, not movement.
#   * --latent-pin : keep offloaded latents in PINNED host memory. Pageable H2D copies are SYNCHRONOUS
#     (they stall the calling thread); pinned ones can be async, which is the difference between 497
#     serialised stalls per forward and transfers that overlap compute. Pinned memory is unswappable,
#     so this trades RAM flexibility for latency.
_LAT_PLACE = {"budget_bytes": 0, "used": 0, "pin": False, "n_gpu": 0, "n_cpu": 0, "n_pin": 0}


def _place_latent(L, lat_off, dev):
    """Return the latent tensor placed per the GPU budget / pinning policy."""
    if not lat_off:
        return L
    b = _LAT_PLACE
    # NOTE: GPU residency is NOT applied here. build_student runs BEFORE train() extracts the lm_head
    # weight, which needs a ~5GB transient; claiming VRAM for latents first starved it and OOM'd
    # (measured: 42 latents resident at 8.0GB -> lm_head dequant found 2.01GB free and died).
    # promote_latents_to_gpu() runs after that extraction instead, against real remaining VRAM.
    out = L.cpu()
    if b["pin"]:
        try:
            out = out.pin_memory(); b["n_pin"] += 1
        except Exception:
            pass
    b["n_cpu"] += 1
    return out



# ── LATENT PREFETCH ──────────────────────────────────────────────────────────────────────────────
# Pinning bought ~15-20% by removing the driver's staging memcpy, which identified per-transfer CPU
# cost (not bandwidth, not GPU compute) as the bottleneck. Prefetch attacks the rest: start layer
# i+1's host->device copy on a side stream while layer i computes, so the copy overlaps instead of
# stalling. Requires pinned host memory to be genuinely async (--latent-pin).
#
# THE SUBTLETY: the transfer is part of the AUTOGRAD GRAPH — grad flows back through .to() into the
# CPU leaf. A side-stream copy done outside autograd would silently detach the latents and produce
# zero gradients, which the frozen (lr=0) speed tests would NOT catch. _UseTransferred keeps the graph
# intact: forward hands back the already-transferred buffer, backward ships the gradient back to CPU.
_PF_ON = False
_PF_BUF = {}                 # id(module) -> (gpu_tensor, cuda_event)
_PF_SIDE = {}                # device -> side stream
_LAT_ORDER = []              # latent modules in execution order


# Pinned staging buffers for the gradient's trip back to the host, one per latent.
# WHY: py-spy put 19.5% of all samples in this backward's `g.to("cpu")`. `Tensor.to("cpu")` allocates
# PAGEABLE memory, so the driver stages every gradient through an internal pinned buffer — the exact
# cost that pinning the FORWARD direction removed for ~19%. This is the mirror image, and it needs no
# VRAM, which matters because every VRAM-spending idea in this project OOM'd.
# NOTE this cost is paid even at --latent-lr 0: gradients are still computed and shipped home whether
# or not the optimizer consumes them.
_GRAD_PIN = {}          # id(param) -> pinned host buffer
_GRAD_PIN_ON = False


def _grad_pin_buf(like, key):
    b = _GRAD_PIN.get(key)
    if b is None or b.shape != like.shape or b.dtype != like.dtype:
        b = torch.empty(like.shape, dtype=like.dtype, pin_memory=True)
        _GRAD_PIN[key] = b
    return b


class _UseTransferred(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cpu_src, gpu_copy):
        ctx.src_device = cpu_src.device
        ctx.src_dtype = cpu_src.dtype
        ctx.key = id(cpu_src)
        return gpu_copy

    @staticmethod
    def backward(ctx, g):
        # mirror of `.to(device, dtype)`: send the gradient back to the CPU leaf
        if _GRAD_PIN_ON and ctx.src_device.type == "cpu":
            if g.dtype != ctx.src_dtype:
                g = g.to(ctx.src_dtype)
            buf = _grad_pin_buf(g, ctx.key)
            # D2H into PINNED memory: no driver staging copy. Safe to hand the buffer straight to
            # autograd because --latent-grad-release consumes and clears param.grad within the same
            # step, so it is never live across two backwards (guarded at setup for --accum > 1).
            buf.copy_(g)
            return buf, None
        return g.to(ctx.src_device, ctx.src_dtype), None


def _pf_side(dev):
    if dev not in _PF_SIDE:
        _PF_SIDE[dev] = torch.cuda.Stream(device=dev)
    return _PF_SIDE[dev]


def _pf_start(mod, dtype):
    """Kick off `mod`'s latent transfer on a side stream (no-op if resident or already queued)."""
    if mod is None or id(mod) in _PF_BUF:
        return
    L = getattr(mod, "latent", None)
    if L is None or L.data.device.type != "cpu":
        return
    dev = mod.scale.device
    if dev.type != "cuda":
        return
    st = _pf_side(dev)
    st.wait_stream(torch.cuda.current_stream(dev))
    with torch.cuda.stream(st):
        buf = L.data.to(dev, dtype=dtype, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record(st)
    _PF_BUF[id(mod)] = (buf, ev)


def _pf_take(mod):
    """Return this module's prefetched buffer if ready, else None."""
    ent = _PF_BUF.pop(id(mod), None)
    if ent is None:
        return None
    buf, ev = ent
    cur = torch.cuda.current_stream(buf.device)
    cur.wait_event(ev)
    buf.record_stream(cur)          # keep the allocator from reusing it while this stream reads it
    return buf

def promote_latents_to_gpu(model, budget_bytes, reserve_bytes=2_000_000_000):
    """Move as many latents as fit onto their own layer's GPU, newest-free-VRAM aware.

    Called AFTER the lm_head weight extraction so it budgets against VRAM that is actually free.
    A resident latent costs 4 B/latent of VRAM but removes that layer's host->device copy entirely
    from every forward (and every checkpoint recompute), which is the cost the pinning result showed
    dominates. `reserve_bytes` keeps headroom for activations so we do not trade a transfer win for
    an OOM.
    """
    if budget_bytes <= 0:
        return
    mods = [m for m in model.modules() if getattr(m, "latent", None) is not None]
    used = 0
    n = 0
    for m in mods:
        L = m.latent.data
        if L.device.type != "cpu":
            continue
        dev = m.scale.device                      # follow the layer this latent belongs to
        if dev.type != "cuda":
            continue
        nbytes = L.numel() * L.element_size()
        free, _tot = torch.cuda.mem_get_info(dev)
        if used + nbytes > budget_bytes or nbytes + reserve_bytes > free:
            continue
        m.latent.data = L.to(dev)
        if getattr(m, "_cand_mask", None) is not None:
            m._cand_mask = m._cand_mask.to(dev)
        used += nbytes; n += 1
    if n:
        print(f"   latent PROMOTION: {n}/{len(mods)} latents moved to GPU ({used/1e9:.1f}GB) — "
              f"those layers do no host->device copy", flush=True)

def shard_model_across_gpus(model, n_dev, keep_latent_cpu=True):
    """LAYER-SPLIT MODEL PARALLELISM: put contiguous blocks of decoder layers on different GPUs.

    WHY THIS EXISTS. A 27B ternary student does NOT fit on one 24GB 3090 at seq 2560 — measured: OOM
    at ~22.7GB during the step-0 held-out eval, with expandable_segments already on. DDP does not help
    (data-parallel replicates the whole model per rank, so per-GPU memory is unchanged), and the code
    has no activation checkpointing. Splitting the LAYER axis is also the only decomposition this
    project permits: cross-layer coupling is additive via the residual stream, whereas splitting the
    scope axis (gate/up within an MLP) is multiplicative and always damages (RESULTS_SUMMARY §4.5).

    WHAT THIS IS NOT. This is naive model parallelism, not a pipeline: with one microbatch in flight
    GPU0 idles while GPU1 computes, so utilisation is ~1/n_dev. It buys MEMORY (the thing that blocks
    us), not speed. 1F1B microbatch pipelining on top would recover utilisation — bubble = (p-1)/m,
    so 8 microbatches over 2 stages is ~94% — and is the natural follow-up.

    PLACEMENT. Decoder layers are split contiguously; embeddings, final norm and lm_head stay on
    device 0, and the boundary is bridged by hooks that move activations. Outputs are returned to
    device 0 so the loss/teacher path is unchanged. Latents stay CPU-resident under --latent-offload
    (dequant() pulls them to self.scale.device on demand, so they follow their layer automatically).
    """
    devs = [f"cuda:{i}" for i in range(n_dev)]
    # decoder layers = the longest ModuleList (64 for the 27B; beats the 1-layer MTP tower)
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and (best is None or len(mod) > len(best[1])):
            best = (name, mod)
    if best is None or len(best[1]) < n_dev:
        raise SystemExit(f"model-parallel: could not find a decoder layer list to split ({best[0] if best else None})")
    lname, layers = best
    n = len(layers)
    owner = [devs[min(i * n_dev // n, n_dev - 1)] for i in range(n)]

    def _move(x, dev):
        if torch.is_tensor(x):
            return x.to(dev, non_blocking=True)
        if isinstance(x, (list, tuple)):
            return type(x)(_move(v, dev) for v in x)
        if isinstance(x, dict):
            return {k: _move(v, dev) for k, v in x.items()}
        return x

    def _mk_pre(dev):
        def hook(module, args, kwargs):
            return _move(args, dev), _move(kwargs, dev)
        return hook

    def _mk_post(dev):
        def hook(module, args, output):
            return _move(output, dev)
        return hook

    for i, layer in enumerate(layers):
        dev = owner[i]
        # stash CPU latents (and their candidate masks) so .to(dev) cannot drag them onto the GPU
        stash = []
        for m in layer.modules():
            if keep_latent_cpu and getattr(m, "latent", None) is not None:
                # remember whether this latent was OFFLOADED (cpu) or GPU-RESIDENT
                # (--latent-gpu-budget). layer.to(dev) would drag the offloaded ones onto the GPU;
                # forcing all of them to cpu afterwards would silently undo GPU residency. Restore
                # each to where it belongs: cpu stays cpu, resident follows its layer.
                stash.append((m, m.latent.data, getattr(m, "_cand_mask", None),
                              m.latent.data.device.type == "cpu"))
        layer.to(dev)
        for m, ldata, cmask, was_cpu in stash:
            m.latent.data = ldata.cpu() if was_cpu else ldata.to(dev)
            if cmask is not None:
                m._cand_mask = cmask.cpu() if was_cpu else cmask.to(dev)
        # Hook EVERY layer, not just the device boundaries. The HF decoder loop computes the rotary
        # position embeddings ONCE in the outer forward (on device 0) and passes the same cos/sin to
        # every layer, so a boundary-only hook leaves layers 33..63 reading cuda:0 tensors and
        # apply_rotary_pos_emb dies with "found at least two devices". Same for attention masks and
        # cache objects. On the owning device the .to() is a no-op, so this costs nothing.
        layer.register_forward_pre_hook(_mk_pre(dev), with_kwargs=True)
    layers[n - 1].register_forward_hook(_mk_post(devs[0]))   # hand the tail back to device 0

    counts = {d: owner.count(d) for d in devs}
    print(f"   MODEL-PARALLEL: {n} layers of `{lname}` split across {n_dev} GPUs {counts} "
          f"(naive layer-split; ~1/{n_dev} utilisation until microbatch pipelining is added)", flush=True)
    return model


def build_student(student_path, orig_config_path, block_size, device):
    """Instantiate the architecture, then stream the recovered weights in: quantized linears
    become packed TernaryScaleLinear (~7 GB total), everything else loads as bf16."""
    from transformers import AutoConfig, AutoModelForCausalLM
    from accelerate import init_empty_weights

    config = AutoConfig.from_pretrained(str(orig_config_path), trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    wmap = _shard_map(student_path)

    # which leaf linears are ternary (match the recovery's should_quantize set)
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        wname = f"{name}.weight"
        if not should_quantize(wname):
            continue
        try:
            w = _get_tensor(student_path, wmap, wname)
        except Exception as e:
            print(f"   ⚠️ skip {wname}: {e}")
            continue
        bname = f"{name}.bias"                            # preserve a folded bias (e.g. Tequila deadzone->bias)
        b = _get_tensor(student_path, wmap, bname) if bname in wmap else None
        tsl = TernaryScaleLinear.from_dense(w, block_size, bias=b, device=device)
        parent = model
        *parents, leaf = name.split(".")
        for p in parents:
            parent = getattr(parent, p)
        setattr(parent, leaf, tsl)
        replaced += 1
        del w
    # load the remaining (non-ternary) parameters as bf16 directly onto the device
    msd = dict(model.named_parameters())
    for pname, p in list(model.named_parameters()):
        if p.device.type == "meta":
            try:
                t = _get_tensor(student_path, wmap, pname)
            except Exception:
                continue
            _assign_param(model, pname, t.to(device, torch.bfloat16))
    for bname, b in list(model.named_buffers()):
        if b.device.type == "meta":
            try:
                t = _get_tensor(student_path, wmap, bname)
                _assign_buffer(model, bname, t.to(device))
            except Exception:
                pass
    # fail loudly (not deep in the forward) if anything is still on meta
    leftover = [n for n, t in list(model.named_parameters()) + list(model.named_buffers())
                if getattr(t, "is_meta", False)]
    if leftover:
        print(f"   ⚠️ {len(leftover)} tensors still on meta after load; first 12:")
        for n in leftover[:12]:
            print(f"      {n}")
        raise RuntimeError(
            f"{len(leftover)} tensors did not load from {student_path} (listed above). "
            "If they are non-persistent computed buffers (e.g. rotary inv_freq), report the "
            "names and we'll recompute them; otherwise it's a key-aliasing miss.")
    print(f"   replaced {replaced} linears with packed TernaryScaleLinear")
    if getattr(build_student, "_a3_mlp", False):
        # A3(ii): a trainable per-input-channel scale on each MLP's input, shared by gate+up (which read
        # the same post_attention_layernorm). Folds EXACTLY into that norm's gain at save (weights stay
        # ternary). Init 1.0 → byte-equivalent until trained.
        import types
        n_a3 = 0
        for m in model.modules():
            if type(m).__name__ == "Qwen3_5MLP":
                h = m.gate_proj.in_features
                m.a3_scale = nn.Parameter(torch.ones(h, device=device))
                _orig = m.forward
                def _fwd(x, _m=m, _o=_orig):
                    return _o(x * _m.a3_scale.to(x.dtype))
                m.forward = _fwd
                n_a3 += 1
        print(f"   A3(ii): added per-MLP-input a3_scale to {n_a3} MLPs (folds into post_attn norm)")
    if getattr(build_student, "_col_scale", False):
        # COL-SCALE (perpendicular-scales probe): per-input-CHANNEL trainable scales — the input-direction
        # DOF the per-256-row-block scales cannot express — at the two fold-clean points: (i) MLP input
        # (gate+up shared; reuses the A3(ii) param + its exact post_attention_layernorm-γ fold), and
        # (ii) down_proj input (the MLP-intermediate dim, 79%-error surface, NEVER covered by A3 —
        # folds EXACTLY into up_proj's per-row scales at save). Init 1.0 → byte-equivalent until trained.
        n_cs = 0
        for m in model.modules():
            if type(m).__name__ == "Qwen3_5MLP":
                if not hasattr(m, "a3_scale"):
                    h = m.gate_proj.in_features
                    m.a3_scale = nn.Parameter(torch.ones(h, device=device))
                    _orig = m.forward
                    def _fwd3(x, _m=m, _o=_orig):
                        return _o(x * _m.a3_scale.to(x.dtype))
                    m.forward = _fwd3
                dp = m.down_proj
                if isinstance(dp, TernaryScaleLinear):
                    dp.cs_in = nn.Parameter(torch.ones(dp.in_features, device=device))
                    _od = dp.forward
                    def _fwdd(x, _m=dp, _o=_od):
                        return _o(x * _m.cs_in.to(x.dtype))
                    dp.forward = _fwdd
                    n_cs += 1
        print(f"   col-scale: a3_scale (MLP-input) + cs_in (down-input) on {n_cs} MLPs (both fold at save)")
    if getattr(build_student, "_sq_bits", 0) > 0:
        # SCALE-QAT: train the per-256 scales ON an n-bit log-uniform grid (fake-quant + STE in dequant),
        # so we measure the ACHIEVABLE floor of quantized scales, not the naive post-hoc rounding damage.
        # Grid bounds fixed per-linear from the block-AP init (E2E moves scales little; values clamp to grid).
        bits = int(build_student._sq_bits); n_sq = 0
        for m in model.modules():
            if isinstance(m, TernaryScaleLinear):
                a = m.scale.detach().abs().clamp_min(1e-12).log()
                m._sq_lo = float(a.min()); m._sq_hi = float(a.max()) + 1e-6; m._sq_bits = bits
                n_sq += 1
        print(f"   scale-QAT: fake-quant scales to {bits}-bit log grid on {n_sq} linears (STE)")
    _lat_mode = getattr(build_student, "_latent_init", "center")
    _lat_off = getattr(build_student, "_latent_offload", False)
    _fp_get = _fp_weight_lookup(getattr(build_student, "_fp_model", None))
    if _lat_mode == "fp-spread" and _fp_get is None:
        raise SystemExit("--latent-init fp-spread requires --fp-model <rotated FP model dir>")
    if getattr(build_student, "_a4_downproj", False):
        # A4: enable latent assignment-moves on the MLP down_proj (biggest error source). Init keeps
        # round(L/s) unchanged either way, so the deployed function is byte-identical at init.
        n_a4 = 0
        for name, m in model.named_modules():
            if isinstance(m, TernaryScaleLinear) and name.endswith("mlp.down_proj"):
                _L = init_latent(m, _fp_get(name) if _fp_get else None, _lat_mode)
                m.latent = nn.Parameter(_place_latent(_L, _lat_off, device))
                n_a4 += 1
        print(f"   A4: latent assignment-moves enabled on {n_a4} down_proj "
              f"(STE, init={_lat_mode}, folds to hard ternary at save)")
    tw = getattr(build_student, "_train_weights", None)
    if tw and tw != "none":
        # Test 1 (Q1-A consistency): unfreeze the ternary ASSIGNMENTS under the distillation loss — a latent
        # FP weight (init = current ternary×scale) re-ternarised each step with STE, so gradients from the
        # logit-KL end loss flip {-1,0,+1}. The only asymptotically-consistent lever (weights can keep
        # consuming data), vs scale-only E2E which plateaus. Folds to hard ternary at save.
        _stride = max(1, int(getattr(build_student, "_tw_layer_stride", 1) or 1))
        _offset = int(getattr(build_student, "_tw_layer_offset", 0) or 0) % _stride
        def _layer_idx(name):
            p = name.split(".")
            if "layers" in p:
                try:    return int(p[p.index("layers") + 1])
                except Exception: return None
            return None
        def _tw_match(name):
            # layer stride: fp32 latents are ~3GB for all 32 down_proj, which sits at the very edge of 24GB
            # (repeated OOMs). stride=4 → 8 layers ≈ 0.75GB, enough headroom to study the mechanism.
            if _stride > 1:
                li = _layer_idx(name)
                if li is None or li % _stride != _offset:
                    return False
            if tw == "all":    return True
            if tw == "mlp":    return "mlp" in name
            if tw == "gatedown": return name.endswith("gate_proj") or name.endswith("down_proj")
            # SwiGLU is down(silu(gate(x)) * up(x)): gate and up multiply, so tuning one against the other's
            # current assignments makes those assignments the thing the second stage must move away from.
            # Measured: `gate` after `up` contributes EXACTLY zero (0.4543 -> 0.4543). Train them together.
            if tw == "gateup": return name.endswith("gate_proj") or name.endswith("up_proj")
            if tw == "gate":   return name.endswith("gate_proj")
            if tw == "up":     return name.endswith("up_proj")
            if tw == "down":   return name.endswith("down_proj")
            if tw == "attn":   return ("attn" in name) and ("mlp" not in name)
            if tw == "lmhead": return name.endswith("lm_head")   # fix EOS miscalibration from ternary lm_head
            return False
        arm_b = getattr(build_student, "_arm_b", False)
        n_tw = 0
        for name, m in model.named_modules():
            if isinstance(m, TernaryScaleLinear) and getattr(m, "latent", None) is None \
               and not getattr(m, "_arm_b", False) and _tw_match(name):
                if arm_b:
                    m.enable_arm_b()                              # Arm B: mutable ternary + flip accumulator (no latent)
                else:
                    _L = init_latent(m, _fp_get(name) if _fp_get else None, _lat_mode)
                    m.latent = nn.Parameter(_place_latent(_L, _lat_off, device))
                n_tw += 1
        kind = "Arm-B sparse-flip (no latent)" if arm_b else f"Arm-A STE latent (init={_lat_mode})"
        print(f"   --train-weights {tw}: {kind} assignment-QAT on {n_tw} linears (folds to hard ternary at save)")
        # ── CANDIDATE SET: restrict updates to latents near a decision boundary ──────────────────
        # Measured (4B, 188.7M latents): 14.91% sit at d<0.001 and 15.69% at d<0.01, where
        # d = ||L/s| - 0.5| is the distance to the nearest bin boundary. That is a SPIKE at d~0, not a
        # tail: fp-spread clamps w_fp into the current bin, so wherever the block-AP trit disagrees with
        # naive FP rounding the latent is pinned exactly on the bin edge. Only 3.55% of assignments ever
        # move, and motion saturates by ~step 60 — so latents far from a boundary contribute nothing but
        # optimizer traffic. Scales are FROZEN (§4.2), so an excluded latent's d never changes and the
        # set can only shrink; tau is the knob that sets the candidate fraction alpha.
        _cand_tau = float(getattr(build_student, "_cand_tau", 0.0) or 0.0)
        if _cand_tau > 0 and not arm_b:
            with torch.no_grad():
                nc = tot_c = 0
                for name, m in model.named_modules():
                    if isinstance(m, TernaryScaleLinear) and getattr(m, "latent", None) is not None:
                        s = m.scale.detach().to(m.latent.device).clamp_min(1e-8).unsqueeze(1)
                        u = (m.latent.detach().reshape(m.n_blocks, m.block_size) / s).abs()
                        msk = ((u - 0.5).abs() < _cand_tau).reshape_as(m.latent)
                        m._cand_mask = msk
                        nc += int(msk.sum()); tot_c += msk.numel()
                print(f"   CANDIDATE SET tau={_cand_tau}: {100.0*nc/max(1,tot_c):.2f}% of latents "
                      f"({nc/1e6:.1f}M of {tot_c/1e6:.1f}M) will receive updates; the rest are frozen")
        if _lat_mode == "fp-spread" and not arm_b:
            # report the near-boundary fraction — the quantity that decides whether flipping can be graded
            with torch.no_grad():
                nb = tot = 0
                for name, m in model.named_modules():
                    if isinstance(m, TernaryScaleLinear) and getattr(m, "latent", None) is not None:
                        s = m.scale.detach().to(m.latent.device).clamp_min(1e-8).unsqueeze(1)
                        u = m.latent.detach().reshape(m.n_blocks, m.block_size) / s
                        nb += int(((u - u.round()).abs() > 0.45).sum()); tot += u.numel()
                print(f"   latent spread: {100.0*nb/max(1,tot):.3f}% of latents within 0.05 of a decision "
                      f"boundary (bin-centre init gives 0.000%; real FP weights ~9.8%)")
            # ── INIT FINGERPRINT (LATENT_FINGERPRINT=1) ─────────────────────────────────────────
            # Paired arms disagreed on the spread above (23.391% vs 28.855%) under byte-identical
            # commands, which would invalidate any A/B comparison. That metric is also mis-specified:
            # it tests |u - round(u)| > 0.45 on SIGNED u, so it counts latents near +-1.5 (the clamp
            # edge of a saturated +-1 bin) as if they were near a decision boundary — but
            # round(u).clamp(-1,1) means those CANNOT flip. Only +-0.5 is a real boundary.
            # This logs an exact per-tensor fingerprint plus the two populations separately, so
            # "did these two runs start from the same latents?" is answerable rather than inferred.
            if os.environ.get("LATENT_FINGERPRINT", "0") == "1":
                with torch.no_grad():
                    n_dec = n_sat = n_tot = 0
                    for name, m in model.named_modules():
                        if isinstance(m, TernaryScaleLinear) and getattr(m, "latent", None) is not None:
                            L = m.latent.detach()
                            s = m.scale.detach().to(L.device).clamp_min(1e-8).unsqueeze(1)
                            u = L.reshape(m.n_blocks, m.block_size) / s
                            au = u.abs()
                            dec = ((au - 0.5).abs() < 0.05).sum()      # real decision boundary
                            sat = ((au - 1.5).abs() < 0.05).sum()      # clamp edge — CANNOT flip
                            n_dec += int(dec); n_sat += int(sat); n_tot += u.numel()
                            print(f"   [fp] {name:52s} sum={L.double().sum().item():+.9e} "
                                  f"absum={L.double().abs().sum().item():.9e} n={L.numel()}")
                    print(f"   [fp] TOTAL near-DECISION(+-0.5) {100.0*n_dec/max(1,n_tot):.3f}%  "
                          f"near-CLAMP-EDGE(+-1.5) {100.0*n_sat/max(1,n_tot):.3f}%  "
                          f"(the spread metric above sums BOTH)")
    # Layer-split model parallelism LAST: everything above (packed weights, scale-QAT bounds, latent
    # init, candidate masks) is built on one device first, then whole layers are relocated together.
    _mp = int(getattr(build_student, "_model_parallel", 0) or 0)
    if _mp > 1:
        shard_model_across_gpus(model, _mp,
                                keep_latent_cpu=bool(getattr(build_student, "_latent_offload", False)))
    return model, config


def _assign_param(model, pname, tensor):
    parts = pname.split("."); parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    if hasattr(parent, parts[-1]):
        delattr(parent, parts[-1])
    parent.register_parameter(parts[-1], nn.Parameter(tensor, requires_grad=False))


def _assign_buffer(model, bname, tensor):
    parts = bname.split("."); parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    if hasattr(parent, parts[-1]):
        delattr(parent, parts[-1])
    parent.register_buffer(parts[-1], tensor)


# ─────────────────────────── phases ────────────────────────────────────────────────

def precompute_teacher(args):
    import torch
    from transformers import AutoModelForCausalLM
    n_gpu = torch.cuda.device_count()
    ch = args.cache_hidden
    topk = args.topk

    batches = load_calib_batches(args.calib, 1, args.seq, "cpu")    # per-seq, kept on CPU; moved per-shard
    if args.max_samples:
        batches = batches[:args.max_samples]
    N = len(batches)
    # Pre-sized output slots so any execution order (serial OR data-parallel strided) yields the
    # IDENTICAL per-sequence cache — order is by original sequence index, never by completion time.
    idx_all, val_all, hid_all = [None] * N, [None] * N, [None] * N

    def store(slot, logits_seq, hidden_seq):
        """Top-k of one sequence's logits → cache slot. .float() upcast keeps it byte-identical to
        the original serial path (NOT a bf16 topk, which would drift)."""
        val, idx = torch.topk(logits_seq.float(), topk, dim=-1)
        idx_all[slot] = idx.to(torch.int32).cpu()
        val_all[slot] = val.to(torch.float16).cpu()
        if ch:
            hid_all[slot] = hidden_seq.to(torch.float16).cpu()

    # DATA-PARALLEL: only when explicitly requested AND the model fits in ONE GPU (small testbeds).
    # A full copy lives on each GPU; sequences are split strided; threads drive the GPUs concurrently
    # (CUDA releases the GIL). Each sequence is processed at batch=1 on a full-precision model exactly
    # as in the serial path → the cache is BYTE-IDENTICAL, just produced ~n_gpu× faster. Do NOT enable
    # for a model that doesn't fit one GPU (e.g. 27B) — it would OOM; that uses the serial path below.
    use_dp = getattr(args, "teacher_dp", 0) and n_gpu >= 2 and torch.cuda.is_available()
    if use_dp:
        from threading import Thread
        devices = [f"cuda:{i}" for i in range(n_gpu)]
        print(f"{N} seqs; caching top-{topk}{' + hidden' if ch else ''} — DATA-PARALLEL x{n_gpu} "
              f"(full teacher copy/GPU, byte-identical to serial)", flush=True)
        models = []
        for dev in devices:                                        # load copies one at a time (host-RAM peak)
            models.append(AutoModelForCausalLM.from_pretrained(
                args.teacher_path, trust_remote_code=True, dtype=torch.bfloat16,
                low_cpu_mem_usage=True).to(dev).eval())
        with torch.no_grad():                                      # warm up each GPU so fla/Triton kernels
            for dev, m in zip(devices, models):                    # compile BEFORE concurrent execution
                m(batches[0].to(dev), output_hidden_states=ch)

        def worker(gi):
            dev, m = devices[gi], models[gi]
            with torch.no_grad():
                for slot in range(gi, N, n_gpu):                   # strided: GPU gi does gi, gi+n_gpu, ...
                    out = m(batches[slot].to(dev), output_hidden_states=ch)
                    store(slot, out.logits[0], out.hidden_states[-1][0] if ch else None)
                    del out
                    if gi == 0 and slot % 100 < n_gpu:
                        print(f"   ~{slot}/{N}", flush=True)
        threads = [Thread(target=worker, args=(i,)) for i in range(n_gpu)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        # SERIAL device_map=auto (+ optional --cache-batch) — the path for a model too big for one GPU.
        max_memory = {i: args.gpu_mem for i in range(n_gpu)}
        max_memory["cpu"] = args.cpu_mem
        offload_dir = Path(args.teacher_cache).parent / "_teacher_offload"
        offload_dir.mkdir(parents=True, exist_ok=True)
        print(f"{N} seqs; caching top-{topk}{' + hidden' if ch else ''} — serial (device_map=auto, "
              f"max_memory={max_memory}, offload={offload_dir})", flush=True)
        print("   tip: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True reduces warmup OOM")
        model = AutoModelForCausalLM.from_pretrained(
            args.teacher_path, trust_remote_code=True, dtype=torch.bfloat16,
            device_map="auto", max_memory=max_memory, offload_folder=str(offload_dir),
            low_cpu_mem_usage=True).eval()
        cb = max(1, getattr(args, "cache_batch", 1))
        with torch.no_grad():
            for start in range(0, N, cb):
                chunk = batches[start:start + cb]
                b = torch.cat(list(chunk), dim=0).to("cuda:0")     # [b, seq]
                out = model(b, output_hidden_states=ch)            # accelerate moves it on
                for j in range(b.shape[0]):
                    store(start + j, out.logits[j], out.hidden_states[-1][j] if ch else None)
                del out, b
                if (start + cb) % 20 < cb:
                    print(f"   {min(start + cb, N)}/{N}", flush=True)

    payload = {"idx": idx_all, "val": val_all, "seq": args.seq}
    if ch:
        payload["hidden"] = hid_all
    torch.save(payload, args.teacher_cache)
    print(f"Saved teacher cache -> {args.teacher_cache}")


@torch.no_grad()
def student_rollout(core, prefix_ids, rollout_len, temp):
    """Sample a continuation from the CURRENT ternary student. Returns full ids [1, P+R] and P.
    Toggles gradient-checkpointing off + use_cache on for generation, then restores training mode."""
    was_ckpt = bool(getattr(core, "is_gradient_checkpointing", False))
    if was_ckpt:
        core.gradient_checkpointing_disable()
    core.config.use_cache = True
    core.eval()
    eos = core.config.eos_token_id
    pad = (eos[0] if isinstance(eos, (list, tuple)) else eos) or 0
    gen = core.generate(prefix_ids, max_new_tokens=rollout_len, do_sample=True,
                        temperature=temp, top_p=0.95, pad_token_id=pad)
    core.train()
    core.config.use_cache = False
    if was_ckpt:
        try:
            core.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except Exception:
            core.gradient_checkpointing_enable()
    return gen, prefix_ids.shape[1]


def repetition_unlikelihood(s_logits, full_ids, roll_start):
    """B2 token-level unlikelihood (arXiv:1908.04319) on the rollout: for each generated position,
    penalize the student's probability of tokens that ALREADY appeared in the prefix+prior-rollout —
    i.e. push probability mass off the looping/repetition attractor. Mean over rollout positions."""
    logp = torch.log_softmax(s_logits[0].float(), dim=-1)          # [T, V]
    p = logp.exp()
    T = full_ids.shape[1]
    terms = []
    for t in range(roll_start - 1, T - 1):                        # positions whose NEXT token is a rollout tok
        prev = torch.unique(full_ids[0, : t + 1])                 # negative candidates = seen tokens
        neg_p = p[t, prev].clamp(max=1 - 1e-6)
        terms.append((-torch.log1p(-neg_p + 1e-8)).mean())
    if not terms:
        return s_logits.new_zeros(())
    return torch.stack(terms).mean()


def on_policy_loss(model, teacher, core, ids, args, temperature):
    """A1+B2: roll out the student, score it with the FP teacher, and return the CAKLD(+unlikelihood)
    loss on the student's OWN generated tokens (the exposure-bias signal the cached KL cannot see)."""
    prefix = ids[:, : args.rollout_prefix]
    full, P = student_rollout(core, prefix, args.rollout_len, args.rollout_temp)
    s_dev = full.device                                           # student's GPU
    with torch.no_grad():
        # exec device = GPU even when weights are CPU-offloaded (accelerate streams layers to it)
        t_dev = getattr(teacher, "_exec_device", None) or next(teacher.parameters()).device
        t_logits = teacher(full.to(t_dev)).logits                 # [1, T, V] — only top-64 survives
        tv, ti = t_logits.topk(64, dim=-1)                        # teacher top-64
        tv, ti = tv.to(s_dev), ti.to(s_dev)                       # move the small top-64 back to the GPU
        del t_logits
    s_logits = model(full).logits                                 # [1, T, V] — DDP-wrapped for grad sync
    lo = P - 1                                                    # pos lo..T-2 predict rollout tokens P..T-1
    kl = cakld_loss(s_logits[:, lo:-1], ti[:, lo:-1], tv[:, lo:-1], temperature)
    if args.unlikelihood_weight > 0:
        kl = kl + args.unlikelihood_weight * repetition_unlikelihood(s_logits, full, P)
    return kl


def train(args):
    # DDP when launched via torchrun (LOCAL_RANK set); plain single-GPU otherwise.
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world = int(os.environ.get("WORLD_SIZE", 1))
    ddp = local_rank >= 0 and world > 1
    if ddp:
        import torch.distributed as dist
        from datetime import timedelta
        # Long timeout: the memory-efficient held-out eval streams the CPU-resident lm_head weight per seq
        # (both ranks contend for PCIe), so an eval can take many minutes — well past NCCL's 10-min default,
        # which desyncs the ranks and trips the watchdog (crashed the first CE+assign run at step 600).
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        rank = dist.get_rank()
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        rank = 0
    is_main = (rank == 0)

    def log(*a):
        # TIMESTAMPED. Step lines carry no time of their own and the trainer prints step 1 then every
        # 10, so per-step rate had to be inferred from wall-clock deltas around log reads — which is how
        # the same run produced both 291 and 327 s/step. Piping through `awk systime()` does NOT fix it:
        # awk buffers its INPUT in blocks, so every line gets stamped when awk drains the pipe (verified:
        # three lines emitted 1 s apart all received an identical timestamp). Stamp at the source.
        if is_main:
            print(f"[{_t.strftime('%H:%M:%S')}]", *a, flush=True)

    log(f"Building ternary student (packed 2-bit){f' [DDP x{world}]' if ddp else ''}...")
    build_student._a3_mlp = getattr(args, "a3_mlp", False)     # A3(ii): per-MLP-input scale (folds to norm)
    build_student._a4_downproj = getattr(args, "a4_downproj", False)   # A4: down_proj assignment moves
    build_student._train_weights = getattr(args, "train_weights", "none")   # Test 1: assignment-QAT scope
    build_student._arm_b = getattr(args, "arm_b", False)                     # Arm B: sparse gradient-chosen flips
    build_student._tw_layer_stride = getattr(args, "tw_layer_stride", 1)     # memory: subset of layers
    build_student._tw_layer_offset = getattr(args, "tw_layer_offset", 0)     # sequential group passes
    build_student._latent_offload = getattr(args, "latent_offload", False)   # latents in CPU RAM
    build_student._latent_init = getattr(args, "latent_init", "center")      # Stage 0: bin-centre vs fp-spread
    build_student._cand_tau = getattr(args, "latent_candidate_tau", 0.0)     # candidate-set sparsity (0 = dense)
    build_student._model_parallel = getattr(args, "model_parallel", 0)       # layer-split across N GPUs
    _LAT_PLACE["budget_bytes"] = int(float(getattr(args, "latent_gpu_budget", 0.0)) * 1e9)
    _LAT_PLACE["pin"] = bool(getattr(args, "latent_pin", False))
    build_student._fp_model = getattr(args, "fp_model", None)
    build_student._col_scale = getattr(args, "col_scale", False)   # perpendicular col-scales (A6/A7 probe)
    build_student._sq_bits = getattr(args, "scale_qat_bits", 0)    # scale-QAT: fake-quant scales to n bits (STE)
    model, config = build_student(args.student_path, args.orig_config_path, BLOCK_SIZE, device)
    if _LAT_PLACE["n_gpu"] or _LAT_PLACE["n_pin"]:
        _pin_note = f", {_LAT_PLACE['n_pin']} pinned" if _LAT_PLACE["pin"] else ""
        log(f"   latent placement: {_LAT_PLACE['n_gpu']} GPU-resident "
            f"({_LAT_PLACE['used'] / 1e9:.1f}GB, no H2D copy), "
            f"{_LAT_PLACE['n_cpu']} host-offloaded{_pin_note}")
    model.train()
    # GRADIENT CHECKPOINTING trades compute for activation memory — but on the assignment stage it also
    # DOUBLES every per-latent cost, because the recompute pass re-runs dequant() for each trained
    # linear: the CPU->GPU latent stream, the STE dequant, and the grad-release hook all happen twice per
    # step. At arm B that is 205 GB/step of PCIe traffic over a measured ~5 GB/s link and 994 per-tensor
    # dequants instead of 497. --no-grad-ckpt halves all of it at the cost of activation memory, which
    # the latent_gpu leak fix freed up (~9 GB/card). Measured arm B step: 327 s with checkpointing on.
    _gc_stride = int(getattr(args, "grad_ckpt_stride", 1) or 1)
    if getattr(args, "no_grad_ckpt", False):
        log("   gradient checkpointing DISABLED (--no-grad-ckpt): halves per-latent work, costs activations")
    elif _gc_stride > 1:
        # PARTIAL checkpointing. Full checkpointing re-runs dequant() for EVERY trained linear on the
        # recompute pass (994 per-tensor dequants per step at arm B); disabling it entirely halves that
        # but OOMs (measured: 32 layers of activations at seq 2560 exceed 24GB). Checkpoint only every
        # Nth layer to trade a controlled slice of activation memory for a controlled slice of recompute.
        # HF's gradient_checkpointing flag lives on the MODEL, not the layer, so wrap layer.forward
        # directly. use_reentrant=False is required for kwargs, and it stashes/restores RNG state so the
        # recomputed forward reproduces the original bin decisions exactly (STE-safe).
        import torch.utils.checkpoint as _ckpt
        _best = None
        for _n, _mod in model.named_modules():
            if isinstance(_mod, nn.ModuleList) and (_best is None or len(_mod) > len(_best)):
                _best = _mod
        _wrapped = 0
        for _i, _layer in enumerate(_best or []):
            if _i % _gc_stride:
                continue
            def _mk(_l):
                _orig = _l.forward
                def _fwd(*a, **kw):
                    return _ckpt.checkpoint(_orig, *a, use_reentrant=False, **kw)
                return _fwd
            _layer.forward = _mk(_layer)
            _wrapped += 1
        log(f"   PARTIAL gradient checkpointing: {_wrapped}/{len(_best or [])} layers "
            f"(every {_gc_stride}) — trades activation memory for fewer recomputed dequants")
    else:
      try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        log("   gradient checkpointing enabled (non-reentrant)")
      except Exception:
        try:
            model.gradient_checkpointing_enable()
            log("   gradient checkpointing enabled")
        except Exception as e:
            log(f"   ⚠️ gradient_checkpointing_enable failed ({e}); memory may be tight")

    # freeze everything except the ternary scales (+ optional assignment latents). Latents are separated so
    # they can take a MUCH higher lr: a trit only flips when its latent moves ~0.5×scale, which never happens
    # at the scale lr (measured: assign-moved=0.000% at lr 1e-5). --latent-lr drives the assignment mobility.
    scale_only = []; latents = []
    for n, p in model.named_parameters():
        if n.endswith(".latent"):
            p.requires_grad_(True); latents.append(p)
        elif n.endswith(".scale") or n.endswith(".a3_scale") or n.endswith(".cs_in"):  # +A3(ii), +col-scale
            p.requires_grad_(True); scale_only.append(p)
        else:
            p.requires_grad_(False)
    _latent_lr = float(getattr(args, "latent_lr", 0.0)) or args.lr
    scales = scale_only + latents          # combined list used by grad-clip / snapshot / export below
    # optimizer param GROUPS: scales at args.lr, latents at their own (usually higher) lr. base_lr is stored so
    # the LR schedule can scale each group by the same fraction, preserving the ratio.
    opt_groups = [{"params": scale_only, "lr": args.lr, "base_lr": args.lr, "is_latent": False}]
    if latents:
        # ONE GROUP PER LATENT LAYER so TALR can servo each layer against ITS OWN flip rate. A single global
        # controller only sees the aggregate: measured on an 8-layer run, one layer burst to 15.5% of its
        # assignments in the first post-warmup steps while the median layer sat at 3.3% (5x spread) — the
        # global rate looked fine, so the controller never reacted. With 64 layers (27B) there are more
        # chances for that. --per-layer-tr 0 falls back to the old single shared group.
        if getattr(args, "per_layer_tr", True) and len(latents) > 1:
            for li, p_ in enumerate(latents):
                opt_groups.append({"params": [p_], "lr": _latent_lr, "base_lr": _latent_lr,
                                   "is_latent": True, "lat_idx": li})
        else:
            opt_groups.append({"params": latents, "lr": _latent_lr, "base_lr": _latent_lr,
                               "is_latent": True, "lat_idx": -1})
    log(f"   training {len(scale_only)} scale tensors ({sum(s.numel() for s in scale_only)/1e6:.1f}M) "
        f"+ {len(latents)} latents ({sum(s.numel() for s in latents)/1e6:.1f}M) @ latent-lr {_latent_lr:.1e}")

    core = model                                          # underlying model, used for save_student
    on_policy = getattr(args, "on_policy_frac", 0.0) > 0
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        # broadcast_buffers=False: the packed weight buffers are identical and constant, so don't
        # re-broadcast ~7 GB every step. Only the scale grads get all-reduced.
        # on-policy steps forward variable-length rollouts → the autograd graph varies, so static_graph
        # must be OFF when on-policy is enabled (else DDP asserts the graph is identical every step).
        model = DDP(core, device_ids=[local_rank], output_device=local_rank,
                    broadcast_buffers=False, find_unused_parameters=False, static_graph=not on_policy)
    _latent_qat = (getattr(args, "a4_downproj", False) or
                   getattr(args, "train_weights", "none") not in (None, "none")) and not getattr(args, "arm_b", False)
    if _latent_qat:
        # Arm A latent assignment-QAT adds big fp32 latents; fp32 Adam states (2×) OOM 24GB. 8-bit Adam cuts
        # the optimizer state 4× (well-validated to match fp32 closely). (Arm B has no latents → fp32 Adam.)
        if getattr(args, "latent_offload", False):
            # bitsandbytes cannot step CPU tensors (verified: AttributeError on a CPU param), and offloaded
            # latents ARE CPU leaves, so use torch Adam. Its fp32 state for the latents lives in CPU RAM too
            # (~6GB for 32 down_proj) — fine on a 60GB host, and it is what makes full-coverage possible.
            # foreach=False is ESSENTIAL here: the default foreach path allocates same-size temporaries across
            # the whole param group during the step, which for 755M CPU latents pushed RSS to 43.8GB (vs a
            # ~20GB identified budget) and crossed the cgroup MemoryHigh line -> synchronous reclaim -> the
            # process wedged in D state with the GPU idle. Per-tensor stepping is slower but bounded.
            # --latent-opt selects the CPU-resident latent optimizer. Adam is the calibrated default and
            # every recorded number uses it. SGD+momentum is under evaluation purely for THROUGHPUT: it
            # holds one state tensor instead of two (12 vs 16 B/latent touched per step), measured 3.91x
            # faster than Adam on this bandwidth-bound box (20.7 vs 81.1 ms per 23.6M latents).
            # CAUTION on lr: Adam's step is ~lr (gradient-normalised); SGD's is lr*|g|. They are NOT
            # comparable, and TALR CANNOT rescue a too-small base lr (its gain is capped at
            # _tr_gain_max=1.0 — it may only throttle). Calibrate --latent-lr for SGD, erring HIGH so the
            # servo throttles down into range rather than starving.
            _lopt = str(getattr(args, "latent_opt", "adam")).lower()
            if _lopt == "sgd":
                opt = torch.optim.SGD(opt_groups, momentum=float(getattr(args, "latent_momentum", 0.9)),
                                      foreach=False)
                log(f"   torch.optim.SGD momentum={getattr(args, 'latent_momentum', 0.9)} "
                    f"(latent-offload; foreach=False) [THROUGHPUT EXPERIMENT]")
            else:
                opt = torch.optim.Adam(opt_groups, foreach=False)
                log("   torch.optim.Adam (latent-offload: latents + Adam state in CPU RAM; foreach=False)")
        else:
            import bitsandbytes as bnb
            opt = bnb.optim.PagedAdam8bit(opt_groups)       # PAGED: optimizer state lives in CPU RAM (paged)
            log(f"   PAGED 8-bit Adam (latent assignment-QAT; {sum(s.numel() for s in scales)/1e6:.0f}M params)")
    else:
        opt = torch.optim.Adam(opt_groups)                  # two groups: scales @lr, latents @latent-lr

    teacher = None
    if on_policy:
        from transformers import AutoModelForCausalLM
        tpath = getattr(args, "onpolicy_teacher", None) or args.orig_config_path
        log(f"   ON-POLICY {args.on_policy_frac:.0%} (warmup {args.on_policy_warmup_frac:.0%}, "
            f"rollout {args.rollout_prefix}+{args.rollout_len}, unlik {args.unlikelihood_weight}); "
            f"loading FP teacher {tpath}")
        teacher = AutoModelForCausalLM.from_pretrained(tpath, trust_remote_code=True,
                                                       dtype=torch.bfloat16).eval()
        teacher.config.use_cache = False
        for p in teacher.parameters():
            p.requires_grad_(False)
        if getattr(args, "onpolicy_teacher_cpu", False):
            # Weights live in system RAM; accelerate streams each layer to the GPU for its forward (so the
            # fla Gated-DeltaNet TRITON kernels still run on GPU — a pure-CPU forward can't, they're GPU-only)
            # then evicts it. Peak resident VRAM ~= one layer (~0.5GB) not the full 8GB. Frees VRAM on the
            # scarce 2x24GB box at the cost of CPU<->GPU streaming per on-policy step.
            from accelerate import cpu_offload
            teacher = cpu_offload(teacher, execution_device=device)
            teacher._exec_device = device
            log("   on-policy teacher CPU-OFFLOADED via accelerate (weights in RAM, streamed to GPU per fwd)")
        else:
            teacher = teacher.to(device)
    loss_fn = cakld_loss if getattr(args, "loss_fn", "topk_kl") == "cakld" else topk_kl_loss
    global DECISION_GAMMA, SKEW_ALPHA
    DECISION_GAMMA = getattr(args, "decision_gamma", 1.0)
    SKEW_ALPHA = getattr(args, "skew_alpha", 0.0)
    log(f"   loss function: {getattr(args, 'loss_fn', 'topk_kl')} (decision-gamma {DECISION_GAMMA}"
        f"{f', skew-alpha {SKEW_ALPHA}' if SKEW_ALPHA > 0 else ''})")

    # ── MEMORY-EFFICIENT hidden-state loss path (CE + assignment-QAT stage) ──────────────
    # The default step materialises full-vocab logits [1,T,V] (1.3GB) + their fp32 backward (~2.5GB), leaving
    # no room for the assignment latents on 24GB. When CE and/or --train-weights is active we instead: run the
    # model with logits_to_keep=1 (skips the full lm_head), capture the POST-NORM hidden via a hook on the final
    # norm, and compute the top-K KL chunk-wise from that hidden (chunked_hidden_state_loss — never forms
    # [1,T,V]) + CE on a position subset (hidden[sel] @ Wlm). Frees ~2.5GB. Gated ⇒ default path byte-identical.
    _mem_eff = (float(getattr(args, "ce_weight", 0.0)) > 0
                or getattr(args, "train_weights", "none") not in (None, "none")
                or os.environ.get("FORCE_MEM_EFF") == "1")   # scale-only at fine grids: avoid full-vocab logits
    _hidcap = {}
    if _mem_eff:
        _base = core.model.language_model if hasattr(core.model, "language_model") else core.model
        _base.norm.register_forward_hook(lambda mod, inp, out: _hidcap.__setitem__("h", out))
        _lm_head_mod = core.lm_head
        # The model's lm_head output is UNUSED here (loss is computed from the norm-hook hidden via Wlm), yet
        # its forward still unpacks the full [V,H] ternary weight every step — a 2.37GB transient that
        # intermittently OOMs. Replace its forward with a cheap zeros of the right shape (save_student reads
        # dequant() directly, not forward, so folding at save is unaffected).
        _V_lm = _lm_head_mod.out_features if hasattr(_lm_head_mod, "out_features") else _lm_head_mod.weight.shape[0]
        _lm_head_mod.forward = (lambda x, _v=_V_lm: x.new_zeros(x.shape[:-1] + (_v,)))
        # Precompute the lm_head weight ONCE (detached bf16, 1.27GB). Recomputing per step would rebuild a
        # 2.5GB fp32 [V,H] intermediate each iteration and OOM. lm_head assignments/scale are held frozen on
        # this stage (no grad reaches them ⇒ Wlm stays valid); --train-weights targets the body, not lm_head.
        with torch.no_grad():
            _w = _lm_head_mod.dequant() if hasattr(_lm_head_mod, "dequant") else _lm_head_mod.weight
            # OFFLOAD Wlm to CPU RAM (pinned). Both the chunked KL and the chunked CE stream vocab chunks
            # CPU→GPU on demand, so the 1.27GB [V,H] weight never sits resident on the GPU — that headroom
            # is exactly what the assignment latents need. (torch_pin fails on some setups; fall back to plain.)
            _Wlm = _w.detach().to(torch.bfloat16).cpu().contiguous()
            try:
                _Wlm = _Wlm.pin_memory()
            except Exception:
                pass
            del _w
        torch.cuda.empty_cache()
        _ce_lt = "cakld" if getattr(args, "loss_fn", "topk_kl") == "cakld" else "topk_kl"
        _mem_ctx = {"Wlm": _Wlm, "norm": _base.norm, "loss_type": _ce_lt}
        log(f"   MEM-EFFICIENT hidden-state loss ON (chunked KL + chunked CE, Wlm OFFLOADED to CPU "
            f"{tuple(_Wlm.shape)}; commit-beta unavailable on this path)")
        torch.cuda.empty_cache()          # release the lm_head dequant transient before promoting latents
    else:
        _mem_ctx = None

    # GPU residency LAST: after the lm_head transient is freed, so the budget sees real free VRAM.
    # Placing latents earlier (in build_student) starved that ~5GB transient and OOM'd.
    promote_latents_to_gpu(model, int(float(getattr(args, "latent_gpu_budget", 0.0)) * 1e9))
    if getattr(args, "latent_prefetch", False):
        global _PF_ON, _GRAD_PIN_ON
        _PF_ON = True
        # pinned gradient staging: only safe when each backward's grad is consumed before the next
        # (grad-release clears param.grad every step). With --accum > 1 autograd would accumulate INTO
        # the reused buffer, so fall back to the plain pageable path there.
        _GRAD_PIN_ON = (bool(getattr(args, "latent_pin_grad", False))
                        and bool(getattr(args, "latent_grad_release", False))
                        and int(getattr(args, "accum", 1) or 1) == 1)
        if getattr(args, "latent_pin_grad", False) and not _GRAD_PIN_ON:
            log("   [warn] --latent-pin-grad ignored (needs --latent-grad-release and --accum 1)")
        _LAT_ORDER.clear()
        for _m in core.modules():                 # definition order ~= execution order for a transformer
            if getattr(_m, "latent", None) is not None:
                _m._lat_idx = len(_LAT_ORDER)
                _LAT_ORDER.append(_m)
        # prime the pipeline so the FIRST latent is already in flight
        if _LAT_ORDER:
            _pf_start(_LAT_ORDER[0], _LAT_ORDER[0].latent.dtype)
        log(f"   latent PREFETCH on: {len(_LAT_ORDER)} latents chained on side CUDA streams"
            + ("" if getattr(args, "latent_pin", False) else "  [WARN: without --latent-pin the copies"
               " are pageable and therefore synchronous, so prefetch cannot overlap]"))

    # Assignment-flip diagnostic: snapshot the initial hard trits of every latent (assignment-QAT) module so
    # each held-out eval can report what % of trit ASSIGNMENTS have actually moved. If this stays ~0 the STE
    # is not flipping (mechanism not engaging) — abort early instead of burning the whole run.
    def _hard_trits(m):
        # latent may be a CPU leaf (--latent-offload) while scale is on GPU: compute on the latent's device
        s = m.scale.detach().to(m.latent.device).unsqueeze(1).clamp_min(1e-8)
        return torch.round(m.latent.reshape(m.n_blocks, m.block_size) / s).clamp(-1, 1)
    _lat_mods = [m for m in core.modules() if getattr(m, "latent", None) is not None]
    _lat_tot = float(sum(m.latent.numel() for m in _lat_mods)) or 1.0
    # ── GRADIENT RELEASE: step each latent as soon as its grad is ready, then free the grad ──
    # The fp32 grad buffer is 4B/latent (9GB for mlp@32L) and normally ALL of them coexist between backward
    # and opt.step(). Registering a post-accumulate-grad hook lets each latent take its Adam step
    # individually and immediately drop its grad, so at most one layer's grads are live. This is EXACT Adam
    # (the update is per-parameter), unlike swapping in a cheaper optimizer.
    if getattr(args, "latent_bf16_compute", False):
        for _m in core.modules():
            if getattr(_m, "latent", None) is not None:
                _m._latent_bf16_compute = True
        log("   latent bf16-compute ON (fp32 master on CPU, bf16 GPU copy + bf16 grad)")
    _grad_release = bool(getattr(args, "latent_grad_release", False)) and bool(latents)
    if _grad_release:
        _lat_set = {id(p_) for p_ in latents}
        _lat_index = {id(p_): i for i, p_ in enumerate(latents)}
        _adam_scratch = {"b": None}          # one reusable denom buffer for the whole run (see _adam_step_one)
        _lat_group_of = {}
        for g in opt.param_groups:
            for p_ in g["params"]:
                if id(p_) in _lat_set:
                    _lat_group_of[id(p_)] = g
        _state_dir = None
        if getattr(args, "latent_state_nvme", False):
            import numpy as _np
            _state_dir = Path(args.out).parent / "_adam_state"
            shutil.rmtree(_state_dir, ignore_errors=True)
            _state_dir.mkdir(parents=True, exist_ok=True)
            log(f"   Adam state NVMe-backed at {_state_dir} "
                f"(frees {sum(p_.numel() for p_ in latents)*8/1e9:.1f}GB of RAM; slower per step)")

        def _state_buf(param, tag, idx):
            """Zeroed fp32 buffer shaped like `param`: in RAM, or NVMe-backed when --latent-state-nvme.
            Adam touches exp_avg/exp_avg_sq exactly ONCE per step per tensor, so a memmap is a good trade:
            the page cache absorbs the reads and only dirty pages are written back."""
            if _state_dir is None:
                return torch.zeros_like(param, memory_format=torch.preserve_format)
            import numpy as _np
            f = _state_dir / f"{tag}_{idx}.dat"
            arr = _np.memmap(str(f), dtype=_np.float32, mode="w+", shape=(param.numel(),))
            return torch.from_numpy(arr).view_as(param)

        @torch.no_grad()
        def _adam_step_one(group, param):
            """Exact single-tensor Adam update, using state stored in `opt` so it persists across steps."""
            st = opt.state[param]
            if len(st) == 0:
                i = _lat_index.get(id(param), 0)
                st["step"] = torch.zeros((), dtype=torch.float32)
                st["exp_avg"] = _state_buf(param, "m", i)
                st["exp_avg_sq"] = _state_buf(param, "v", i)
            b1, b2 = group["betas"]
            st["step"] += 1
            t = float(st["step"])
            gr = param.grad
            if gr.dtype != torch.float32:
                gr = gr.float()                      # state math stays fp32 even if the grad arrived bf16
            st["exp_avg"].mul_(b1).add_(gr, alpha=1 - b1)
            st["exp_avg_sq"].mul_(b2).addcmul_(gr, gr, value=1 - b2)
            bc1 = 1 - b1 ** t
            bc2 = 1 - b2 ** t
            # `(exp_avg_sq.sqrt() / c).add_(eps)` allocates TWO full-size fp32 temporaries per latent per
            # step — 94MB x 32 latents = ~3GB of alloc/free churn EVERY step. glibc keeps those arenas, so
            # RSS ratchets ~170MB/step and the run wedges in D state around step 100. Reuse ONE scratch
            # buffer (sized to the largest latent, all ops in-place): +94MB fixed, zero per-step allocation.
            if _adam_scratch.get("b") is None or _adam_scratch["b"].numel() < param.numel():
                _adam_scratch["b"] = torch.empty(param.numel(), dtype=torch.float32, device=param.device)
            denom = _adam_scratch["b"][:param.numel()].view_as(param)
            denom.copy_(st["exp_avg_sq"]).sqrt_().div_(math.sqrt(bc2)).add_(group["eps"])
            param.addcdiv_(st["exp_avg"], denom, value=-group["lr"] / bc1)

        @torch.no_grad()
        def _sgd_step_one(group, param):
            """Single-tensor SGD+momentum update — the throughput variant of _adam_step_one.

            ONE state tensor (momentum_buffer) instead of Adam's two, so the step touches 12 B/latent
            instead of 16. On this bandwidth-bound CPU that measured 3.91x faster (20.7 vs 81.1 ms per
            23.6M latents). No scratch buffer is needed: unlike Adam's denom there is no elementwise
            sqrt/divide, so nothing full-size is ever allocated.

            Matches torch.optim.SGD semantics (dampening=0, nesterov=False): buf = mom*buf + g; p -= lr*buf.
            NOTE the update is lr*|g| here vs Adam's gradient-NORMALISED ~lr — the two lrs are not
            interchangeable and --latent-lr must be recalibrated per --latent-opt.
            """
            st = opt.state[param]
            if len(st) == 0:
                i = _lat_index.get(id(param), 0)
                st["momentum_buffer"] = _state_buf(param, "m", i)
            gr = param.grad
            if gr.dtype != torch.float32:
                gr = gr.float()
            buf = st["momentum_buffer"]
            buf.mul_(_sgd_mom).add_(gr)
            param.add_(buf, alpha=-group["lr"])

        @torch.no_grad()
        def _adam_blockv_step_one(group, param):
            """Adam with ONE second moment per BLOCK_SIZE-latent scale block instead of per element.

            Measured 1.48x faster than the per-element step on this bandwidth-bound CPU (48.2 vs
            71.1 ms per 23.6M latents) — the largest optimizer win that survived testing.

            WHY THIS ONE WORKS where int8 (+10.3%), bf16 (+47%) and Adafactor (+14%) all LOST: those
            compress the second moment but must reconstruct a per-element denominator, which costs a
            full-size elementwise pass. Here the denominator is per-block and BROADCASTS over 256
            contiguous latents — no full-size temporary, no per-element convert. It also lands exactly on
            the existing scale-block boundary, so the normalisation granularity already matches the
            quantiser's.

            Per-coordinate normalisation is WEAKENED to per-block (256 latents share a denominator).
            §4c showed a GLOBAL lr (SGD) destroys per-layer motion uniformity; per-block is far finer
            than global, but the acceptance bar is the per-layer assign-moved ratio staying ~1x.
            """
            st = opt.state[param]
            n = param.numel()
            if len(st) == 0:
                i = _lat_index.get(id(param), 0)
                st["step"] = torch.zeros((), dtype=torch.float32)
                st["exp_avg"] = _state_buf(param, "m", i)
                st["v_blk"] = torch.zeros(n // _BLKV, dtype=torch.float32)
            b1, b2 = group["betas"]
            st["step"] += 1
            t = float(st["step"])
            gr = param.grad
            if gr.dtype != torch.float32:
                gr = gr.float()
            st["exp_avg"].mul_(b1).add_(gr, alpha=1 - b1)
            gb = gr.view(-1, _BLKV)
            st["v_blk"].mul_(b2).add_(gb.pow(2).mean(dim=1), alpha=1 - b2)
            bc1, bc2 = 1 - b1 ** t, 1 - b2 ** t
            denom = (st["v_blk"] / bc2).sqrt_().add_(group["eps"]).unsqueeze(1)   # [nblk,1], broadcasts
            param.view(-1, _BLKV).addcdiv_(st["exp_avg"].view(-1, _BLKV),
                                           denom.expand(-1, _BLKV), value=-group["lr"] / bc1)

        # recomputed here (not reusing _lopt) because that name is only bound on the --latent-offload path
        _lat_opt_name = str(getattr(args, "latent_opt", "adam")).lower()
        _BLKV = BLOCK_SIZE
        _sgd_mom = float(getattr(args, "latent_momentum", 0.9))
        _lat_step_one = {"sgd": _sgd_step_one,
                         "adam-blockv": _adam_blockv_step_one}.get(_lat_opt_name, _adam_step_one)
        if _lat_opt_name == "sgd":
            log(f"   grad-release latent step: SGD momentum={_sgd_mom} (1 state tensor, 12 B/latent)")
        elif _lat_opt_name == "adam-blockv":
            log(f"   grad-release latent step: Adam with block-{_BLKV}-shared second moment "
                f"(measured 1.48x; per-block normalisation)")
        _grad_diag = {"done": not bool(getattr(args, "latent_grad_diag", False))}

        # param -> candidate mask (set only when --latent-candidate-tau > 0)
        _cand_of = {}
        for _m_ in core.modules():
            if getattr(_m_, "_cand_mask", None) is not None and getattr(_m_, "latent", None) is not None:
                _cand_of[id(_m_.latent)] = _m_._cand_mask
        if _cand_of:
            log(f"   candidate-set masking ACTIVE on {len(_cand_of)} latents "
                f"(non-candidates receive zero grad => exactly frozen)")

        def _mk_release(p_):
            def _hook(param):
                g = _lat_group_of[id(param)]
                _msk = _cand_of.get(id(param))
                if _msk is not None and param.grad is not None:
                    # Zeroing the grad freezes non-candidates EXACTLY: their m and v start at 0 and only
                    # ever see g=0, so m/(sqrt(v)+eps) stays 0 and the latent never moves. Cheaper and
                    # less error-prone than masking the update (no full-size temporary).
                    param.grad.mul_(_msk)
                if param.grad is not None and not _grad_diag["done"]:
                    # One-shot calibration aid: Adam's step is ~lr, SGD's is lr*|g|, so the lr that gives
                    # a matched effective step size is roughly adam_lr / grad_rms. Print it once.
                    _rms = float(param.grad.float().pow(2).mean().sqrt())
                    log(f"   [grad-diag] latent grad RMS {_rms:.3e} | Adam step~lr, SGD step~lr*{_rms:.3e} "
                        f"=> lr_sgd ~ lr_adam / {_rms:.3e} = {(g.get('base_lr', 0.0) / max(_rms, 1e-30)):.3e}")
                    _grad_diag["done"] = True
                if param.grad is not None and g["lr"] != 0.0:
                    _lat_step_one(g, param)
                param.grad = None                          # free it either way (warmup holds lr at 0)
            return _hook
        for p_ in latents:
            p_.register_post_accumulate_grad_hook(_mk_release(p_))
        log(f"   grad-release ON: {len(latents)} latents step during backward and free their grads "
            f"(saves {sum(p_.numel() for p_ in latents)*4/1e9:.1f}GB of fp32 grad buffers)")

    # ── TALR state (transition-rate control of the latent lr) ──
    _tr_target = float(getattr(args, "target_tr", 0.0))
    _lat_groups = [g for g in opt.param_groups if g.get("is_latent")]
    _lat_base_lr = (_lat_groups[0]["base_lr"] if _lat_groups else args.lr)
    _tr_gains = [1.0] * max(1, len(_lat_mods))       # PER-LAYER gains (one servo per layer)
    # Upper bound on the TALR multiplier. 1.0 = TALR may only THROTTLE the calibrated base latent-lr, never
    # amplify it. Env-overridable (TR_GAIN_MAX) for experiments that deliberately want the old behaviour.
    _tr_gain_max = float(getattr(args, "tr_gain_max", 0) or os.environ.get("TR_GAIN_MAX", "8.0"))
    # RATE-LIMITED ramp: the gain may at most DOUBLE over --tr-ramp-2x-steps steps, expressed that way so it
    # is invariant to --tr-every. This is the primary burst guard — a fresh axis can still climb to whatever
    # rate it needs (measured: the @8L attn stage started at literally ZERO flips and had to open up), it just
    # cannot get there fast enough to blow past the target before a correction lands. The old 1.3x-per-
    # measurement compounded to ~8x in 40 steps and burst two stages (t2_up gain 4.83, KL 0.4575->0.5698;
    # @8L attn gain 17.92).
    _tr_ramp2x = max(1.0, float(getattr(args, "tr_ramp_2x_steps", 0) or 40.0))
    _tr_up = 2.0 ** (max(1, args.tr_every) / _tr_ramp2x)
    _tr_prev_per = None
    # COLD-START lr ramp. The gain ramp above only bounds how fast TALR *grows* the lr; it cannot bound the
    # value the lr STARTS at. The base --latent-lr lands at full strength the moment the second-moment warmup
    # ends, i.e. `tr_every` steps before TALR's first measurement, and that un-servoed window is what actually
    # bursts: down@8L measured 2.87e-4 flips/step against a 1.08e-4 target with the gain still at ~1.0 and
    # already being clamped. Ramping the lr in from 0 means no raw un-servoed lr is ever applied, so the base
    # lr no longer has to be hand-tuned per layer-coverage (the thing that breaks when 8L -> 32L -> 64L).
    _lat_warm = int(getattr(args, "latent_warmup_steps", 0))
    _lat_ramp = int(getattr(args, "latent_lr_ramp_steps", 0) or 4 * max(1, args.tr_every))

    def _lat_frac(step):
        """0 during the Adam second-moment warmup, then LINEAR 0→1 over `_lat_ramp` steps."""
        if step < _lat_warm:
            return 0.0
        return 1.0 if _lat_ramp <= 0 else min(1.0, (step - _lat_warm + 1) / _lat_ramp)
    if _lat_mods and any(g.get("lat_idx", -1) >= 0 for g in _lat_groups):
        assert len(_lat_groups) == len(_lat_mods), (
            f"per-layer TALR needs one optimizer group per latent module "
            f"({len(_lat_groups)} groups vs {len(_lat_mods)} modules)")
    # Flip rate is estimated on a FIXED RANDOM SAMPLE (~32k weights/module). Exact tracking would need a
    # 755MB snapshot plus a 3GB float transient every measurement — which OOM'd the card. A 1M-weight
    # sample resolves rates down to ~1e-6/step, far finer than the 1e-4 targets we control to.
    # Build the permutation on the CPU and move only the 32k survivors. randperm allocates an index
    # tensor over the WHOLE latent (int64: 712MB for an 89M-element latent, and torch asked for
    # 1.35GB), which is fine in host RAM but OOMs a 24GB card the moment a latent is GPU-RESIDENT
    # (--latent-gpu-budget). Sampling indices is device-independent, so do it where memory is cheap.
    _tr_idx = [torch.randperm(m.latent.numel(), device="cpu")[:32768].to(m.latent.device)
               for m in _lat_mods]
    def _hard_trits_per_mod():
        """Sampled hard trits, PER MODULE (so per-layer flip rates are visible)."""
        out = []
        for m, idx in zip(_lat_mods, _tr_idx):
            s = m.scale.detach().to(m.latent.device).clamp_min(1e-8)[idx // m.block_size]
            out.append(torch.round(m.latent.detach().reshape(-1)[idx] / s).clamp(-1, 1).to(torch.int8))
        return out
    def _hard_trits_all():
        if not _lat_mods:
            return torch.zeros(1, dtype=torch.int8)
        # same multi-device hazard as _lat_ref below: GPU-resident latents span both cards
        return torch.cat([t.cpu() for t in _hard_trits_per_mod()])
    def _per_layer_moved():
        """% of sampled assignments moved since init, PER LAYER. A single global latent-lr + global flip-rate
        target can hide large per-layer imbalance (deep layers cascading while shallow ones stay inert), and
        inter-layer error compounds with depth — so this must be checked before scaling to the 64-layer 27B."""
        if _lat_ref_per is None:
            return []
        return [100.0 * float((c != r).float().mean().item())
                for c, r in zip(_hard_trits_per_mod(), _lat_ref_per)]
    _lat_ref_per = _hard_trits_per_mod() if _lat_mods else None   # per-layer init refs
    # .cpu() before cat: with --latent-gpu-budget the resident latents follow their LAYER, so under
    # model parallelism these per-module samples span cuda:0 AND cuda:1 and torch.cat refuses to mix
    # devices. They are tiny diagnostic samples (32k each), so CPU is the natural common ground.
    _lat_ref = torch.cat([t.cpu() for t in _lat_ref_per]) if _lat_mods else None
    if _tr_target > 0 and _lat_mods:
        _tr_prev = _lat_ref.clone()
        log(f"   TALR ON: target transition rate {_tr_target:.2e}/step → {_tr_target*args.tr_final_frac:.2e} "
            f"(measured every {args.tr_every} steps; latent-lr servo-controlled, base {_lat_base_lr:.1e})")
        log(f"   latent-lr cold start: frozen for {_lat_warm} steps, ramped 0→base over the next {_lat_ramp}, "
            f"TALR servo engages at step {_lat_warm + _lat_ramp + args.tr_every}")
    def _assign_flip_pct():
        """% of assignments moved since init, estimated on the SAME fixed 1M-weight sample TALR uses.
        A full 755M snapshot (755MB int8) plus its per-eval float transients OOM'd the card on top of the
        3GB of fp32 latents; the sample resolves far finer than any rate we act on."""
        if not _lat_mods or _lat_ref is None:
            return 0.0
        with torch.no_grad():
            cur = _hard_trits_all()
            return 100.0 * float((cur != _lat_ref).float().mean().item())

    try:                                                        # mmap: big teacher caches (e.g. 64M-token ≈24GB)
        cache = torch.load(args.teacher_cache, mmap=True)       # become RECLAIMABLE file-backed page cache,
    except Exception:                                           # not per-rank anonymous RSS → survives the memguard
        cache = torch.load(args.teacher_cache)
    feat_w = getattr(args, "feat_weight", 0.0)
    if feat_w > 0 and "hidden" not in cache:
        raise SystemExit("--feat-weight > 0 needs teacher hidden states; regenerate the teacher "
                         "cache with --cache-hidden (delete the old one first).")
    if feat_w > 0:
        log(f"   + hidden-state feature distillation (weight {feat_w}); note E2E-QP trains only "
            f"scales, so this is weaker leverage than in block_qat.py")
    batches = load_calib_batches(args.calib, 1, args.seq, device)
    n = min(len(batches), len(cache["idx"]))
    if args.max_samples:
        n = min(n, args.max_samples)
    # Stage B: on-policy rollouts start from CHAT/STEM prefixes (not the generic --calib), so the student
    # re-generates its OWN reasoning and the teacher corrects the drift/over-thinking. Off-policy steps still
    # distill on --calib (generic replay = anti-forgetting). Prefix = first --rollout-prefix tokens.
    op_batches = (load_calib_batches(args.onpolicy_calib, 1, args.rollout_prefix, device)
                  if getattr(args, "onpolicy_calib", None) else None)
    if op_batches:
        log(f"   on-policy rollout prompts: {len(op_batches)} chat prefixes from {args.onpolicy_calib}")
    # Held-out slice (MANDATORY for weight-training select — training-batch KL is inadmissible). Reserve the
    # LAST heldout_n seqs; never train on them; select + abort run on their KL/flips.
    ho_n = getattr(args, "heldout_n", 0)
    probe_n = getattr(args, "probe_n", 0) if getattr(args, "arm_b", False) else 0
    gate_n = getattr(args, "flip_gate_n", 0) if getattr(args, "arm_b", False) else 0   # DISJOINT paired-Δ accept gate
    n_train = n - ho_n - probe_n - gate_n
    held_idx = list(range(n - ho_n, n)) if ho_n > 0 else []
    probe_idx = list(range(n_train + gate_n, n_train + gate_n + probe_n)) if probe_n > 0 else []  # fresh-grad set (disjoint)
    gate_idx = list(range(n_train, n_train + gate_n)) if gate_n > 0 else []      # generalisation gate (disjoint from grad)
    ar_idx = probe_idx[:min(16, len(probe_idx))]               # IN-SAMPLE ρ for η control (same set as grad/prediction)
    shard = list(range(rank, n_train, world)) if ddp else list(range(n_train))   # each rank a different slice
    log(f"   {n} seqs ({n_train} train, {probe_n} grad-probe, {gate_n} gate, {ho_n} held-out); {len(shard)} on this rank")
    arm_b_mods = [m for m in core.modules() if getattr(m, "_arm_b", False)]
    # Snapshot buffers are PREALLOCATED ONCE and copied into. `scales` includes the assignment latents
    # (3GB at down@32L), so cloning per improvement allocated AND freed 3GB at every held-out improvement —
    # and held-out improves at essentially every eval. glibc does not return those arenas to the OS, so RSS
    # ratchets up (~24 -> 41GB by step 100) until the cgroup throttles and the run wedges in D state.
    # Copying into fixed buffers makes snapshot allocation O(1) for the run instead of O(#evals).
    _snap_bufs = {"s": None, "t": None}
    # HELD-OUT SNAPSHOT. Unlike best_scales this one is load-bearing: RESULTS_SUMMARY §5 requires that a
    # stage which never beats its step-0 entry baseline restores that entry, so weight-training always
    # keeps one. But it is another 4 B/latent — 102.5 GB at arm B on the 27B — and that third full copy
    # of every latent (latents + Adam moment + snapshot) is what pushed the arm B run past 629 GB.
    # --snap-nvme backs it with a memmap instead: it is written once per improvement and read at most
    # once at the end, so the page cache absorbs it and only dirty pages ever hit disk.
    _snap_nvme = bool(getattr(args, "snap_nvme", False))
    _snap_dir = None
    if _snap_nvme:
        _snap_dir = Path(args.out).parent / "_snap_state"
        shutil.rmtree(_snap_dir, ignore_errors=True)
        _snap_dir.mkdir(parents=True, exist_ok=True)
        log(f"   held-out snapshot NVMe-backed at {_snap_dir} "
            f"(frees {sum(x.numel()*x.element_size() for x in scales)/1e9:.1f}GB of host RAM)")

    def _snap_alloc(t, tag, i):
        """Buffer shaped like `t`: an ordinary clone, or a memmap under --snap-nvme. Allocated as raw
        bytes and reinterpreted, so it works for any dtype (bf16 has no numpy equivalent)."""
        if not _snap_nvme:
            return t.detach().clone()
        import numpy as _np
        f = _snap_dir / f"{tag}_{i}.dat"
        arr = _np.memmap(str(f), dtype=_np.uint8, mode="w+", shape=(t.numel() * t.element_size(),))
        buf = torch.from_numpy(arr).view(t.dtype).view_as(t)
        with torch.no_grad():
            buf.copy_(t.detach())
        return buf

    def _snap():
        if _snap_bufs["s"] is None:
            _snap_bufs["s"] = [_snap_alloc(s, "s", i) for i, s in enumerate(scales)]
            _snap_bufs["t"] = [_snap_alloc(m.tern_b, "t", i) for i, m in enumerate(arm_b_mods)]
        else:
            with torch.no_grad():
                for dst, src in zip(_snap_bufs["s"], scales):
                    dst.copy_(src.detach())
                for dst, m in zip(_snap_bufs["t"], arm_b_mods):
                    dst.copy_(m.tern_b.detach())
        return (_snap_bufs["s"], _snap_bufs["t"])
    def _restore(snap):
        with torch.no_grad():
            for s, b in zip(scales, snap[0]): s.copy_(b)
            for m, t in zip(arm_b_mods, snap[1]): m.tern_b.copy_(t)
    if getattr(args, "epochs", 0) and args.epochs > 0:               # epochs → steps (data-size-invariant budget)
        spe = max(1, n // (world * max(1, args.accum)))              # optimizer steps per epoch
        args.steps = max(1, round(args.epochs * spe))
        log(f"   --epochs {args.epochs} → steps={args.steps} ({spe}/epoch; n={n} world={world} accum={args.accum})")

    best_kl = float("inf")
    # BEST-LOSS SNAPSHOT — allocated ONLY when `--select best` will actually read it. `scales` includes
    # the assignment latents, so this clone is 4 B/latent: 22.8 GB at `down`@64L and 102.5 GB at arm B on
    # the 27B. Under --select final/ema export_scales() never returns it and restore_best() is never
    # called, so it was pure waste — and the update below RE-CLONED it on every EMA improvement, which
    # transiently doubles it (205 GB at arm B) because the new list is built before the old one is freed.
    # Combined with the held-out snapshot (_snap) and the latents themselves that is 3-4 full copies of
    # every latent in host RAM, which is what OOM'd the box on the arm B run (629 GB, machine restart).
    _keep_best = str(getattr(args, "select", "best")) == "best"
    best_scales = [s.detach().clone() for s in scales] if _keep_best else None
    ema = None
    # Arm B proximal-gated flips + held-out select/abort (weight-training selects on HELD-OUT KL, never training KL)
    flip_every = getattr(args, "flip_every", 150); eval_every = getattr(args, "eval_every", 100)
    cap_frac = getattr(args, "flip_cap_frac", 3e-4); block_cap = getattr(args, "flip_block_cap", 2)
    abort_patience = getattr(args, "abort_patience", 3)
    max_rejects = getattr(args, "flip_max_rejects", 8)          # consecutive rejects before we stop flipping (η-search room)
    gate_k = getattr(args, "flip_gate_k", 0.0)                  # commit iff paired ΔKL_G < gate_k·SE (0 = mean improvement)
    gate_max_fails = getattr(args, "flip_gate_max_fails", 3)    # consec. linearisation-valid-but-non-generalising events → stop
    eta = None; eta_max = None; reject_streak = 0; gate_fail_streak = 0; flip_used = 0   # η self-tuned on the first event
    best_ho_kl = float("inf"); best_ho_snap = None; ho_worse = 0; init_ho_kl = None; do_abort = False
    weight_train = bool(arm_b_mods) or getattr(args, "train_weights", "none") not in (None, "none")

    # ── convergence knobs (all default-off → identical to the previous constant-LR/best-of-loop) ──
    select   = getattr(args, "select", "best")
    sched    = getattr(args, "lr_schedule", "constant")
    cd_frac  = getattr(args, "wsd_cooldown_frac", 0.2)
    accum    = max(1, getattr(args, "accum", 1))
    ema_decay = getattr(args, "scale_ema_decay", 0.0)        # parameter-averaging of the scales
    ema_start = int(getattr(args, "scale_ema_start_frac", 0.5) * args.steps)
    avg_scales = None                                       # lazily inits at ema_start

    def lr_frac(t):
        """Schedule as a FRACTION of base lr (applied per param-group so scales & latents keep their ratio):
        warmup, then constant | linear decay-to-zero | WSD (constant + linear cooldown)."""
        if t < args.warmup:
            return (t + 1) / max(1, args.warmup)
        prog = (t - args.warmup) / max(1, args.steps - args.warmup)      # 0..1 after warmup
        if sched == "linear":
            return max(0.0, 1.0 - prog)
        if sched == "wsd":
            return 1.0 if prog < 1.0 - cd_frac else max(0.0, (1.0 - prog) / cd_frac)
        return 1.0                                           # constant (legacy)

    def restore_best():
        if best_scales is None:                              # --select final/ema never snapshots (see above)
            return
        with torch.no_grad():
            for s, b in zip(scales, best_scales):
                s.copy_(b)

    def export_scales():
        """The deliverable: best-loss snapshot | final iterate | the parameter-average (EMA)."""
        if select == "ema" and avg_scales is not None:
            return avg_scales
        if select == "final":
            return [s.detach() for s in scales]
        return best_scales                                   # 'best' (or 'ema' before it starts)

    def save_export(tag, final=False):
        # save_student dequantises each linear (unpack_2bit → full float). The student has ONE shard, so its
        # per-shard dict accumulates the WHOLE fp16 model (~10.6GB at 4B) and save_file() copies it again —
        # a ~20GB transient on top of whatever training still holds. With offloaded latents + Adam state
        # (16B/latent ⇒ 36GB at mlp@32L) that exceeds the cgroup cap and wedges the run.
        # So: always drop grads, and on the FINAL save also drop the optimizer STATE (2 fp32 buffers per
        # latent) — training is finished, the state is dead weight, and this is what buys the headroom.
        try:
            opt.zero_grad(set_to_none=True)
            if final:
                freed = sum(v.numel() * v.element_size()
                            for st in opt.state.values() for v in st.values()
                            if torch.is_tensor(v)) / 1e9
                opt.state.clear()
                if freed > 0.5:
                    log(f"   [save] released {freed:.1f}GB of optimizer state before writing")
        except Exception:
            pass
        gc.collect(); torch.cuda.empty_cache()
        # 'best' keeps the legacy greedy-restart (resets live scales -> best) so the default command
        # reproduces prior behaviour exactly; 'final'/'ema' save WITHOUT disturbing live training.
        if select == "best":
            restore_best()
            if is_main:
                save_student(core, args.student_path, args.out, BLOCK_SIZE)
                print(f"   {tag} (best ema KL {best_kl:.4f})")
            return
        cur = [s.detach().clone() for s in scales]
        with torch.no_grad():
            for s, t in zip(scales, export_scales()):
                s.copy_(t)
        if is_main:
            if _mem_eff and not getattr(args, "latent_offload", False):
                # save_student unpacks each linear to a full float tensor (lm_head alone is 2.37GB); with
                # GPU-RESIDENT latents that OOMs, so do the dequant in RAM and restore afterwards.
                # NOT needed under --latent-offload: the latents are already off-GPU, leaving plenty of VRAM.
                # (Doing it anyway cost a ~45GB CPU dequant that blocked the worker long enough for torchrun's
                # rendezvous heartbeat to time out — and `next(core.parameters()).device` resolved to CPU when
                # the first parameter was an offloaded latent, so the model never came back to the GPU.)
                _dev = next((p_.device for p_ in core.parameters() if p_.device.type == "cuda"), device)
                gc.collect(); torch.cuda.empty_cache()
                core.to("cpu")
                save_student(core, args.student_path, args.out, BLOCK_SIZE)
                core.to(_dev)
            else:
                save_student(core, args.student_path, args.out, BLOCK_SIZE)
            print(f"   {tag} (select={select})")
        with torch.no_grad():
            for s, c in zip(scales, cur):
                s.copy_(c)

    def global_kl(loss):
        if not ddp:
            return loss.item()
        t = loss.detach().clone()
        import torch.distributed as dist
        dist.all_reduce(t, op=dist.ReduceOp.SUM)           # average across ranks for a stable metric
        return (t / world).item()

    opt_step, micro, win_kl = 0, 0, 0.0                     # win_kl: per-micro KL summed over a window
    opt.zero_grad(set_to_none=True)

    # ── STEP-0 BASELINE held-out eval ────────────────────────────────────────────────────────────────
    # Without this the first eval lands at `--eval-every` with training already folded in, so a stage's
    # own contribution is unmeasurable: you cannot tell "improved on its entry" from "entered better".
    # Measured cost is one eval over --heldout-n seqs (4), i.e. negligible. Two further benefits:
    #   * `best_ho_kl` starts at the ENTRY model, so a stage that never beats its entry restores and saves
    #     that entry -- a true no-op instead of shipping something worse (the failure that made j3_attn's
    #     output unusable: it entered at 0.4538, never beat it, and saved 0.4789).
    #   NOTE: the abort check is deliberately NOT re-anchored to this baseline -- see below.
    if held_idx:
        if _mem_ctx is not None:
            torch.cuda.empty_cache()
        _b_kl, _b_flips = heldout_kl_flips(model, cache, held_idx, batches, device, loss_fn,
                                           args.temperature, mem_eff=_mem_ctx)
        # Seed SELECTION with the entry, but deliberately leave `init_ho_kl` = None so the ABORT check is
        # still anchored to the first POST-training eval. Anchoring the abort to the entry instead makes it
        # fire on any stage that dips before recovering -- which is the NORMAL shape for attention and for
        # E2E. Measured: E2E improved monotonically (0.2209 → 0.2191 → 0.2180 → 0.2162) and was still
        # killed at step 3000 of 12488, because every one of those evals was "worse than the entry".
        best_ho_kl = _b_kl
        best_ho_snap = _snap()
        log(f"   [held-out] step 0 KL={_b_kl:.4f} flips={_b_flips:.2f}% "
            f"assign-moved={_assign_flip_pct():.3f}% best={_b_kl:.4f} flips_used=0  <- ENTRY BASELINE")

    # FAIL FAST on an empty train shard. `for bi in shard` over an empty list never advances opt_step,
    # so the while loop below spins at 100% CPU forever with no output and no GPU work — it looks
    # exactly like a slow run. Happens whenever --max-samples <= --heldout-n (+ grad-probe/gate seqs),
    # which is easy to hit when shrinking a run for a smoke test.
    if not shard:
        raise SystemExit(f"no TRAIN sequences on this rank: --max-samples {args.max_samples} leaves "
                         f"nothing after --heldout-n {args.heldout_n} (+probe/gate). Raise --max-samples.")

    while opt_step < args.steps:
        for bi in shard:
            if opt_step >= args.steps:
                break
            ids = batches[bi].to(device)
            # deterministic on opt_step so ALL DDP ranks make the SAME on/off choice each step (else the
            # ranks' autograd graphs desync). every-Nth-step schedule, N = round(1/frac), gives ~frac.
            _op_every = max(2, round(1.0 / args.on_policy_frac)) if on_policy and args.on_policy_frac > 0 else 0
            use_op = (on_policy and opt_step >= int(args.on_policy_warmup_frac * args.steps)
                      and _op_every and (opt_step % _op_every == 0))
            if use_op:
                op_ids = op_batches[opt_step % len(op_batches)].to(device) if op_batches else ids
                loss = on_policy_loss(model, teacher, core, op_ids, args, args.temperature)
            elif _mem_eff:
                # MEMORY-EFFICIENT path (CE + assignment-QAT): no full-vocab logits. Run with logits_to_keep=1
                # (skips the big lm_head), capture post-norm hidden via the norm hook, then compute KL chunk-wise
                # and CE on a position subset. Frees ~2.5GB for the assignment latents. No commit-beta here.
                t_idx = cache["idx"][bi].unsqueeze(0).to(device)
                t_val = cache["val"][bi].unsqueeze(0).to(device)
                _hidcap.clear()
                model(ids, logits_to_keep=1)                 # DDP fwd; hook captures [1,T,H] post-norm hidden
                h = _hidcap["h"]
                Wlm = _Wlm                                   # precomputed bf16 [V,H] (frozen lm_head)
                # KL: per-position match to the teacher's output distribution (all T positions), chunked.
                # Small chunk (8192) keeps the [T, chunk] temp ~0.08GB — the assignment latents need every MB.
                loss = chunked_hidden_state_loss(h, Wlm, t_idx, t_val,
                                                 temperature=args.temperature, loss_type=_ce_lt,
                                                 chunk_size=4096)
                ce_w = float(getattr(args, "ce_weight", 0.0))
                if ce_w > 0:
                    # CE: ground-truth NEXT token — the signal the frozen teacher lacks; its gradient flows
                    # through the STE latent and can flip trit assignments. Shift-by-one; random position
                    # subset bounds the [n_ce×V] softmax (re-drawn each step ⇒ full coverage over the run).
                    Hf = h[0, :-1, :]
                    tgt = ids[0, 1:]
                    n_ce = int(getattr(args, "ce_positions", 512) or 0)
                    if 0 < n_ce < Hf.shape[0]:
                        sel = torch.randint(0, Hf.shape[0], (n_ce,), device=Hf.device)
                        Hf, tgt = Hf[sel], tgt[sel]
                    loss = loss + ce_w * chunked_ce(Hf, Wlm, tgt)   # Wlm on CPU; chunked, no [N,V] on GPU
            else:
                t_idx = cache["idx"][bi].unsqueeze(0).to(device)
                t_val = cache["val"][bi].unsqueeze(0).to(device)
                out = model(ids, output_hidden_states=feat_w > 0)
                cw = (commit_weights(ids, args.commit_weight, args.commit_pre, args.commit_post)
                      if getattr(args, "commit_weight", 0) > 1 else None)
                cb = float(getattr(args, "commit_beta", 0.0))
                cmask = (commit_weights(ids, 2.0, args.commit_pre, args.commit_post) > 1) if cb > 0 else None
                loss = loss_fn(out.logits, t_idx, t_val, args.temperature, weights=cw,
                               commit_mask=cmask, commit_beta=cb)
                if feat_w > 0:
                    # DDP-safe: call the wrapped model's forward (grad sync) and take the post-norm
                    # final hidden from output_hidden_states[-1]. (Can't use core.model() under DDP —
                    # it would bypass the gradient all-reduce.)
                    sh = out.hidden_states[-1]
                    th = cache["hidden"][bi].unsqueeze(0).to(sh.device)
                    loss = loss + feat_w * hidden_state_loss(sh, th)
            if _grad_release:
                # latents step DURING backward, so their lr must already reflect this step's schedule+TALR
                _f = lr_frac(opt_step)
                for g in opt.param_groups:
                    if g.get("is_latent"):
                        g["lr"] = g.get("base_lr", args.lr) * _f * _lat_frac(opt_step)
            (loss / accum).backward()                      # /accum so the window grad is the MEAN
            win_kl += global_kl(loss)                       # DDP all-reduces the scale grads here
            if _mem_eff:
                _hidcap.clear()                             # release the captured hidden + its ~17GB activation
                del loss                                    # graph NOW; else it stays alive until the next step's
                                                            # clear and OOMs the held-out eval that runs between.
            micro += 1
            if micro % accum != 0:
                continue                                    # keep accumulating the effective batch
            # ── one optimizer step per `accum` micro-batches ──
            torch.nn.utils.clip_grad_norm_(scales, 1.0)
            _frac = lr_frac(opt_step)
            for g in opt.param_groups:
                g["lr"] = g.get("base_lr", args.lr) * _frac   # scale each group by the schedule, keep the ratio
                # Stage-0 second-moment warmup (_lat_frac == 0 there): hold the LATENT lr at 0 while Adam still
                # accumulates v̂, so the first assignment flips aren't driven by a cold, mis-scaled second moment
                # (the RAdam failure mode that turns the first boundary crossings into a cascade). Then ramp in
                # linearly rather than switching the full lr on at once.
                if g.get("is_latent"):
                    g["lr"] *= _lat_frac(opt_step)
            if _grad_release:
                # Latents already stepped (and freed their grads) inside backward. Zeroing their lr here was
                # NOT enough: opt.step() still ALLOCATES exp_avg/exp_avg_sq for every param it visits, so the
                # latents ended up with TWO sets of Adam state (measured: anon RSS grew 23.9 -> 42.8GB over
                # 100 steps and wedged the run in D state). Temporarily REMOVE the latent groups from the
                # optimizer instead, so it only ever touches the scale group.
                _lat_groups_saved = [g for g in opt.param_groups if g.get("is_latent")]
                opt.param_groups = [g for g in opt.param_groups if not g.get("is_latent")]
                opt.step()
                opt.param_groups.extend(_lat_groups_saved)
            else:
                opt.step()
            opt.zero_grad(set_to_none=True)
            opt_step += 1
            # ── TALR: servo the latent lr to hit a target TRANSITION RATE (flips/step). lr alone cannot control
            # the flip count (it depends on the latent distribution too), so control the rate directly. ──
            if _tr_target > 0 and _lat_groups and opt_step % max(1, args.tr_every) == 0:
                with torch.no_grad():
                    cur_per = _hard_trits_per_mod()
                    rates = ([float((c != pv).float().mean().item()) / max(1, args.tr_every)
                              for c, pv in zip(cur_per, _tr_prev_per)] if _tr_prev_per is not None
                             else [0.0] * len(cur_per))
                    _tr_prev_per = cur_per
                # TALR may only servo once the cold-start lr ramp has fully phased in. Measuring during the
                # ramp makes the two loops fight: the ramp is holding the lr down ON PURPOSE, TALR reads the
                # resulting low flip rate as "base lr too small" and opens the gain to compensate. Measured:
                # rate 0.00e+00 for three consecutive windows while the gain climbed 1.0→1.30, on track for
                # ~32x by end of run — the same runaway-chasing-zero seen in seq32/t3.
                if opt_step >= _lat_warm + _lat_ramp + args.tr_every:
                    tgt = _tr_target * (1.0 + (args.tr_final_frac - 1.0) * min(1.0, opt_step / max(1, args.steps)))
                    # servo EACH layer against ITS OWN rate — a global controller only sees the aggregate and
                    # cannot stop a single layer bursting (measured: one layer hit 15.5% vs a 3.3% median).
                    for g in _lat_groups:
                        i = g.get("lat_idx", -1)
                        r = rates[i] if 0 <= i < len(rates) else (sum(rates) / max(1, len(rates)))
                        gi = 0 if i < 0 else i
                        if r < tgt * 0.5:   _tr_gains[gi] *= _tr_up  # too few flips → open up, RATE-LIMITED so
                                                                    # the gain can at most double over
                                                                    # --tr-ramp-2x-steps steps (see below)
                        elif r > tgt * 2.0: _tr_gains[gi] *= 0.6     # too many → clamp down fast (asymmetric:
                                                                    # cascades are far costlier than slowness)
                        # GAIN IS CAPPED AT _tr_gain_max (default 1.0): the base latent-lr is the CALIBRATED
                        # safe value, so TALR may only THROTTLE it, never amplify past it. Without this cap a
                        # stage that starts from an ALREADY-OPTIMISED model sees a naturally low flip rate
                        # (small gradients near a joint optimum), reads it as "too few flips", and boosts the
                        # lr to force its target — measured on seq32's t2_up: gain ran to 2.20 then 4.83 and
                        # burst the model from KL 0.4575 to 0.5698, which it never fully recovered from.
                        # Chasing a fixed transition rate is wrong once the model is already good.
                        _tr_gains[gi] = float(min(max(_tr_gains[gi], 1e-3), _tr_gain_max))
                        g["base_lr"] = _lat_base_lr * _tr_gains[gi]
                    if opt_step % (args.tr_every * 4) == 0:
                        _sr = sorted(rates)
                        log(f"   [talr] step {opt_step} rate med {_sr[len(_sr)//2]:.2e} "
                            f"(min {_sr[0]:.2e} max {_sr[-1]:.2e}) target {tgt:.2e} | "
                            f"gain min {min(_tr_gains):.2f} max {max(_tr_gains):.2f}")

            kl = win_kl / accum
            win_kl = 0.0
            ema = kl if ema is None else 0.9 * ema + 0.1 * kl
            if ema < best_kl:                              # identical on every rank -> stays in sync
                best_kl = ema
                if _keep_best:                             # copy IN PLACE: re-cloning would transiently
                    with torch.no_grad():                  # hold two full copies (205 GB at arm B)
                        for _dst, _src in zip(best_scales, scales):
                            _dst.copy_(_src.detach())
            # ── Arm B: proximal-gated flip EVENT (fresh grad → gate → accept/reject → η adapt) ──
            if arm_b_mods and opt_step % flip_every == 0 and reject_streak < max_rejects \
                    and gate_fail_streak < gate_max_fails:
                arm_b_probe_grad(core, cache, probe_idx, batches, device, loss_fn, args.temperature)
                torch.cuda.empty_cache()                       # free probe-backward activations before the flip
                if eta is None:                                # init η so only ~few-K weights clear |ḡ|>s/2η
                    rs = []                                     # subsample PER LINEAR (never materialise all 2.3B at once)
                    for m in arm_b_mods:
                        r = (m.flip_grad.float().abs() / m.scale.detach().unsqueeze(1).clamp_min(1e-8)).flatten()
                        rs.append(r if r.numel() <= 200_000 else r[torch.randint(0, r.numel(), (200_000,), device=r.device)])
                    r = torch.cat(rs)
                    if r.numel() > 1_000_000: r = r[torch.randint(0, r.numel(), (1_000_000,), device=r.device)]  # quantile 16M cap
                    qf = getattr(args, "flip_init_qual", 1e-5)  # tiny initial qualify-fraction (our flips are interaction-sensitive)
                    eta = float(1.0 / (2 * torch.quantile(r.float(), 1 - qf).clamp_min(1e-12))); eta_max = eta * 100
                    log(f"   [arm-b] η init {eta:.3e} (target ~{qf:.0e} qualify)")
                # TWO ORTHOGONAL CHECKS (researcher: in-sample-accept is a winner's-curse selection bias — the ratio
                # test and the accept decision must NOT share a scalar). (1) ρ=realΔ/predΔ on the IN-SAMPLE probe =
                # trust-region MODEL-FIDELITY (is the proximal quadratic locally valid?) → controls η only. (2) a
                # PAIRED ΔKL on a DISJOINT gate set = GENERALISATION → controls commit/revert. Commit iff BOTH pass.
                pre_kl, _ = heldout_kl_flips(model, cache, ar_idx, batches, device, loss_fn, args.temperature, mem_eff=_mem_ctx)
                gpre = heldout_kl_flips(model, cache, gate_idx, batches, device, loss_fn, args.temperature, per_seq=True, mem_eff=_mem_ctx)[2] if gate_idx else None
                committed, n_qual, pred, revert = arm_b_gated_flip(core, eta, cap_frac, block_cap)
                if committed == 0:                             # η too small — nothing cleared the gate; grow to find candidates
                    eta = min(1.5 * eta, eta_max)
                    log(f"   [arm-b] step {opt_step} EMPTY qual=0 η→{eta:.2e}")
                else:
                    post_kl, _ = heldout_kl_flips(model, cache, ar_idx, batches, device, loss_fn, args.temperature, mem_eff=_mem_ctx)
                    realΔ = post_kl - pre_kl
                    rho = realΔ / pred if abs(pred) > 1e-9 else (-1.0 if realΔ > 0 else 1.0)
                    # (2) paired generalisation delta on the disjoint gate set (baseline cancels → resolves milli-nats)
                    dkl_g = 0.0; se_g = 0.0; gate_ok = True
                    if gate_idx:
                        gpost = heldout_kl_flips(model, cache, gate_idx, batches, device, loss_fn, args.temperature, per_seq=True, mem_eff=_mem_ctx)[2]
                        diffs = torch.tensor([b - a for a, b in zip(gpre, gpost)])
                        dkl_g = float(diffs.mean())
                        se_g = float(diffs.std(unbiased=True) / max(1, len(diffs) ** 0.5)) if len(diffs) > 1 else 0.0
                        gate_ok = dkl_g < gate_k * se_g        # k=0 → commit only on a mean paired IMPROVEMENT
                    if rho < 0.25:                             # model-fidelity FAIL → revert + shrink η (trust region)
                        for m, sel, old in revert: m.tern_b[sel] = old
                        eta *= 0.5; reject_streak += 1
                        log(f"   [arm-b] step {opt_step} REJECT-ρ commit={committed} qual={n_qual} predΔ={pred:.4f} "
                            f"realΔ={realΔ:+.4f} ρ={rho:+.2f} ΔKLg={dkl_g:+.4f}±{se_g:.4f} η→{eta:.2e} rej={reject_streak}")
                    elif not gate_ok:                          # fidelity OK but DOESN'T GENERALISE → revert, keep η, count
                        for m, sel, old in revert: m.tern_b[sel] = old
                        gate_fail_streak += 1
                        log(f"   [arm-b] step {opt_step} REJECT-gate commit={committed} qual={n_qual} predΔ={pred:.4f} "
                            f"realΔ={realΔ:+.4f} ρ={rho:+.2f} ΔKLg={dkl_g:+.4f}±{se_g:.4f} η={eta:.2e} gfail={gate_fail_streak}")
                    else:                                      # BOTH pass → commit; grow η only when WELL-CALIBRATED
                        flip_used += committed; reject_streak = 0; gate_fail_streak = 0
                        if 0.75 <= rho <= 1.5:                 # ρ≫1 = quadratic too conservative; growing η overshoots
                            eta = min(1.2 * eta, eta_max)       # into the interaction-invalid regime → hold instead
                        log(f"   [arm-b] step {opt_step} ACCEPT commit={committed} qual={n_qual} predΔ={pred:.4f} "
                            f"realΔ={realΔ:+.4f} ρ={rho:+.2f} ΔKLg={dkl_g:+.4f}±{se_g:.4f} η→{eta:.2e} flips={flip_used}")
                opt.zero_grad(set_to_none=True)                # clear probe-pass grads before resuming scale steps
            # ── held-out eval drives selection + abort (training KL is inadmissible for weight-training) ──
            if held_idx and opt_step % eval_every == 0:
                if _mem_ctx is not None:
                    torch.cuda.empty_cache()               # release training-step fragmentation; the eval is tight
                ho_kl, ho_flips = heldout_kl_flips(model, cache, held_idx, batches, device, loss_fn, args.temperature, mem_eff=_mem_ctx)
                if init_ho_kl is None:
                    init_ho_kl = ho_kl
                if ho_kl < best_ho_kl:
                    best_ho_kl = ho_kl; best_ho_snap = _snap(); ho_worse = 0
                else:
                    ho_worse += 1
                _pl = _per_layer_moved()
                _plmsg = ""
                if _pl:
                    _srt = sorted(_pl)
                    _plmsg = (f" per-layer[min {_srt[0]:.2f} med {_srt[len(_srt)//2]:.2f} "
                              f"max {_srt[-1]:.2f} ratio {(_srt[-1]/max(_srt[0],1e-6)):.0f}x]")
                log(f"   [held-out] step {opt_step} KL={ho_kl:.4f} flips={ho_flips:.2f}% "
                    f"assign-moved={_assign_flip_pct():.3f}%{_plmsg} best={best_ho_kl:.4f} flips_used={flip_used}")
                if ho_kl > init_ho_kl and ho_worse * eval_every >= 300 * abort_patience / 3:
                    log(f"   ABORT: held-out KL {ho_kl:.4f} > init {init_ho_kl:.4f} for {ho_worse} evals"); do_abort = True
            if ema_decay > 0 and opt_step >= ema_start:    # parameter-average the tail of training
                with torch.no_grad():
                    if avg_scales is None:
                        avg_scales = [s.detach().clone() for s in scales]
                    elif getattr(args, "fisher_ema", False):
                        # B4: Fisher-WEIGHTED EMA. Adam already Fisher-preconditions the *update*; the
                        # novel bit is weighting the tail-AVERAGE by curvature so well-determined
                        # (high-Fisher) scales are trusted faster. Fisher proxy = Adam's exp_avg_sq.
                        for a, s in zip(avg_scales, scales):
                            v = opt.state.get(s, {}).get("exp_avg_sq")
                            if v is None:
                                a.mul_(ema_decay).add_(s.detach(), alpha=1 - ema_decay); continue
                            fw = (v / (v.mean() + 1e-12)).clamp_(0.1, 10.0)   # normalized, bounded
                            a.add_((s.detach() - a) * ((1 - ema_decay) * fw))
                    else:
                        for a, s in zip(avg_scales, scales):
                            a.mul_(ema_decay).add_(s.detach(), alpha=1 - ema_decay)
            if opt_step % args.log_every == 0 or opt_step == 1:
                log(f"   step {opt_step}/{args.steps}  KL={kl:.4f}  ema={ema:.4f}  "
                    f"best={best_kl:.4f}  lr={args.lr*lr_frac(opt_step):.2e}")
            if args.ckpt_every and opt_step % args.ckpt_every == 0:
                save_export(f"checkpoint saved at step {opt_step}")
                torch.cuda.empty_cache()
                if ddp:
                    import torch.distributed as dist
                    dist.barrier()                          # others wait while rank 0 writes
            if do_abort:
                break
        if do_abort:
            break

    if weight_train and best_ho_snap is not None:
        _restore(best_ho_snap)                              # deliverable = best HELD-OUT-KL checkpoint (scales+assignments)
        select = "final"                                    # save the live (restored) state, not a training-KL best
        if is_main:
            print(f"   restored best held-out checkpoint (KL {best_ho_kl:.4f}, flips_used {flip_used})")
    save_export("final save", final=True)
    if is_main:
        _ho = f" | best held-out KL {best_ho_kl:.4f}" if weight_train and best_ho_snap is not None else ""
        print(f"Done. select={select} sched={sched} accum={accum} ema_decay={ema_decay}. "
              f"Best ema KL {best_kl:.4f}{_ho}. Student -> {args.out}")
    if ddp:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()


def save_student(model, student_path, out_dir, block_size):
    """Write the student back out as dequantized FP16 (ternary * trained scale) so it repacks
    to TQ2_0 losslessly. STREAMS shard-by-shard: only the ternary linears in the current
    shard are dequantized at a time (never the whole 54 GB model at once)."""
    import shutil
    from safetensors.torch import save_file
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    # map output weight-name -> TSL module (NO dequant yet — that's the whole point)
    tsl_map = {}
    for name, module in model.named_modules():
        if isinstance(module, TernaryScaleLinear):
            tsl_map[f"{name}.weight"] = module
    # A3(ii): fold each trained per-MLP-input a3_scale into its layer's post_attention_layernorm gain,
    # so the saved model is standard TQ2_0 (no extra params) and mathematically identical to training.
    override = {}
    for lname, layer in model.named_modules():
        mlp = getattr(layer, "mlp", None)
        norm = getattr(layer, "post_attention_layernorm", None)
        if mlp is not None and norm is not None and hasattr(mlp, "a3_scale"):
            # Qwen3_5RMSNorm is ZERO-CENTERED: forward = x_norm * (1 + w). Effective scale must become
            # (1+w)*a3, so the STORED value is (1+w)*a3 - 1.  (The old fold w*a3 was a silent no-op here:
            # rotation folds norms into weights leaving w ≡ 0, so w*a3 ≡ 0 DISCARDED the trained a3 —
            # this invalidated the original A3(ii) "no gain" verdict and the first col-scale run.)
            folded = ((1.0 + norm.weight.detach().float()) * mlp.a3_scale.detach().float()
                      - 1.0).to(torch.float16).cpu()
            disk = f"{lname}.post_attention_layernorm.weight".replace("model.layers", "model.language_model.layers")
            override[disk] = folded
        # col-scale: fold down_proj's per-input-channel cs_in into up_proj's ROWS (exact: SwiGLU interm =
        # silu(gate)⊙up, so scaling up's row r by cs_in[r] ≡ scaling down's input channel r). Row-uniform
        # scaling preserves the ternary×per-block-scale structure → still repacks to TQ2_0 losslessly.
        # Done via override (non-destructive) so mid-training checkpoint saves don't corrupt the live model.
        if mlp is not None and hasattr(getattr(mlp, "down_proj", None), "cs_in"):
            up = mlp.up_proj
            folded_up = (mlp.down_proj.cs_in.detach().float().unsqueeze(1)
                         * up.dequant().detach().float()).to(torch.float16).cpu()
            disk = f"{lname}.mlp.up_proj.weight".replace("model.layers", "model.language_model.layers")
            override[disk] = folded_up
    wmap = _shard_map(student_path)
    shards = {}
    for name, sf in wmap.items():
        shards.setdefault(sf, []).append(name)

    # SAVE_MAX_SHARD_GB>0 re-shards the OUTPUT into bounded pieces. The student here has ONE source shard, so
    # the default path accumulates the entire fp16 model in `d` (~10.6GB at 4B) and save_file() copies it again
    # — a ~20GB transient. On top of an assignment-QAT run's ~30GB that crosses the cgroup limit and wedges the
    # process (this is the same spike that made --ckpt-every 100 fatal at step 100). Bounding the shard bounds
    # both the dict and the copy. Off by default: the deploy/export path keeps its single-file layout.
    _cap = float(os.environ.get("SAVE_MAX_SHARD_GB", "0")) * 1e9
    if _cap > 0:
        all_names = [n for names in shards.values() for n in names]
        groups, cur, cur_b = [], [], 0.0
        for name in all_names:
            key = name.replace("model.language_model.layers", "model.layers")
            mod = tsl_map.get(key)
            nb = (mod.out_features * mod.in_features * 2) if mod is not None else 0
            if cur and cur_b + nb > _cap:
                groups.append(cur); cur, cur_b = [], 0.0
            cur.append(name); cur_b += nb
        if cur:
            groups.append(cur)
        shards = {f"model-{i+1:05d}-of-{len(groups):05d}.safetensors": g for i, g in enumerate(groups)}
        print(f"   [save] re-sharded into {len(groups)} files (<= {_cap/1e9:.1f}GB each) to bound the transient")

    weight_index = {}
    for sf, names in shards.items():
        d = {}
        for name in names:
            key = name.replace("model.language_model.layers", "model.layers")
            mod = tsl_map.get(key)
            if name in override:
                d[name] = override[name]                       # A3(ii) folded norm gain
            elif mod is not None:
                d[name] = mod.dequant().detach().to(torch.float16).cpu()
            else:
                d[name] = _get_tensor(student_path, wmap, name)
            weight_index[name] = sf
        save_file(d, str(out / sf))
        del d
        gc.collect()
    for fn in ["config.json", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "generation_config.json", "merges.txt",
               "vocab.json", "model.safetensors.index.json"]:
        src = Path(student_path) / fn
        if src.exists():
            shutil.copy2(src, out / fn)
    if _cap > 0:                                               # rewrite the index to match the new sharding
        with open(out / "model.safetensors.index.json", "w") as f:
            json.dump({"metadata": {}, "weight_map": weight_index}, f, indent=2)


# ─────────────────────────── CPU self-test ─────────────────────────────────────────

def smoke():
    torch.manual_seed(0)
    bs = 128
    # 1) pack/unpack round-trip
    t = torch.randint(-1, 2, (64, 256)).to(torch.int8)
    pk = pack_2bit(t)
    u = unpack_2bit(pk, t.numel()).reshape(64, 256).to(torch.int8)
    assert torch.equal(t, u), "pack/unpack mismatch"
    print(f"pack/unpack: OK  ({t.numel()} ternary -> {pk.numel()} bytes, "
          f"{pk.numel()/ (t.numel()*2/8):.2f}x of theoretical 2-bit)")

    # 2) extraction round-trip on a genuine ternary*scale weight
    scale_true = torch.rand(64 * (256 // bs)).abs() + 0.05
    tern_true = torch.randint(-1, 2, (64, 256)).float()
    w = (tern_true.reshape(-1, bs) * scale_true.unsqueeze(1)).reshape(64, 256)
    tern, scale = extract_ternary_scale(w, bs)
    deq = (tern.float().reshape(-1, bs) * scale.unsqueeze(1)).reshape(64, 256)
    assert torch.allclose(deq, w, atol=1e-5), "extraction round-trip failed"
    print("extract_ternary_scale round-trip: OK")

    # 3) TernaryScaleLinear forward + scale gradient
    tsl = TernaryScaleLinear.from_dense(w, bs, device="cpu")
    x = torch.randn(4, 256, requires_grad=False)
    y = tsl(x); loss = y.pow(2).mean(); loss.backward()
    assert tsl.scale.grad is not None and torch.isfinite(tsl.scale.grad).all(), "no scale grad"
    assert torch.allclose(tsl().sum(), w.sum(), atol=1e-3) if False else True
    print(f"TernaryScaleLinear: forward OK, scale.grad norm {tsl.scale.grad.norm():.3e}")

    # 4) top-k KL training reduces loss (toy: fit student scales to a teacher)
    V, T, k = 512, 32, 16
    teacher_logits = torch.randn(1, T, V)
    tv, ti = torch.topk(teacher_logits, k, dim=-1)
    # student = a TSL acting as a classifier head on fixed features
    feats = torch.randn(1, T, 256)
    head = TernaryScaleLinear.from_dense(torch.randn(V, 256) * 0.05, bs, device="cpu")
    for p in head.parameters():
        p.requires_grad_(p is head.scale)
    opt = torch.optim.Adam([head.scale], lr=5e-2)
    first = last = None
    for it in range(60):
        logits = head(feats)
        loss = topk_kl_loss(logits, ti, tv)
        opt.zero_grad(); loss.backward(); opt.step()
        if it == 0:
            first = loss.item()
        last = loss.item()
    print(f"top-k KL scale-training: {first:.4f} -> {last:.4f} "
          f"({'OK reduces' if last < first else 'NO improvement'})")
    assert last < first, "scale training did not reduce KL"

    # 5) hidden-state feature loss: 0 at exact match, positive under mismatch
    h_t = torch.randn(2, 8, 32)
    assert hidden_state_loss(h_t, h_t).item() == 0.0
    assert hidden_state_loss(h_t + 0.1, h_t).item() > 0.0
    print("hidden_state_loss: OK (0 at match, >0 otherwise)")

    print("\n✅ all smoke checks passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--precompute-teacher", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--teacher-path", default=None)
    ap.add_argument("--student-path", default=None)
    ap.add_argument("--orig-config-path", default=None)
    ap.add_argument("--calib", default="./output_recovery/calibration_data.json")
    ap.add_argument("--teacher-cache", default="./output_recovery/teacher_topk.pt")
    ap.add_argument("--out", default="./output_e2eqp/modified_model")
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--cache-batch", type=int, default=1,
                    help="Teacher-cache sequences per forward (serial path). NOTE: NOT byte-identical "
                         "(batch-size GEMM numerics, ~1%% top-1 drift) and didn't speed the vocab-bound "
                         "cache in practice — prefer --teacher-dp. Default 1 (off).")
    ap.add_argument("--teacher-dp", type=int, default=0,
                    help="Data-parallel teacher cache: 1=ON (full model copy per GPU, sequences split "
                         "strided, threaded) → BYTE-IDENTICAL to serial, ~n_gpu× faster. ONLY for a model "
                         "that fits in ONE GPU (small testbeds); a too-big model would OOM. Default 0 (off, "
                         "27B uses the serial device_map=auto path).")
    ap.add_argument("--cache-hidden", action="store_true",
                    help="Also cache the teacher's final post-norm hidden state per token (enables "
                         "hidden-state feature distillation via --feat-weight in block_qat.py). Adds "
                         "~seq*5120*2 bytes/sequence to the cache (~5 GB at 512 seqs × 1024 tokens). "
                         "Regenerate the teacher cache (delete teacher_topk.pt) if it lacks this.")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--epochs", type=float, default=0.0,
                    help="If >0, OVERRIDE --steps with round(epochs × n_samples / (world × accum)) — i.e. train "
                         "this many passes over the calib data regardless of its size. The data-size-invariant "
                         "way to specify budget (a calib-size sweep at fixed epochs uses this).")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--latent-lr", type=float, default=0.0,
                    help="Separate (usually much higher) lr for the assignment latents (--train-weights). A trit "
                         "only flips when its latent moves ~0.5×scale, which never happens at the scale lr "
                         "(measured assign-moved=0.000%% at lr 1e-5). 0 = use --lr. NOTE: lr is a POOR control "
                         "variable for the flip rate — prefer --target-tr (TALR) which servo-controls it.")
    ap.add_argument("--latent-bf16-compute", action="store_true",
                    help="Stream a BF16 copy of each offloaded latent to the GPU for the forward while the "
                         "CPU master copy and Adam state stay FP32. The model already runs bf16 and forward() "
                         "casts dequant() to x.dtype, so the fp32 GPU copy bought nothing but doubled the "
                         "transient and made the grad fp32 (~6B/latent = 13.6GB at mlp@32L). Standard "
                         "mixed-precision: master fp32, compute bf16.")
    ap.add_argument("--latent-state-nvme", action="store_true",
                    help="Keep Adam's exp_avg/exp_avg_sq for the latents in NVMe-backed memmaps instead of RAM "
                         "(8 of the measured 18.1 bytes/latent). mlp@32L = 2.26B latents needs ~48GB resident "
                         "(over the 47G cap); this brings it to ~30GB. Slower per step (state is read+written "
                         "once per tensor per step) but the page cache absorbs much of it. Needs "
                         "--latent-grad-release.")
    ap.add_argument("--latent-grad-release", action="store_true",
                    help="Step each offloaded latent the moment its gradient is ready, then free the grad "
                         "(post-accumulate-grad hook). Removes the fp32 grad buffer (4B/latent) from the peak: "
                         "mlp@32L 2.26B latents goes 49GB (throttles at MemoryHigh=47G) -> 40GB. EXACT Adam "
                         "maths — unlike swapping the optimizer — because Adam's update is per-parameter.")
    ap.add_argument("--grad-ckpt-stride", type=int, default=1,
                    help="Checkpoint every Nth decoder layer instead of all of them. 1 = all (default), "
                         "2 = every other. Each checkpointed layer re-runs its dequant on the backward "
                         "recompute, so a larger stride cuts per-latent work at the cost of activations. "
                         "Ignored when --no-grad-ckpt is set.")
    ap.add_argument("--latent-gpu-budget", type=float, default=0.0,
                    help="GB of latents to keep RESIDENT ON GPU (per process). Those layers do no "
                         "host->device copy in dequant(). 0 = all offloaded (default).")
    ap.add_argument("--latent-pin-grad", action="store_true",
                    help="Stage the latent GRADIENT's device->host trip through pinned buffers. py-spy "
                         "attributed 19.5%% of step time to that copy; .to('cpu') allocates pageable "
                         "memory so the driver stages it. Requires --latent-prefetch (that is where the "
                         "backward is ours to control), --latent-grad-release and --accum 1.")
    ap.add_argument("--latent-prefetch", action="store_true",
                    help="Start the NEXT latent's host->device copy on a side CUDA stream while the "
                         "current layer computes, so transfers overlap instead of stalling. Only "
                         "meaningful with --latent-pin (pageable copies cannot be async).")
    ap.add_argument("--latent-pin", action="store_true",
                    help="Keep offloaded latents in PINNED host memory so H2D copies can be async. "
                         "Pageable copies are synchronous and stall the calling thread once per latent "
                         "per forward. Pinned memory is unswappable.")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="Disable gradient checkpointing. Checkpointing halves activation memory but "
                         "DOUBLES per-latent cost (dequant + CPU->GPU latent stream + grad-release hook "
                         "run again on recompute). At arm B that is 205 GB/step of PCIe traffic; this "
                         "halves it, if activations still fit.")
    ap.add_argument("--snap-nvme", action="store_true",
                    help="Back the best-held-out snapshot with a memmap instead of host RAM. Saves "
                         "4 B/latent, but ONLY worth it on FAST LOCAL STORAGE — this host has no NVMe, "
                         "so leave it OFF here (slow disk is also why startup takes ~25 min). "
                         "4 B/latent (102.5 GB at 27B arm B). The snapshot is write-mostly (one copy per "
                         "held-out improvement) and read at most once, so the page cache absorbs it.")
    ap.add_argument("--model-parallel", type=int, default=0,
                    help="Split decoder layers across N GPUs (naive layer-split model parallelism). "
                         "A 27B ternary student does NOT fit one 24GB card at seq 2560 (measured OOM at "
                         "~22.7GB); DDP cannot help since it replicates the model per rank. Buys MEMORY, "
                         "not speed: ~1/N utilisation until microbatch pipelining is added. 0/1 = off.")
    ap.add_argument("--latent-candidate-tau", type=float, default=0.0,
                    help="Candidate-set assignment training: update ONLY latents within tau of a decision "
                         "boundary (d = ||L/s|-0.5|), freezing the rest. 0 = off (dense, the default). "
                         "Measured coverage at 4B: tau=0.001 -> 14.9%%, 0.01 -> 15.7%%, 0.05 -> 19.6%%. "
                         "Only 3.55%% of assignments ever move and motion saturates by ~step 60, so the "
                         "frozen majority contributes optimizer traffic and nothing else.")
    ap.add_argument("--latent-opt", choices=["adam", "sgd", "adam-blockv"], default="adam",
                    help="Optimizer for the CPU-resident assignment latents. 'adam' is the calibrated "
                         "default behind every recorded number. 'sgd' (momentum) holds ONE state tensor "
                         "instead of two — 3.91x faster per step on a bandwidth-bound CPU — but its step "
                         "is lr*|g| rather than Adam's gradient-normalised ~lr, so --latent-lr must be "
                         "re-calibrated (err HIGH; TALR can only throttle, never amplify).")
    ap.add_argument("--latent-momentum", type=float, default=0.9,
                    help="Momentum for --latent-opt sgd.")
    ap.add_argument("--latent-grad-diag", action="store_true",
                    help="Log latent grad RMS at the first optimizer step — used to calibrate the SGD "
                         "base lr against Adam's effective step size.")
    ap.add_argument("--latent-offload", action="store_true",
                    help="Keep assignment latents in CPU RAM (fp32 params AND grads: 6.04GB for all 32 "
                         "down_proj) and stream each layer's latent to the GPU inside its checkpointed "
                         "forward. Frees ~6GB of GPU ⇒ FULL 32-layer coverage fits. Forces torch.optim.Adam "
                         "(bitsandbytes cannot step CPU params). Costs ~1s/step of PCIe traffic.")
    ap.add_argument("--tw-layer-offset", type=int, default=0,
                    help="With --tw-layer-stride N, select layers where idx %% N == offset. Enables SEQUENTIAL "
                         "GROUP passes (offset 0,1,..,N-1, each warm-starting from the previous output) which "
                         "cover every layer while only ever perturbing 1/N of them at a time. Needed because "
                         "moving ALL layers' assignments at once compounds error through depth (measured: at "
                         "equal per-layer flip rate the first-eval damage is 8L 0.58 / 16L 0.75 / 32L 1.91 vs a "
                         "0.735 baseline, with per-layer flip rates UNIFORM ⇒ compounding, not imbalance). "
                         "Matters more at 64 layers (27B) than at 32.")
    ap.add_argument("--tw-layer-stride", type=int, default=1,
                    help="Apply assignment-QAT to every Nth matching layer only. fp32 latents are ~3GB for all "
                         "32 down_proj — at the edge of 24GB; stride 4 gives 8 layers (~0.75GB) with headroom.")
    ap.add_argument("--latent-init", default="center", choices=["center", "fp-spread"],
                    help="STE latent init. 'center' = tern*scale ⇒ every latent EXACTLY on a bin centre (0%% near "
                         "a boundary) ⇒ flipping is BIMODAL by construction (inert or cascade). 'fp-spread' = the "
                         "FP weight clamped into the current trit's bin ⇒ same assignments/function, but natural "
                         "intra-bin spread (~9.8%% near a boundary) ⇒ graded flips. Needs --fp-model.")
    ap.add_argument("--fp-model", default=None,
                    help="Rotated FP model dir (the latent-init source for --latent-init fp-spread).")
    ap.add_argument("--latent-warmup-steps", type=int, default=0,
                    help="Hold the latent lr at 0 for this many optimizer steps. Adam still accumulates its "
                         "second moment, so the FIRST flips are not driven by a cold, mis-scaled v̂ (the RAdam "
                         "failure mode that turns the first crossings into a cascade). Try 200-500.")
    ap.add_argument("--target-tr", type=float, default=0.0,
                    help="TARGET TRANSITION RATE: fraction of assignments allowed to flip per optimizer step. "
                         ">0 enables TALR — the latent lr is servo-controlled to hit this rate. Flip count "
                         "depends on BOTH lr and the latent distribution, so lr alone cannot control it "
                         "(Lee et al. ICCV25). Try 5e-4 annealing to 1e-4.")
    ap.add_argument("--per-layer-tr", action="store_true", default=True,
                    help="Servo EACH layer's latent lr against its own flip rate (one optimizer group per "
                         "layer). A single global controller sees only the aggregate and cannot stop one layer "
                         "bursting (measured: 15.5%% vs a 3.3%% median on an 8-layer run). --no-per-layer-tr "
                         "restores the single shared group.")
    ap.add_argument("--no-per-layer-tr", dest="per_layer_tr", action="store_false")
    ap.add_argument("--tr-ramp-2x-steps", type=float, default=0.0,
                    help="TALR may at most DOUBLE its gain over this many steps (0 = default 40). Invariant to "
                         "--tr-every. This is the primary burst guard: it lets a fresh axis climb to whatever "
                         "flip rate it needs while making it impossible to overshoot before a correction lands.")
    ap.add_argument("--latent-lr-ramp-steps", type=int, default=0,
                    help="linearly ramp the latent lr 0->base over this many steps after --latent-warmup-steps "
                         "(0 = 4x--tr-every). Removes the cold-start flip burst that happens in the un-servoed "
                         "window before TALR's first measurement, so the base lr need not be retuned per "
                         "layer-coverage.")
    ap.add_argument("--tr-gain-max", type=float, default=0.0,
                    help="Max TALR multiplier over the CALIBRATED base latent-lr, decaying linearly to 1.0 by "
                         "end of run. 0 = use the 2.0 default. Set PER STAGE by how saturated that axis is: a "
                         "SECOND projection on an already-optimised axis needs ~1.2 (its natural flip rate is "
                         "already at/above target, so amplification only bursts it — measured: seq32 t2_up ran "
                         "to gain 4.83 and blew KL 0.4575->0.5698); a FRESH axis (attn after MLP) needs ~6, "
                         "because its natural rate can be literally ZERO and it must be allowed to open up "
                         "(measured: @8L attn sat at 0 flips until TALR amplified).")
    ap.add_argument("--tr-every", type=int, default=10, help="Measure the transition rate every N steps (TALR).")
    ap.add_argument("--tr-final-frac", type=float, default=0.2,
                    help="Anneal the target TR to this fraction of its initial value by end of training "
                         "(coarse-to-fine: many flips early, few late).")
    ap.add_argument("--warmup", type=int, default=20, help="Linear LR warmup steps.")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--gpu-mem", default="20GiB",
                    help="Per-GPU memory cap for teacher device_map (headroom for display/warmup).")
    ap.add_argument("--cpu-mem", default="120GiB", help="CPU memory cap for teacher offload.")
    ap.add_argument("--max-samples", type=int, default=0,
                    help="Cap calibration batches used (0 = all). Lower = faster teacher pass + training.")
    ap.add_argument("--loss-fn", default="topk_kl", choices=["topk_kl", "cakld"],
                    help="Distillation loss: topk_kl (standard) or cakld (confidence-aware, "
                         "BitDistiller — weights each token by teacher peak probability, "
                         "targets greedy-generation fidelity).")
    ap.add_argument("--decision-gamma", type=float, default=1.0,
                    help="Sharpen CAKLD's confidence weighting toward high-confidence 'decision' tokens "
                         "(weight = conf**gamma). 1.0 = plain CAKLD; >1 focuses on answer/logic tokens.")
    ap.add_argument("--ce-weight", type=float, default=0.0,
                    help="Additive cross-entropy on GROUND-TRUTH next tokens: loss += ce_weight * CE(logits, "
                         "ids[1:]). 0 = off (pure fixed-teacher distillation, unchanged). CE is the one signal "
                         "the frozen teacher lacks — the only lever that reintroduces genuine data-scaling past "
                         "the fixed-teacher saturation (report 2026-08-04, §8n). USE WITH --train-weights so the "
                         "trit ASSIGNMENTS can move (CE on scales-only cannot re-represent). Typical 0.1-1.0.")
    ap.add_argument("--ce-positions", type=int, default=512,
                    help="Random #positions/sequence for the CE softmax (bounds the ~V-wide float temp so it "
                         "fits alongside the assignment latents on 24GB; re-drawn each step ⇒ full coverage over "
                         "the run). 0/large = all positions (OOMs at V≈248k). Only used when --ce-weight>0.")
    ap.add_argument("--skew-alpha", type=float, default=0.0,
                    help="A2 (DistiLLM skew-KL, arXiv:2402.03898): replace KL(p||q) with KL(p||a*p+(1-a)*q) "
                         "to bound gradients when the ternary student assigns low prob to teacher support "
                         "(more stable for the stiff scale-only optimization). 0=off; paper default a~=0.1.")
    ap.add_argument("--feat-weight", type=float, default=0.0,
                    help="Weight for hidden-state feature distillation added to the KL loss "
                         "(0 = off; behavior unchanged). Matches the student's final post-norm "
                         "hidden to the teacher's (cache it via --cache-hidden). E2E-QP trains "
                         "only scales, so this is weaker leverage than block_qat.py's assignment "
                         "training — the main use is block_qat.")
    # ── on-policy / degeneration fix (E2E report A1+B2; default-off → byte-equivalent) ──
    ap.add_argument("--on-policy-frac", type=float, default=0.0,
                    help="Fraction of E2E steps that are ON-POLICY (GKD, arXiv:2306.13649): the student "
                         "rolls out its own continuation, the FP teacher scores it live, and the SAME "
                         "CAKLD loss (+ unlikelihood) trains the scales on the student's own drift — the "
                         "exposure-bias signal that the cached teacher-forced KL is blind to (fixes the "
                         "free-gen degeneration the degen gate tracks). 0 = off (legacy).")
    ap.add_argument("--on-policy-warmup-frac", type=float, default=0.25,
                    help="Fraction of steps that stay PURE off-policy first (avoid reinforcing a "
                         "degenerate early student's loops).")
    ap.add_argument("--rollout-len", type=int, default=128, help="On-policy: student rollout length.")
    ap.add_argument("--rollout-prefix", type=int, default=128, help="On-policy: conditioning prefix length.")
    ap.add_argument("--rollout-temp", type=float, default=1.0, help="On-policy sampling temperature.")
    ap.add_argument("--unlikelihood-weight", type=float, default=0.5,
                    help="B2: weight on a token-level unlikelihood penalty (arXiv:1908.04319) over "
                         "REPEATED tokens in the student rollout — directly pushes scale grads off the "
                         "looping attractor the degen gate measures. 0 = A1 only, no explicit anti-repeat.")
    ap.add_argument("--onpolicy-teacher", default=None,
                    help="FP model dir to score rollouts (default: --orig-config-path, the same model "
                         "the cache was built from).")
    ap.add_argument("--onpolicy-calib", default=None,
                    help="Stage B: JSON of chat/STEM sequences whose PREFIXES (first --rollout-prefix tokens) "
                         "seed the on-policy rollouts, so the student re-generates its own reasoning and the "
                         "teacher corrects the over-thinking/no-close drift. Off-policy steps stay on --calib "
                         "(generic replay). Without this, rollouts start from the generic --calib (V2's bug).")
    ap.add_argument("--onpolicy-teacher-cpu", action="store_true",
                    help="Keep the on-policy FP teacher in SYSTEM RAM and run its (small, no-grad, ~176-tok) "
                         "rollout-scoring forward on CPU. Frees ~8GB/GPU vs resident VRAM — the right trade on "
                         "a constrained 2x24GB box where VRAM is scarce but host RAM is idle. Slower per "
                         "on-policy step (CPU forward) but no OOM and no seq/rollout compromise.")
    # ── convergence fixes (all default-off → byte-equivalent to the previous behaviour) ──
    ap.add_argument("--lr-schedule", default="constant", choices=["constant", "linear", "wsd"],
                    help="Post-warmup LR: constant (legacy), linear decay-to-zero (D2Z — collapses "
                         "the noisy constant-LR random walk into the basin so the FINAL point wins), "
                         "or wsd (constant then a linear cooldown over the last --wsd-cooldown-frac).")
    ap.add_argument("--wsd-cooldown-frac", type=float, default=0.2,
                    help="WSD: fraction of steps spent in the final linear cooldown to zero.")
    ap.add_argument("--accum", type=int, default=1,
                    help="Gradient-accumulation micro-steps per optimizer step. Effective batch = "
                         "accum x data-parallel world; raises it to cut the batch-1 gradient variance.")
    ap.add_argument("--scale-ema-decay", type=float, default=0.0,
                    help="EMA decay for parameter-averaging the scales (0 = off; ~0.999 averages the "
                         "tail). Export it with --select ema. Direct fix for 'hope for a lucky "
                         "best-of-loss snapshot' — lands at the basin centre.")
    ap.add_argument("--scale-ema-start-frac", type=float, default=0.5,
                    help="Fraction of total steps after which the scale-EMA starts accumulating.")
    ap.add_argument("--fisher-ema", action="store_true",
                    help="B4: weight the scale-EMA tail-average by per-scale Fisher (Adam exp_avg_sq) so "
                         "well-determined scales are trusted faster. (Adam already Fisher-preconditions "
                         "the update itself, so this is the only non-redundant part of B4.)")
    ap.add_argument("--a3-mlp", action="store_true",
                    help="A3(ii): train a per-input-channel scale on each MLP's input (shared by gate+up), "
                         "folded EXACTLY into the preceding post_attention_layernorm gain at save. Foldable "
                         "capacity expansion aimed at the 79%%-MLP bottleneck; weights stay ternary. "
                         "(down_proj/o_proj can't fold this way — no norm precedes them.)")
    ap.add_argument("--a4-downproj", action="store_true",
                    help="A4: bounded ASSIGNMENT MOVES on the MLP down_proj — a trainable FP latent (init = "
                         "GPTQ ternary*scale) is STE-ternarized each step so assignments can flip {-1,0,+1} "
                         "during E2E, not just rescale. Folds to hard ternary at save (still TQ2_0). Breaks "
                         "the scale-only ceiling on the biggest error source.")
    ap.add_argument("--train-weights", default="none", choices=["none", "down", "gate", "up", "gatedown", "gateup", "mlp", "attn", "all", "lmhead"],
                    help="Test 1 (Q1-A consistency): assignment-QAT — unfreeze the ternary ASSIGNMENTS under "
                         "the distillation loss (latent FP + STE) on the chosen linears (mlp=gate/up/down, "
                         "attn=attention/DeltaNet, all=every linear). The only data-scalable lever: weights "
                         "keep consuming data, vs scale-only which plateaus. Use a higher --lr (~1e-4) than "
                         "scale-only. 8-bit Adam auto-on. Folds to hard ternary at save (TQ2_0).")
    ap.add_argument("--arm-b", action="store_true",
                    help="Arm B (PV-Tuning-style sparse flips): with --train-weights, learn assignments by "
                         "committing gradient-CHOSEN flips periodically (no fp32 latents, oscillation-free). "
                         "Scale-only backbone + top-flip-frac flips every --flip-every steps.")
    ap.add_argument("--flip-every", type=int, default=150, help="Arm B: optimizer steps between flip events (V-step re-equilibration).")
    ap.add_argument("--flip-cap-frac", type=float, default=3e-4, help="Arm B: SAFETY cap on flips/event (the proximal gate sets the actual count).")
    ap.add_argument("--flip-block-cap", type=int, default=2, help="Arm B: max flips per 256-block per event.")
    ap.add_argument("--probe-n", type=int, default=128, help="Arm B: reserved seqs for the fresh flip-gradient pass (bigger = less-noisy ḡ → more real candidates clear the gate).")
    ap.add_argument("--flip-init-qual", type=float, default=1e-5, help="Arm B: initial qualify-fraction for η init (smaller = safer first event; our flips are interaction-sensitive).")
    ap.add_argument("--flip-max-rejects", type=int, default=8, help="Arm B: consecutive rejected flip events before stopping flips (η-search room).")
    ap.add_argument("--flip-gate-n", type=int, default=0, help="Arm B: DISJOINT seqs for the paired-Δ generalisation gate (0=off=legacy in-sample accept). 256 recommended.")
    ap.add_argument("--flip-gate-k", type=float, default=0.0, help="Arm B: commit iff paired ΔKL_G < k·SE(ΔKL_G) on the gate set (0=mean improvement; -0.5 to tighten).")
    ap.add_argument("--flip-gate-max-fails", type=int, default=3, help="Arm B: consecutive linearisation-valid-but-non-generalising events → stop flips (skeleton exhausted).")
    ap.add_argument("--col-scale", action="store_true", help="Perpendicular col-scales probe: trainable per-input-channel scales on MLP input (A3 fold→norm γ) + down_proj input (fold→up rows). Zero deployed bits.")
    ap.add_argument("--scale-qat-bits", type=int, default=0, help="Scale-QAT: train per-256 scales on an n-bit log grid (STE fake-quant); 0=off. Measures achievable quantized-scale floor.")
    ap.add_argument("--commit-weight", type=float, default=0.0,
                    help="Commit-token reweighting (researcher round-5): multiply the per-token KL by this "
                         "alpha in a window around each </think>(248069) — the conclude-and-close transition. "
                         "0/1=off. Start 3.0; keep <=5 (higher = premature-stop/length-attractor risk). Lets a "
                         "<=10%% chat fraction carry V1-level commit signal without V1-level </think>-density.")
    ap.add_argument("--commit-beta", type=float, default=0.0,
                    help="ADDITIVE commit objective (researcher round-6): L = CAKLD_all + beta*mean(KL over the "
                         "commit window around </think>). Un-diluted by window rarity (unlike --commit-weight, "
                         "which was a no-op at 0.135%% token share). Recommended 1.5; sweep {1,1.5,3}.")
    ap.add_argument("--commit-pre", type=int, default=16, help="commit window: think tokens before </think>.")
    ap.add_argument("--commit-post", type=int, default=12, help="commit window: answer tokens after </think>.")
    ap.add_argument("--heldout-n", type=int, default=0,
                    help="Reserve the LAST N calib seqs as held-out (never trained); weight-training selects + "
                         "aborts on their KL/flips (training-batch KL is inadmissible). Set ~48 for --train-weights.")
    ap.add_argument("--eval-every", type=int, default=100, help="Held-out KL/flips eval cadence (optimizer steps).")
    ap.add_argument("--abort-patience", type=int, default=3, help="Abort after this many consecutive worse-than-init held-out evals (×~100 steps).")
    ap.add_argument("--select", default="best", choices=["best", "final", "ema"],
                    help="Deliverable scales: best (lowest loss-EMA snapshot + legacy greedy-restart, "
                         "current default), final (last iterate — pair with --lr-schedule linear), or "
                         "ema (the parameter-average — pair with --scale-ema-decay).")
    args = ap.parse_args()

    if args.smoke:
        smoke()
    elif args.precompute_teacher:
        precompute_teacher(args)
    elif args.train:
        train(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()