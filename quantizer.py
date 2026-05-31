#!/usr/bin/env python3
"""
Ternary quantizer — RTN and calibration-aware quantization to {-1, 0, +1}.

Phase 1: Round-To-Nearest (RTN)
  - Per-block (g128) absolute-max scaling
  - Normalize to [-1, 1] and round

Phase 2: Calibration-aware (GPTQ-style)
  - Uses calibration data to minimize reconstruction error
  - Iteratively adjusts ternary assignments per block
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class QuantizedTensor:
    """Result of ternary quantization."""
    ternary: torch.Tensor      # int8 tensor of {-1, 0, 1}
    scales: torch.Tensor       # fp16 per-block scale factors
    original_shape: tuple      # shape before flattening/padding
    block_size: int
    num_valid: int             # number of valid (non-padding) weights


def quantize_rtn(
    weight: torch.Tensor,
    block_size: int = 128,
) -> QuantizedTensor:
    """
    Round-To-Nearest ternary quantization with per-block scaling.

    For each block of `block_size` weights:
      1. scale = max(|w_i|) for i in block
      2. normalized = w_i / scale
      3. ternary = round(normalized) clamped to {-1, 0, 1}

    Args:
        weight: Float weight tensor of any shape.
        block_size: Number of weights per quantization group.

    Returns:
        QuantizedTensor with ternary values and scales.
    """
    original_shape = weight.shape
    flat = weight.reshape(-1).float()
    num_valid = len(flat)

    # Pad to multiple of block_size
    pad_len = (block_size - num_valid % block_size) % block_size
    if pad_len > 0:
        flat = F.pad(flat, (0, pad_len))

    blocks = flat.reshape(-1, block_size)
    num_blocks = blocks.shape[0]

    # Per-block absolute max (scale factor)
    scales = blocks.abs().amax(dim=1)
    scales = torch.clamp(scales, min=1e-10)  # avoid division by zero

    # Normalize and round
    normalized = blocks / scales.unsqueeze(1)
    ternary = torch.round(normalized).clamp(-1, 1).to(torch.int8)

    # Store scales as FP16
    scales = scales.to(torch.float16)

    return QuantizedTensor(
        ternary=ternary.reshape(-1),
        scales=scales,
        original_shape=original_shape,
        block_size=block_size,
        num_valid=num_valid,
    )


def quantize_absmean(
    weight: torch.Tensor,
    block_size: int = 128,
) -> QuantizedTensor:
    """
    AbsMean ternary quantization (BitNet b1.58 style).

    Uses mean absolute value as the scale factor instead of max.
    This can be more robust to outliers.

    For each block:
      1. scale = mean(|w_i|)
      2. normalized = w_i / scale
      3. ternary = round(normalized) clamped to {-1, 0, 1}
    """
    original_shape = weight.shape
    flat = weight.reshape(-1).float()
    num_valid = len(flat)

    pad_len = (block_size - num_valid % block_size) % block_size
    if pad_len > 0:
        flat = F.pad(flat, (0, pad_len))

    blocks = flat.reshape(-1, block_size)

    # AbsMean scale
    scales = blocks.abs().mean(dim=1)
    scales = torch.clamp(scales, min=1e-10)

    normalized = blocks / scales.unsqueeze(1)
    ternary = torch.round(normalized).clamp(-1, 1).to(torch.int8)
    scales = scales.to(torch.float16)

    return QuantizedTensor(
        ternary=ternary.reshape(-1),
        scales=scales,
        original_shape=original_shape,
        block_size=block_size,
        num_valid=num_valid,
    )


def dequantize(qt: QuantizedTensor) -> torch.Tensor:
    """
    Reconstruct approximate FP16 weights from ternary + scales.

    w_reconstructed = ternary * scale (broadcast per block)
    """
    ternary = qt.ternary.float().reshape(-1, qt.block_size)
    scales = qt.scales.float().unsqueeze(1)
    reconstructed = (ternary * scales).reshape(-1)

    # Remove padding
    reconstructed = reconstructed[:qt.num_valid]
    return reconstructed.reshape(qt.original_shape).to(torch.float16)


def compute_quantization_error(
    original: torch.Tensor,
    qt: QuantizedTensor,
) -> dict:
    """Compute reconstruction error metrics."""
    reconstructed = dequantize(qt)
    original_f = original.float()
    recon_f = reconstructed.float()

    diff = original_f - recon_f
    mse = (diff ** 2).mean().item()
    rmse = mse ** 0.5
    mae = diff.abs().mean().item()

    # Signal-to-quantization-noise ratio
    signal_power = (original_f ** 2).mean().item()
    sqnr = 10 * torch.log10(torch.tensor(signal_power / (mse + 1e-20))).item()

    # Cosine similarity
    cos_sim = F.cosine_similarity(
        original_f.reshape(1, -1),
        recon_f.reshape(1, -1),
    ).item()

    # Ternary value distribution
    t = qt.ternary[:qt.num_valid]
    n_neg = (t == -1).sum().item()
    n_zero = (t == 0).sum().item()
    n_pos = (t == 1).sum().item()
    total = qt.num_valid

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "sqnr_db": sqnr,
        "cosine_similarity": cos_sim,
        "ternary_distribution": {
            "-1": n_neg / total,
            "0": n_zero / total,
            "+1": n_pos / total,
        },
        "sparsity": n_zero / total,  # fraction of zero weights
    }


def quantize_gptq_ternary(
    weight: torch.Tensor,
    hessian: torch.Tensor,
    block_size: int = 128,
    damp_pct: float = 0.01,
) -> QuantizedTensor:
    """
    GPTQ-style ternary quantization with Hessian-guided error compensation.

    Phase 2 implementation — uses calibration data to compute the Hessian
    (H = X^T @ X) and iteratively quantizes columns to minimize
    reconstruction error.

    Args:
        weight: [out_features, in_features] weight matrix.
        hessian: [in_features, in_features] Hessian matrix (X^T @ X).
        block_size: Quantization group size.
        damp_pct: Dampening factor for Hessian diagonal.

    Returns:
        QuantizedTensor with optimized ternary assignments.
    """
    W = weight.float().clone()
    n_out, n_in = W.shape

    # Dampening
    H = hessian.float()
    damp = damp_pct * H.diag().mean()
    H.diagonal().add_(damp)

    # Cholesky decomposition of Hessian
    try:
        H_inv = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(H_inv)
    except torch.linalg.LinAlgError:
        # Fallback: add more dampening
        H.diagonal().add_(damp * 10)
        H_inv = torch.linalg.cholesky(H)
        H_inv = torch.cholesky_inverse(H_inv)

    H_inv_diag = H_inv.diag()

    # Quantize column by column with error compensation
    all_ternary = torch.zeros_like(W, dtype=torch.int8)
    all_scales = []

    # Process in groups of block_size columns
    for col_start in range(0, n_in, block_size):
        col_end = min(col_start + block_size, n_in)
        block_cols = W[:, col_start:col_end]
        actual_block_size = col_end - col_start

        # Scale for this block — abs-mean (BitNet b1.58); abs-max over-sparsifies
        # rotated/near-Gaussian blocks (~84% zeros), starving GPTQ's error feedback.
        scale = block_cols.abs().mean(dim=1)
        scale = torch.clamp(scale, min=1e-10)
        all_scales.append(scale)

        # Keep a copy of original block weights to compute block quantization error E
        W_block_orig = block_cols.clone()

        # Local column-by-column quantization within the block (very fast)
        for j in range(col_start, col_end):
            w_col = W[:, j]

            # Quantize this column
            normalized = w_col / scale
            q = torch.round(normalized).clamp(-1, 1)
            all_ternary[:, j] = q.to(torch.int8)

            # Compute quantization error
            w_hat = q * scale
            err = (w_col - w_hat) / H_inv_diag[j]

            # Only update remaining columns INSIDE the current block locally
            if j + 1 < col_end:
                W[:, j + 1:col_end] -= err.unsqueeze(1) * H_inv[j, j + 1:col_end].unsqueeze(0)

        # Compute total block error E
        W_hat_block = all_ternary[:, col_start:col_end].float() * scale.unsqueeze(1)
        E = W_block_orig - W_hat_block

        # Batched matrix-multiplication (GEMM) update for all columns OUTSIDE the block
        if col_end < n_in:
            inv_H_block = H_inv[col_start:col_end, col_start:col_end]
            # Solve inv_H_block @ P = H_inv[col_start:col_end, col_end:]
            P = torch.linalg.solve(inv_H_block, H_inv[col_start:col_end, col_end:])
            # Single highly optimized GEMM update
            W[:, col_end:] -= E @ P

    # Reshape to standard format
    flat_ternary = all_ternary.reshape(-1)
    # Stack along dim 1 to get shape [n_out, n_groups] so that flattening
    # aligns with the row-major block ordering of flat_ternary
    scales_tensor = torch.stack(all_scales, dim=1).to(torch.float16)

    return QuantizedTensor(
        ternary=flat_ternary,
        scales=scales_tensor.reshape(-1),
        original_shape=weight.shape,
        block_size=block_size,
        num_valid=weight.numel(),
    )