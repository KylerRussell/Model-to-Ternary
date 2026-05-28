#!/usr/bin/env python3
"""
Hadamard rotation utilities for pre-quantization outlier smoothing.

Applies the Fast Walsh-Hadamard Transform (FWHT) to weight matrices
to spread outlier values across dimensions, making the distribution
more amenable to ternary quantization.

Supports:
  - CUDA-accelerated FWHT via Dao-AILab's fast-hadamard-transform
  - Pure PyTorch fallback for dimensions that are powers of 2
  - Automatic padding for non-power-of-2 dimensions
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# Try to import CUDA-accelerated Hadamard
try:
    from fast_hadamard_transform import hadamard_transform as _cuda_hadamard
    HAS_CUDA_FHT = True
except ImportError:
    HAS_CUDA_FHT = False


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def _pytorch_hadamard_1d(x: torch.Tensor) -> torch.Tensor:
    """
    Pure PyTorch Walsh-Hadamard transform along the last dimension.
    Input dimension must be a power of 2.
    Uses the recursive butterfly structure: O(n log n) operations.
    """
    n = x.shape[-1]
    assert n > 0 and (n & (n - 1)) == 0, f"Dimension must be power of 2, got {n}"

    # Butterfly passes
    h = 1
    while h < n:
        # Reshape to pair elements
        x_reshaped = x.view(*x.shape[:-1], n // (2 * h), 2, h)
        a = x_reshaped[..., 0, :]  # even
        b = x_reshaped[..., 1, :]  # odd
        x_reshaped[..., 0, :] = a + b
        x_reshaped[..., 1, :] = a - b
        x = x_reshaped.view(*x.shape)
        h *= 2

    return x


def hadamard_transform(
    x: torch.Tensor,
    scale: Optional[float] = None,
    use_cuda: bool = True,
) -> torch.Tensor:
    """
    Apply the Hadamard transform along the last dimension of x.

    Args:
        x: Input tensor of shape (..., dim). dim will be padded to next
           power of 2 if necessary.
        scale: Multiplicative scale applied to output. If None, uses
               1/sqrt(dim) for orthonormal normalization.
        use_cuda: Whether to use the CUDA kernel if available.

    Returns:
        Transformed tensor of same shape as input (padded dims are truncated).
    """
    original_dim = x.shape[-1]
    padded_dim = _next_power_of_2(original_dim)

    if scale is None:
        scale = 1.0 / math.sqrt(padded_dim)

    # Pad if needed
    if padded_dim != original_dim:
        x = F.pad(x, (0, padded_dim - original_dim))

    # Apply transform
    if use_cuda and HAS_CUDA_FHT and x.is_cuda:
        result = _cuda_hadamard(x, scale=scale)
    else:
        result = _pytorch_hadamard_1d(x.float()) * scale
        result = result.to(x.dtype)

    # Truncate back to original dimension
    if padded_dim != original_dim:
        result = result[..., :original_dim]

    return result


def rotate_weight_matrix(
    weight: torch.Tensor,
    dim: str = "input",
) -> torch.Tensor:
    """
    Apply Hadamard rotation to a weight matrix.

    For a linear layer y = Wx:
      - dim="input":  rotates input dimension  → W' = W @ H^T
                       (each row of W is transformed)
      - dim="output": rotates output dimension → W' = H @ W
                       (each column of W is transformed)

    The orthonormal Hadamard satisfies H = H^T = H^{-1} (up to scale),
    so H^T @ H = I.

    Args:
        weight: Weight tensor of shape [out_features, in_features].
        dim: Which dimension to rotate ("input" or "output").

    Returns:
        Rotated weight tensor of same shape.
    """
    assert weight.ndim == 2, f"Expected 2D weight, got shape {weight.shape}"

    if dim == "input":
        # Rotate each row (input dimension = columns)
        # W' = W @ H^T = hadamard(W, along last dim)
        return hadamard_transform(weight)
    elif dim == "output":
        # Rotate each column (output dimension = rows)
        # W' = H @ W = hadamard(W^T, along last dim)^T
        return hadamard_transform(weight.T).T
    else:
        raise ValueError(f"dim must be 'input' or 'output', got '{dim}'")


def compute_rotation_error(
    original: torch.Tensor,
    rotated: torch.Tensor,
) -> dict:
    """
    Diagnostic: check that rotation is orthogonal (preserves norms).
    For a proper Hadamard transform, row norms should be preserved.
    """
    orig_norms = original.float().norm(dim=-1)
    rot_norms = rotated.float().norm(dim=-1)
    rel_error = ((orig_norms - rot_norms).abs() / (orig_norms + 1e-10)).mean()

    return {
        "mean_relative_norm_error": rel_error.item(),
        "max_relative_norm_error": ((orig_norms - rot_norms).abs() / (orig_norms + 1e-10)).max().item(),
        "original_weight_range": (original.min().item(), original.max().item()),
        "rotated_weight_range": (rotated.min().item(), rotated.max().item()),
        "original_abs_mean": original.abs().mean().item(),
        "rotated_abs_mean": rotated.abs().mean().item(),
    }


def analyze_quantization_friendliness(weight: torch.Tensor, block_size: int = 128) -> dict:
    """
    Analyze how well a weight matrix will quantize to ternary.
    Returns metrics on outlier distribution and block-wise uniformity.
    """
    flat = weight.float().reshape(-1)
    abs_vals = flat.abs()

    # Global stats
    global_mean = abs_vals.mean().item()
    global_max = abs_vals.max().item()
    outlier_ratio = (abs_vals > 3 * global_mean).float().mean().item()

    # Block-wise stats
    n_blocks = (len(flat) + block_size - 1) // block_size
    padded = F.pad(flat, (0, n_blocks * block_size - len(flat)))
    blocks = padded.reshape(n_blocks, block_size)
    block_maxes = blocks.abs().amax(dim=1)
    block_means = blocks.abs().mean(dim=1)

    # Ratio of max to mean per block (lower = more uniform = better for ternary)
    max_to_mean = (block_maxes / (block_means + 1e-10))

    return {
        "global_abs_mean": global_mean,
        "global_abs_max": global_max,
        "outlier_ratio_3sigma": outlier_ratio,
        "block_max_to_mean_avg": max_to_mean.mean().item(),
        "block_max_to_mean_std": max_to_mean.std().item(),
        "num_blocks": n_blocks,
    }
