#!/usr/bin/env python
"""tq164.py — reference encoder/decoder for the TQ1_64 ternary format (1.7812 bpw).

This is the SPEC. The ggml/CUDA kernels must reproduce `decode_row` bit-for-bit.

LAYOUT — SELF-CONTAINED superblock of 512 weights, 114 bytes (no per-row side data):
    qs[96]  480 trits, 5 per byte, base-3   b = t0 + 3*t1 + 9*t2 + 27*t3 + 81*t4
    qh[8]    32 trits, 4 per byte, base-3   b = t0 + 3*t1 + 9*t2 + 27*t3
    sc[8]     8 uint8 sub-scales, one per g64 group:  scale[g] = (sc[g] / 255) * d
    d[2]      fp16 super = max over this superblock's 8 g64 scales
  => 114 B / 512 weights = 1.7812 bpw   (27B model ≈ 6.01 GB)
  trit storage value t ∈ {0,1,2} represents the weight sign t-1 ∈ {-1,0,+1}.
  Decode order is little-digit-first: element 5*i+k uses base-3 digit k of qs[i].

WHY THIS SHAPE (all measured; see logs/RESULTS_SUMMARY.md §8 and TQ1_64_SPEC.md):
  * 5 trits/byte because 3^5 = 243 <= 256 (1.6 bpw). 64 is not divisible by 5, so a superblock splits into
    (5-per-byte bulk + a 4-per-byte remainder), exactly as TQ1_0 does => 1.625 bpw for the weights.
  * uint8 sub-scales work because --scale-qat-bits 8 already put the trained scales on a 256-value grid.
    fp8-e4m3 was REJECTED: 4.23% median scale error (its 3-bit mantissa is the wrong 8 bits).
  * the super lives INSIDE the superblock so the type is SELF-CONTAINED. ggml requires a uniform block layout
    (nbytes = nelems/blck_size * type_size) with no per-row side data. A per-row super would be 1.756 bpw but
    needs ggml core surgery; this costs +0.025 bpw (80 MB at 27B) and is MORE precise (0.090% vs 0.116%
    median) because the super spans 8 groups instead of a whole row.
  * 512 (not 256 or 1024): 256 costs 1.8125 bpw; 1024 would be 1.766 bpw but does NOT divide a 2560-wide row.
    512 divides every row width we have (2560 / 4096 / 9216 / 248320).
"""
import numpy as np

QK_K = 512                  # weights per superblock (self-contained)
GROUP = 64                  # scale granularity (what the model is trained at)
N_4 = 4 * (QK_K // GROUP)   # 32 trits packed 4-per-byte (the remainder)
N_5 = QK_K - N_4            # 480 trits packed 5-per-byte
QS_BYTES = N_5 // 5         # 96
QH_BYTES = N_4 // 4         # 8
SC_BYTES = QK_K // GROUP    # 8 uint8 sub-scales
D_BYTES = 2                 # fp16 super
BLOCK_BYTES = QS_BYTES + QH_BYTES + SC_BYTES + D_BYTES   # 114
_OFF_QH = QS_BYTES
_OFF_SC = QS_BYTES + QH_BYTES
_OFF_D = _OFF_SC + SC_BYTES


def encode_row(w_row, group=GROUP):
    """Encode one weight row [cols] (already ternary x per-g64 scale, i.e. on-grid).
    Returns packed uint8 [n_super, BLOCK_BYTES]. Self-contained — no extra return value."""
    w = np.asarray(w_row, dtype=np.float32)
    cols = w.shape[0]
    assert cols % QK_K == 0, f"row length {cols} must be a multiple of {QK_K}"
    g = w.reshape(-1, group)
    s = np.abs(g).max(axis=1)                                    # per-g64 scale
    nz = s > 0
    trits = np.zeros_like(g, dtype=np.int8)
    trits[nz] = np.rint(g[nz] / s[nz, None]).astype(np.int8)     # {-1,0,+1}; all-zero groups stay 0
    trits = np.clip(trits, -1, 1)
    t = (trits.reshape(-1) + 1).astype(np.uint8)                 # {0,1,2}

    gps = QK_K // group                                          # 8 groups per superblock
    n_super = cols // QK_K
    out = np.zeros((n_super, BLOCK_BYTES), dtype=np.uint8)
    for b in range(n_super):
        blk = t[b * QK_K:(b + 1) * QK_K]
        a = blk[:N_5].reshape(QS_BYTES, 5)
        out[b, :QS_BYTES] = a[:, 0] + 3 * a[:, 1] + 9 * a[:, 2] + 27 * a[:, 3] + 81 * a[:, 4]
        c = blk[N_5:].reshape(QH_BYTES, 4)
        out[b, _OFF_QH:_OFF_SC] = c[:, 0] + 3 * c[:, 1] + 9 * c[:, 2] + 27 * c[:, 3]
        sb = s[b * gps:(b + 1) * gps]                            # this superblock's 8 scales
        d = np.float16(sb.max())
        if float(d) > 0:
            sc = np.clip(np.rint(sb / np.float32(d) * 255.0), 1, 255).astype(np.uint8)
        else:
            sc = np.ones(gps, dtype=np.uint8)
        out[b, _OFF_SC:_OFF_D] = sc
        out[b, _OFF_D:] = np.frombuffer(np.array([d], dtype=np.float16).tobytes(), dtype=np.uint8)
    return out


def decode_row(packed, cols, group=GROUP):
    """Exact inverse of encode_row -> float32 [cols]. The ggml/CUDA kernels must match THIS."""
    n_super = packed.shape[0]
    assert n_super * QK_K == cols
    w = np.zeros(cols, dtype=np.float32)
    for b in range(n_super):
        blk = packed[b]
        t = np.empty(QK_K, dtype=np.int16)
        v = blk[:QS_BYTES].astype(np.int16)
        t[:N_5] = np.stack([(v // (3 ** k)) % 3 for k in range(5)], axis=1).reshape(-1)
        u = blk[_OFF_QH:_OFF_SC].astype(np.int16)
        t[N_5:] = np.stack([(u // (3 ** k)) % 3 for k in range(4)], axis=1).reshape(-1)
        d = np.frombuffer(blk[_OFF_D:].tobytes(), dtype=np.float16)[0].astype(np.float32)
        s = blk[_OFF_SC:_OFF_D].astype(np.float32) / 255.0 * d
        tr = (t.astype(np.float32) - 1.0).reshape(-1, group)
        w[b * QK_K:(b + 1) * QK_K] = (tr * s[:, None]).reshape(-1)
    return w


def encode_tensor(W, group=GROUP):
    """Vectorised whole-tensor encode -> uint8 [rows * n_super_per_row, BLOCK_BYTES], row-major.
    Identical output to stacking encode_row over rows (enforced by tools/check_tq164_consistency.py).
    Required for real tensors: a 248320-row lm_head is far too slow row-by-row."""
    w = np.ascontiguousarray(np.asarray(W, dtype=np.float32))
    rows, cols = w.shape
    assert cols % QK_K == 0, f"cols {cols} must be a multiple of {QK_K}"
    gps = QK_K // group

    g = w.reshape(-1, group)
    s = np.abs(g).max(axis=1)
    nz = s > 0
    tf = np.zeros(g.shape, dtype=np.float32)
    tf[nz] = g[nz] / s[nz, None]                                  # all-zero groups stay 0
    trits = np.clip(np.rint(tf), -1, 1).astype(np.int8)
    t = (trits.reshape(-1) + 1).astype(np.uint8)                  # {0,1,2}

    n_super = (rows * cols) // QK_K
    tb = t.reshape(n_super, QK_K)
    out = np.empty((n_super, BLOCK_BYTES), dtype=np.uint8)
    a = tb[:, :N_5].reshape(n_super, QS_BYTES, 5).astype(np.uint16)
    out[:, :QS_BYTES] = (a[..., 0] + 3 * a[..., 1] + 9 * a[..., 2]
                         + 27 * a[..., 3] + 81 * a[..., 4]).astype(np.uint8)
    c = tb[:, N_5:].reshape(n_super, QH_BYTES, 4).astype(np.uint16)
    out[:, _OFF_QH:_OFF_SC] = (c[..., 0] + 3 * c[..., 1] + 9 * c[..., 2] + 27 * c[..., 3]).astype(np.uint8)

    sb = s.reshape(n_super, gps)
    d16 = sb.max(axis=1).astype(np.float16)                       # fp16 super per superblock
    d32 = d16.astype(np.float32)
    safe = np.where(d32 > 0, d32, 1.0)
    sc = np.clip(np.rint(sb / safe[:, None] * 255.0), 1, 255).astype(np.uint8)
    sc[d32 <= 0] = 1
    out[:, _OFF_SC:_OFF_D] = sc
    out[:, _OFF_D:] = d16.view(np.uint8).reshape(n_super, 2)
    return out


def bpw():
    """bits per weight (self-contained: no per-row side data)."""
    return BLOCK_BYTES * 8 / QK_K
