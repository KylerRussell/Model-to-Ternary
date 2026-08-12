#!/usr/bin/env python
"""sim_tq164.py — SIMULATE the proposed TQ1_64 packing in fp, before writing any ggml/CUDA code.

Proposed format (1.756 bpw, 27B ≈ 5.93 GB — matches Bonsai's 5.9 GB):
  superblock = 256 weights
    52 B  trits, base-3 packed   (reuse TQ1_0's layout verbatim: 240 @5/byte + 16 @4/byte = 1.625 bpw)
     4 B  4x uint8 sub-scale     (one per g64 group; LINEAR fraction of the row max)
  per row : 1x fp16 "super" max  (16 bits / row_len ≈ 0.006 bpw; natural for ggml — vec_dot is row-at-a-time,
                                  so the row scalar just multiplies the final dot product)

This script applies the ENCODE→DECODE round-trip to every ternary tensor and writes the result as a normal HF
checkpoint, so the existing eval2k + loop_gate can measure the REAL quality cost of the format. The trits are
exact by construction; the only loss is the uint8 sub-scale quantization (measured ~0.13% median).

  ./.venv/bin/python src/sim_tq164.py --in <model_dir> --out <model_dir> [--block 64]
"""
import argparse, shutil
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import should_quantize


SUPERBLOCK = 512          # must match tq164.QK_K (self-contained layout)


def encode_decode(W, block=64):
    """Vectorised TQ1_64 round-trip (fast path for whole-model sims). Must stay bit-identical to
    src/tq164.py encode_row/decode_row — enforced by tools/check_tq164_consistency.py."""
    rows, cols = W.shape
    gps = SUPERBLOCK // block                                # 8 g64 groups per self-contained superblock
    r = W.float().reshape(-1, block)
    s = r.abs().max(dim=1).values                            # per-g64 scale
    nz = s > 0
    trits = torch.zeros_like(r)
    trits[nz] = (r[nz] / s[nz].unsqueeze(1)).round().clamp(-1, 1)    # EXACT (the model is on-grid)
    sb = s.reshape(-1, gps)                                  # [n_super, 8]
    d = sb.max(dim=1, keepdim=True).values.half().float()     # fp16 super INSIDE the superblock
    q = torch.where(d > 0, (sb / d.clamp_min(1e-30) * 255).round().clamp(1, 255), torch.ones_like(sb))
    s_rec = (q / 255.0) * d
    out = (trits.reshape(-1, gps, block) * s_rec.unsqueeze(-1)).reshape(rows, cols)
    m = sb > 0
    err = ((s_rec[m] - sb[m]).abs() / sb[m]) if bool(m.any()) else torch.zeros(1)
    return out.to(W.dtype), err.median().item(), err.max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--block", type=int, default=64)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.suffix != ".safetensors":
            shutil.copy2(f, dst / f.name)

    n_enc = 0; worst = 0.0; meds = []
    for sh in sorted(src.glob("*.safetensors")):
        t = load_file(str(sh))
        for k in list(t.keys()):
            # match the model's own quantized set (body linears + lm_head); embed stays as-is
            probe = k.replace("model.language_model.", "model.")
            if not should_quantize(probe) or t[k].dim() != 2:
                continue
            if t[k].shape[1] % args.block:
                continue
            enc, med, mx = encode_decode(t[k], args.block)
            t[k] = enc; n_enc += 1; meds.append(med); worst = max(worst, mx)
        save_file(t, str(dst / sh.name), metadata={"format": "pt"})
    meds.sort()
    print(f"TQ1_64 sim: encoded {n_enc} tensors | sub-scale rel-err median {100*meds[len(meds)//2]:.3f}% "
          f"worst {100*worst:.2f}%")
    print(f"wrote {dst}  -> now run eval2k + loop_gate on it and compare to the unencoded model")


if __name__ == "__main__":
    main()
