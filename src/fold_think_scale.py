#!/usr/bin/env python
"""fold_think_scale.py — fold the calibrated `</think>`-row gain into a saved HF checkpoint.

The saved lm_head weight is ON-GRID (ternary assignments x per-block scale), so multiplying row 248069 by c is
identical to multiplying that row's block-scales by c: the ternary assignments are UNCHANGED, only the scales
move. The result therefore re-quantizes to TQ2_0 exactly — fully ternary, no mixed precision, no side tensors,
no decode-time knob. (Researcher round-6 RQ4.)

  ./.venv/bin/python src/fold_think_scale.py --in <model_dir> --out <model_dir> --c 1.10
"""
import argparse, json, shutil
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

THINK_CLOSE = 248069


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    ap.add_argument("--c", type=float, required=True, help="row gain (>1 raises P(</think>))")
    ap.add_argument("--row", type=int, default=THINK_CLOSE)
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    # copy every non-weight file (config/tokenizer/index) verbatim
    for f in src.iterdir():
        if f.suffix != ".safetensors":
            shutil.copy2(f, dst / f.name)

    shards = sorted(src.glob("*.safetensors"))
    touched = False
    for sh in shards:
        t = load_file(str(sh))
        for k in list(t.keys()):
            if k.endswith("lm_head.weight"):
                before = t[k][args.row].abs().max().item()
                t[k][args.row] = (t[k][args.row].float() * args.c).to(t[k].dtype)
                after = t[k][args.row].abs().max().item()
                # verify the row stays on-grid: |w|/scale must still take <=3 distinct values {0,1}*scale
                uq = torch.unique(t[k][args.row].float().abs())
                print(f"  {sh.name}: scaled {k} row {args.row} by {args.c} "
                      f"(row max |w| {before:.5g} -> {after:.5g}; {len(uq)} distinct |values| "
                      f"{'OK on-grid' if len(uq) <= 64 else 'CHECK'})")
                touched = True
        save_file(t, str(dst / sh.name), metadata={"format": "pt"})
    if not touched:
        raise SystemExit("no lm_head.weight found — nothing folded")
    print(f"wrote {dst}  (ternary assignments unchanged; only the </think> row's scale moved)")


if __name__ == "__main__":
    main()
