#!/usr/bin/env python3
"""compare.py <results_dir> — 3-way: FP-27B (original) vs ternary-27B (ours) vs 9B-q8 (same-ish mem).

The question: does the 8 GB ternary-27B land BETWEEN the FP-27B and the 9B (TQ2_0 worth it),
or BELOW both (you'd be better off with a small high-precision model -> method needs work)?
"""
import json, os, sys

RES = sys.argv[1] if len(sys.argv) > 1 else "eval_harness/results"
GGUF = {"fp": "eval_gguf/fp_27b-Q8_0.gguf", "ternary": "eval_gguf/ternary_4M-TQ2_0.gguf",
        "9b": "eval_gguf/qwen9b-Q8_0.gguf"}
LABEL = {"fp": "FP-27B q8", "ternary": "ternary-27B", "9b": "9B q8"}


def gb(m):
    p = GGUF[m]
    return os.path.getsize(p) / 1e9 if os.path.exists(p) else float("nan")


def metric(ev, m):
    p = os.path.join(RES, f"{ev}_{m}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return d.get("pass@1", d.get("accuracy"))


EVALS = [("Knowledge", "MMLU-Pro", "mmlu_pro"), ("Knowledge", "GPQA-Diamond", "gpqa"),
         ("STEM", "AIME26", "aime26"), ("STEM", "HMMT26", "hmmt26")]
mem = {m: gb(m) for m in GGUF}
print(f"\n  memory:  FP-27B {mem['fp']:.1f} GB   |   ternary-27B {mem['ternary']:.1f} GB   |   9B-q8 {mem['9b']:.1f} GB")
print(f"  {'eval':<14}{'FP-27B':>9}{'ternary':>9}{'9B-q8':>9}{'tern vs FP':>12}{'tern vs 9B':>12}  verdict")
print("  " + "-" * 78)
between = below = 0
for sec, name, key in EVALS:
    fp, tn, nb = metric(key, "fp"), metric(key, "ternary"), metric(key, "9b")
    if tn is None:
        print(f"  {name:<14}{'(ternary pending)':>30}")
        continue
    cells = "".join(f"{(v):>9.3f}" if v is not None else f"{'--':>9}" for v in (fp, tn, nb))
    dfp = f"{tn-fp:+.3f}" if fp is not None else "--"
    d9b = f"{tn-nb:+.3f}" if nb is not None else "--"
    verdict = ""
    if fp is not None and nb is not None:
        if nb <= tn <= fp:
            verdict = "BETWEEN ✓"; between += 1
        elif tn < nb:
            verdict = "below 9B ✗"; below += 1
        elif tn > fp:
            verdict = "above FP ??"
        else:
            verdict = "above 9B, >FP?"
    print(f"  {name:<14}{cells}{dfp:>12}{d9b:>12}  {verdict}")
print("  " + "-" * 78)
print("  tern vs FP  = degradation from the original (negative = worse, expected)")
print("  tern vs 9B  = vs a same-memory small model (POSITIVE = ternary-27B wins; it's also smaller)")
if between or below:
    print(f"\n  SCORE: ternary lands BETWEEN on {between}/{between+below} evals, BELOW the 9B on {below}.")
    print("  -> mostly BETWEEN = TQ2_0 is worth it (big-model-low-bit beats small-model-high-bit at ~same mem).")
    print("  -> mostly BELOW 9B = method needs work (a plain 9B q8 would be better at similar memory).")
