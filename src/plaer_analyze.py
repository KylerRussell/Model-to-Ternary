#!/usr/bin/env python
"""plaer_analyze.py — Pre-Loop Answer Extraction Rate, offline, from a loop_gate sample dump.

WHY THIS MEASUREMENT AND NOT ANOTHER. Gate B fails on commit_rate 0.4250 (bar 0.68) and loop_rate
0.5625 (bar 0.30). Two pathologies produce exactly those numbers and need OPPOSITE fixes:

  COMMITMENT failure  — the model derives a usable answer, then cannot stop: it keeps re-verifying
                        ("Wait", "Alternatively") until the token budget runs out. Fixable at
                        DECODING time, 0 bpw, no retraining.
  PATH-FINDING failure — no answer is ever derived; the repetition is a symptom of not knowing.
                        No sampler fixes this; it needs better weights.

PLAER separates them: of the rollouts that LOOPED, what fraction already contained a plausible final
answer BEFORE the loop began? High -> commitment. Low -> path-finding.

Read the number, not the vibe: a decoding-time programme is worth running only if PLAER is high.

Usage: ./.venv/bin/python src/plaer_analyze.py output_sweep/plaer_samples.json
"""
import json, re, sys, zlib
from collections import Counter

# Detector v2. v1 looked only for \boxed{} and "the answer is" and scored PLAER 0.120 -- but
# \boxed appears in ZERO of 48 real rollouts, and v1 fired on only 30.4% of the SUCCESSFUL ones,
# so it was under-sensitive and the 0.120 was an artifact. Always calibrate an answer detector
# against the rollouts that worked before trusting it on the ones that failed. v2 matches the forms
# this model actually emits ("Common knowledge: Paris", "*Response:* ...") and fires on 69.6% of
# successful rollouts, giving PLAER 0.400.
ANSWER = re.compile(
    r"(?:answer|response|conclusion|result)\s*(?:is|:)\s*\**\s*([A-Za-z0-9][^\n.;*]{0,60})"
    r"|common knowledge\s*:\s*([^\n.;*]{1,60})"
    r"|\*(?:draft|response|final)[^:]*:\*\s*([^\n]{1,80})"
    r"|\\boxed\s*\{([^{}]{1,80})\}", re.I)


def loop_onset(text, n=5, times=3):
    """Character index where the repeating n-gram attractor starts, or None.

    Mirrors loop_gate's ngram_loop detector: the first position at which some word-level n-gram
    that eventually repeats `times` times makes its SECOND appearance. Everything before that index
    is trace the model produced while still making progress."""
    words = text.split()
    if len(words) < n * times:
        return None
    seen, first_rep = {}, None
    counts = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    repeated = {g for g, c in counts.items() if c >= times}
    if not repeated:
        return None
    for i in range(len(words) - n + 1):
        g = tuple(words[i:i + n])
        if g in repeated:
            if g in seen:
                first_rep = seen[g]          # index of its FIRST occurrence
                break
            seen[g] = i
    if first_rep is None:
        return None
    return len(" ".join(words[:first_rep]))  # char offset of the attractor's first cycle


def has_answer(seg):
    m = ANSWER.search(seg)
    return bool(m and any(g and g.strip() for g in m.groups()))


def main(path):
    rows = json.load(open(path))
    looped = [r for r in rows if r.get("looped")]
    print(f"rollouts={len(rows)}  looped={len(looped)} ({100*len(looped)/max(len(rows),1):.1f}%)")
    if not looped:
        print("no looped rollouts — nothing to attribute")
        return
    pre_ans = onset_known = 0
    pre_tok = []
    for r in looped:
        t = r["text"]
        cut = loop_onset(t)
        if cut is None:
            cut = len(t)                      # flagged by comp-ratio only; use the whole trace
        else:
            onset_known += 1
        seg = t[:cut]
        if has_answer(seg):
            pre_ans += 1
            pre_tok.append(len(seg.split()))
    plaer = pre_ans / len(looped)
    print(f"loop onset located in {onset_known}/{len(looped)} (rest flagged by comp-ratio only)")
    print(f"\nPLAER = {plaer:.3f}   ({pre_ans}/{len(looped)} looped rollouts already held an answer "
          f"before the loop)")
    if pre_tok:
        pre_tok.sort()
        print(f"  answer appears at ~{pre_tok[len(pre_tok)//2]} words (median) into the trace")
    print()
    if plaer >= 0.50:
        print("=> COMMITMENT failure dominates. The model finds the answer and cannot stop.")
        print("   Decoding-time control (loop rescue / DRY / stop-token boosting) is the right lever,")
        print("   costs 0 bpw and needs no retraining. Weight-space reconstruction work will NOT help.")
    elif plaer < 0.20:
        print("=> PATH-FINDING failure dominates (the report's own falsifier threshold, PLAER<0.20).")
        print("   No sampler can commit an answer the model never derived. Decoding-time candidates")
        print("   1-5 should be DROPPED and effort put into capability at 1.78 bpw.")
    else:
        print("=> MIXED. Decoding control can only address the commitment share; size any expected")
        print(f"   Gate B gain by roughly this fraction ({plaer:.2f}), not by the paper's headline.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "output_sweep/plaer_samples.json")
