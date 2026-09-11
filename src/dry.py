"""dry.py — the DRY (Don't Repeat Yourself) logits processor, extracted VERBATIM from loop_gate.py.

WHY IT LIVES HERE. loop_gate.py builds a model and runs its ENTIRE 48-prompt gate at module level, so
`from loop_gate import DRYLogitsProcessor` executed all of it -- every math_correct.py run paid for a
full discarded generation sweep AND had its RNG stream advanced by 48 sampled rollouts. This module is
side-effect free: importing it builds nothing and consumes no RNG.

DRY is CURBED as of 13ap-ii -- it stays OFF by default (DRY_MULT=0) and should not be enabled in new
work. It buys no capability and it confounds off-argmax measurements. Kept here because the code is
validated and the Gate B history depends on it.
"""
import torch
from transformers import LogitsProcessor


class DRYLogitsProcessor(LogitsProcessor):
    """DRY (Don't Repeat Yourself) suffix-continuation penalty — arXiv:2608.22761, and the sampler
    shipped in llama.cpp / ExLlamaV2 / text-generation-webui.

    Unlike a flat repetition penalty (which penalises tokens uniformly wherever they occurred and
    wrecks code/LaTeX/structured text), DRY penalises ONLY the tokens that would EXTEND a repeat.
    For the current context s, find the longest suffix s[n-L:n] that also occurred earlier ending at
    j; the token s[j+1] that followed it is the one about to continue the loop, and it is penalised by

        multiplier * base ** (L - allowed_length)      for L >= allowed_length

    so the penalty grows exponentially with how much context is already repeating.

    Longest-suffix matching is the Z-algorithm on the reversed context: for reversed r, Z[i] is the
    longest common prefix of r and r[i:], which is exactly the longest common suffix of s[:n-i] and
    s[:n]. The continuation token is then s[n-i]. O(n) per step.

    WHY THIS ONE FIRST (13ag): PLAER = 0.400, so only ~40% of our loops hold an extractable answer.
    DRY is the only candidate whose value does NOT scale with PLAER -- it suppresses verbatim
    continuation whether or not an answer was ever derived.

    Deviation from llama.cpp, recorded: sequence breakers are applied by capping the match length at
    the distance to the most recent breaker token, rather than by llama.cpp's per-restart bookkeeping.
    Same intent (do not let a match run across a structural boundary), simpler implementation."""

    CAP = 1e4          # an absolute ban; keeps the exponential from overflowing on long loops

    def __init__(self, multiplier, base, allowed_length, breaker_ids, penalty_last_n=0):
        self.mult = float(multiplier); self.base = float(base)
        self.allowed = int(allowed_length); self.breakers = set(int(b) for b in breaker_ids)
        self.last_n = int(penalty_last_n)          # 0 = whole context

    @staticmethod
    def _z(r):
        n = len(r); z = [0] * n
        if n:
            z[0] = n
        l = rgt = 0
        for i in range(1, n):
            zi = 0
            if i < rgt:
                zi = min(rgt - i, z[i - l])
            while i + zi < n and r[zi] == r[i + zi]:
                zi += 1
            z[i] = zi
            if i + zi > rgt:
                l, rgt = i, i + zi
        return z

    def __call__(self, input_ids, scores):
        if self.mult <= 0:
            return scores
        for b in range(input_ids.shape[0]):
            s = input_ids[b].tolist()
            if self.last_n > 0:
                s = s[-self.last_n:]
            n = len(s)
            if n < self.allowed + 1:
                continue
            cap = n                                  # do not let a match cross a sequence breaker
            for k in range(n - 1, -1, -1):
                if s[k] in self.breakers:
                    cap = n - 1 - k
                    break
            if cap < self.allowed:
                continue
            z = self._z(s[::-1])
            pen = {}
            for i in range(1, n):
                L = min(z[i], cap)
                if L >= self.allowed:
                    tok = s[n - i]
                    e = L - self.allowed
                    # CLAMP. A real loop repeats for hundreds of tokens, so base**e overflows a
                    # Python float (1.75**1100 -> OverflowError, which killed the first run; the
                    # unit test had only gone to a 32-token repeat). Past ~1e4 the distinction is
                    # meaningless anyway: subtracting 1e4 from a logit is already an absolute ban
                    # on that token after softmax.
                    p = self.CAP if e > 64 else min(self.mult * (self.base ** e), self.CAP)
                    if p > pen.get(tok, 0.0):
                        pen[tok] = p
            for tok, p in pen.items():
                scores[b, tok] -= p
        return scores

from transformers import AutoModelForCausalLM, AutoTokenizer
