"""answer_score.py — GSM8K gold/prediction extraction, shared and SIDE-EFFECT FREE.

Moved verbatim out of math_correct.py so loop_gate can score correctness without importing a module
that builds a model at import time. Importing this costs nothing and touches no global state --
notably it does NOT consume RNG, which is what made the old cross-imports change sampled output.
"""
import re

NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def gold(ans):
    return ans.split("####")[-1].strip().replace(",", "")


def pred(text):
    """Answer = last \\boxed{} if present, else the last number AFTER </think> (the answer block),
    else the last number anywhere. Matching the gate's convention that the committed answer is what
    follows the close tag."""
    b = text.rfind(r"\boxed")
    if b >= 0:
        j = text.find("{", b)
        if j >= 0:
            depth, k = 0, j
            for k in range(j, len(text)):
                depth += (text[k] == "{") - (text[k] == "}")
                if depth == 0:
                    break
            m = NUM.findall(text[j:k + 1])
            if m:
                return m[-1].replace(",", "").rstrip(".")
    tail = text.split("</think>")[-1] if "</think>" in text else text
    m = NUM.findall(tail) or NUM.findall(text)
    return m[-1].replace(",", "").rstrip(".") if m else None   # "42." -> "42"


def same(a, b):
    if a is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return a.strip() == b.strip()
