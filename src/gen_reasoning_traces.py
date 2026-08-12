#!/usr/bin/env python
"""gen_reasoning_traces.py — generate long-CoT thinking-mode traces from the FP teacher, for the
reasoning-trace proxy (researcher round-9 §H). Bonsai's sub-4-bit collapse is "qualitative, not gradual,"
concentrated in long-CoT generation where per-token error compounds across the KV-cache — the generic
teacher-forced eval2k HIDES it. This produces the on-distribution traces the ternary model must be scored on.

Each trace = FP teacher's own thinking-mode rollout on a reasoning prompt (math/code/logic). We save the full
token-id sequence + the prompt length + the <think>/</think> boundary positions + a structural-validity flag,
so reasoning_trace_eval.py can segment agreement/KL by phase (prompt vs gen), position (early/mid/late gen),
and block (inside-think vs answer).

Env-free CLI:
  python gen_reasoning_traces.py --model output_4b/untied_4b --out output_4b/reasoning_traces.json \
      --n 64 --max-new 1536 --temp 0.6 --top-p 0.95 [--prompts-file extra.json]
"""
import argparse, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

THINK_OPEN, THINK_CLOSE = 248068, 248069

# Curated reasoning prompts that reliably elicit multi-step CoT (math / competition / code / logic).
PROMPTS = [
    "What is the remainder when 7^100 is divided by 13? Show your reasoning.",
    "A bag has 5 red, 3 blue, and 2 green balls. Two are drawn without replacement. What is the probability both are the same color?",
    "Find all integer solutions to x^2 - y^2 = 45.",
    "A train travels 60 km at 40 km/h, then 90 km at 60 km/h. What is its average speed for the whole trip?",
    "How many positive divisors does 2^4 * 3^2 * 5 have, and what is their sum?",
    "Prove that the sum of the first n odd numbers equals n^2.",
    "If f(x) = 3x + 2 and g(x) = x^2 - 1, what is f(g(2)) - g(f(2))?",
    "A rectangle's length is 3 more than twice its width. If the perimeter is 36, find the area.",
    "What is the smallest positive integer divisible by every integer from 1 to 6?",
    "Three consecutive even integers sum to 78. What is the product of the smallest and largest?",
    "In how many ways can 4 people be seated in a row if two specific people must not sit next to each other?",
    "Solve for x: log2(x) + log2(x - 2) = 3.",
    "A number is 4 times the sum of its digits. If it is a two-digit number, find all such numbers.",
    "What is the 2026th digit after the decimal point in the decimal expansion of 1/7?",
    "Write a Python function that returns the nth Fibonacci number using memoization, and explain its time complexity.",
    "Write a Python function to check whether a string is a valid palindrome ignoring case and non-alphanumeric characters. Reason about edge cases first.",
    "Given a list of integers, write Python to find the length of the longest strictly increasing subsequence. Explain the approach.",
    "Implement binary search in Python and prove its loop invariant maintains correctness.",
    "Write Python to detect a cycle in a singly linked list, and explain why Floyd's algorithm works.",
    "A farmer has chickens and cows totaling 30 heads and 74 legs. How many of each? Reason step by step.",
    "Five houses in a row are painted different colors. The blue house is immediately left of the green. The red house is first. Where can the yellow house be? Enumerate.",
    "If today is Wednesday, what day of the week will it be in 100 days? Show the modular arithmetic.",
    "A clock shows 3:15. What is the angle between the hour and minute hands? Reason carefully.",
    "You have a 3-liter and a 5-liter jug and unlimited water. Describe how to measure exactly 4 liters.",
    "What is the sum of the interior angles of a convex polygon with 12 sides, and the measure of each if regular?",
    "Two dice are rolled. What is the probability the sum is a prime number? Enumerate the favorable outcomes.",
    "Simplify: (1 - 1/2)(1 - 1/3)(1 - 1/4)...(1 - 1/100). Reason about the telescoping.",
    "A car depreciates 15% per year. What fraction of its original value remains after 3 years? Round to a percentage.",
    "How many trailing zeros are in 100! ? Explain the counting method.",
    "Determine whether 1 + 3 + 5 + ... + 99 is a perfect square, and identify which one.",
    "Write a Python one-liner using reduce to compute the product of a list, and explain what reduce does.",
    "Given a binary tree, write Python to compute its maximum depth recursively, and state the base case reasoning.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="output_4b/untied_4b")
    ap.add_argument("--out", default="output_4b/reasoning_traces.json")
    ap.add_argument("--n", type=int, default=64, help="number of traces (cycles through the prompt list)")
    ap.add_argument("--max-new", type=int, default=1536)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seq-cap", type=int, default=2048, help="hard cap on total sequence length (eval memory)")
    ap.add_argument("--prompts-file", default=None, help="optional JSON list of extra prompt strings")
    args = ap.parse_args()

    prompts = list(PROMPTS)
    if args.prompts_file:
        prompts = json.load(open(args.prompts_file)) + prompts
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"loading FP teacher {args.model} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
                                                 dtype=torch.bfloat16).to("cuda:0").eval()

    traces, valid = [], 0
    for i in range(args.n):
        p = prompts[i % len(prompts)]
        msgs = [{"role": "user", "content": p}]
        try:
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=True,
                                          return_tensors="pt", tokenize=True)
        except TypeError:
            ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", tokenize=True)
        if not torch.is_tensor(ids):                       # some versions return a BatchEncoding/dict
            ids = ids["input_ids"] if "input_ids" in ids else torch.tensor(ids)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        prompt_len = ids.shape[1]
        with torch.no_grad():
            out = model.generate(ids.to("cuda:0"), max_new_tokens=args.max_new, do_sample=True,
                                 temperature=args.temp, top_p=args.top_p, top_k=20,
                                 pad_token_id=tok.eos_token_id)
        full = out[0].tolist()[: args.seq_cap]
        gen = full[prompt_len:]
        tc = full.index(THINK_CLOSE) if THINK_CLOSE in full else -1
        to = full.index(THINK_OPEN) if THINK_OPEN in full else (prompt_len - 1)
        eos_emitted = (tok.eos_token_id in gen)
        # structural validity: emitted a </think>, non-empty answer after it, and stopped (not truncated)
        answer_len = (len(full) - tc - 1) if tc >= 0 else 0
        structural = "valid" if (tc >= 0 and answer_len >= 2 and eos_emitted) else \
                     ("truncated" if tc < 0 else ("empty_answer" if answer_len < 2 else "no_stop"))
        traces.append({"prompt": p, "ids": full, "prompt_len": prompt_len,
                       "think_open": to, "think_close": tc, "structural": structural})
        valid += (structural == "valid")
        if (i + 1) % 8 == 0:
            print(f"  {i+1}/{args.n}  last_len={len(full)} struct={structural} valid_so_far={valid}", flush=True)

    json.dump({"model": args.model, "n": len(traces), "traces": traces}, open(args.out, "w"))
    from collections import Counter
    sc = Counter(t["structural"] for t in traces)
    avg_len = sum(len(t["ids"]) for t in traces) / len(traces)
    print(f"\nwrote {len(traces)} traces -> {args.out}")
    print(f"  avg len {avg_len:.0f} | structural: {dict(sc)} | valid {valid}/{len(traces)} ({100*valid/len(traces):.0f}%)")


if __name__ == "__main__":
    main()
