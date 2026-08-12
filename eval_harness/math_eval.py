#!/usr/bin/env python3
"""math_eval.py — score served model(s) on a MathArena set (AIME/HMMT), concurrent + multi-endpoint.

  python math_eval.py --base-urls http://127.0.0.1:8081/v1,http://127.0.0.1:8082/v1 \
      --dataset MathArena/aime_2026 --samples 4 --concurrency 8 --max-tokens 12288 \
      --out results/aime26_ternary.json
"""
import argparse, json, re, time
from pathlib import Path
from datasets import load_dataset
from eval_common import chat, run_concurrent


def extract_boxed(text):
    i = text.rfind(r"\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k].strip()
    return None


def norm(a):
    if a is None:
        return None
    a = str(a).strip().replace(" ", "").replace("$", "").replace(",", "")
    a = re.sub(r"\\(text|mathrm|left|right)\b", "", a)
    try:
        return str(int(float(a)))
    except ValueError:
        return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-urls", required=True, help="comma-separated OpenAI endpoints")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=12288)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    endpoints = args.base_urls.split(",")

    ds = load_dataset(args.dataset, split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    n = len(ds)
    PROMPT = "{}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
    golds = [norm(ex["answer"]) for ex in ds]
    probs = [ex["problem"] for ex in ds]
    tasks = [(p, s) for p in range(n) for s in range(args.samples)]  # (problem, sample)

    def worker(i, ep):
        p, _ = tasks[i]
        txt, fin = chat(ep, PROMPT.format(probs[p]), args.max_tokens, args.temp)
        pred = norm(extract_boxed(txt))
        return {"p": p, "correct": int(pred is not None and pred == golds[p]), "finish": fin}

    t0 = time.time()
    res = run_concurrent(len(tasks), endpoints, args.concurrency, worker, label=args.dataset.split("/")[-1])
    hits = [0] * n
    for r in res:
        if r and "correct" in r:
            hits[r["p"]] += r["correct"]
    per = [{"idx": p, "gold": golds[p], "pass@1": hits[p] / args.samples} for p in range(n)]
    out = {"dataset": args.dataset, "n": n, "samples": args.samples,
           "pass@1": sum(x["pass@1"] for x in per) / max(n, 1),
           "minutes": (time.time() - t0) / 60, "per": per}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n{args.dataset}: pass@1={out['pass@1']:.4f} (n={n}, k={args.samples}, {out['minutes']:.1f} min) -> {args.out}")


if __name__ == "__main__":
    main()
