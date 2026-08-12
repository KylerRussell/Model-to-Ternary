#!/usr/bin/env python3
"""mcq_eval.py — score served model(s) on a multiple-choice set (MMLU-Pro / GPQA-Diamond),
concurrent + multi-endpoint. Same prompt/extraction for FP and ternary so the DELTA is clean
(absolute won't match the model card — that's fine, we run our own FP reference).

  python mcq_eval.py --base-urls http://127.0.0.1:8081/v1,http://127.0.0.1:8082/v1 \
      --type mmlu_pro --limit 300 --samples 4 --concurrency 8 --out results/mmlupro_ternary.json
"""
import argparse, json, random, re, time
from pathlib import Path
from datasets import load_dataset
from eval_common import chat, run_concurrent

LETTERS = "ABCDEFGHIJ"


def load_mcq(kind, limit):
    """Return list of {question, options(list), gold_letter}."""
    items = []
    if kind == "mmlu_pro":
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        idx = list(range(len(ds)))
        random.Random(0).shuffle(idx)
        if limit:
            idx = idx[:limit]
        for i in idx:
            ex = ds[i]
            items.append({"q": ex["question"], "opts": list(ex["options"]),
                          "gold": LETTERS[ex["answer_index"]]})
    elif kind == "gpqa":
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", token=True)  # gated; uses cached HF token
        idx = list(range(len(ds)))
        if limit:
            idx = idx[:limit]
        for i in idx:
            ex = ds[i]
            opts = [ex["Correct Answer"], ex["Incorrect Answer 1"],
                    ex["Incorrect Answer 2"], ex["Incorrect Answer 3"]]
            order = [0, 1, 2, 3]
            random.Random(1000 + i).shuffle(order)
            shuffled = [opts[o] for o in order]
            gold = LETTERS[order.index(0)]
            items.append({"q": ex["Question"], "opts": shuffled, "gold": gold})
    else:
        raise SystemExit(f"unknown --type {kind}")
    return items


def make_prompt(it):
    lines = [it["q"], ""]
    for k, o in enumerate(it["opts"]):
        lines.append(f"{LETTERS[k]}. {o}")
    lines.append("\nRespond with ONLY your final choice in the form: The answer is (X)   — where X is the "
                 "letter. Do not explain or show any working.")
    return "\n".join(lines)


def extract_letter(text, nopt):
    valid = LETTERS[:nopt]
    # prefer the explicit "answer is (X)" near the end
    m = list(re.finditer(r"answer\s*is\s*\(?\s*([" + valid + r"])\s*\)?", text, re.I))
    if m:
        return m[-1].group(1).upper()
    m = list(re.finditer(r"\(([" + valid + r"])\)", text))
    if m:
        return m[-1].group(1).upper()
    m = list(re.finditer(r"\b([" + valid + r"])\b", text))
    return m[-1].group(1).upper() if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-urls", required=True)
    ap.add_argument("--type", required=True, choices=["mmlu_pro", "gpqa"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--thinking", action="store_true",
                    help="enable CoT thinking (default OFF for MCQ — ~100x faster; delta still valid)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    endpoints = args.base_urls.split(",")
    tkw = None if args.thinking else {"enable_thinking": False}

    items = load_mcq(args.type, args.limit)
    n = len(items)
    tasks = [(p, s) for p in range(n) for s in range(args.samples)]

    def worker(i, ep):
        p, _ = tasks[i]
        txt, fin = chat(ep, make_prompt(items[p]), args.max_tokens, args.temp, template_kwargs=tkw)
        pred = extract_letter(txt, len(items[p]["opts"]))
        return {"p": p, "correct": int(pred == items[p]["gold"]), "finish": fin}

    t0 = time.time()
    res = run_concurrent(len(tasks), endpoints, args.concurrency, worker, label=args.type)
    hits = [0] * n
    for r in res:
        if r and "correct" in r:
            hits[r["p"]] += r["correct"]
    per = [{"idx": p, "gold": items[p]["gold"], "acc": hits[p] / args.samples} for p in range(n)]
    out = {"type": args.type, "n": n, "samples": args.samples,
           "accuracy": sum(x["acc"] for x in per) / max(n, 1),
           "minutes": (time.time() - t0) / 60, "per": per}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n{args.type}: accuracy={out['accuracy']:.4f} (n={n}, k={args.samples}, {out['minutes']:.1f} min) -> {args.out}")


if __name__ == "__main__":
    main()
