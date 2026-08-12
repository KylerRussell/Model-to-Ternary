#!/usr/bin/env python
"""augment_corpus.py — Stage-1 (researcher round-4) data augmentation for the exposure-bias residual. Produces
TWO auxiliary training streams, each trained with its OWN loss in e2e_qp_distill (no FP teacher rollouts):

  ditto_data.json  — DITTO pseudo-repetitive sequences (arXiv:2206.02369): prefix + SPAN repeated N times.
      Trained with the DITTO repetition-penalization loss so the model learns to EXPONENTIALLY DECAY the
      probability of continuing a repetition as its count grows — directly dissolves the self-reinforcement
      attractor. Entry: {ids, span_start, span_len, n_copies}.

  commit_data.json — truncated-trace </think>-commit supervision (budget-forcing baked into data; s1
      arXiv:2501.19393): FP reasoning truncated at varied depths, then </think> + the REAL answer + EOS
      appended. Trained with HARD-TARGET CE at the forced </think> + answer tokens so the model learns to STOP
      and commit from many mid-reasoning states. Entry: {ids, close_pos} (close_pos = index of </think>).

  python augment_corpus.py --traces output_4b/reasoning_traces.json --out-dir output_4b \
     --ditto-copies 5 --ditto-n 400 --commit-depths 0.2,0.4,0.6,0.8
"""
import argparse, json, random, os

THINK_OPEN, THINK_CLOSE, EOS, NL = 248068, 248069, 248046, 198


def spans_from_reasoning(ids, think_open, think_close, lo=8, hi=40):
    """split the reasoning region into newline-delimited lines; keep lines of lo..hi tokens as candidate spans.
    returns list of (start_idx, span_tokens)."""
    r0 = think_open + 1
    r1 = think_close if think_close > r0 else len(ids)
    reg = ids[r0:r1]
    spans, cur, cstart = [], [], r0
    for i, t in enumerate(reg):
        cur.append(t)
        if t == NL or i == len(reg) - 1:
            if lo <= len(cur) <= hi:
                spans.append((cstart, list(cur)))
            cstart = r0 + i + 1
            cur = []
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", default="output_4b/reasoning_traces.json")
    ap.add_argument("--extra-traces", default=None, help="optional 2nd traces json to pool in")
    ap.add_argument("--out-dir", default="output_4b")
    ap.add_argument("--ditto-copies", type=int, default=5)
    ap.add_argument("--ditto-n", type=int, default=400, help="# DITTO pseudo sequences")
    ap.add_argument("--ditto-prefix-cap", type=int, default=256)
    ap.add_argument("--commit-depths", default="0.2,0.4,0.6,0.8")
    ap.add_argument("--commit-answer-cap", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    traces = json.load(open(args.traces))["traces"]
    if args.extra_traces and os.path.exists(args.extra_traces):
        traces = traces + json.load(open(args.extra_traces))["traces"]
    closed = [t for t in traces if 0 <= t.get("think_close", -1) < len(t["ids"]) - 3]
    print(f"{len(traces)} traces ({len(closed)} closed) ", flush=True)

    # ---- DITTO pseudo-repetitive data ----
    all_spans = []
    for t in traces:
        to = t.get("think_open", t["prompt_len"] - 1)
        tc = t.get("think_close", -1)
        for st, span in spans_from_reasoning(t["ids"], to, tc):
            all_spans.append((t["ids"], st, span))
    random.shuffle(all_spans)
    ditto = []
    for ids, st, span in all_spans[: args.ditto_n]:
        prefix = ids[max(0, st - args.ditto_prefix_cap): st]        # realistic context up to the span
        seq = list(prefix) + span * args.ditto_copies
        ditto.append({"ids": seq, "span_start": len(prefix), "span_len": len(span),
                      "n_copies": args.ditto_copies})
    json.dump(ditto, open(f"{args.out_dir}/ditto_data.json", "w"))
    print(f"wrote {len(ditto)} DITTO seqs (span x{args.ditto_copies}) -> {args.out_dir}/ditto_data.json")

    # ---- truncated-trace commit data ----
    depths = [float(x) for x in args.commit_depths.split(",")]
    commit = []
    for t in closed:
        ids = t["ids"]; to = t.get("think_open", t["prompt_len"] - 1); tc = t["think_close"]
        answer = ids[tc + 1: tc + 1 + args.commit_answer_cap]
        if EOS not in answer:
            answer = answer + [EOS]
        rlen = tc - (to + 1)
        if rlen < 8:
            continue
        # full (natural) close + several truncations
        for d in [1.0] + depths:
            cut = to + 1 + max(4, int(rlen * d))
            reasoning = ids[to + 1: cut]
            seq = ids[: to + 1] + reasoning + [THINK_CLOSE, NL] + answer
            close_pos = (to + 1) + len(reasoning)              # index of THINK_CLOSE in seq
            commit.append({"ids": seq, "close_pos": close_pos})
    random.shuffle(commit)
    json.dump(commit, open(f"{args.out_dir}/commit_data.json", "w"))
    lens = [len(c["ids"]) for c in commit]
    print(f"wrote {len(commit)} commit seqs (depths {depths}+full, len {min(lens)}-{max(lens)}) "
          f"-> {args.out_dir}/commit_data.json")


if __name__ == "__main__":
    main()
