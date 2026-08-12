#!/usr/bin/env python
"""gen_student_rollouts.py — Stage B phase 1: the STUDENT generates its own long reasoning rollouts from
chat/STEM prefixes. These are the drifted, over-long trajectories the teacher will then correct (offline
semi-on-policy / DAgger — DistiLLM arXiv:2402.03898). Student loaded ALONE (teacher not) → no OOM, so we can
use LONG rollouts. NOTE: close-RATE is 31%@1792 vs 49%@2560 (1786 is the median close POSITION among closers,
not a close rate) — and Stage B does NOT want only closers: the over-long UNCLOSED rollouts carry the signal.

Output = JSON list of token-id lists (>= --seq), consumed by the normal teacher-cache precompute + E2E.

  ORIG=output_4b/untied_4b STUDENT=output_4b/combined2560_g64q8/e2e/modified_model \
  ./.venv/bin/python src/gen_student_rollouts.py --n 900 --max-new 2496 --seq 2560 --out output_4b/student_rollouts.json
"""
import argparse, json, os, random, torch
from transformers import AutoTokenizer
from e2e_qp_distill import build_student, BLOCK_SIZE

THINK_CLOSE, EOS = 248069, 248046


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default=os.environ.get("ORIG", "output_4b/untied_4b"))
    ap.add_argument("--student", default=os.environ.get("STUDENT", "output_4b/combined2560_g64q8/e2e/modified_model"))
    ap.add_argument("--prompts", default="output_4b/chat_pool_rc2560.json",
                    help="chat/STEM seqs whose PREFIXES seed the rollouts (uses --prefix tokens of each)")
    ap.add_argument("--prefix", type=int, default=64, help="prompt-prefix tokens to condition on")
    ap.add_argument("--n", type=int, default=900)
    ap.add_argument("--max-new", type=int, default=2496, help="prefix64+2496=2560 (one seq). close-rate 49 pct @2560 vs 31 pct @1792; unclosed over-long rollouts are the Stage-B signal")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.6, help="match deployment sampling (NOT 1.0)")
    ap.add_argument("--seq", type=int, default=2560, help="pack rollouts to this length")
    ap.add_argument("--out", default="output_4b/student_rollouts.json")
    ap.add_argument("--device", default="cuda:0", help="GPU for this shard")
    ap.add_argument("--shard", type=int, default=0, help="this shard index (data-parallel across GPUs)")
    ap.add_argument("--nshards", type=int, default=1, help="total shards; each takes a strided slice")
    args = ap.parse_args()
    random.seed(0)                                      # same seed => identical prefix list across shards

    tok = AutoTokenizer.from_pretrained(args.orig, trust_remote_code=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Prefixes must END at the `…assistant\n<think>\n` boundary — that is where the student begins its own
    # reasoning and where the over-thinking/no-close failure happens. Taking s[:prefix] instead would grab
    # arbitrary MID-reasoning fragments (the packed pool's seq starts do NOT align with prompt starts: only
    # 1/917 begins at <|im_start|>), and the student would merely continue someone else's text.
    THINK_OPEN, NL = 248068, 198
    pool = json.load(open(args.prompts))
    random.shuffle(pool)
    prefixes = []
    for s in pool:
        for p in (i for i, t in enumerate(s) if t == THINK_OPEN):
            end = p + 2 if p + 1 < len(s) and s[p + 1] == NL else p + 1   # include <think> (and its newline)
            if end >= args.prefix:                       # need a full-length prefix for clean batching
                prefixes.append(s[end - args.prefix:end])
            if len(prefixes) >= args.n:
                break
        if len(prefixes) >= args.n:
            break
    if not prefixes:
        raise SystemExit(f"no <think> boundaries with >={args.prefix} tokens of context in {args.prompts}")
    while len(prefixes) < args.n:                        # cycle if the pool yielded fewer than n
        prefixes.append(prefixes[len(prefixes) % max(1, len(prefixes))])
    prefixes = prefixes[:args.n]
    if args.nshards > 1:                                # data-parallel: strided slice for this shard
        prefixes = prefixes[args.shard::args.nshards]

    print(f"student rollouts: {len(prefixes)} prefixes x max_new {args.max_new} (temp {args.temp}) shard {args.shard}/{args.nshards} on {args.device}", flush=True)
    print(f"  student={args.student}", flush=True)
    st, _ = build_student(args.student, args.orig, BLOCK_SIZE, args.device); st.eval(); st.config.use_cache = True

    rollouts, n_closed = [], 0
    with torch.no_grad():
        for b0 in range(0, len(prefixes), args.batch):
            chunk = prefixes[b0:b0 + args.batch]
            ii = torch.tensor(chunk, dtype=torch.long).to(args.device)
            am = torch.ones_like(ii)                    # all prefixes are exactly --prefix long → no padding
            out = st.generate(ii, attention_mask=am, max_new_tokens=args.max_new, do_sample=True,
                              temperature=args.temp, top_p=0.95, top_k=20, eos_token_id=EOS, pad_token_id=EOS)
            for j in range(out.shape[0]):
                full = out[j].tolist()
                gen = full[ii.shape[1]:]
                if EOS in gen:                          # trim batch padding past the stop
                    full = full[: ii.shape[1] + gen.index(EOS) + 1]
                rollouts.append(full)
                n_closed += (THINK_CLOSE in full)
            if (b0 // args.batch + 1) % 5 == 0:
                print(f"  {b0+len(chunk)}/{len(prefixes)} | closed {n_closed}", flush=True)

    avg = sum(len(r) for r in rollouts) / len(rollouts)
    print(f"generated {len(rollouts)} rollouts, avg len {avg:.0f}, closed-think {n_closed} "
          f"({100*n_closed/len(rollouts):.0f}%)", flush=True)

    # pack to --seq (rollouts are the student's OWN trajectories; the teacher will score them next)
    random.shuffle(rollouts)
    seqs, buf = [], []
    for r in rollouts:
        buf.extend(r)
        while len(buf) >= args.seq:
            seqs.append(buf[:args.seq]); buf = buf[args.seq:]
    json.dump(seqs, open(args.out, "w"))
    cl = sum(1 for s in seqs if THINK_CLOSE in s)
    print(f"wrote {len(seqs)} packed seqs of {args.seq} ({len(seqs)*args.seq/1e6:.2f}M tok), "
          f"{cl} with </think> ({100*cl/max(len(seqs),1):.0f}%) -> {args.out}")


if __name__ == "__main__":
    main()
