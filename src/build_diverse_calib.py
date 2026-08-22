#!/usr/bin/env python
"""Build a DIVERSE calibration set from the 17-source manifest (5 HF Nemotron-CC-v2/Math/Code +
12 local CTM data_cache folders), all uniform `text` schema. Reads only the row-groups needed per
source (pyarrow iter_batches — NOT a full multi-GB load; no streaming penalty since files are local/
cached), tokenizes, chunks to --seq, samples each source's token quota, shuffles, and writes a calib
JSON (+ a disjoint held-out slice). Scale via --tokens (0.5M / 4M / 16M); same proportions at each.

  python build_diverse_calib.py --tokens 500000 --out output_recovery/calib_diverse_0p5M.json
"""
import argparse, json, os, glob, random, sys
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

# The 17-source manifest now lives in calib_sources.py so tools/fetch_data.py (which downloads the
# parquets) and this builder (which reads them) share one definition. SOURCES is unchanged.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calib_sources import (SOURCES, HUB, dirname as src_dirname,  # noqa: E402
                           MAX_UNIQUE_TRAIN_TOKENS, BINDING_SOURCE)

ap = argparse.ArgumentParser()
ap.add_argument("--orig-model", required=True, help="original HF model snapshot (for the tokenizer)")
ap.add_argument("--ctm-data", required=True, help="local CTM data_cache dir (the 12 'ctm' sources)")
ap.add_argument("--tokens", type=int, default=500_000, help="total calibration tokens (train)")
ap.add_argument("--seq", type=int, default=1024)
ap.add_argument("--eval-frac", type=float, default=0.10, help="extra held-out fraction (disjoint)")
ap.add_argument("--out", default="output_recovery/calib_diverse.json")
ap.add_argument("--eval-out", default="output_recovery/calib_diverse_eval.json")
ap.add_argument("--chat-frac", type=float, default=0.0,
                help="fraction of --tokens drawn from CHAT <think> rollouts (--chat-src); the remaining "
                     "(1-chat_frac) is split across the 17 generic sources by their fractions. 0 = pure "
                     "generic (original behavior). Lets you dial total size (--tokens) AND chat share here.")
ap.add_argument("--chat-src", default=None,
                help="JSON of pre-tokenized chat seqs (uses those containing <think>=248068); oversampled "
                     "with replacement if the chat quota exceeds the unique pool. Train-only (never held-out).")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()
random.seed(args.seed)
assert abs(sum(s[3] for s in SOURCES) - 1.0) < 1e-6, "fractions must sum to 1"
ORIG = args.orig_model
CTM = os.path.expanduser(args.ctm_data)

tok = AutoTokenizer.from_pretrained(ORIG, trust_remote_code=True)

def resolve(label, kind, loc):
    """Local `data/` first, Hub as fallback.

    Prefer $CTM_DATA/<dirname>/*.parquet for EVERY source (both kinds) so a repo populated by
    tools/fetch_data.py is fully self-contained and needs no network. The old behaviour is preserved
    as the fallback: 'hf' sources still download on demand, 'ctm' sources still error if absent.
    """
    fs = sorted(glob.glob(f"{CTM}/{src_dirname(label)}/*.parquet"))
    if fs:
        return fs[0]
    if kind == "hf":
        return hf_hub_download(loc[0], loc[1], repo_type="dataset")
    raise FileNotFoundError(
        f"no parquet in {CTM}/{src_dirname(label)} — run `python tools/fetch_data.py` to populate it")

def chunks_from(path, need_seqs, seq):
    """Up to need_seqs packed token-id chunks of length seq. PACK documents into one token stream
    and cut seq-length pieces — most docs are < seq tokens, so per-doc chunking would discard them
    and force a full-file scan. Packing uses every token and lets us stop after ~need_seqs*seq tokens
    (a few hundred rows), so we read only the first row-group(s)."""
    out, buf, start = [], [], 0
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=512, columns=["text"]):
        for t in batch.column("text").to_pylist():
            if not t:
                continue
            buf.extend(tok(t, add_special_tokens=False).input_ids)
            buf.append(tok.eos_token_id if tok.eos_token_id is not None else 0)  # doc separator
            while len(buf) - start >= seq:
                out.append(buf[start:start + seq]); start += seq
                if len(out) >= need_seqs:
                    return out
    return out

train, ev = [], []
gen_scale = 1.0 - args.chat_frac                          # generic sources share the non-chat remainder
# Unique-token ceiling (see calib_sources.py): past this the scarcest source (Economics) runs dry and
# its `got` silently falls short of `target`, quietly shifting the mixture away from the documented
# fractions. It does NOT reuse tokens — it under-delivers — but the resulting draw is no longer the
# documented mixture, so warn loudly rather than let a too-large --tokens pass unnoticed.
_gen_tokens = args.tokens * (1.0 - args.chat_frac)
if _gen_tokens > MAX_UNIQUE_TRAIN_TOKENS:
    print(f"!! WARNING: {_gen_tokens/1e6:.0f}M generic tokens requested but the mixture caps at "
          f"{MAX_UNIQUE_TRAIN_TOKENS/1e6:.0f}M (binding source: {BINDING_SOURCE}).\n"
          f"!! Sources will fall short of quota and the mixture will NOT match the documented "
          f"fractions — results are not comparable to recorded numbers.", flush=True)

print(f"building diverse calib: {args.tokens} train tokens, seq {args.seq}, +{int(args.eval_frac*100)}% held-out"
      f"{f', {int(args.chat_frac*100)}% CHAT' if args.chat_frac>0 else ''}", flush=True)
for label, kind, loc, frac in SOURCES:
    quota_tok = int(frac * gen_scale * args.tokens)
    n_train = max(1, quota_tok // args.seq)
    n_eval = max(1, int(n_train * args.eval_frac))
    path = resolve(label, kind, loc)
    got = chunks_from(path, n_train + n_eval, args.seq)
    random.shuffle(got)
    tr, ev_ = got[:n_train], got[n_train:n_train + n_eval]
    train += tr; ev += ev_
    print(f"  {label:22s} frac {frac:.2f}  target {n_train} seqs  got {len(tr)} train + {len(ev_)} eval  "
          f"({'HF' if kind=='hf' else 'CTM'} {os.path.basename(path)})", flush=True)

# CHAT source: the <think> rollouts that fix the free-gen collapse (train-only; kept out of held-out eval)
if args.chat_frac > 0 and args.chat_src:
    THINK_OPEN = 248068
    pool = [s[:args.seq] for s in json.load(open(args.chat_src)) if THINK_OPEN in s and len(s) >= args.seq]
    if not pool:
        raise SystemExit(f"--chat-src {args.chat_src} has no seqs with <think>=248068 of len>={args.seq}")
    n_chat = max(1, int(args.chat_frac * args.tokens) // args.seq)
    chat_seqs = pool[:n_chat] if n_chat <= len(pool) else \
                pool + [random.choice(pool) for _ in range(n_chat - len(pool))]
    train += chat_seqs
    tag = f"oversample {n_chat/len(pool):.1f}x" if n_chat > len(pool) else f"from {len(pool)} unique"
    print(f"  {'CHAT-<think>':22s} frac {args.chat_frac:.2f}  target {n_chat} seqs  got {len(chat_seqs)}  ({tag})",
          flush=True)

random.shuffle(train); random.shuffle(ev)
json.dump(train, open(args.out, "w"))
json.dump(ev, open(args.eval_out, "w"))
print(f"\nwrote {len(train)} train seqs (~{len(train)*args.seq/1e6:.2f}M tok) -> {args.out}")
print(f"wrote {len(ev)} held-out seqs -> {args.eval_out}")
