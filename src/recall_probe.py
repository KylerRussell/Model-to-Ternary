#!/usr/bin/env python3
"""
Multi-key associative-recall probe for the Gated-DeltaNet 4B — the cheap retrieval signal the
moonshot checklist asks for (C2 smudging / C3 contraction both act on DeltaNet memory, which
seq-1024 ppl barely exercises).

Task: present N (key = value) pairs, then query one key; the model must recall its value.
Keys/values are single tokens (leading-space word tokens mined from the vocab), so scoring is a
SINGLE forward pass — argmax of the next-token logits at the answer position (no generation).
We report:
  - full  : top-1 over the whole vocab == target value
  - restr : target value has the highest logit among the N in-context value tokens (isolates
            retrieval from calibration/formatting)
Broken down by N (memory load) and by query position (early keys must survive longer → probes
long-range retention, the FM3/C3 axis). Sweeping N probes multi-key interference (FM2/C2).

Loads FP (--kind fp, AutoModel) or ternary (--kind ternary, build_student). Honors CONTRACTION_EPS
(C3 clamp) via contraction_clamp.maybe_install.
"""
import os, sys, argparse, random, math
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM, AutoTokenizer
import contraction_clamp


def mine_word_tokens(tok, need, seed=0):
    """Single tokens that decode to a leading-space lowercase alphabetic word (stable, distinct)."""
    pool = []
    for tid in range(min(len(tok), 152000)):
        s = tok.convert_ids_to_tokens(tid)
        if s is None:
            continue
        d = tok.decode([tid])
        if d.startswith(" ") and d[1:].isalpha() and d[1:].islower() and 4 <= len(d[1:]) <= 9:
            pool.append((tid, d))
    rng = random.Random(seed)
    rng.shuffle(pool)
    if len(pool) < need:
        raise RuntimeError(f"only mined {len(pool)} word tokens, need {need}")
    return pool[:need]


def build_trial(tok, keys, vals, n_pairs, rng):
    """Return (input_ids list, target_value_id, query_index, list of in-context value ids)."""
    idx = rng.sample(range(len(keys)), n_pairs)
    vidx = rng.sample(range(len(vals)), n_pairs)
    pair_k = [keys[i] for i in idx]
    pair_v = [vals[i] for i in vidx]
    q = rng.randrange(n_pairs)                      # which pair we query
    intro = tok("Here is a list of items and their codes.\n", add_special_tokens=True).input_ids
    ids = list(intro)
    sep = tok(" is", add_special_tokens=False).input_ids
    nl = tok(".\n", add_special_tokens=False).input_ids
    for (ktid, _), (vtid, _) in zip(pair_k, pair_v):
        ids += [ktid] + sep + [vtid] + nl
    qintro = tok("\nQuestion: the code for", add_special_tokens=False).input_ids
    ids += qintro + [pair_k[q][0]] + sep
    return ids, pair_v[q][0], q, [v[0] for v in pair_v]


@torch.no_grad()
def run(model, tok, dev, n_pairs_list, trials, seed):
    keys = mine_word_tokens(tok, 256, seed=seed)
    vals = mine_word_tokens(tok, 256, seed=seed + 777)
    # ensure disjoint key/value token ids
    kset = {t for t, _ in keys}
    vals = [(t, d) for t, d in vals if t not in kset][:200]
    keys = keys[:200]
    rng = random.Random(seed + 12345)
    results = {}
    for N in n_pairs_list:
        full_hit = restr_hit = 0
        # 3 position buckets: early / mid / late third of the pair list
        buck = {"early": [0, 0], "mid": [0, 0], "late": [0, 0]}
        for _ in range(trials):
            ids, tgt, q, vids = build_trial(tok, keys, vals, N, rng)
            x = torch.tensor([ids], device=dev)
            logits = model(x).logits[0, -1].float()
            pred = int(logits.argmax())
            full = int(pred == tgt)
            vids_t = torch.tensor(vids, device=logits.device)
            restr = int(vids[int(logits[vids_t].argmax())] == tgt)
            full_hit += full
            restr_hit += restr
            b = "early" if q < N / 3 else ("mid" if q < 2 * N / 3 else "late")
            buck[b][0] += restr
            buck[b][1] += 1
        results[N] = {
            "full": full_hit / trials,
            "restr": restr_hit / trials,
            "pos": {k: (v[0] / v[1] if v[1] else float("nan")) for k, v in buck.items()},
        }
    return results


def load_model(kind, path, orig_cfg, dev):
    if kind == "ternary":
        from e2e_qp_distill import build_student, BLOCK_SIZE
        model, _ = build_student(path, orig_cfg, BLOCK_SIZE, dev)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, dtype=torch.bfloat16).to(dev)
    model.eval()
    contraction_clamp.maybe_install(model)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--kind", choices=["fp", "ternary"], required=True)
    ap.add_argument("--orig-config", default=None)
    ap.add_argument("--tok", default=None, help="tokenizer path (default: --orig-config or --model)")
    ap.add_argument("--pairs", default="8,16,32,48")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    dev = "cuda:0"
    tok_path = args.tok or args.orig_config or args.model
    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    n_list = [int(x) for x in args.pairs.split(",")]
    model = load_model(args.kind, args.model, args.orig_config, dev)
    res = run(model, tok, dev, n_list, args.trials, args.seed)
    tag = args.tag or f"{args.kind}:{os.path.basename(args.model.rstrip('/'))}"
    print(f"\n===== RECALL PROBE [{tag}]  eps={os.environ.get('CONTRACTION_EPS','0')}  trials={args.trials} =====")
    print(f"  {'Npairs':>7} {'full':>7} {'restr':>7} {'early':>7} {'mid':>7} {'late':>7}")
    for N in n_list:
        r = res[N]
        print(f"  {N:>7} {r['full']:>7.3f} {r['restr']:>7.3f} "
              f"{r['pos']['early']:>7.3f} {r['pos']['mid']:>7.3f} {r['pos']['late']:>7.3f}")
    avg_restr = sum(res[N]['restr'] for N in n_list) / len(n_list)
    print(f"  MEAN restr over N = {avg_restr:.4f}")


if __name__ == "__main__":
    main()
