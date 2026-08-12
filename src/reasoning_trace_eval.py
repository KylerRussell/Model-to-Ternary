#!/usr/bin/env python
"""reasoning_trace_eval.py — the reasoning-trace proxy (researcher round-9 §H). Teacher-force FP and ternary
over FP-generated long-CoT traces and report top-1 agreement + top-64 KL SEGMENTED, because generic teacher-
forced eval2k hides the compounding generation cost:
  - phase:     prompt tokens vs generation tokens
  - position:  generation split into early / mid / LATE thirds (late = where KV-cache error compounds)
  - block:     inside <think>..</think> vs post-think answer
Plus structural-validity distribution (from the trace file). The whole point is LATE-position + answer
agreement, which the single-token generic metric is blind to.

Env: ORIG, FP_DIR (rotated FP), E2E_MODEL (ternary), TRACES (from gen_reasoning_traces.py), SEQ_CAP.
FP->cuda:0, tern->cuda:1. Streams per-trace (no logit cache)."""
import os, json, math, torch
import torch.nn.functional as F
from collections import Counter
from transformers import AutoModelForCausalLM
from e2e_qp_distill import build_student, BLOCK_SIZE

ORIG = os.environ.get("ORIG", "output_4b/untied_4b")
FP_DIR = os.environ.get("FP_DIR", "output_4b/rot/modified_model")
E2E = os.environ.get("E2E_MODEL", "output_4b/ternhead/e2e/modified_model")
TRACES = os.environ.get("TRACES", "output_4b/reasoning_traces.json")
SEQ_CAP = int(os.environ.get("SEQ_CAP", "2048"))
THINK_OPEN, THINK_CLOSE = 248068, 248069
CHUNK = 512

data = json.load(open(TRACES))
traces = data["traces"]
print(f"reasoning-trace eval: {len(traces)} traces | FP={FP_DIR} vs tern={E2E}", flush=True)
print(f"  structural: {dict(Counter(t['structural'] for t in traces))}", flush=True)

fp = AutoModelForCausalLM.from_pretrained(FP_DIR, trust_remote_code=True, dtype=torch.bfloat16).to("cuda:0").eval()
fp.config.use_cache = False
st, _ = build_student(E2E, ORIG, BLOCK_SIZE, "cuda:1"); st.eval(); st.config.use_cache = False

# accumulators: segment -> [sum_kl, n_kl, flips, n_tok]
SEGS = ["prompt", "gen", "gen_early", "gen_mid", "gen_late", "think", "answer", "ALL"]
acc = {s: [0.0, 0, 0, 0] for s in SEGS}

def add(seg, kl_vec, flip_vec):
    a = acc[seg]
    a[0] += float(kl_vec.sum()); a[1] += kl_vec.numel()
    a[2] += int(flip_vec.sum()); a[3] += flip_vec.numel()

with torch.no_grad():
    for ti, t in enumerate(traces):
        ids = torch.tensor(t["ids"][:SEQ_CAP], dtype=torch.long)
        T = ids.shape[0]
        if T < 4:
            continue
        plen = min(t["prompt_len"], T)
        tc = t["think_close"] if 0 <= t["think_close"] < T else T          # answer starts after </think>
        x = ids.unsqueeze(0)
        lf = fp(x.to("cuda:0")).logits[0, :-1]                              # [T-1, V]
        lt = st(x.to("cuda:1")).logits[0, :-1]
        # per-token KL(fp||tern) + argmax flip, chunked in fp32
        kl = torch.empty(T - 1); fl = torch.empty(T - 1)
        for c0 in range(0, T - 1, CHUNK):
            a = lf[c0:c0+CHUNK].float().to("cuda:1"); b = lt[c0:c0+CHUNK].float()
            pf = F.softmax(a, -1)
            kl[c0:c0+a.shape[0]] = (pf * (F.log_softmax(a, -1) - F.log_softmax(b, -1))).sum(-1).cpu()
            fl[c0:c0+a.shape[0]] = (a.argmax(-1) != b.argmax(-1)).float().cpu()
            del a, b, pf
        pos = torch.arange(T - 1)                                          # predicting token pos+1
        gen_mask = pos >= plen - 1
        add("ALL", kl, fl)
        add("prompt", kl[~gen_mask], fl[~gen_mask])
        add("gen", kl[gen_mask], fl[gen_mask])
        g_idx = pos[gen_mask]
        if g_idx.numel() > 0:
            g0, g1 = int(g_idx[0]), int(g_idx[-1]); span = max(1, g1 - g0)
            for lo, hi, seg in [(0.0, 1/3, "gen_early"), (1/3, 2/3, "gen_mid"), (2/3, 1.01, "gen_late")]:
                m = gen_mask & (pos >= g0 + lo*span) & (pos < g0 + hi*span)
                add(seg, kl[m], fl[m])
        think_mask = gen_mask & (pos < tc - 1)                             # inside <think>..</think>
        ans_mask = pos >= tc - 1
        add("think", kl[think_mask], fl[think_mask])
        add("answer", kl[ans_mask], fl[ans_mask])
        del lf, lt, kl, fl
        torch.cuda.empty_cache()
        if (ti + 1) % 16 == 0:
            a = acc["gen_late"]
            print(f"  {ti+1}/{len(traces)}  late-gen agree={100*(1-a[2]/max(a[3],1)):.2f}%", flush=True)

print("\n" + "=" * 72)
print(f"  {'segment':10s} {'mean KL':>10s} {'agreement':>11s} {'%flips':>9s} {'tokens':>10s}")
for s in SEGS:
    sk, nk, fz, nt = acc[s]
    if nt == 0:
        continue
    print(f"  {s:10s} {sk/max(nk,1):>10.4f} {100*(1-fz/nt):>10.2f}% {100*fz/max(nt,1):>8.2f}% {nt:>10d}")
print("=" * 72)
out = {s: {"kl": acc[s][0]/max(acc[s][1],1), "agree": 100*(1-acc[s][2]/max(acc[s][3],1)),
           "flips": 100*acc[s][2]/max(acc[s][3],1), "tokens": acc[s][3]} for s in SEGS if acc[s][3] > 0}
if os.environ.get("RT_OUT"):
    json.dump(out, open(os.environ["RT_OUT"], "w"), indent=1)
    print(f"wrote {os.environ['RT_OUT']}")
