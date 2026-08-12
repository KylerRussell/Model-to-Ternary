#!/usr/bin/env python
"""sweep_maxnew_close.py — how high must --max-new be for the STEM-derived chat rollouts to CLOSE </think>?
Generate ONCE at a large cap, record the position of the first </think>(=248069) in each rollout, then read
off the close-rate at every shorter cutoff from that single pass (no per-length regeneration). Prompts are the
same derived-from-generic STEM-heavy distribution the real chat pool uses (build_chat_calib --n-derived), so
the curve is representative. Target: the max-new where close-rate >= ~0.5.

  N=96 MAXNEW=4096 ./.venv/bin/python src/sweep_maxnew_close.py
"""
import os, json, random, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("MODEL", "output_4b/untied_4b")
GENERIC = os.environ.get("GENERIC", "output_4b/test1_data/calib_16M.json")
N = int(os.environ.get("N", "96"))
MAXNEW = int(os.environ.get("MAXNEW", "4096"))
BATCH = int(os.environ.get("BATCH", "16"))
TEMP = float(os.environ.get("TEMP", "0.7"))
THINK_CLOSE, EOS = 248069, 248046
CUTOFFS = [768, 1024, 1280, 1792, 2560, 3328, 4096]
random.seed(0)

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
tok.padding_side = "left"
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

# derived STEM-heavy prompts (same style as build_chat_calib): "Continue and explain: <generic snippet>"
gen = json.load(open(GENERIC)); random.shuffle(gen)
prompts = []
for s in gen:
    txt = tok.decode(s[:40]).strip().replace("\n", " ")
    if len(txt) > 20:
        prompts.append(f"Continue and explain: {txt}")
    if len(prompts) >= N:
        break

print(f"sweep max-new close-rate: {len(prompts)} STEM-derived prompts, gen@{MAXNEW} temp {TEMP}", flush=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16).to("cuda:0").eval()

close_pos = []   # position (in generated tokens) of first </think>, or None
gen_lens = []
with torch.no_grad():
    for b0 in range(0, len(prompts), BATCH):
        chunk = prompts[b0:b0 + BATCH]
        msgs = [[{"role": "user", "content": p}] for p in chunk]
        try:
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=True,
                                          return_tensors="pt", tokenize=True, padding=True, return_dict=True)
        except TypeError:
            enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                          tokenize=True, padding=True, return_dict=True)
        ii = enc["input_ids"].to("cuda:0"); am = enc["attention_mask"].to("cuda:0")
        out = model.generate(ii, attention_mask=am, max_new_tokens=MAXNEW, do_sample=True,
                             temperature=TEMP, top_p=0.95, top_k=20, eos_token_id=EOS, pad_token_id=EOS)
        for j in range(out.shape[0]):
            g = out[j, ii.shape[1]:].tolist()
            if EOS in g:                              # strip batch padding past the stop
                g = g[: g.index(EOS) + 1]
            gen_lens.append(len(g))
            close_pos.append(g.index(THINK_CLOSE) if THINK_CLOSE in g else None)
        print(f"  {b0+len(chunk)}/{len(prompts)}", flush=True)

n = len(close_pos)
avg_len = sum(gen_lens) / n
closed = [p for p in close_pos if p is not None]
print("\n" + "=" * 56)
print(f"generated {n} | avg gen len {avg_len:.0f} | closed-ever {len(closed)}/{n} ({100*len(closed)/n:.0f}%)")
print(f"  {'max-new':>8s} {'close-rate':>11s}  (frac of rollouts whose </think> lands <= cutoff)")
for c in CUTOFFS:
    r = sum(1 for p in closed if p < c) / n
    flag = " <- >=50%" if r >= 0.5 else ""
    print(f"  {c:>8d} {100*r:>10.0f}%{flag}")
print("=" * 56)
if closed:
    import statistics
    print(f"  median close position: {int(statistics.median(closed))} tok | "
          f"75th pct: {int(sorted(closed)[int(0.75*len(closed))])} tok")
