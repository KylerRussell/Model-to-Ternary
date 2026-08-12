#!/usr/bin/env python
"""freegen_gate.py — the CHEAP free-gen collapse gate (researcher round-2, Metrics 1-4). Teacher-forced KL is
PROVEN BLIND to the <think>-loop; these are the cheap proxies that DO detect the context-specific distribution
collapse, computed ONLY at induced chat-context boundary positions (never generic mid-sequence). Metric 5 (the
served-GGUF loop rate) is the non-negotiable ground truth and is checked BY HAND, not here.

At each boundary position we compare FP (teacher) vs ternary next-token logits:
  M1 entropy ratio   H_tern / H_FP            flag < 0.50   (collapse = destroyed entropy)
  M2 magnitude ratio max|logit_tern|/max|FP|  flag < 0.70   (our ~16/~30 = 0.53 fails)
  M3 argmax agree    frac(argmax_tern==FP)    need >= 0.90
  M4 top1-top2 margin (ternary)               flag < 1.0    (0.2 margin => GGUF tips it)

Boundary contexts:
  start_think : chat-template end `...assistant\n<think>\n`  -> predict 1st thinking token  (THE failing ctx)
  post_think  : the answer-start position right after `</think>` in FP traces (if TRACES given)

Env: ORIG (tokenizer+config), FP_DIR (rotated FP teacher), E2E_MODEL (ternary HF dir),
     TRACES (optional, from gen_reasoning_traces.py), GATE_OUT (optional json), N_EXTRA_CHAT (plain prompts).
FP->cuda:0, tern->cuda:1."""
import os, json, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from e2e_qp_distill import build_student, BLOCK_SIZE
from gen_reasoning_traces import PROMPTS as REASON_PROMPTS

ORIG = os.environ.get("ORIG", "output_4b/untied_4b")
FP_DIR = os.environ.get("FP_DIR", "output_4b/rot/modified_model")
E2E = os.environ.get("E2E_MODEL", "output_4b/lmheadfix/e2e/modified_model")
TRACES = os.environ.get("TRACES", "output_4b/reasoning_traces.json")
GATE_OUT = os.environ.get("GATE_OUT", "")
THINK_OPEN, THINK_CLOSE = 248068, 248069

# a few plain (non-reasoning) chat prompts — the boundary must hold generally, not just on math
PLAIN = [
    "What is the capital of France?",
    "Write a haiku about the ocean.",
    "Explain photosynthesis in one sentence.",
    "Who wrote Pride and Prejudice?",
    "Give me three tips for better sleep.",
    "Translate 'good morning' into Spanish.",
    "What is 2 plus 2?",
    "Recommend a book for a rainy afternoon.",
]

tok = AutoTokenizer.from_pretrained(ORIG, trust_remote_code=True)


def start_think_ids(prompt):
    """chat-template sequence ending exactly at `assistant\n<think>\n` (add_generation_prompt=True)."""
    msgs = [{"role": "user", "content": prompt}]
    try:
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=True,
                                      return_tensors="pt", tokenize=True)
    except TypeError:
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", tokenize=True)
    if not torch.is_tensor(ids):
        ids = ids["input_ids"] if "input_ids" in ids else torch.tensor(ids)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    return ids[0].tolist()


# ---- build the boundary position list: (ids_prefix, tag) where we score the LAST position's next-token ----
positions = []
prompts = list(REASON_PROMPTS) + PLAIN
for p in prompts:
    ids = start_think_ids(p)
    # sanity: the sequence should end with <think>\n (THINK_OPEN then a newline token). we score the last pos.
    positions.append((ids, "start_think"))

# post-</think> answer-start positions from FP traces (reuse existing traces; cheap, on-distribution)
if os.path.exists(TRACES):
    tr = json.load(open(TRACES))["traces"]
    for t in tr:
        tc = t["think_close"]
        full = t["ids"]
        if 0 <= tc < len(full) - 2:
            # score the position that predicts the FIRST answer token (right after </think>\n)
            positions.append((full[: tc + 2], "post_think"))
        if len(positions) > 400:
            break

print(f"freegen-gate: {len(positions)} boundary positions "
      f"({sum(1 for _,t in positions if t=='start_think')} start_think, "
      f"{sum(1 for _,t in positions if t=='post_think')} post_think)", flush=True)
print(f"  FP={FP_DIR}\n  tern={E2E}", flush=True)

fp = AutoModelForCausalLM.from_pretrained(FP_DIR, trust_remote_code=True, dtype=torch.bfloat16).to("cuda:0").eval()
fp.config.use_cache = False
st, _ = build_student(E2E, ORIG, BLOCK_SIZE, "cuda:1"); st.eval(); st.config.use_cache = False


def metrics_at(logits_fp, logits_t):
    """all in fp32 on cpu; return (H_fp,H_t,max_fp,max_t,agree,margin_t,argmax_fp,argmax_t)."""
    af = logits_fp.float(); at = logits_t.float()
    pf = F.softmax(af, -1); pt = F.softmax(at, -1)
    Hf = -(pf * torch.log(pf + 1e-12)).sum().item()
    Ht = -(pt * torch.log(pt + 1e-12)).sum().item()
    amf = int(af.argmax()); amt = int(at.argmax())
    top2t = at.topk(2).values
    margin_t = float(top2t[0] - top2t[1])
    return Hf, Ht, float(af.max()), float(at.max()), int(amf == amt), margin_t, amf, amt


agg = {}  # tag -> lists
with torch.no_grad():
    for i, (ids, tag) in enumerate(positions):
        x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        lf = fp(x.to("cuda:0")).logits[0, -1].cpu()
        lt = st(x.to("cuda:1")).logits[0, -1].cpu()
        m = metrics_at(lf, lt)
        agg.setdefault(tag, []).append(m)
        agg.setdefault("ALL", []).append(m)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(positions)}", flush=True)


def report(tag, rows):
    Hf = sum(r[0] for r in rows) / len(rows); Ht = sum(r[1] for r in rows) / len(rows)
    mf = sum(r[2] for r in rows) / len(rows); mt = sum(r[3] for r in rows) / len(rows)
    agree = sum(r[4] for r in rows) / len(rows)
    margin = sum(r[5] for r in rows) / len(rows)
    m1 = Ht / max(Hf, 1e-6); m2 = mt / max(mf, 1e-6)
    def flag(v, thr, lo=True):
        bad = (v < thr) if lo else (v > thr)
        return "FLAG" if bad else "ok  "
    print(f"\n[{tag}]  n={len(rows)}")
    print(f"  M1 entropy ratio   {m1:6.3f}  (H_t={Ht:.2f} / H_fp={Hf:.2f})   {flag(m1,0.50)}  thr>=0.50")
    print(f"  M2 magnitude ratio {m2:6.3f}  (|t|={mt:.2f} / |fp|={mf:.2f})   {flag(m2,0.70)}  thr>=0.70")
    print(f"  M3 argmax agree    {agree:6.3f}                          {flag(agree,0.90)}  thr>=0.90")
    print(f"  M4 tern margin     {margin:6.3f}  (top1-top2)              {flag(margin,1.0)}  thr>=1.0")
    return {"n": len(rows), "m1_entropy_ratio": m1, "m2_magnitude_ratio": m2,
            "m3_argmax_agree": agree, "m4_margin": margin, "H_fp": Hf, "H_t": Ht, "max_fp": mf, "max_t": mt}


print("\n" + "=" * 68)
out = {}
for tag in ["start_think", "post_think", "ALL"]:
    if tag in agg:
        out[tag] = report(tag, agg[tag])
print("=" * 68)
# show a few concrete start_think examples (argmax tokens) for eyeballing
st_rows = [(ids, r) for (ids, tg), r in zip(positions, agg.get("ALL", [])) if tg == "start_think"][:6]
print("\nsample start_think argmax (fp -> tern):")
for ids, r in st_rows:
    _,_,_,_,_,_,amf,amt = r
    print(f"  fp={tok.decode([amf])!r}({amf})  tern={tok.decode([amt])!r}({amt})")
if GATE_OUT:
    json.dump(out, open(GATE_OUT, "w"), indent=1)
    print(f"\nwrote {GATE_OUT}")
