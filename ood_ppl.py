#!/usr/bin/env python
"""Out-of-domain perplexity: does the 1.2302x ratio hold off the (in-domain Nemotron code/math)
calibration distribution? Compute teacher + ternary ppl on WikiText-2 (general English) and report
the OOD ratio vs the in-domain 1.2302x.
"""
import math, gc, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from e2e_qp_distill import build_student, BLOCK_SIZE

import os
ORIG = os.environ.get("ORIG", "/home/kyler/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9")
FP_DIR = os.environ.get("FP_DIR", "output/modified_model")
E2E = os.environ.get("E2E_MODEL", "output_e2eqp/modified_model")
SEQ, NCHUNK, DEV = 1024, 48, "cuda:0"
tok = AutoTokenizer.from_pretrained(ORIG, trust_remote_code=True)

print("loading WikiText-2 test (OOD vs in-domain Nemotron)...", flush=True)
try:
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
except Exception as e:
    print(f"❌ could not load wikitext ({e}); need network or a cached copy. Aborting OOD."); raise
ids = tok(text, return_tensors="pt").input_ids[0]
chunks = [ids[i:i + SEQ] for i in range(0, len(ids) - SEQ, SEQ)][:NCHUNK]
print(f"  {len(chunks)} chunks × {SEQ} tokens", flush=True)

@torch.no_grad()
def ppl(forward, on):
    tot = n = 0
    for c in chunks:
        lg = forward(c.unsqueeze(0).to(on)).float()[:-1]
        tgt = c[1:].to(lg.device); logZ = torch.logsumexp(lg, dim=-1)
        tot += (logZ - lg.gather(1, tgt.unsqueeze(1)).squeeze(1)).sum().item(); n += tgt.numel()
    return math.exp(tot / n)

mm = {i: "20GiB" for i in range(torch.cuda.device_count())}; mm["cpu"] = "120GiB"
fp = AutoModelForCausalLM.from_pretrained(FP_DIR, trust_remote_code=True, dtype=torch.bfloat16,
                                          device_map="auto", max_memory=mm,
                                          offload_folder="output/_ood_off", low_cpu_mem_usage=True).eval()
fp_ppl = ppl(lambda x: fp(x).logits[0], "cuda:0")
print(f"  teacher WikiText ppl {fp_ppl:.4f}", flush=True)
del fp; gc.collect(); torch.cuda.empty_cache()

st, _ = build_student(E2E, ORIG, BLOCK_SIZE, DEV); st.eval()
tn_ppl = ppl(lambda x: st(x).logits[0], DEV)
print(f"  ternary WikiText ppl {tn_ppl:.4f}", flush=True)

print("\n" + "=" * 60)
print(f"  OUT-OF-DOMAIN (WikiText-2) vs IN-DOMAIN (Nemotron)")
print(f"  teacher ppl {fp_ppl:.3f}  ternary ppl {tn_ppl:.3f}")
print(f"  OOD ppl ratio {tn_ppl/fp_ppl:.4f}   (in-domain was 1.2302)")
print("-" * 60)
r = tn_ppl / fp_ppl
print(f"  VERDICT: {'OOD degradation MUCH worse — in-domain 1.23x is optimistic' if r > 1.35 else ('OOD comparable to in-domain — 1.23x generalizes' if r < 1.28 else 'OOD modestly worse than in-domain')}")
print("=" * 60)
