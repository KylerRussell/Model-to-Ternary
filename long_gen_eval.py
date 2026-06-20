#!/usr/bin/env python
"""Long free-generation eval: the deployment-realistic test the teacher-forced accumulation diagnostic
can't see. Both models greedily generate 960 tokens (full 1024-context) from the same prompts, with
KV cache (O(n)). Tracks, binned by GENERATED position, whether the ternary model's FREE generation
falls apart with length (the report's truncation/degeneration concern; our known free-gen weak spot):

  - exact-match vs teacher rollout (per gen-position bin)  — decays naturally; rate of decay is the signal
  - tokens-to-first-divergence
  - truncation rate (early EOS)                            — teacher vs ternary
  - degeneration: repeated-4-gram fraction per bin         — RISING with position = free-gen breakdown
  - student NLL on the TEACHER's rollout, per bin          — RISING = student diverges more on generated text
"""
import json, gc, math, os, torch, torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from e2e_qp_distill import build_student, load_calib_batches, BLOCK_SIZE

SAMPLE = os.environ.get("SAMPLE") == "1"      # SAMPLE=1 -> deployment-realistic temp/top-p decoding
TEMP, TOP_P = 0.7, 0.9
MODE = f"sampled(T={TEMP},p={TOP_P})" if SAMPLE else "greedy"

ORIG = os.environ.get("ORIG", "/home/kyler/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9")
FP_DIR = os.environ.get("FP_DIR", "output/modified_model")
E2E = os.environ.get("E2E_MODEL", "output_e2eqp/modified_model")
EVAL = os.environ.get("EVAL_DATA", "output_recovery/eval_data.json")
PROMPT, NBINS = 64, 8
GEN = int(os.environ.get("GEN", "960"))
NP = int(os.environ.get("NP", "16"))
DEV = "cuda:0"
tok = AutoTokenizer.from_pretrained(ORIG, trust_remote_code=True)
EOS = tok.eos_token_id if tok.eos_token_id is not None else -1
batches = load_calib_batches(EVAL, 1, 1024, "cpu")
prompts = [b[0, :PROMPT].tolist() for b in batches[:NP]]

def rep4_per_bin(toks):
    """fraction of positions completing a repeated 4-gram, per bin."""
    seen = set(); bs = torch.zeros(NBINS); bc = torch.zeros(NBINS)
    for i in range(len(toks)):
        b = i * NBINS // len(toks); bc[b] += 1
        if i >= 3:
            g = tuple(toks[i-3:i+1])
            if g in seen: bs[b] += 1
            seen.add(g)
    return bs / bc.clamp_min(1)

from transformers import LogitsProcessor, LogitsProcessorList
class _F32(LogitsProcessor):                            # upcast+sanitize so temp/top-p/softmax run in fp32
    def __call__(self, ids, scores):
        return torch.nan_to_num(scores.float(), nan=-1e4, posinf=1e4, neginf=-1e4)

@torch.no_grad()
def gen_all(model, on_device):
    rolls, truncs = [], 0
    kw = (dict(do_sample=True, temperature=TEMP, top_p=TOP_P,
               logits_processor=LogitsProcessorList([_F32()])) if SAMPLE else dict(do_sample=False))
    for pi, p in enumerate(prompts):
        if SAMPLE: torch.manual_seed(1000 + pi)        # same seed per prompt across models
        ids = torch.tensor([p], device=on_device)
        out = model.generate(ids, max_new_tokens=GEN, use_cache=True,
                             pad_token_id=EOS if EOS >= 0 else tok.pad_token_id, **kw)
        new = out[0, PROMPT:].tolist()
        if EOS in new:
            truncs += 1; new = new[:new.index(EOS) + 1]
        rolls.append(new)
    return rolls, truncs

print("FP teacher generating...", flush=True)
mm = {i: "20GiB" for i in range(torch.cuda.device_count())}; mm["cpu"] = "120GiB"
fp = AutoModelForCausalLM.from_pretrained(FP_DIR, trust_remote_code=True, dtype=torch.bfloat16,
                                          device_map="auto", max_memory=mm,
                                          offload_folder="output/_lg_off", low_cpu_mem_usage=True).eval()
fp.config.use_cache = True
t_rolls, t_trunc = gen_all(fp, "cuda:0")
del fp; gc.collect(); torch.cuda.empty_cache()

print("ternary E2E generating...", flush=True)
st, _ = build_student(E2E, ORIG, BLOCK_SIZE, DEV); st.eval()
st.config.use_cache = True
try: st.gradient_checkpointing_disable()
except Exception: pass
s_rolls, s_trunc = gen_all(st, DEV)

# ---- metrics ----
em = torch.zeros(NBINS); emc = torch.zeros(NBINS); fdivs = []
t_rep = torch.zeros(NBINS); s_rep = torch.zeros(NBINS)
for tr, sr in zip(t_rolls, s_rolls):
    L = min(len(tr), len(sr), GEN)
    eq = [tr[i] == sr[i] for i in range(L)]
    for i in range(L):
        b = i * NBINS // GEN; emc[b] += 1; em[b] += int(eq[i])
    fd = next((i for i, e in enumerate(eq) if not e), L); fdivs.append(fd)
    t_rep += rep4_per_bin(tr); s_rep += rep4_per_bin(sr)
em = em / emc.clamp_min(1); t_rep /= len(t_rolls); s_rep /= len(s_rolls)

print("\n" + "=" * 74)
print(f"  LONG FREE-GENERATION ({NP} prompts × {GEN} tokens, {MODE})")
if SAMPLE: print("  [exact-match/first-div are informational under sampling — degeneration rep4% is the signal]")
print(f"  truncation (early EOS):   teacher {t_trunc}/{NP}   ternary {s_trunc}/{NP}")
print(f"  tokens-to-first-divergence: mean {sum(fdivs)/len(fdivs):.1f} / {GEN}  (median {sorted(fdivs)[len(fdivs)//2]})")
print("-" * 74)
print(f"  {'gen-position bin':22s}{'exact-match%':>14s}{'teacher rep4%':>15s}{'ternary rep4%':>15s}")
for k in range(NBINS):
    lo, hi = k * GEN // NBINS, (k + 1) * GEN // NBINS
    print(f"  b{k} [{lo:4d}-{hi:4d})       {100*em[k]:>13.1f}%{100*t_rep[k]:>14.1f}%{100*s_rep[k]:>14.1f}%")
print("-" * 74)
rep_slope = (s_rep[NBINS-1] - s_rep[0]).item()
print(f"  ternary degeneration (rep4) b0={100*s_rep[0]:.1f}% → b{NBINS-1}={100*s_rep[NBINS-1]:.1f}%  "
      f"(slope {100*rep_slope:+.1f}pp); teacher b0={100*t_rep[0]:.1f}%→b{NBINS-1}={100*t_rep[NBINS-1]:.1f}%")
print(f"  VERDICT: {'ternary FREE-GEN DEGENERATES with length (deployment risk ppl misses)' if (rep_slope > 0.05 and s_rep[NBINS-1] > t_rep[NBINS-1] + 0.05) or s_trunc > t_trunc + NP//5 else 'free-gen degeneration/truncation comparable to teacher — ppl is representative for generation too'}")
print("=" * 74)
