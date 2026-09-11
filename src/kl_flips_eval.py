#!/usr/bin/env python
"""KL-divergence-vs-FP and %flips — the behavioral metric the GPTQ-vs-QAT follow-up report demands
("Accuracy is Not All You Need", arXiv:2407.09141: aggregate ppl/accuracy deltas hide answer flips;
KL–flips Spearman 0.981 on MMLU). Perplexity (even OOD) is a SCREEN; KL/flips is the verdict for whether
a skeleton change is behaviorally real or cosmetic.

For matched prompts, run FP and ternary teacher-forced over the same tokens and, per next-token position:
  - mean KL( P_fp || P_tern )                  — distributional drift (nats)
  - %flips = argmax(P_fp) != argmax(P_tern)     — decision changes
  - top-1 agreement, and mean KL restricted to positions where FP is confident (p_max>0.5)

Env: ORIG, FP_DIR (rotated FP), E2E_MODEL (ternary), EVAL_DATA, NP (#seqs), SEQ. FP→cuda:0, tern→cuda:1
(falls back to sequential fp-logits-on-cpu when only 1 GPU is visible).
2-GPU mode STREAMS per-seq (no FP-logit cache) so NP can be ~2000 (eval2k) without 600GB of cached logits;
PER_SEQ_OUT=<path.json> additionally dumps per-sequence mean-KL/%flips for PAIRED cross-model comparison."""
import os, gc, math, json, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from e2e_qp_distill import build_student, load_calib_batches, BLOCK_SIZE

ORIG = os.environ.get("ORIG", "output_4b/untied_4b")
FP_DIR = os.environ.get("FP_DIR", "output_4b/rot/modified_model")
E2E = os.environ.get("E2E_MODEL", "output_4b/gptq_baseE2E/modified_model")
EVAL = os.environ.get("EVAL_DATA", "output_4b/calib_eval.json")
NP = int(os.environ.get("NP", "24"))
SEQ = int(os.environ.get("SEQ", "1024"))
two_gpu = torch.cuda.device_count() >= 2
FP_DEV, T_DEV = ("cuda:0", "cuda:1") if two_gpu else ("cuda:0", "cuda:0")

batches = load_calib_batches(EVAL, 1, SEQ, "cpu")[:NP]
print(f"KL/flips eval: {len(batches)} seqs × {SEQ} tok | FP={FP_DIR} vs tern={E2E}", flush=True)

@torch.no_grad()
def fp_logits(ids):
    return fp(ids.to(FP_DEV)).logits[:, :-1].float().cpu()

print("loading FP...", flush=True)
if two_gpu:
    fp = AutoModelForCausalLM.from_pretrained(FP_DIR, trust_remote_code=True, dtype=torch.bfloat16).to(FP_DEV).eval()
    fp_cache = None                                            # STREAM per-seq — no 600GB logit cache at NP~2000
else:
    mm = {0: "20GiB", "cpu": "80GiB"}
    fp = AutoModelForCausalLM.from_pretrained(FP_DIR, trust_remote_code=True, dtype=torch.bfloat16,
                                              device_map="auto", max_memory=mm).eval()
    if NP > 64:
        print(f"WARN: 1-GPU cache path with NP={NP} will hold ~{NP*0.6:.0f}GB of logits — use 2 GPUs", flush=True)
    fp_cache = [fp_logits(b) for b in batches]                 # [NP][1,T-1,V] on cpu
    del fp; gc.collect(); torch.cuda.empty_cache()

print("loading ternary E2E...", flush=True)
st, _ = build_student(E2E, ORIG, BLOCK_SIZE, T_DEV); st.eval()
st.config.use_cache = False

if os.environ.get("HEAD_MODE"):                                # see head_swap.py; 'fp' = upper bound
    from head_swap import swap_head
    swap_head(st, os.environ.get("HEAD_SRC", "output_4bpipe/rotbase/modified_model"),
              os.environ["HEAD_MODE"])
if os.environ.get("EMBED_MODE"):
    from head_swap import swap_embed
    swap_embed(st, os.environ.get("HEAD_SRC", "output_4bpipe/rotbase/modified_model"),
               os.environ["EMBED_MODE"])
if float(os.environ.get("THINK_ROW_SCALE", "1.0")) != 1.0:
    from head_swap import apply_think_gain
    apply_think_gain(st, float(os.environ["THINK_ROW_SCALE"]), 248069)

SQB = int(os.environ.get("SCALE_QBITS", "0"))                  # Test 1a: POST-HOC scale quantization —
if SQB > 0:                                                    # round trained per-256 scales to an n-bit
    from e2e_qp_distill import TernaryScaleLinear              # log-uniform grid (per linear), then eval.
    nq = 0
    with torch.no_grad():
        for m in st.modules():
            if isinstance(m, TernaryScaleLinear):
                s = m.scale.data
                a = s.abs().clamp_min(1e-12).log()
                lo, hi = float(a.min()), float(a.max()) + 1e-6
                step = max(hi - lo, 1e-6) / (2 ** SQB - 1)
                q = ((a - lo) / step).round().clamp(0, 2 ** SQB - 1) * step + lo
                m.scale.data = q.exp() * torch.sign(s)
                nq += 1
    print(f"SCALE_QBITS={SQB}: post-hoc quantized scales on {nq} linears (log grid)", flush=True)

tot_kl = tot_klc = 0.0; n = nc = 0; flips = 0; agree = 0; N = 0
seq_kl, seq_fl = [], []                                        # per-seq means (paired-comparison dump)
for i, b in enumerate(batches):
    with torch.no_grad():
        lt = st(b.to(T_DEV)).logits[:, :-1].float()
        lf = (fp_logits(b) if fp_cache is None else fp_cache[i]).to(lt.device)
    pf = F.softmax(lf, -1); logpf = F.log_softmax(lf, -1); logpt = F.log_softmax(lt, -1)
    kl = (pf * (logpf - logpt)).sum(-1).reshape(-1)            # KL(fp||tern) per token
    af, at = lf.argmax(-1).reshape(-1), lt.argmax(-1).reshape(-1)
    conf = pf.max(-1).values.reshape(-1) > 0.5                 # FP-confident positions
    tot_kl += kl.sum().item(); n += kl.numel()
    tot_klc += kl[conf].sum().item(); nc += int(conf.sum())
    fl = int((af != at).sum())
    flips += fl; agree += int((af == at).sum()); N += af.numel()
    seq_kl.append(float(kl.mean())); seq_fl.append(100.0 * fl / af.numel())
    del lt, lf, pf, logpf, logpt, kl
    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(batches)} seqs  running KL={tot_kl/max(n,1):.4f}", flush=True)

if os.environ.get("PER_SEQ_OUT"):
    with open(os.environ["PER_SEQ_OUT"], "w") as f:
        json.dump({"eval": EVAL, "model": E2E, "seq_kl": seq_kl, "seq_flips": seq_fl}, f)
    print(f"per-seq dump -> {os.environ['PER_SEQ_OUT']}", flush=True)

print("\n" + "=" * 64)
print(f"  KL / FLIPS  ({N} next-token positions)")
print(f"  mean KL(fp||tern)          : {tot_kl / max(n,1):.4f} nats")
print(f"  mean KL (FP-confident>0.5) : {tot_klc / max(nc,1):.4f} nats   ({nc} pos)")
print(f"  %flips (argmax fp≠tern)    : {100*flips/max(N,1):.2f}%")
print(f"  top-1 agreement            : {100*agree/max(N,1):.2f}%")
print("=" * 64)
