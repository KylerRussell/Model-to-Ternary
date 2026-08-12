#!/usr/bin/env python
"""B3/A8 LICENSING measurement (axes battery): does the ternary-vs-FP KL gap GROW with sequence length?

Gated DeltaNet's transition A_t = alpha_t(I - beta_t k_t k_t^T) is strictly contractive (||A_t||_2 =
alpha_t < 1), so recurrent quantization error e_T = Sigma A^{T-t} eps_t should be geometrically damped ->
prediction: FLAT gap, axis dies. License development ONLY if gap(8192) - gap(512) > 0.01 KL.

Design: ONE forward per model per sequence at L_max=8192; per-token full-vocab KL(fp||tern); the nested
prefix means at L in {512,1024,2048,4096,8192} are perfectly PAIRED per sequence. KL computed in chunks
(fp32 logits at 8192 x 152k vocab would be ~5GB otherwise).

Env: ORIG, FP_DIR, TERN_DIR, EVAL_DATA (long-seq calib json), NP (#seqs), OUT_JSON. FP->cuda:0, tern->cuda:1.
"""
import os, json, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from e2e_qp_distill import build_student, load_calib_batches, BLOCK_SIZE

ORIG = os.environ.get("ORIG", "output_4b/untied_4b")
FP_DIR = os.environ.get("FP_DIR", "output_4b/rot/modified_model")
TERN_DIR = os.environ.get("TERN_DIR", "output_4b/scaling/16M_2ep/modified_model")
EVAL = os.environ.get("EVAL_DATA", "output_4b/eval_long8k.json")
NP = int(os.environ.get("NP", "96"))
OUT_JSON = os.environ.get("OUT_JSON", "logs/b3_klgap.json")
LENS = [512, 1024, 2048, 4096, 8192]
CHUNK = 1024

batches = load_calib_batches(EVAL, 1, LENS[-1], "cpu")[:NP]
print(f"B3 licensing: {len(batches)} seqs x {LENS[-1]} tok | FP={FP_DIR} vs TERN={TERN_DIR}", flush=True)

fp = AutoModelForCausalLM.from_pretrained(FP_DIR, trust_remote_code=True, dtype=torch.bfloat16).to("cuda:0").eval()
fp.config.use_cache = False
st, _ = build_student(TERN_DIR, ORIG, BLOCK_SIZE, "cuda:1")
st.eval(); st.config.use_cache = False

per_seq = {L: [] for L in LENS}
with torch.no_grad():
    for i, b in enumerate(batches):
        lf = fp(b.to("cuda:0")).logits[:, :-1]                 # [1, T-1, V] bf16 on gpu0
        lt = st(b.to("cuda:1")).logits[:, :-1]
        T = lf.shape[1]
        kl_tok = torch.empty(T)
        for c0 in range(0, T, CHUNK):                          # chunked full-vocab KL (fp32 only per chunk)
            a = lf[:, c0:c0+CHUNK].float().to("cuda:1")
            t = lt[:, c0:c0+CHUNK].float()
            pf = F.softmax(a, -1)
            kl = (pf * (F.log_softmax(a, -1) - F.log_softmax(t, -1))).sum(-1)
            kl_tok[c0:c0+kl.shape[1]] = kl[0].cpu()
            del a, t, pf, kl
        for L in LENS:                                         # nested prefix means -> paired across L
            n = min(L - 1, T)
            per_seq[L].append(float(kl_tok[:n].mean()))
        del lf, lt
        torch.cuda.empty_cache()
        if (i + 1) % 16 == 0:
            print(f"  {i+1}/{len(batches)}  " +
                  "  ".join(f"L{L}:{sum(per_seq[L])/len(per_seq[L]):.4f}" for L in LENS), flush=True)

import statistics as stats
print("\n== KL(fp||tern) by prefix length (paired, full-vocab) ==")
means = {}
for L in LENS:
    v = per_seq[L]; means[L] = sum(v) / len(v)
    print(f"  L={L:5d}: mean KL = {means[L]:.4f}")
d = [b - a for a, b in zip(per_seq[512], per_seq[8192])]       # paired per-seq gap growth 512 -> 8192
mean_d = sum(d) / len(d)
se = stats.stdev(d) / len(d) ** 0.5
print(f"\n  paired gap growth (L8192 - L512): {mean_d:+.5f} +/- {se:.5f}  (t={mean_d/max(se,1e-12):+.2f})")
lic = mean_d > 0.01
print(f"VERDICT: {'LICENSED — gap grows >0.01, build the contractivity-aware pass' if lic else 'B3/A8 DEAD — gap does not grow with sequence length (contractivity holds); axis retired unlaunched'}")
json.dump({"means": means, "per_seq": {str(k): v for k, v in per_seq.items()},
           "gap_growth": mean_d, "se": se, "licensed": lic}, open(OUT_JSON, "w"))
print(f"wrote {OUT_JSON}")
