"""e2e_gptq.py — model-wide GPTQ perplexity, per format.

The verdict F18 could not give. Proxy loss is a better predictor than reconstruction error but is
still a per-tensor surrogate; this measures the thing itself.

SEQUENTIAL, not one-shot. Layers are quantized in forward order and Hessians are captured from the
CURRENT (partially quantized) model, so each layer compensates for the error its predecessors already
introduced. That is what production GPTQ does, and one-shot Hessians from the FP model would
understate every format -- equally, but understating everything is still the wrong measurement.

Embeddings are quantized RTN in every arm: GPTQ needs an input activation covariance and an embedding
lookup has none. Handling them identically across arms keeps the comparison clean and is stated
rather than hidden.
"""
import os, sys, glob, json, math, gc, torch
sys.path.insert(0, os.path.dirname(__file__))
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from family_sweep import LOG2_3, q_sym, q_trellis, q_vq, fit_vq_codebook, rel_err
from gptq_encode import gptq_quantize

SRC = os.environ.get("SRC") or glob.glob(
    "/home/kasm-user/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*")[0]
FMT = os.environ.get("FMT", "ternary64")
DEV = "cuda:0"
NB = int(os.environ.get("NB", "16"))
CSEQ = int(os.environ.get("CSEQ", "512"))
NP = int(os.environ.get("NP", "48"))
SEQ = int(os.environ.get("SEQ", "1024"))

FORMATS = {
    "ternary64": (1.835, 64,  lambda W: q_sym(W, LOG2_3, 64)),
    "trellis2":  (2.062, 256, lambda W: q_trellis(W, 2, 10, 256, device=DEV, n_scan=5)),
    # VQ carries ONE codebook per tensor, so it is fit per tensor and blocks only assign against
    # it (see fit_vq_codebook). A per-block codebook would be a different, more expensive format.
    "vq4096":    (1.562, 256, ("vq", 4096, 8)),
    "vq8192":    (1.688, 256, ("vq", 8192, 8)),
}
bpw, GROUP, QFN = FORMATS[FMT]

print(f"[{FMT}] loading stock model", flush=True)
m = AutoModelForCausalLM.from_pretrained(SRC, trust_remote_code=True,
                                         dtype=torch.bfloat16).to(DEV).eval()
m.config.use_cache = False
calib = [torch.tensor(t[:CSEQ]).unsqueeze(0).to(DEV)
         for t in json.load(open("output_4b/eval2k.json"))[:NB] if len(t) >= 64]

lins = [(n, mod) for n, mod in m.named_modules()
        if isinstance(mod, torch.nn.Linear) and mod.weight.ndim == 2
        and mod.weight.shape[1] % 256 == 0 and min(mod.weight.shape) >= 256]
# group by transformer layer so Hessians for a whole layer are captured in one pass
groups = {}
for n, mod in lins:
    key = n.rsplit(".", 2)[0] if "." in n else n
    groups.setdefault(key, []).append((n, mod))
print(f"[{FMT}] {len(lins)} linears in {len(groups)} groups", flush=True)

done = 0
tot_err = 0.0
for gi, (key, members) in enumerate(groups.items()):
    H = {}
    hooks = []
    def mk(nm, din):
        H[nm] = torch.zeros(din, din, device=DEV, dtype=torch.float32)
        cnt = [0]
        def hk(_mod, inp, _out):
            x = inp[0].detach().reshape(-1, din).float()
            H[nm] += x.T @ x
            cnt[0] += x.shape[0]
        return hk, cnt
    counts = {}
    for n, mod in members:
        hk, c = mk(n, mod.in_features)
        counts[n] = c
        hooks.append(mod.register_forward_hook(hk))
    with torch.no_grad():
        for b in calib:
            m(b)
    for h in hooks:
        h.remove()
    for n, mod in members:
        Hn = H[n] / max(counts[n][0], 1)
        d = torch.diag(Hn).mean().clamp_min(1e-8)
        Hn += torch.eye(Hn.shape[0], device=DEV) * (0.01 * d)
        W = mod.weight.data.float()
        fn = (fit_vq_codebook(W, QFN[1], QFN[2], device=DEV)
              if isinstance(QFN, tuple) else QFN)
        Q = gptq_quantize(W, Hn, fn, GROUP)
        tot_err += rel_err(Q, W)
        mod.weight.data.copy_(Q.to(mod.weight.dtype))
        done += 1
        del Hn, W, Q
    H.clear()
    gc.collect(); torch.cuda.empty_cache()
    if (gi + 1) % 5 == 0 or gi == len(groups) - 1:
        print(f"   group {gi+1}/{len(groups)}  {done} linears  mean recon {tot_err/done:.4f}",
              flush=True)

# embeddings: RTN in every arm (GPTQ needs an input covariance; a lookup has none)
with torch.no_grad():
    for n, p in m.named_parameters():
        if "embed" in n and p.ndim == 2 and p.shape[1] % 256 == 0:
            efn = (fit_vq_codebook(p.data.float(), QFN[1], QFN[2], device=DEV)
                   if isinstance(QFN, tuple) else QFN)
            p.data.copy_(efn(p.data.float()).to(p.dtype))
            print(f"   embedding {n} quantized RTN {tuple(p.shape)}", flush=True)

toks = json.load(open("output_4b/eval2k.json"))[:NP]
nll, ntok = 0.0, 0
with torch.no_grad():
    for t in toks:
        ids = torch.tensor(t[:SEQ], device=DEV).unsqueeze(0)
        if ids.shape[1] < 16:
            continue
        lg = m(ids).logits[0, :-1].float()
        nll += F.cross_entropy(lg, ids[0, 1:], reduction="sum").item()
        ntok += ids.shape[1] - 1
        del lg
ppl = math.exp(nll / max(ntok, 1))
print(f"\n[{FMT}] GPTQ bpw {bpw:.3f}  perplexity {ppl:.4f}  ({ntok} tokens, "
      f"mean recon {tot_err/max(done,1):.4f})", flush=True)
json.dump({"fmt": FMT, "bpw": bpw, "ppl": ppl, "encoder": "gptq-sequential",
           "mean_recon": tot_err / max(done, 1), "n_lin": done},
          open(os.environ.get("OUT", f"output_sweep/e2egptq_{FMT}.json"), "w"))
