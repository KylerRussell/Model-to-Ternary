"""e2e_format.py — does the reconstruction knee at ~2.06 bpw survive end to end?

THE CLAIM UNDER TEST. On reconstruction error, trellis k2 (2.062 bpw) beats symmetric ternary g64
(1.835 bpw) by ~40% for 12% more bits. This project has documented that layer-wise reconstruction
metrics NEVER ONCE caught a real end-to-end failure -- block-MSE improved 99.8% on a model whose
residual stream had collapsed -- so a reconstruction knee is a hypothesis, not a result.

METHOD. Simulated deployment: every quantizable 2D weight of the STOCK model is replaced by its
dequantized form, and the resulting model is evaluated against the untouched FP model on the same
tokens. Nothing from this project's pipeline is applied -- no untying, no QuaRot, no trained
assignments -- because the question is what the FORMAT preserves, not what our pipeline can recover.

METRICS, in increasing order of trustworthiness for this project:
  perplexity        the literature's standard, and a screen only
  top-1 agreement   teacher-forced; proven blind to free-generation capability here
  KL(fp||quant)     distributional, still teacher-forced
All three are teacher-forced, so all three are screens. They are reported because they are what the
format literature reports; a capability claim would need a scored benchmark on top.
"""
import os, sys, json, glob, math, torch
sys.path.insert(0, os.path.dirname(__file__))
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from family_sweep import LOG2_3, q_sym, q_trellis, q_vq, q_nonuniform, rel_err

SRC = os.environ.get("SRC") or glob.glob(
    "/home/kasm-user/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*")[0]
FMT = os.environ.get("FMT", "fp")
NP = int(os.environ.get("NP", "48"))
SEQ = int(os.environ.get("SEQ", "1024"))
DEV = "cuda:0"

QUANT = {
    "fp":        (None, 16.0),
    "ternary64": (lambda W: q_sym(W, LOG2_3, 64), 1.835),
    "ternary256":(lambda W: q_sym(W, LOG2_3, 256), 1.647),
    "trellis2":  (lambda W: q_trellis(W, 2, 10, 256, device=DEV, n_scan=5), 2.062),
    "trellis1":  (lambda W: q_trellis(W, 1, 10, 256, device=DEV, n_scan=5), 1.062),
    "int3":      (lambda W: q_sym(W, 3.0, 256), 3.062),
    # the two sub-2 picks, chosen from the filled band on measured error, not on family loyalty
    "vq8192":    (lambda W: q_vq(W, 8192, 8, 256, device=DEV), 1.688),
    "vq4096":    (lambda W: q_vq(W, 4096, 8, 256, device=DEV), 1.562),
}
fn, bpw = QUANT[FMT]

print(f"[{FMT}] loading stock model: {SRC}", flush=True)
m = AutoModelForCausalLM.from_pretrained(SRC, trust_remote_code=True,
                                         dtype=torch.bfloat16).to(DEV).eval()
m.config.use_cache = False

if fn is not None:
    n_q = n_skip = 0
    tot_err = 0.0
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.ndim != 2 or p.shape[1] % 256 or min(p.shape) < 256:
                n_skip += 1
                continue
            W = p.data.float()
            Q = fn(W).float()
            tot_err += rel_err(Q, W)
            p.data.copy_(Q.to(p.dtype))
            n_q += 1
            if n_q % 25 == 0:
                print(f"   quantized {n_q} tensors (mean rel_err {tot_err/n_q:.4f})", flush=True)
            del W, Q
            torch.cuda.empty_cache()
    print(f"[{FMT}] quantized {n_q} tensors, skipped {n_skip}, "
          f"mean rel_err {tot_err/max(n_q,1):.5f}", flush=True)

toks = json.load(open(os.environ.get("EVAL", "output_4b/eval2k.json")))[:NP]
nll, ntok = 0.0, 0
logits_out = []
with torch.no_grad():
    for i, t in enumerate(toks):
        ids = torch.tensor(t[:SEQ], device=DEV).unsqueeze(0)
        if ids.shape[1] < 16:
            continue
        lg = m(ids).logits[0, :-1].float()
        tgt = ids[0, 1:]
        nll += F.cross_entropy(lg, tgt, reduction="sum").item()
        ntok += tgt.numel()
        logits_out.append(lg.argmax(-1).cpu())
        if os.environ.get("DUMP_LOGITS") and i < 8:
            torch.save(lg.cpu(), f"output_sweep/lg_{FMT}_{i}.pt")
        del lg
ppl = math.exp(nll / max(ntok, 1))
res = {"fmt": FMT, "bpw": bpw, "ppl": ppl, "n_tok": ntok,
       "argmax": [a.tolist() for a in logits_out]}
print(f"\n[{FMT}] bpw {bpw:.3f}   perplexity {ppl:.4f}   ({ntok} tokens)", flush=True)
out = os.environ.get("OUT", f"output_sweep/e2e_{FMT}.json")
json.dump(res, open(out, "w"))
print(f"wrote {out}")
