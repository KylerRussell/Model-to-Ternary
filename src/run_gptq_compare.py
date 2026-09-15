"""run_gptq_compare.py — how much of each format's Shannon gap does Hessian-aware encoding close?

Answers the report's unsourced claim (scalar ternary 64.8% -> ~71.5% under second-order PTQ) with our
own measurement, on stock weights, with real calibration Hessians.
"""
import os, sys, glob, json, math, torch
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoModelForCausalLM
from family_sweep import LOG2_3, q_sym, q_trellis, q_vq, rel_err
from gptq_encode import capture_hessian, gptq_quantize, proxy_loss

SRC = os.environ.get("SRC") or glob.glob(
    "/home/kasm-user/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*")[0]
DEV = "cuda:0"
NB = int(os.environ.get("NB", "16"))
SEQ = int(os.environ.get("SEQ", "512"))

FORMATS = {
    "ternary g64":  (1.835, 64,  lambda W: q_sym(W, LOG2_3, 64)),
    "ternary g256": (1.647, 256, lambda W: q_sym(W, LOG2_3, 256)),
    "vq k4096 d8":  (1.562, 256, lambda W: q_vq(W, 4096, 8, 256, device=DEV)),
    "trellis k2":   (2.062, 256, lambda W: q_trellis(W, 2, 10, 256, device=DEV, n_scan=5)),
}

print(f"loading stock model {SRC}", flush=True)
m = AutoModelForCausalLM.from_pretrained(SRC, trust_remote_code=True,
                                         dtype=torch.bfloat16).to(DEV).eval()
m.config.use_cache = False
toks = json.load(open("output_4b/eval2k.json"))[:NB]
batches = [torch.tensor(t[:SEQ]).unsqueeze(0) for t in toks if len(t) >= 64]
print(f"calibration: {len(batches)} sequences x {SEQ} tokens", flush=True)

# one mid-depth tensor of each MLP kind -- the same stratification the family sweep uses
names = []
for kind in ("down_proj", "gate_proj", "up_proj"):
    c = sorted(n for n, mod in m.named_modules()
               if isinstance(mod, torch.nn.Linear) and n.endswith(kind))
    if c:
        names.append(c[len(c) // 2])

rows = []
for name in names:
    mod = m.get_submodule(name)
    W = mod.weight.data.float()
    print(f"\n=== {name}  {tuple(W.shape)}", flush=True)
    H = capture_hessian(m, name, batches, DEV)
    print(f"    Hessian captured, cond ~ {torch.linalg.cond(H).item():.3e}", flush=True)
    for fmt, (bpw, group, fn) in FORMATS.items():
        q_rtn = fn(W).float()
        q_gptq = gptq_quantize(W, H.clone(), fn, group)
        bound = 2 ** -bpw
        r = {"tensor": name, "fmt": fmt, "bpw": bpw,
             "rtn_recon": rel_err(q_rtn, W), "gptq_recon": rel_err(q_gptq, W),
             "rtn_proxy": proxy_loss(W, q_rtn, H), "gptq_proxy": proxy_loss(W, q_gptq, H),
             "bound": bound}
        r["rtn_eff"] = bound / r["rtn_recon"]
        r["gptq_eff"] = bound / r["gptq_recon"]
        rows.append(r)
        print(f"    {fmt:14s} recon {r['rtn_recon']:.4f}->{r['gptq_recon']:.4f}   "
              f"proxy {r['rtn_proxy']:.4f}->{r['gptq_proxy']:.4f}   "
              f"eff {100*r['rtn_eff']:.1f}%->{100*r['gptq_eff']:.1f}%", flush=True)
    del H
    torch.cuda.empty_cache()

json.dump(rows, open(os.environ.get("OUT", "output_sweep/gptq_compare.json"), "w"), indent=1)
print("\nwrote", os.environ.get("OUT", "output_sweep/gptq_compare.json"))
