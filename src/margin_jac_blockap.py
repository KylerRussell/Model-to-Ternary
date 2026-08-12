#!/usr/bin/env python3
"""#4 Margin-Jacobian GPTQ (tractable form). Re-quantize the MLP linears (the 79%-error bottleneck) with
a GPTQ Hessian WEIGHTED by each token's margin-sensitivity g_t = ‖∂(logit-margin)/∂(this linear's
output)‖ — so placement protects the directions that move the final DECISION MARGIN, not high-variance
syntax. GPTQ ≡ Babai on the Hessian lattice, so weighting the Gram changes which ternary values are
chosen. Distinct from the FAILED decision-token weighting (C1 weighted by teacher CONFIDENCE, same for
every layer; this weights by THIS layer's causal influence on the margin, per-layer, via a real backprop).

Non-MLP weights are copied from the plain-GPTQ block-AP (--gptq). Emits a standard dense block-AP model
(ternary*scale) that repacks to TQ2_0 and feeds E2E. Self-contained; reuses quantize_gptq_ternary.
"""
import os, sys, json, glob, shutil, argparse
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM
from quantizer import quantize_gptq_ternary, dequantize

BS = 256
MLP = "Qwen3_5MLP"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rot", default="output_4b/rot/modified_model")           # rotated FP (weights + logits)
    ap.add_argument("--gptq", default="output_4b/gptq_op/modified_model")      # non-MLP weights source
    ap.add_argument("--orig-config", default="output_4b/untied_4b")
    ap.add_argument("--calib", default="output_4b/calibration_data.json")
    ap.add_argument("--out", default="output_4b/margin_jac/modified_model")
    ap.add_argument("--n-seq", type=int, default=48)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--margin-lambda", type=float, default=1.0,
                    help="Shrinkage blend H=(1-λ)·H_uniform + λ·H_margin (trace-matched). λ=0 plain GPTQ, "
                         "λ=1 full margin. The raw g² Hessian over-concentrates on a few tokens; λ<1 is the "
                         "report's shrinkage-toward-uniform fix.")
    args = ap.parse_args()
    dev = "cuda:0"

    print(f"[mj] loading rotated FP {args.rot}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.rot, trust_remote_code=True, dtype=torch.bfloat16).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    mlps = [(n, m) for n, m in model.named_modules() if type(m).__name__ == MLP]
    print(f"[mj] {len(mlps)} MLPs; computing margin-sensitivity-weighted Hessians (down on CPU)", flush=True)

    lins = {}
    for i, (name, m) in enumerate(mlps):
        lins[(i, name, "gate")] = m.gate_proj
        lins[(i, name, "up")] = m.up_proj
        lins[(i, name, "down")] = m.down_proj
    Hm = {}           # key -> [H, N]
    xbuf = {}         # key -> current-batch input
    fh, bh = [], []
    def mk_fwd(key):
        def h(mod, inp):
            xbuf[key] = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
        return h
    def mk_bwd(key):
        def h(mod, gin, gout):
            x = xbuf.pop(key, None)
            if x is None or gout[0] is None:
                return
            gt = gout[0].detach().reshape(-1, gout[0].shape[-1]).float().norm(dim=-1)   # [ntok] margin sens.
            xg = x * gt.unsqueeze(-1)                                                    # g_t·x_t
            Hu = x.t() @ x                                                               # Σ xxᵀ (uniform)
            Hg = xg.t() @ xg                                                             # Σ g²xxᵀ (margin)
            on_cpu = key[2] == "down"
            if on_cpu:
                Hu, Hg = Hu.cpu(), Hg.cpu()
            st = Hm.get(key)
            if st is None:
                Hm[key] = [Hu, Hg, x.shape[0]]
            else:
                st[0] += Hu; st[1] += Hg; st[2] += x.shape[0]
            del Hu, Hg, xg
        return h
    for key, m in lins.items():
        fh.append(m.register_forward_pre_hook(mk_fwd(key)))
        bh.append(m.register_full_backward_hook(mk_bwd(key)))

    calib = json.load(open(args.calib))
    emb = model.get_input_embeddings()
    for s in range(min(args.n_seq, len(calib))):
        ids = torch.tensor([calib[s][:args.seq]], device=dev)
        e = emb(ids).detach().requires_grad_(True)          # leaf so backward has something to seed the graph
        out = model(inputs_embeds=e)
        lg = out.logits[0].float()
        top2 = lg.topk(2, dim=-1).values
        margin = (top2[:, 0] - top2[:, 1]).sum()            # sum of per-token (top1−top2) logit margins
        model.zero_grad(set_to_none=True)
        margin.backward()
        xbuf.clear()
        del out, lg, e
    for h in fh + bh:
        h.remove()

    lam = args.margin_lambda
    print(f"[mj] quantizing MLP with blended Hessian λ={lam} (H=(1-λ)·Huniform + λ·Hmargin, trace-matched)", flush=True)
    newmlp = {}                                             # model-name weight -> dequantized ternary (cpu fp16)
    for i, (name, m) in enumerate(mlps):
        for which, lin in [("gate", m.gate_proj), ("up", m.up_proj), ("down", m.down_proj)]:
            Hu, Hg, N = Hm[(i, name, which)]
            Hu, Hg = Hu.to(dev), Hg.to(dev)
            tu, tg = torch.trace(Hu), torch.trace(Hg).clamp_min(1e-12)
            H = (1 - lam) * Hu + lam * Hg * (tu / tg)        # trace-matched shrinkage blend
            qt = quantize_gptq_ternary(lin.weight.data.float(), H, block_size=BS)
            newmlp[f"{name}.{which}_proj.weight"] = dequantize(qt).to(torch.float16).cpu()
            del Hu, Hg, H
        torch.cuda.empty_cache()

    # write output = copy the plain-GPTQ block-AP, override MLP weights with the margin-Jacobian ones.
    print(f"[mj] writing {args.out} (non-MLP from {args.gptq})", flush=True)
    os.makedirs(args.out, exist_ok=True)
    idx = json.load(open(os.path.join(args.gptq, "model.safetensors.index.json")))
    wmap = idx["weight_map"]
    def disk_of(model_name):                                # model.layers.X -> disk model.language_model.layers.X
        return model_name.replace("model.layers", "model.language_model.layers")
    override = {disk_of(k): v for k, v in newmlp.items()}
    shards = {}
    for dname, sf in wmap.items():
        shards.setdefault(sf, []).append(dname)
    for sf, names in shards.items():
        with safe_open(os.path.join(args.gptq, sf), framework="pt", device="cpu") as f:
            d = {n: (override[n] if n in override else f.get_tensor(n)) for n in names}
        save_file(d, os.path.join(args.out, sf))
        del d
    for fn in ["config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "generation_config.json", "merges.txt", "vocab.json"]:
        src = os.path.join(args.gptq, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out, fn))
    print(f"[mj] done — {len(override)} MLP weights overridden with margin-Jacobian placement", flush=True)


if __name__ == "__main__":
    main()
