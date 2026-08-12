#!/usr/bin/env python3
"""A4 saliency precompute (GuidedQuant-style, arXiv:2505.07004). Block-AP is streaming (one decoder layer
at a time), so the end-loss saliency of each layer's OUTPUT can't be computed inside the loop — it needs a
full-model backward. This does exactly that once, up front, on the ROTATED FP model (the same model whose
per-layer outputs block-AP reconstructs):

  saliency[l][c] = mean_token ( ∂(LM cross-entropy)/∂ layer_l_output[...,c] )²   (diagonal Fisher of the
                   block output w.r.t. the final loss), per hidden channel c.

Cached as {layer_idx: [hidden] fp32}, normalised to mean 1 per layer so it reweights (not rescales) the
block-output MSE. block_ap_recovery.py --qat-loss saliency --qat-saliency-cache <out> then weights each
channel of the reconstruction objective by this — aligning the polish with the directions that actually
move the final loss (the generalisation channel), rather than raw output variance.
"""
import os, sys, json, argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="output_4b/rot/modified_model")     # rotated FP (matches block-AP target)
    ap.add_argument("--orig-config", default="output_4b/untied_4b")
    ap.add_argument("--calib", default="output_4b/calibration_data.json")
    ap.add_argument("--out", default="output_4b/saliency.pt")
    ap.add_argument("--n-seq", type=int, default=32)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--temp", type=float, default=1.0,
                    help="Saliency tempering exponent applied before mean-1 normalisation. temp=1.0 = raw "
                         "diagonal Fisher (empirically well-spread here, eff-rank≈dim — not concentrated); "
                         "temp=0.5 (√) softens it further; temp=0.0 = uniform (≡ plain MSE). Sweep knob.")
    args = ap.parse_args()
    dev = "cuda:0"

    print(f"[sal] loading rotated FP {args.model}", flush=True)
    nd = torch.cuda.device_count()
    if nd >= 2:                                                # shard big models; keep grads (no offload)
        mm = {i: "20GiB" for i in range(nd)}
        model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, dtype=torch.bfloat16,
                                                     device_map="auto", max_memory=mm, low_cpu_mem_usage=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
                                                     dtype=torch.bfloat16).to(dev)
    model.eval()
    layers = model.model.layers
    H = model.config.hidden_size
    sal = {l: torch.zeros(H, dtype=torch.float64) for l in range(len(layers))}
    ntok = 0

    captured = {}
    def mk(l):
        def hook(mod, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            o.retain_grad()                                    # keep grad on this intermediate
            captured[l] = o
        return hook
    hooks = [layers[l].register_forward_hook(mk(l)) for l in range(len(layers))]

    calib = json.load(open(args.calib))
    emb_dev = model.get_input_embeddings().weight.device
    for s in range(min(args.n_seq, len(calib))):
        ids = torch.tensor([calib[s][:args.seq]], device=emb_dev)
        captured.clear()
        out = model(ids)
        logits = out.logits[:, :-1].float()
        tgt = ids[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        model.zero_grad(set_to_none=True)
        loss.backward()
        for l in range(len(layers)):
            g = captured.get(l)
            if g is None or g.grad is None:
                continue
            sq = (g.grad.detach().double() ** 2).sum(dim=(0, 1))    # [H] Σ over batch,token of grad²
            sal[l] += sq.cpu()
        ntok += tgt.numel()
        if s % 8 == 0:
            print(f"[sal] seq {s+1}/{min(args.n_seq, len(calib))} loss {loss.item():.3f}", flush=True)
    for h in hooks:
        h.remove()

    out_d = {}
    for l in range(len(layers)):
        v = (sal[l] / max(ntok, 1)).float()
        v = v.clamp_min(0).pow(args.temp)                     # temper: √-Fisher avoids 1-channel degeneracy
        v = v / v.mean().clamp_min(1e-30)                      # normalise to mean 1 → reweight, not rescale
        out_d[str(l)] = v
    torch.save(out_d, args.out)
    mn = torch.stack(list(out_d.values()))                     # [L,H], each row mean-1
    pr = (mn.sum(1) ** 2 / (mn ** 2).sum(1)).mean().item()    # participation ratio = eff # channels
    print(f"[sal] saved {len(out_d)} layers → {args.out}  (per-layer eff-channels mean {pr:.0f}/{H}; "
          f"temp={args.temp} — eff≈dim means a mild reweight, not a 1-channel collapse)", flush=True)


if __name__ == "__main__":
    main()
