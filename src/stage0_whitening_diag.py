#!/usr/bin/env python3
"""
Stage-0 diagnostic for the "place/hide the error" report: does the QuaRot Hadamard rotation leave any
LOW-ENERGY or NULL activation subspace to hide the ternarization residual in — or did it whiten the
activation covariance flat (in which case mechanisms #1/#5/#8/#9 are no-ops)?

For a linear y = W x, the output error of a weight perturbation is  ΔW x, whose per-output variance is
(ΔW row)·Cov(x)·(ΔW row)ᵀ. So "hideable" error space = the LOW-eigenvalue directions of the INPUT
activation covariance Cov(x) = the same Gram H = E[x xᵀ] GPTQ uses. We measure its eigenspectrum on the
ROTATED model for the MLP linears (gate/up share the post-attn-norm input; down reads the SwiGLU
intermediate — the report claims the wide down_proj should have an expanded input null space).

Verdict:
  - anisotropic (eff_rank ≪ dim, tiny energy in the bottom directions, a real near-null subspace)
    → there IS room to hide error → the hide-the-error family is VIABLE.
  - isotropic/whitened (eff_rank ≈ dim, energy spread evenly, no null space)
    → nowhere to hide → that family is DEAD; pursue placement-optimality / metric ideas instead.
"""
import os, sys, json, argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM

MLP = "Qwen3_5MLP"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="output_4b/rot/modified_model")
    ap.add_argument("--orig-config", default="output_4b/untied_4b")
    ap.add_argument("--calib", default="output_4b/calibration_data.json")
    ap.add_argument("--n-seq", type=int, default=24)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--every", type=int, default=4, help="sample every Nth MLP layer (memory)")
    args = ap.parse_args()
    dev = "cuda:0"

    print(f"[s0] loading rotated FP {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, dtype=torch.bfloat16).to(dev).eval()
    mlps = [(n, m) for n, m in model.named_modules() if type(m).__name__ == MLP]
    sampled = mlps[:: args.every]
    print(f"[s0] {len(mlps)} MLP layers, sampling {len(sampled)} (every {args.every})", flush=True)

    # Gram accumulators keyed by (layer_idx, which)  which in {gateup_in, down_in}
    grams = {}
    hooks = []
    def mk(key):
        def hook(mod, inp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).to(torch.float32)
            g = grams.get(key)
            if g is None:
                d = x.shape[1]
                grams[key] = [torch.zeros(d, d, device=dev), 0]
                g = grams[key]
            g[0].addmm_(x.t(), x); g[1] += x.shape[0]
        return hook
    for i, (name, m) in enumerate(sampled):
        hooks.append(m.gate_proj.register_forward_pre_hook(mk((i, "gateup_in"))))
        hooks.append(m.down_proj.register_forward_pre_hook(mk((i, "down_in"))))

    calib = json.load(open(args.calib))
    for s in range(min(args.n_seq, len(calib))):
        model(torch.tensor([calib[s][:args.seq]], device=dev))
    for h in hooks:
        h.remove()

    def spectrum_stats(H, n):
        C = (H / max(n, 1)).double()                       # covariance
        ev = torch.linalg.eigvalsh(C).flip(0).clamp_min(0) # descending eigenvalues
        d = ev.numel(); tot = ev.sum()
        p = ev / tot.clamp_min(1e-30)
        eff_rank = float(torch.exp(-(p * (p + 1e-30).log()).sum()))    # participation (entropy) rank
        cum = torch.cumsum(ev, 0) / tot
        # energy fraction in the BOTTOM directions = 1 - cum at the split from the top
        bot50 = float(1 - cum[d // 2 - 1])                 # energy in bottom 50% of directions
        bot25 = float(1 - cum[int(0.75 * d) - 1])          # energy in bottom 25%
        null = float((ev < 1e-6 * ev[0]).float().mean())   # ~null-space fraction
        cond = float(ev[0] / ev[ev > 0].min())
        # how much of the block's error could hide: dims whose cumulative-from-bottom energy < 1% of total
        hide_dims = int((torch.cumsum(ev.flip(0), 0) / tot < 0.01).sum())
        return dict(dim=d, eff_rank=eff_rank, eff_frac=eff_rank / d, bot50=bot50, bot25=bot25,
                    null=null, cond=cond, hide_frac=hide_dims / d)

    def agg(which):
        rows = [spectrum_stats(*grams[(i, which)]) for i in range(len(sampled)) if (i, which) in grams]
        import statistics as st
        keys = rows[0].keys()
        return {k: (st.mean(r[k] for r in rows) if k != "dim" else rows[0]["dim"]) for k in keys}

    print("\n===== STAGE-0 WHITENING DIAGNOSTIC (MLP input covariance, rotated model) =====")
    print(f"{'input':<12}{'dim':>6}{'eff_rank':>10}{'eff/dim':>9}{'E_bot50%':>10}{'E_bot25%':>10}"
          f"{'nullfrac':>9}{'hidedims%':>10}{'cond':>10}")
    for which, label in [("gateup_in", "gate/up in"), ("down_in", "down in")]:
        a = agg(which)
        print(f"{label:<12}{a['dim']:>6}{a['eff_rank']:>10.1f}{a['eff_frac']:>9.3f}"
              f"{a['bot50']*100:>9.2f}%{a['bot25']*100:>9.2f}%{a['null']*100:>8.2f}%"
              f"{a['hide_frac']*100:>9.1f}%{a['cond']:>10.1e}")
    print("\nRead: eff/dim≈1 + E_bot50%≈50% + nullfrac≈0 = WHITENED (nowhere to hide → hide-family DEAD).")
    print("      eff/dim≪1 + E_bot50%≈0% + big hidedims%/nullfrac = ANISOTROPIC (room to hide → VIABLE).")
    print("hidedims% = fraction of input dirs holding <1% total energy = where error can be parked ~free.")


if __name__ == "__main__":
    main()
