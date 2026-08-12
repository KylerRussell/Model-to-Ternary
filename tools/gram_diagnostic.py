#!/usr/bin/env python
"""gram_diagnostic.py — Stage-0 capacity-vs-estimation diagnostic for the QAT saturation (research report Q1/Q3).

The report's core claim: block-AP reconstruction saturates at ~1.3M tokens because the objective depends on data
ONLY through the activation Gram H = XᵀX (an O(d²) statistic), which is fully estimated in O(r_eff(H)/ε²) tokens.
If true then (a) r_eff(H) ≪ d (a few hundred, not 2560), (b) H converges by ~512 samples (the observed knee), and
(c) the calibration H already spans the held-out activation energy — so the plateau is a CAPACITY ceiling on the
ternary grid, not a data/estimation problem.

This tests all three from the FROZEN FP model alone — NO retraining. It hooks the residual-stream INPUT to a few
decoder layers, accumulates H incrementally over calibration tokens (snapshotting at 128/256/512/1024 seqs), and
compares against a held-out Gram.

  ORIG_MODEL=<orig-config> ./.venv/bin/python tools/gram_diagnostic.py \
      --model output_4bpipe/rotbase/modified_model \
      --calib output_4bpipe/calibration_data.json --held output_4bpipe/calib_eval.json \
      --layers 8,16,24 --max-seqs 1024 --held-seqs 256 --seq 2560 --out logs/gram_diag.json
"""
import argparse, json, os, sys
import numpy as np
import torch
from transformers import AutoModelForCausalLM


def eff_rank(eigs):
    eigs = np.clip(eigs, 0, None)
    s = eigs.sum()
    if s <= 0:
        return 0.0, 0.0
    p = eigs / s
    r_eff = float(s / eigs.max())                       # tr(H)/λ_max  (participation/effective rank)
    stable = float((s ** 2) / (eigs ** 2).sum())        # stable rank ‖H‖²_F-based
    return r_eff, stable


def energy_captured(H_cal_eigvecs, H_held, ks):
    """Fraction of held-out Gram energy (trace) captured by the top-k eigvectors of the CALIB Gram."""
    tot = float(np.trace(H_held))
    out = {}
    for k in ks:
        V = H_cal_eigvecs[:, -k:]                        # top-k (eigh returns ascending)
        out[k] = float(np.trace(V.T @ H_held @ V) / tot) if tot > 0 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--held", required=True)
    ap.add_argument("--layers", default="8,16,24")
    ap.add_argument("--max-seqs", type=int, default=1024)
    ap.add_argument("--held-seqs", type=int, default=256)
    ap.add_argument("--seq", type=int, default=2560)
    ap.add_argument("--snapshots", default="128,256,512,1024")
    ap.add_argument("--out", default="logs/gram_diag.json")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    snaps = [int(x) for x in args.snapshots.split(",") if int(x) <= args.max_seqs]
    dev = "cuda:0"

    print(f"loading FP model {args.model} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
                                                 dtype=torch.bfloat16).to(dev).eval()
    # locate the decoder layer list (hybrid Qwen3.5)
    core = model.model.language_model if hasattr(model.model, "language_model") else model.model
    dec = core.layers
    d = model.config.text_config.hidden_size if hasattr(model.config, "text_config") else model.config.hidden_size
    print(f"  hidden={d}, {len(dec)} layers; hooking inputs to layers {layers}", flush=True)

    # per-layer running Gram (fp64 on CPU) + per-snapshot copies
    H = {l: np.zeros((d, d), dtype=np.float64) for l in layers}
    Hsnap = {l: {} for l in layers}
    ntok = {l: 0 for l in layers}
    cur = {"X": {}}

    def mk_hook(l):
        def hook(mod, inp):
            x = inp[0] if isinstance(inp, tuple) else inp
            cur["X"][l] = x.detach().reshape(-1, x.shape[-1]).float()
        return hook
    handles = [dec[l].register_forward_pre_hook(mk_hook(l)) for l in layers]

    def run(seqs, accumulate=True):
        Hh = {l: np.zeros((d, d), dtype=np.float64) for l in layers}
        nt = 0
        for i, s in enumerate(seqs):
            ids = torch.tensor(s[:args.seq], dtype=torch.long, device=dev).unsqueeze(0)
            with torch.no_grad():
                model(ids)
            for l in layers:
                X = cur["X"][l]
                g = (X.t() @ X).double().cpu().numpy()
                if accumulate:
                    H[l] += g; ntok[l] += X.shape[0]
                else:
                    Hh[l] += g
            nt = i + 1
            if accumulate and nt in snaps:
                for l in layers:
                    Hsnap[l][nt] = H[l].copy()
                print(f"  calib snapshot {nt} seqs (~{ntok[layers[0]]/1e6:.2f}M tok)", flush=True)
            if (i + 1) % 128 == 0:
                print(f"  {'calib' if accumulate else 'held'} {i+1}/{len(seqs)}", flush=True)
        return Hh, nt

    calib = json.load(open(args.calib))[:args.max_seqs]
    held = json.load(open(args.held))[:args.held_seqs]
    print(f"calib {len(calib)} seqs, held {len(held)} seqs", flush=True)
    run(calib, accumulate=True)
    Hheld, _ = run(held, accumulate=False)
    for h in handles:
        h.remove()
    del model
    torch.cuda.empty_cache()

    # Persist the raw Grams BEFORE any analysis — the forwards are the expensive part (~15 min); a bug in the
    # cheap post-processing must never throw them away again.
    gram_npz = os.path.splitext(args.out)[0] + "_grams.npz"
    save = {}
    for l in layers:
        save[f"Hfull_{l}"] = H[l]; save[f"Hheld_{l}"] = Hheld[l]
        for n, hs in Hsnap[l].items():
            save[f"Hsnap_{l}_{n}"] = hs
    np.savez(gram_npz, **save)
    print(f"saved raw Grams -> {gram_npz}", flush=True)

    ks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    ks = [k for k in ks if k < d]
    report = {"model": args.model, "d": d, "layers": {}}
    fref = args.max_seqs
    for l in layers:
        Hfull = H[l]
        eigs, V = np.linalg.eigh(Hfull)                 # ascending
        eigs = np.clip(eigs, 0, None)
        r_eff, stable = eff_rank(eigs)
        # spectral convergence: ‖H_n − H_full‖_F / ‖H_full‖_F  (H_full = max-seqs)
        nf = np.linalg.norm(Hfull)
        conv = {}
        for n in snaps:
            if n == fref or n not in Hsnap[l]:
                conv[n] = 0.0 if n == fref else None
                continue
            conv[n] = float(np.linalg.norm(Hsnap[l][n] - Hfull) / nf)
        # held-out energy captured by calib top-k eigenbasis
        cap_cal = energy_captured(V, Hheld[l], ks)
        # baseline: held-out captured by its OWN top-k (upper bound)
        eh, Vh = np.linalg.eigh(Hheld[l]); eh = np.clip(eh, 0, None)
        cap_self = energy_captured(Vh, Hheld[l], ks)
        # eigenvalue decay (top fractions of total energy)
        tot = eigs.sum()
        cum = np.cumsum(eigs[::-1]) / tot
        frac_energy = {k: float(cum[k-1]) for k in ks}
        report["layers"][str(l)] = {
            "r_eff_tr/lmax": round(r_eff, 1),
            "stable_rank": round(stable, 1),
            "n_tokens": int(ntok[l]),
            "spectral_conv_Fnorm_rel": {str(k): (round(v, 4) if v is not None else None) for k, v in conv.items()},
            "heldout_energy_in_calib_topk": {str(k): round(v, 4) for k, v in cap_cal.items()},
            "heldout_energy_in_own_topk": {str(k): round(v, 4) for k, v in cap_self.items()},
            "calib_cum_energy_by_rank": {str(k): round(v, 4) for k, v in frac_energy.items()},
        }
        print(f"\n== layer {l} ==  r_eff(tr/λmax)={r_eff:.1f}  stable_rank={stable:.1f}  (d={d})")
        print(f"   spectral convergence ‖H_n−H_full‖/‖H_full‖: " +
              " ".join(f"{k}:{(conv[k] if conv[k] is not None else 0):.3f}" for k in snaps))
        print(f"   calib cum-energy by rank: " + " ".join(f"{k}:{frac_energy[k]:.2f}" for k in [8,32,128,256] if k in frac_energy))
        print(f"   held-out energy in calib top-k: " + " ".join(f"{k}:{cap_cal[k]:.3f}" for k in [32,128,256] if k in cap_cal))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")
    print("READ: r_eff ≪ d and spectral_conv → ~0 by 512 seqs ⇒ H saturates fast (data-limited claim CONFIRMED).")
    print("      held-out energy in calib top-k ≈ its own top-k ⇒ calib H spans held-out ⇒ plateau is CAPACITY, not estimation.")


if __name__ == "__main__":
    main()
