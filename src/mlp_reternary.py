#!/usr/bin/env python3
"""
MLP-targeted ternary reconstruction (C1 / C6), motivated by the component ablation: the MLP is
~79% of the ternary retrieval collapse, so better MLP ternarization is the highest-value source fix.

Uses GPTQ (the pipeline's proven Hessian-error-feedback method) so the baseline reconstruction is
strong and stable (a hand-rolled STE was both unstable and too weak to be a fair baseline).
Re-ternarizes every MLP (gate/up/down) from FP with a chosen objective, leaves everything else FP,
and measures multi-key associative recall. Anchors: FP=0.9975, real-ternary-MLP≈0.560, full=0.4425.

Objectives:
  independent  each linear GPTQ'd against its own FP-input Hessian; down fit to the FP intermediate.
  joint (C6)   gate/up GPTQ'd, then the down_proj target is CORRECTED: solve the ternary-input least
               squares  min_D || D h_st - out_fp ||  (h_st = ternary intermediate, out_fp = FP output)
               so down absorbs the gate/up quantization error — intra-block error correction that
               exploits the excess width (anti-correlated-clone / TMR spirit, no hand-pairing).
  decision(C1) weight the calibration Hessian (and the joint regression) by FP low-margin "decision"
               tokens (gamma=2, matching the E2E gamma=2 win) — protects decision-relevant directions.
Combine C1+C6 with --method joint --decision.
"""
import os, sys, argparse, json
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.activations import ACT2FN
from quantizer import quantize_gptq_ternary, dequantize
import recall_probe

BS = 256
MLP = "Qwen3_5MLP"


def gptq_tern(W, H, dev):
    """Ternarize W [out,in] via GPTQ against Hessian H [in,in]; return dequantized bf16 weight."""
    qt = quantize_gptq_ternary(W.float().to(dev), H.to(dev), block_size=BS)
    return dequantize(qt).float().to(dev)


@torch.no_grad()
def collect(model, mlps, calib, tok_cap, seq, n_seq, dev, want_weights, gamma):
    store = {i: [] for i in range(len(mlps))}
    got = {i: 0 for i in range(len(mlps))}
    tokw = []
    hooks = []
    def mk(i):
        def hook(mod, inp):
            if got[i] < tok_cap:
                x = inp[0].detach().reshape(-1, inp[0].shape[-1])
                store[i].append(x[: tok_cap - got[i]].to("cpu", torch.float32))
                got[i] += x.shape[0]
        return hook
    for i, (_, m) in enumerate(mlps):
        hooks.append(m.register_forward_pre_hook(mk(i)))
    for s in range(min(n_seq, len(calib))):
        out = model(torch.tensor([calib[s][:seq]], device=dev))
        if want_weights:
            p = out.logits[0].float().softmax(-1)
            top2 = p.topk(2, dim=-1).values
            tokw.append((top2[:, 0] - top2[:, 1]).cpu())          # margin; low = decision token
        if all(g >= tok_cap for g in got.values()):
            break
    for h in hooks:
        h.remove()
    X = {i: torch.cat(store[i], 0) for i in store}
    W = None
    if want_weights:
        m = torch.cat(tokw, 0)[: X[0].shape[0]]
        w = torch.ones_like(m) + gamma * (m < torch.quantile(m, 0.33)).float()
        W = (w / w.mean())                                        # mean-1 -> redistribute, not upscale
    return X, W


def wh(A, w):
    """Weighted Hessian A^T diag(w) A for rows-as-tokens A [ntok,d]."""
    if w is None:
        return A.T @ A
    Aw = A * w.sqrt().unsqueeze(1)
    return Aw.T @ Aw


def reconstruct_mlp(mlp, X, tokw, act_fn, method, decision, dev, damp=1e-2):
    Wg, Wu, Wd = mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight
    Wg0, Wu0, Wd0 = Wg.data.float().to(dev), Wu.data.float().to(dev), Wd.data.float().to(dev)
    X = X.to(dev)
    w = tokw.to(dev) if (decision and tokw is not None) else None
    with torch.no_grad():
        g_fp, u_fp = X @ Wg0.T, X @ Wu0.T
        h_fp = act_fn(g_fp) * u_fp
        out_fp = h_fp @ Wd0.T
        Hx = wh(X, w)
        Gq = gptq_tern(Wg0, Hx, dev)
        Uq = gptq_tern(Wu0, Hx, dev)
        if method == "joint":
            h_st = act_fn(X @ Gq.T) * (X @ Uq.T)                  # ternary intermediate
            Hh = wh(h_st, w)
            reg = Hh + damp * torch.diag(Hh).mean() * torch.eye(Hh.shape[0], device=dev)
            rhs = (h_st * (w.unsqueeze(1) if w is not None else 1.0)).T @ out_fp   # h_st^T (w) out_fp
            Dstar = torch.linalg.solve(reg, rhs).T                # [hidden, inter] corrected down
            Dq = gptq_tern(Dstar, Hh, dev)
        else:
            Hh = wh(h_fp, w)
            Dq = gptq_tern(Wd0, Hh, dev)
        Wg.data.copy_(Gq.to(Wg.dtype)); Wu.data.copy_(Uq.to(Wu.dtype)); Wd.data.copy_(Dq.to(Wd.dtype))
        h_final = act_fn(X @ Gq.T) * (X @ Uq.T)
        out_st = h_final @ Dq.T
        rel = ((out_st - out_fp).norm() / (out_fp.norm() + 1e-8)).item()
    return rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="output_4b/rot/modified_model")
    ap.add_argument("--orig-config", default="output_4b/untied_4b")
    ap.add_argument("--calib", default="output_4b/calibration_data.json")
    ap.add_argument("--method", choices=["independent", "joint"], default="independent")
    ap.add_argument("--decision", action="store_true")
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--n-seq", type=int, default=48)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--tok-cap", type=int, default=6000)
    ap.add_argument("--n-layers", type=int, default=0)
    ap.add_argument("--pairs", default="16,32,48,64")
    ap.add_argument("--trials", type=int, default=100)
    args = ap.parse_args()
    dev = "cuda:0"
    tok = AutoTokenizer.from_pretrained(args.orig_config, trust_remote_code=True)
    print(f"[mlp] loading FP {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, dtype=torch.bfloat16).to(dev).eval()
    act_fn = ACT2FN[model.config.get_text_config().hidden_act]
    mlps = [(n, m) for n, m in model.named_modules() if type(m).__name__ == MLP]
    if args.n_layers:
        mlps = mlps[: args.n_layers]
    print(f"[mlp] {len(mlps)} MLPs | GPTQ | method={args.method}{'+decision' if args.decision else ''}", flush=True)
    calib = json.load(open(args.calib))
    print(f"[mlp] collecting inputs ({args.n_seq}x{args.seq}) decision={args.decision}...", flush=True)
    X, tokw = collect(model, mlps, calib, args.tok_cap, args.seq, args.n_seq, dev, args.decision, args.gamma)
    rels = []
    for i, (name, m) in enumerate(mlps):
        tw = tokw[: X[i].shape[0]] if tokw is not None else None
        rels.append(reconstruct_mlp(m, X[i], tw, act_fn, args.method, args.decision, dev))
        del X[i]; torch.cuda.empty_cache()
    print(f"[mlp] mean per-MLP relative output error = {sum(rels)/len(rels):.4f}", flush=True)
    n_list = [int(x) for x in args.pairs.split(",")]
    res = recall_probe.run(model, tok, dev, n_list, args.trials, seed=0)
    tag = f"MLP-{args.method}{'+decision' if args.decision else ''}"
    print(f"\n===== RECALL [{tag}] (MLP ternary, rest FP) =====")
    print(f"  {'Npairs':>7} {'full':>7} {'restr':>7}")
    for N in n_list:
        print(f"  {N:>7} {res[N]['full']:>7.3f} {res[N]['restr']:>7.3f}")
    print(f"  MEAN restr = {sum(res[N]['restr'] for N in n_list)/len(n_list):.4f}")


if __name__ == "__main__":
    main()
