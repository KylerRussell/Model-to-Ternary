#!/usr/bin/env python3
"""
C2 — Angular / Gram key-projection calibration (checklist cluster C2; reports R2-A1/A2, R3-3 OPD).

DeltaNet L2-normalizes q & k inside the kernel (use_qk_l2norm_in_kernel=True), so ternary MAGNITUDE
error on the key projection is cancelled — only the ANGULAR error of the (per-head, 128-dim) key
survives, and that angular error rotates the delta-rule erase axis (I - beta k k^T), smudging
unrelated memories (FM2). Standard block-AP/GPTQ minimizes weight MSE, which is blind to this.

This ternarizes ONLY the K-rows of the fused in_proj_qkv on every DeltaNet layer, two ways, on
cached calibration activations, leaving everything else FP:
  --objective mse      min ||W_k x - W~_k x||^2         (the standard objective, as a control)
  --objective angular  min mean_h (1 - cos(k_h, k~_h))  (per-head direction, the C2 objective)
then evaluates multi-key associative recall inline (recall_probe). If K-only-MSE collapses recall
and angular recovers it, C2 is a win. If K-only-MSE barely dents recall, K is not the bottleneck.

Foldable & TQ2_0-clean: output is ternary + per-256 scale for the same in_proj_qkv rows.
"""
import os, sys, argparse, json, math
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM, AutoTokenizer
import recall_probe

BS = 256
DELTA = "Qwen3_5GatedDeltaNet"


def ste_tern(L, s):
    """L [out,in] latent, s [out,nblk] positive block scale -> ternary-dequant with STE passthrough."""
    out, inp = L.shape
    nb = inp // BS
    Lb = L.view(out, nb, BS)
    q = torch.clamp(torch.round(Lb / s.unsqueeze(-1).clamp_min(1e-8)), -1, 1)
    W = (q * s.unsqueeze(-1)).reshape(out, inp)
    return L + (W - L).detach()


@torch.no_grad()
def collect_acts(model, layers, calib, tok_cap, seq, n_seq, dev):
    """Capture the input to each DeltaNet in_proj_qkv over calib; return {layer_idx: X [ntok,hidden] cpu}."""
    store = {i: [] for i in range(len(layers))}
    got = {i: 0 for i in range(len(layers))}
    hooks = []
    def mk(i):
        def hook(mod, inp):
            if got[i] < tok_cap:
                x = inp[0].detach().reshape(-1, inp[0].shape[-1])
                store[i].append(x[: tok_cap - got[i]].to("cpu", torch.float32))
                got[i] += x.shape[0]
        return hook
    for i, (_, dn) in enumerate(layers):
        hooks.append(dn.in_proj_qkv.register_forward_pre_hook(mk(i)))
    for s in range(min(n_seq, len(calib))):
        ids = torch.tensor([calib[s][:seq]], device=dev)
        model(ids)
        if all(g >= tok_cap for g in got.values()):
            break
    for h in hooks:
        h.remove()
    return {i: torch.cat(store[i], 0) for i in store}


def calibrate_layer(Wk, X, num_k_heads, head_k_dim, objective, steps, dev):
    """Return dequantized ternary Wk [out,in]. X [ntok,in] on dev. Optimize latent+scale via STE."""
    Wk = Wk.float().to(dev)
    X = X.to(dev)
    out, inp = Wk.shape
    nb = inp // BS
    L = Wk.clone().requires_grad_(True)
    s0 = Wk.view(out, nb, BS).abs().mean(-1).clamp_min(1e-8).clone()
    s = s0.requires_grad_(True)
    opt = torch.optim.Adam([L, s], lr=1e-3)
    with torch.no_grad():
        k_fp = (X @ Wk.T).view(-1, num_k_heads, head_k_dim)          # [ntok,H,dk]
        k_fp_n = torch.nn.functional.normalize(k_fp, dim=-1)
    for _ in range(steps):
        Wq = ste_tern(L, s)
        k_q = (X @ Wq.T).view(-1, num_k_heads, head_k_dim)
        if objective == "angular":
            k_q_n = torch.nn.functional.normalize(k_q, dim=-1)
            loss = (1 - (k_q_n * k_fp_n).sum(-1)).mean()
        else:  # mse on the projection output (equivalent objective to weight-MSE reconstruction)
            loss = ((k_q - k_fp) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            s.clamp_(min=1e-8)
    with torch.no_grad():
        Wout = ste_tern(L, s).detach()
        # report resulting angular error for logging
        k_q = (X @ Wout.T).view(-1, num_k_heads, head_k_dim)
        ang = (1 - torch.nn.functional.cosine_similarity(k_q, k_fp, dim=-1)).mean().item()
        sp = (Wout.view(out, nb, BS).abs() < 1e-9).float().mean().item()
    return Wout.to(torch.bfloat16).cpu(), ang, sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="FP rotated model")
    ap.add_argument("--orig-config", required=True)
    ap.add_argument("--calib", default="output_4b/calibration_data.json")
    ap.add_argument("--objective", choices=["mse", "angular"], required=True)
    ap.add_argument("--also-q", action="store_true", help="also ternarize the Q-rows (q is L2-normed too)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--n-seq", type=int, default=32)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--tok-cap", type=int, default=6000)
    ap.add_argument("--pairs", default="16,32,48,64")
    ap.add_argument("--trials", type=int, default=100)
    args = ap.parse_args()
    dev = "cuda:0"

    tok = AutoTokenizer.from_pretrained(args.orig_config, trust_remote_code=True)
    print(f"[c2] loading FP {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,
                                                 dtype=torch.bfloat16).to(dev).eval()
    cfg = model.config.get_text_config()
    num_k_heads, head_k_dim = cfg.linear_num_key_heads, cfg.linear_key_head_dim
    key_dim = num_k_heads * head_k_dim
    layers = [(n, m) for n, m in model.named_modules() if type(m).__name__ == DELTA]
    print(f"[c2] {len(layers)} DeltaNet layers | key_dim={key_dim} ({num_k_heads}x{head_k_dim}) | obj={args.objective}", flush=True)

    calib = json.load(open(args.calib))
    print(f"[c2] collecting activations ({args.n_seq} seq x {args.seq})...", flush=True)
    acts = collect_acts(model, layers, calib, args.tok_cap, args.seq, args.n_seq, dev)

    angs = []; sps = []
    for i, (name, dn) in enumerate(layers):
        W = dn.in_proj_qkv.weight.data
        X = acts[i]
        Wk = W[key_dim:2 * key_dim, :]                     # K rows
        Wk_q, ang, sp = calibrate_layer(Wk, X, num_k_heads, head_k_dim, args.objective, args.steps, dev)
        W[key_dim:2 * key_dim, :] = Wk_q.to(W.device)
        angs.append(ang); sps.append(sp)
        if args.also_q:
            Wq_ = W[0:key_dim, :]
            Wq_q, _, _ = calibrate_layer(Wq_, X, num_k_heads, head_k_dim, args.objective, args.steps, dev)
            W[0:key_dim, :] = Wq_q.to(W.device)
        del X, acts[i]
        torch.cuda.empty_cache()
    print(f"[c2] mean per-layer key angular-err={sum(angs)/len(angs):.4f}  mean K sparsity={sum(sps)/len(sps):.3f}", flush=True)

    n_list = [int(x) for x in args.pairs.split(",")]
    res = recall_probe.run(model, tok, dev, n_list, args.trials, seed=0)
    tag = f"C2-{args.objective}{'+q' if args.also_q else ''} (K-only ternary, rest FP)"
    print(f"\n===== RECALL PROBE [{tag}]  trials={args.trials} =====")
    print(f"  {'Npairs':>7} {'full':>7} {'restr':>7} {'early':>7} {'mid':>7} {'late':>7}")
    for N in n_list:
        r = res[N]
        print(f"  {N:>7} {r['full']:>7.3f} {r['restr']:>7.3f} {r['pos']['early']:>7.3f} {r['pos']['mid']:>7.3f} {r['pos']['late']:>7.3f}")
    print(f"  MEAN restr over N = {sum(res[N]['restr'] for N in n_list)/len(n_list):.4f}")


if __name__ == "__main__":
    main()
