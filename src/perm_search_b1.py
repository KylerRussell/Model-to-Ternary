#!/usr/bin/env python
"""B1 permutation search (A1/B1 axis, stage-2 report objective (2)): find a per-layer permutation of the
MLP-intermediate channels that minimizes the summed per-(row,256-block) TERNARY distortion of down_proj —
the objective matched to the deployed TQ2_0 grid, unlike SSR's range-sort (objective 1).

Per (row, block) with values v (256): TWN threshold D = 0.7*mean|v|; m = |v|>D; s = mean(|v|[m]);
mse = sum_{~m} v^2 + sum_m (|v|-s)^2.   (TWN arXiv:1605.04711 optimal-threshold form.)

Search: init from the SSR sort (good 1-D start), then batched pair-swap local search on GPU, greedy
accept of non-conflicting improving swaps. Reports the objective for identity/ssr/random/optimized and
saves {layer_idx: LongTensor} to --out for block_ap_recovery --perm-mode optimized --perm-file.

Env-free CLI: --model output_4b/rot/modified_model --out output_4b/perm_b1.pt --iters 400 --cand 192
"""
import argparse, json, time
import torch
from pathlib import Path
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="output_4b/rot/modified_model")
ap.add_argument("--out", default="output_4b/perm_b1.pt")
ap.add_argument("--report", default="logs/perm_b1_search.json")
ap.add_argument("--block", type=int, default=256)
ap.add_argument("--iters", type=int, default=400, help="swap-batch iterations per layer")
ap.add_argument("--cand", type=int, default=192, help="candidate swaps per batch")
ap.add_argument("--device", default="cuda:0")
args = ap.parse_args()
B = args.block
dev = args.device

idx = json.load(open(Path(args.model) / "model.safetensors.index.json"))["weight_map"]
def get(name):
    with safe_open(str(Path(args.model) / idx[name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name).float()

import re
dn_names = {}
for n in idx:
    m = re.search(r"layers\.(\d+)\.mlp\.down_proj\.weight$", n)
    if m:
        dn_names[int(m.group(1))] = n
layers = sorted(dn_names)
print(f"B1 perm search: {len(layers)} layers | block={B} | iters={args.iters}x{args.cand}", flush=True)


def block_mse(Wb):
    """Wb [..., B] -> ternary TWN distortion per block [...]."""
    a = Wb.abs()
    D = 0.7 * a.mean(-1, keepdim=True)
    m = a > D
    cnt = m.sum(-1).clamp_min(1)
    s = (a * m).sum(-1) / cnt
    mse = (Wb.pow(2) * (~m)).sum(-1) + ((a - s.unsqueeze(-1)).pow(2) * m).sum(-1)
    return mse


def total_obj(W, perm):
    Wp = W[:, perm].view(W.shape[0], -1, B)
    return block_mse(Wp).sum().item()


perms, report = {}, {}
t00 = time.time()
for L in layers:
    W = get(dn_names[L]).to(dev)                            # [O, Dint]
    O, Dn = W.shape
    nb = Dn // B
    ident = torch.arange(Dn, device=dev)
    ssr = torch.argsort(W.norm(dim=0))                       # the SSR sort (objective-1 baseline)
    g = torch.Generator(device="cpu").manual_seed(L)
    rnd = torch.randperm(Dn, generator=g).to(dev)
    obj_i, obj_s, obj_r = total_obj(W, ident), total_obj(W, ssr), total_obj(W, rnd)

    # ---- local search from the SSR init on the TRUE ternary objective ----
    perm = ssr.clone()
    Wp = W[:, perm].view(O, nb, B)
    bm = block_mse(Wp)                                       # [O, nb]
    blk = bm.sum(0)                                          # per-block objective [nb]
    gcpu = torch.Generator().manual_seed(1000 + L)
    accepted = 0
    for it in range(args.iters):
        K = args.cand
        # half uniform, half sourced from the worst blocks (where distortion lives)
        ba = torch.randint(0, nb, (K,), generator=gcpu)
        worst = torch.topk(blk, max(2, nb // 4)).indices.cpu()
        ba[: K // 2] = worst[torch.randint(0, worst.numel(), (K // 2,), generator=gcpu)]
        bb = torch.randint(0, nb, (K,), generator=gcpu)
        ok = ba != bb
        ba, bb = ba[ok].to(dev), bb[ok].to(dev)
        K = ba.numel()
        if K == 0:
            continue
        pa = torch.randint(0, B, (K,), device=dev)
        pb = torch.randint(0, B, (K,), device=dev)
        A = Wp[:, ba, :].permute(1, 0, 2).clone()            # [K, O, B] candidate block-a contents
        Bc = Wp[:, bb, :].permute(1, 0, 2).clone()
        va = A[torch.arange(K, device=dev), :, pa].clone()
        vb = Bc[torch.arange(K, device=dev), :, pb].clone()
        A[torch.arange(K, device=dev), :, pa] = vb           # swap the two columns
        Bc[torch.arange(K, device=dev), :, pb] = va
        newA = block_mse(A).sum(1)                           # [K]
        newB = block_mse(Bc).sum(1)
        delta = (newA + newB) - (blk[ba] + blk[bb])
        order = torch.argsort(delta)
        used = set()
        for k in order.tolist():
            if delta[k] >= -1e-7:
                break
            a_, b_ = int(ba[k]), int(bb[k])
            if a_ in used or b_ in used:
                continue
            used.add(a_); used.add(b_)
            ia, ib = a_ * B + int(pa[k]), b_ * B + int(pb[k])
            perm[ia], perm[ib] = perm[ib].clone(), perm[ia].clone()
            accepted += 1
        if used:                                             # refresh views/caches after commits
            Wp = W[:, perm].view(O, nb, B)
            bm = block_mse(Wp)
            blk = bm.sum(0)
    obj_o = blk.sum().item()
    perms[L] = perm.cpu()
    report[L] = {"identity": obj_i, "ssr": obj_s, "random": obj_r, "optimized": obj_o,
                 "swaps_accepted": accepted,
                 "gain_vs_ident_pct": 100 * (obj_i - obj_o) / max(obj_i, 1e-9),
                 "gain_vs_ssr_pct": 100 * (obj_s - obj_o) / max(obj_s, 1e-9)}
    print(f"  L{L:2d}: ident={obj_i:.1f} rnd={obj_r:.1f} ssr={obj_s:.1f} opt={obj_o:.1f} "
          f"(vs ident −{report[L]['gain_vs_ident_pct']:.2f}%, vs ssr −{report[L]['gain_vs_ssr_pct']:.2f}%, "
          f"{accepted} swaps)", flush=True)
    del W, Wp
    torch.cuda.empty_cache()

torch.save(perms, args.out)
json.dump(report, open(args.report, "w"), indent=1)
gi = sum(r["gain_vs_ident_pct"] for r in report.values()) / len(report)
gs = sum(r["gain_vs_ssr_pct"] for r in report.values()) / len(report)
print(f"\nDONE in {time.time()-t00:.0f}s. Mean ternary-MSE gain: −{gi:.2f}% vs identity, −{gs:.2f}% vs ssr.")
print(f"perms -> {args.out} | report -> {args.report}")
