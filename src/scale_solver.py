#!/usr/bin/env python3
"""Q2-A "GPTQ-for-E2E": closed-form output-MSE-optimal per-256 SCALE solver, JOINT over blocks (uses the
FULL input Gram H = E[xxᵀ] with its cross-block terms — our activations are strongly cross-block correlated,
so the block-DIAGONAL approximation is wrong). For each output row i, the per-256 scales solve the exact
least-squares  minₛ ‖W_i x − Σ_b s_ib (T_ib·x_b)‖²  →  s_i = A_i⁻¹ b_i  with
    A_i[b,b'] = T_ib H_{b,b'} T_ib'ᵀ            (needs the full Gram, incl. cross-block b≠b')
    b_i[b]    = T_ib (H W_iᵀ)_b
Output folds to pure TQ2_0 (ternary×one fp16 scale/256). Warm-starts E2E (Test 2) / scale step of the
alternating round (Test 3).

MEMORY: full Grams are big (27B down_proj = 16384²·4 = 1GB × 64 layers = 64GB). We do NOT approximate them
away — we let them use RAM when they fit and spill to **NVMe memmap** when they don't, and query the OS
(memutil) to bail cleanly rather than crash. The joint solve loads ONE linear's Gram to GPU at a time.
"""
import os, sys, json, shutil, argparse, gc
import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM
import memutil

BS = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tern", required=True)
    ap.add_argument("--fp", default="output_4b/rot/modified_model")
    ap.add_argument("--orig-config", default="output_4b/untied_4b")
    ap.add_argument("--calib", default="output_4b/calibration_data.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-seq", type=int, default=64)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--block-size", type=int, default=BS)
    ap.add_argument("--ram-headroom-gb", type=float, default=10.0,
                    help="if OS-available RAM would drop below this, back Grams with NVMe memmap (else RAM).")
    ap.add_argument("--nvme-dir", default=None,
                    help="scratch dir for NVMe-backed Grams (default: <out>/../_gram_scratch).")
    args = ap.parse_args()
    dev = "cuda:0"; bs = args.block_size

    print(f"[ss] loading ternary {args.tern}  (OS-avail RAM {memutil.available_gb():.1f}G)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.tern, trust_remote_code=True, dtype=torch.bfloat16).to(dev).eval()
    lin = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
           and m.weight.shape[1] % bs == 0
           and any(k in n for k in ("proj", "fc", "mlp", "attn"))}

    # ---- decide backing: total full-Gram bytes vs OS-available RAM ----
    tot_bytes = sum(m.weight.shape[1] ** 2 * 4 for m in lin.values())
    avail = memutil.available_bytes() or 0
    use_nvme = (tot_bytes + args.ram_headroom_gb * 1e9) > avail
    scratch = args.nvme_dir or os.path.join(os.path.dirname(args.out.rstrip("/")) or ".", "_gram_scratch")
    if use_nvme:
        os.makedirs(scratch, exist_ok=True)
        print(f"[ss] full Grams = {tot_bytes/1e9:.1f}G > RAM headroom → NVMe-backed memmap in {scratch}", flush=True)
    else:
        print(f"[ss] full Grams = {tot_bytes/1e9:.1f}G fit in RAM ({avail/1e9:.1f}G avail) → in-memory", flush=True)

    grams, counts = {}, {}
    def _alloc(n, D):
        if use_nvme:
            path = os.path.join(scratch, n.replace("/", "_").replace(".", "_") + ".gram.npy")
            mm = np.memmap(path, dtype=np.float32, mode="w+", shape=(D, D)); mm[:] = 0
            return mm
        return torch.zeros(D, D, dtype=torch.float32)             # CPU RAM

    def mk(n):
        def h(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            xtx = (x.t() @ x)                                     # GPU
            g = grams.get(n)
            if g is None:
                grams[n] = g = _alloc(n, x.shape[1]); counts[n] = 0
            if use_nvme:
                g[:] += xtx.cpu().numpy()                         # accumulate on NVMe (page-cached; spills under pressure)
            else:
                g += xtx.cpu()
            counts[n] += x.shape[0]
        return h
    hooks = [m.register_forward_hook(mk(n)) for n, m in lin.items()]
    calib = json.load(open(args.calib))
    with torch.no_grad():
        for s in range(min(args.n_seq, len(calib))):
            memutil.require_ram_ok(args.ram_headroom_gb, "gram capture")   # clean bail, never crash the host
            model(torch.tensor([calib[s][:args.seq]], device=dev))
    for h in hooks:
        h.remove()

    fp_map = json.load(open(os.path.join(args.fp, "model.safetensors.index.json")))["weight_map"]
    def load_fp(model_name):
        for cand in (f"{model_name}.weight", f"{model_name}.weight".replace("model.layers", "model.language_model.layers")):
            if cand in fp_map:
                with safe_open(os.path.join(args.fp, fp_map[cand]), framework="pt", device="cpu") as f:
                    return f.get_tensor(cand)
        return None

    print(f"[ss] JOINT per-row solve for {len(lin)} linears (full Gram, one at a time on GPU)", flush=True)
    newW = {}
    for n, m in lin.items():
        if n not in grams or counts.get(n, 0) == 0:
            continue
        g = grams[n]
        H = torch.from_numpy(np.array(g)) if use_nvme else g     # [D,D] cpu
        H = (H / counts[n]).to(dev)                              # input covariance on GPU
        W = load_fp(n)
        if W is None:
            del H; continue
        W = W.to(dev, torch.float32)
        T = torch.sign(m.weight.data.to(dev, torch.float32))     # {-1,0,+1}
        out, D = W.shape; B = D // bs
        # b_i[b] = Σ_{j∈b} T[i,j] (W H)[i,j]      (H symmetric)
        WH = W @ H                                                # [out,D]
        b = (T * WH).view(out, B, bs).sum(-1)                    # [out,B]
        # A_i[b,b'] = Σ_{k∈b'} (Σ_{j∈b} T[i,j] H[j,k]) T[i,k]  — B block-restricted matmuls
        A = torch.empty(out, B, B, device=dev)
        for bb in range(B):
            sl = slice(bb * bs, (bb + 1) * bs)
            THb = T[:, sl] @ H[sl, :]                             # [out,D] = block-bb of T times H
            A[:, bb, :] = (THb * T).view(out, B, bs).sum(-1)      # [out,B]
        A += 1e-4 * torch.eye(B, device=dev)                     # ridge for singular blocks (all-zero support)
        s = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)   # [out,B]
        newW[n] = (s.unsqueeze(-1) * T.view(out, B, bs)).view(out, D).to(torch.bfloat16).cpu()
        del H, W, T, WH, b, A, s; grams[n] = None
        torch.cuda.empty_cache()

    print(f"[ss] writing {args.out} ({len(newW)} linears re-scaled)", flush=True)
    os.makedirs(args.out, exist_ok=True)
    wmap = json.load(open(os.path.join(args.tern, "model.safetensors.index.json")))["weight_map"]
    override = {}
    for n, W in newW.items():
        for cand in (f"{n}.weight", f"{n}.weight".replace("model.layers", "model.language_model.layers")):
            if cand in wmap:
                override[cand] = W; break
    shards = {}
    for dname, sf in wmap.items():
        shards.setdefault(sf, []).append(dname)
    for sf, names in shards.items():
        with safe_open(os.path.join(args.tern, sf), framework="pt", device="cpu") as f:
            d = {nm: (override[nm] if nm in override else f.get_tensor(nm)) for nm in names}
        save_file(d, os.path.join(args.out, sf)); del d; gc.collect()
    for fn in ["config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "generation_config.json", "merges.txt", "vocab.json"]:
        src = os.path.join(args.tern, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.out, fn))
    if use_nvme:
        shutil.rmtree(scratch, ignore_errors=True)
    print(f"[ss] done — {len(override)} linears re-scaled with JOINT closed-form s*", flush=True)


if __name__ == "__main__":
    main()
