#!/usr/bin/env python3
"""
block_qat.py — Block-wise GLOBAL QAT for ternary Qwen3.6-27B.

Scale-only E2E-QP plateaued at ~1.70x perplexity because it can't change which weights are
-1/0/+1. This sweeps the network block by block: one block at a time gets full-precision
LATENT weights (straight-through ternary, so the ASSIGNMENTS can flip), trained against the
GLOBAL teacher KL through the ordinary full-model forward, while every other block stays
frozen as packed 2-bit ternary. Then the block is re-packed and we move on. Only one block's
latents+optimizer are ever resident, so it fits a single 24 GB 3090.

This trains the thing scale-only couldn't (assignments), keeps the model 100% ternary, and
uses the final-logit teacher cache you already built. It is SLOW — every step is a full 27B
forward+backward — so use --max-blocks to do a few blocks first and re-eval before committing
to all 64, and it checkpoints after every block (resume with --student-path <out> --start-block N).

Keep DeltaNet on its pure-PyTorch path (no flash-linear-attention / causal-conv1d) so backprop
flows through the recurrence.

Usage:
  python block_qat.py \
      --student-path ./output_e2eqp_cont/modified_model \
      --orig-config-path /path/to/Qwen3.6-27B/snapshots/<hash> \
      --calib ./output_recovery/calibration_data.json \
      --teacher-cache ./output_recovery/teacher_topk.pt \
      --out ./output_qat/modified_model --seq 512 --group-size 2 --passes 3 --steps-per-block 60

Each PASS sweeps the whole model in GROUPS of --group-size adjacent blocks trained jointly
against the global teacher KL; repeating for several --passes lets every block co-adapt to the
others (Gauss-Seidel toward the joint optimum the way true full QAT would). Eval at the end of
each pass: if perplexity keeps dropping pass over pass, the limit is method/compute-bound (keep
passing); if it flattens, you've found the 1.58-bit information ceiling. Resume more passes with
--start-pass. Use --adam8bit for memory headroom to raise --group-size or --seq.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import gc
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent))
from config import BLOCK_SIZE, NUM_HIDDEN_LAYERS
from e2e_qp_distill import (build_student, save_student, topk_kl_loss,
                            load_calib_batches, TernaryScaleLinear, extract_ternary_scale)


def ste_ternary(W, s, block_size):
    """Differentiable ternary: forward s*round(clamp(W/s)); straight-through on round."""
    out, inp = W.shape
    flat = W.reshape(out * (inp // block_size), block_size)
    wc = (flat / s.unsqueeze(1)).clamp(-1, 1)
    wr = torch.round(wc)
    wq = wc + (wr - wc).detach()
    return (wq * s.unsqueeze(1)).reshape(out, inp)


class STETernaryLinear(nn.Module):
    """Trainable full-precision latent + trainable per-g128 scale, quantized via STE on the
    forward. Initialized from the current dequantized ternary so it starts where the packed
    model is, then floats during QAT (the round() can re-pick assignments)."""

    def __init__(self, dequant_weight, block_size, device):
        super().__init__()
        out, inp = dequant_weight.shape
        self.out_features, self.in_features, self.block_size = out, inp, block_size
        _, scale = extract_ternary_scale(dequant_weight.float(), block_size)
        self.latent = nn.Parameter(dequant_weight.float().to(device))
        self.scale = nn.Parameter(scale.float().to(device))

    def forward(self, x):
        w = ste_ternary(self.latent, self.scale, self.block_size).to(x.dtype)
        return F.linear(x, w)

    @torch.no_grad()
    def deployed_dequant(self):
        out, inp = self.latent.shape
        flat = self.latent.reshape(out * (inp // self.block_size), self.block_size)
        q = torch.round((flat / self.scale.unsqueeze(1)).clamp(-1, 1))
        return (q * self.scale.unsqueeze(1)).reshape(out, inp).to(torch.float16)


def _named_child(parent, dotted):
    obj = parent
    for p in dotted.split("."):
        obj = getattr(obj, p)
    return obj


def _set_child(parent, dotted, value):
    *pre, leaf = dotted.split(".")
    obj = parent
    for p in pre:
        obj = getattr(obj, p)
    setattr(obj, leaf, value)


def swap_block_to_ste(layer, block_size, device):
    """Replace each TernaryScaleLinear in the layer with a trainable STETernaryLinear.
    Returns the list of (name, original_module) so we can re-pack afterwards."""
    swapped = []
    for name, mod in list(layer.named_modules()):
        if isinstance(mod, TernaryScaleLinear):
            ste = STETernaryLinear(mod.dequant(), block_size, device)
            _set_child(layer, name, ste)
            swapped.append(name)
    return swapped


def repack_block_from_ste(layer, names, block_size, device):
    """Replace each trained STETernaryLinear back with a frozen packed TernaryScaleLinear."""
    for name in names:
        ste = _named_child(layer, name)
        tsl = TernaryScaleLinear.from_dense(ste.deployed_dequant(), block_size, device=device)
        for p in tsl.parameters():
            p.requires_grad_(False)
        _set_child(layer, name, tsl)


def _module_device(module):
    """Device of any param/buffer in a module (its placement after dispatch)."""
    for t in list(module.parameters()) + list(module.buffers()):
        return t.device
    return torch.device("cuda:0")


def distribute_model(model, single_gpu):
    """Split the model across all visible GPUs with an EXPLICIT even layer split, so both cards
    carry ~half the layers (accelerate's get_balanced_memory under-loads GPU 0 and over-loads the
    last GPU, which OOMs when a group's latents land there). Embedding goes with the first chunk,
    norm + LM head with the last. Activations cross the GPU boundary via align-device hooks."""
    n = torch.cuda.device_count()
    if single_gpu or n < 2:
        model = model.to("cuda:0")
        print(f"   single-GPU: everything on cuda:0")
        return model, torch.device("cuda:0")
    from accelerate import dispatch_model
    L = len(model.model.layers)
    per = -(-L // n)                                   # ceil: contiguous even chunks
    dmap = {f"model.layers.{i}": min(i // per, n - 1) for i in range(L)}
    for name, _ in model.model.named_children():       # non-layer submodules of the inner model
        if name == "layers":
            continue
        dmap[f"model.{name}"] = 0 if name in ("embed_tokens", "rotary_emb") else n - 1
    dmap["lm_head"] = n - 1
    model = dispatch_model(model, device_map=dmap)
    in_dev = _module_device(model.model.embed_tokens)
    counts = {}
    for i in range(L):
        d = str(_module_device(model.model.layers[i]))
        counts[d] = counts.get(d, 0) + 1
    print(f"   dispatched across {n} GPUs (even split); layers/GPU {counts}; input {in_dev}")
    return model, in_dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-path", required=True, help="Current best ternary model (E2E-QP out).")
    ap.add_argument("--orig-config-path", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--teacher-cache", required=True, help="final-logit top-k cache (same as E2E-QP).")
    ap.add_argument("--out", default="./output_qat/modified_model")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--steps-per-block", type=int, default=60,
                    help="Training steps per GROUP per pass.")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--group-size", type=int, default=2,
                    help="Adjacent blocks trained jointly (more leverage; more memory).")
    ap.add_argument("--passes", type=int, default=3,
                    help="Full sweeps over the model; more passes -> closer to true full QAT.")
    ap.add_argument("--start-pass", type=int, default=0, help="Resume from this pass index.")
    ap.add_argument("--adam8bit", action="store_true",
                    help="Use bitsandbytes 8-bit Adam (frees memory for bigger --group-size/--seq).")
    ap.add_argument("--single-gpu", action="store_true",
                    help="Force everything onto cuda:0 (default: split the model across all GPUs).")
    ap.add_argument("--save-every", type=int, default=8,
                    help="Checkpoint every N groups (and always at end of each pass).")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-batches", type=int, default=4,
                    help="Held-out sequences used to score keep-best (same set every eval).")
    ap.add_argument("--eval-every", type=int, default=15,
                    help="Run the held-out keep-best eval every N steps.")
    args = ap.parse_args()

    multi = torch.cuda.device_count() > 1 and not args.single_gpu
    build_dev = "cpu" if multi else ("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Building packed ternary student (frozen context)...")
    model, _ = build_student(args.student_path, args.orig_config_path, BLOCK_SIZE, build_dev)
    for p in model.parameters():                       # freeze everything to start
        p.requires_grad_(False)
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        print("   gradient checkpointing enabled (non-reentrant)")
    except Exception as e:
        try:
            model.gradient_checkpointing_enable()
            print("   gradient checkpointing enabled")
        except Exception as e2:
            print(f"   ⚠️ gradient_checkpointing_enable failed ({e2}); reduce --seq if OOM")
    model, input_device = distribute_model(model, args.single_gpu)

    cache = torch.load(args.teacher_cache)
    batches = load_calib_batches(args.calib, 1, args.seq, "cpu")   # moved to GPU per-step
    n = min(len(batches), len(cache["idx"]))
    if args.max_samples:
        n = min(n, args.max_samples)
    print(f"{n} calibration sequences x {args.seq} tokens")

    # Reserve a fixed held-out slice for the keep-best signal; train on the rest. Scoring every
    # candidate on the SAME sequences makes KL comparable step-to-step (the per-batch training
    # loss is not — each training step sees a different sequence).
    n_eval = min(args.eval_batches, max(1, n // 4))
    eval_idx = list(range(n - n_eval, n))
    train_idx = list(range(0, n - n_eval)) or eval_idx
    print(f"   keep-best on {len(eval_idx)} held-out sequences; {len(train_idx)} for training")

    def make_opt(params, lr):
        if args.adam8bit:
            try:
                import bitsandbytes as bnb
                return bnb.optim.Adam8bit(params, lr=lr)
            except Exception as e:
                print(f"   ⚠️ bitsandbytes unavailable ({e}); using fp32 Adam")
        return torch.optim.Adam(params, lr=lr)

    @torch.no_grad()
    def eval_kl(idxs):
        """Mean top-k KL vs the teacher over a FIXED set of sequences (no grad)."""
        was_training = model.training
        model.eval()
        tot = 0.0
        for bi in idxs:
            seqL = min(args.seq, cache["idx"][bi].shape[0])
            ids = batches[bi][:, :seqL].to(input_device)
            logits = model(ids).logits[:, :seqL]
            t_idx = cache["idx"][bi][:seqL].unsqueeze(0).to(logits.device)
            t_val = cache["val"][bi][:seqL].unsqueeze(0).to(logits.device)
            tot += topk_kl_loss(logits, t_idx, t_val, args.temperature).item()
        if was_training:
            model.train()
        return tot / max(len(idxs), 1)

    layers = model.model.layers
    G = max(1, args.group_size)
    group_starts = list(range(0, NUM_HIDDEN_LAYERS, G))
    groups_done = 0
    for pass_i in range(args.start_pass, args.passes):
        print(f"\n========== PASS {pass_i + 1}/{args.passes} (group size {G}) ==========")
        for gs in group_starts:
            block_ids = list(range(gs, min(gs + G, NUM_HIDDEN_LAYERS)))
            grp_layers = [layers[b] for b in block_ids]
            # Pin the current CUDA device to this group's card. The group never straddles the
            # device split, and bitsandbytes' 8-bit optimizer (plus any current-device-dependent
            # kernel) acts on torch.cuda.current_device(): if that's cuda:0 while the params live
            # on cuda:1, its kernels read/write the wrong addresses -> illegal memory access.
            grp_dev = _module_device(grp_layers[0])
            restore_dev = (torch.cuda.current_device()
                           if (torch.cuda.is_available() and grp_dev.type == "cuda") else None)
            if restore_dev is not None:
                torch.cuda.set_device(grp_dev)
            swapped = [swap_block_to_ste(L, BLOCK_SIZE, _module_device(L)) for L in grp_layers]
            model.train()
            params = [pp for L in grp_layers for _, pp in L.named_parameters() if pp.requires_grad]
            opt = make_opt(params, args.lr)
            print(f"\n■ Pass {pass_i + 1} | blocks {block_ids[0]}–{block_ids[-1]}: "
                  f"QAT on {sum(pp.numel() for pp in params) / 1e6:.0f}M latent params")

            # The incoming (pre-training) state is the baseline candidate. The STE init reproduces
            # the packed weights exactly, so if training never beats this the group repacks
            # bit-identically — QAT can only help, never regress.
            best_kl = eval_kl(eval_idx)
            best = [pp.detach().to("cpu", copy=True) for pp in params]  # keep snapshot off-GPU
            print(f"   baseline held-out KL {best_kl:.4f}")

            step = 0
            while step < args.steps_per_block:
                for bi in train_idx:
                    if step >= args.steps_per_block:
                        break
                    seqL = min(args.seq, cache["idx"][bi].shape[0])
                    ids = batches[bi][:, :seqL].to(input_device)
                    logits = model(ids).logits[:, :seqL]
                    t_idx = cache["idx"][bi][:seqL].unsqueeze(0).to(logits.device)
                    t_val = cache["val"][bi][:seqL].unsqueeze(0).to(logits.device)
                    loss = topk_kl_loss(logits, t_idx, t_val, args.temperature)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    if step < args.warmup:
                        for g in opt.param_groups:
                            g["lr"] = args.lr * (step + 1) / max(1, args.warmup)
                    opt.step()
                    step += 1
                    # held-out eval drives keep-best; the final step always scores
                    if step % args.eval_every == 0 or step == args.steps_per_block:
                        ek = eval_kl(eval_idx)
                        if ek < best_kl:
                            best_kl = ek
                            best = [pp.detach().to("cpu", copy=True) for pp in params]  # keep snapshot off-GPU
                        print(f"   step {step}/{args.steps_per_block}  "
                              f"train_KL={loss.item():.4f}  held-out_KL={ek:.4f}  best={best_kl:.4f}")
                    elif step % args.log_every == 0:
                        print(f"   step {step}/{args.steps_per_block}  "
                              f"train_KL={loss.item():.4f}  best={best_kl:.4f}")

            opt.zero_grad(set_to_none=True)                 # drop ~3GB of grads before repacking
            with torch.no_grad():                           # restore best (≥ incoming), then re-pack
                for pp, b in zip(params, best):
                    pp.copy_(b)                             # b is on CPU; copy_ handles CPU->GPU
            for L, names in zip(grp_layers, swapped):
                repack_block_from_ste(L, names, BLOCK_SIZE, _module_device(L))
            del opt, best, params
            gc.collect()
            torch.cuda.empty_cache()
            if restore_dev is not None:
                torch.cuda.set_device(restore_dev)

            groups_done += 1
            tag = f"pass {pass_i + 1} blocks {block_ids[0]}–{block_ids[-1]}"
            if groups_done % args.save_every == 0:
                save_student(model, args.student_path, args.out, BLOCK_SIZE)
                print(f"   ✓ {tag} (best held-out KL {best_kl:.4f}); checkpoint -> {args.out}")
            else:
                print(f"   ✓ {tag} (best held-out KL {best_kl:.4f})")

        save_student(model, args.student_path, args.out, BLOCK_SIZE)   # end-of-pass checkpoint
        print(f"\n== end of pass {pass_i + 1}: checkpoint -> {args.out} "
              f"(eval here; if ppl still dropping, keep passing) ==")

    print(f"\nDone ({args.passes - args.start_pass} passes, group size {G}). Model -> {args.out}")
    print("Resume more passes with:  --student-path <out> --start-pass <next> --passes <total>")


if __name__ == "__main__":
    main()