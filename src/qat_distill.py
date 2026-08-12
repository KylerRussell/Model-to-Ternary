#!/usr/bin/env python3
"""qat_distill.py — distillation-QAT for 1.58-bit ternary (TQ2_0-foldable).

The structural fix our prior assignment-QAT lacked (ParetoQ): train LATENT full-precision weights,
INITIALIZED FROM THE FP (rotated) TEACHER — not the frozen post-hoc grid — through a learnable
SEQ/LSQ ternary quantizer with STE, distilling against the FP teacher (CAKLD over cached top-64).
Per-256 scales co-trained (LSQ gradient scaling). Output folds to pure ternary {-1,0,+1} + one fp
scale per 256 (no extra params, no mixed precision) -> standard HF dir that the TQ2_0 export reads.

  python qat_distill.py --rot-model output_2b_ab/rot/modified_model --orig-config output_2b_ab/untied_2b \
      --calib output_2b_ab/calibration_data.json --teacher-cache output_2b_ab/teacher_topk.pt \
      --out output_2b_ab/qat/modified_model --steps 4000 --lr 2e-5 --scale-lr 1e-5
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse, math, time, sys
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F

sys.path.append(str(Path(__file__).parent))
from config import should_quantize, BLOCK_SIZE
from e2e_qp_distill import cakld_loss, topk_kl_loss, load_calib_batches


def grad_scale(x, g):                       # LSQ: scale the gradient by g, value unchanged
    return (x - x * g).detach() + x * g


def ste_ternary(x):                         # clip to [-1,1] (grad 0 outside) then round via STE (grad 1)
    c = x.clamp(-1, 1)
    return c + (c.round() - c).detach()


class LatentSEQLinear(nn.Module):
    """FP-init latent weight + learnable per-256 step (SEQ/LSQ), STE -> ternary {-1,0,+1}*scale."""
    def __init__(self, weight, bias, block_size, lora_rank=0, latent_dtype=torch.float32):
        super().__init__()
        out, inp = weight.shape
        assert inp % block_size == 0, f"{inp} % {block_size}"
        self.out, self.inp, self.bs, self.nb = out, inp, block_size, out * (inp // block_size)
        # full QAT: train the latent. E2E polish (lora_rank>0): FREEZE the latent base, train a low-rank
        # delta + the scales; merge & re-quantize at export so the output stays pure ternary+per-256-scale.
        # latent_dtype=bf16 halves latent+grad memory so end-to-end QAT fits a 24GB GPU on mid models (4B).
        self.latent = nn.Parameter(weight.detach().to(latent_dtype).clone(), requires_grad=(lora_rank == 0))
        flat = weight.detach().float().reshape(self.nb, block_size)
        s0 = (2.0 * flat.abs().mean(dim=1)).clamp_min(1e-8)                       # ParetoQ-style step init
        self.log_scale = nn.Parameter(s0.log())                                  # per-256, log-space (>0)
        if lora_rank > 0:
            self.lora_A = nn.Parameter(torch.randn(lora_rank, inp) * (inp ** -0.5))
            self.lora_B = nn.Parameter(torch.zeros(out, lora_rank))              # B=0 -> zero delta at init
        else:
            self.lora_A = self.lora_B = None
        self.register_buffer("bias", None if bias is None else bias.detach().float().clone())

    def _eff(self):
        return self.latent if self.lora_A is None else self.latent + self.lora_B @ self.lora_A

    def quant_weight(self):
        s = grad_scale(self.log_scale.exp().clamp_min(1e-8), 1.0 / math.sqrt(self.bs)).unsqueeze(1)
        wq = ste_ternary(self._eff().reshape(self.nb, self.bs) / s) * s
        return wq.reshape(self.out, self.inp)

    def forward(self, x):
        b = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, self.quant_weight().to(x.dtype), b)


def swap_linears(model, lora_rank=0, latent_dtype=torch.float32):
    """Replace every should_quantize() nn.Linear with a LatentSEQLinear (FP-init). Returns count."""
    n = 0
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear) or not should_quantize(f"{name}.weight"):
            continue
        if mod.in_features % BLOCK_SIZE != 0:
            continue
        new = LatentSEQLinear(mod.weight.data, mod.bias.data if mod.bias is not None else None,
                              BLOCK_SIZE, lora_rank=lora_rank, latent_dtype=latent_dtype)
        *parents, leaf = name.split(".")
        p = model
        for q in parents:
            p = getattr(p, q)
        setattr(p, leaf, new.to(mod.weight.device))
        n += 1
    return n


@torch.no_grad()
def fold_to_ternary(model):
    """Replace each LatentSEQLinear with a plain nn.Linear holding the dequantized ternary weight
    (each 256-block exactly in {-s,0,+s}) so save_pretrained writes a TQ2_0-readable HF model."""
    for name, mod in list(model.named_modules()):
        if not isinstance(mod, LatentSEQLinear):
            continue
        lin = nn.Linear(mod.inp, mod.out, bias=mod.bias is not None)
        lin.weight.data = mod.quant_weight().detach().to(torch.float16)
        if mod.bias is not None:
            lin.bias.data = mod.bias.detach().to(torch.float16)
        *parents, leaf = name.split(".")
        p = model
        for q in parents:
            p = getattr(p, q)
        setattr(p, leaf, lin.to(mod.latent.device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rot-model", required=True, help="rotated FP model dir (latent init source)")
    ap.add_argument("--orig-config", required=True, help="config/tokenizer source (untied model dir)")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--teacher-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=2e-5, help="latent-weight (or LoRA) LR")
    ap.add_argument("--scale-lr", type=float, default=1e-5, help="per-256 scale LR")
    ap.add_argument("--lora-rank", type=int, default=0,
                    help="E2E-polish mode: freeze the latent base, train a rank-r LoRA delta + scales "
                         "(merged & re-quantized at export). 0 = full latent QAT.")
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--loss-fn", default="cakld", choices=["cakld", "topk_kl"])
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--latent-bf16", action="store_true",
                    help="Store trainable latents in bf16 (halves latent+grad mem) so end-to-end QAT fits "
                         "a single 24GB GPU on mid-size models (e.g. the 4B). Slightly coarser STE grads.")
    args = ap.parse_args()
    dev = "cuda:0"
    loss_fn = cakld_loss if args.loss_fn == "cakld" else topk_kl_loss
    latent_dtype = torch.bfloat16 if args.latent_bf16 else torch.float32

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("loading rotated FP model (latent init)...")
    model = AutoModelForCausalLM.from_pretrained(args.rot_model, trust_remote_code=True,
                                                 dtype=torch.bfloat16).to(dev)
    n = swap_linears(model, lora_rank=args.lora_rank, latent_dtype=latent_dtype)
    mode = f"LoRA r={args.lora_rank} (frozen base)" if args.lora_rank else "full latent"
    print(f"  swapped {n} linears -> LatentSEQLinear [{mode}], learnable per-256 SEQ scale")
    # train the weight params (latent OR lora) + scales; freeze everything else (norms/embed/lm_head FP)
    weight_p, scale_p = [], []
    for nm, p in model.named_parameters():
        if nm.endswith(".log_scale"):
            p.requires_grad_(True); scale_p.append(p)
        elif nm.endswith(".lora_A") or nm.endswith(".lora_B"):
            p.requires_grad_(True); weight_p.append(p)
        elif nm.endswith(".latent"):
            if args.lora_rank == 0:
                p.requires_grad_(True); weight_p.append(p)
            else:
                p.requires_grad_(False)
        else:
            p.requires_grad_(False)
    latent_p = weight_p   # keep downstream references (optimizer/clip) working
    print(f"  trainable: {sum(x.numel() for x in weight_p)/1e6:.1f}M weight ({mode}) + {sum(x.numel() for x in scale_p)/1e6:.1f}M scales")
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.train()

    try:
        import bitsandbytes as bnb
        # Paged 8-bit AdamW: optimizer states live in CPU-pageable memory and stream to GPU on demand,
        # freeing several GB of VRAM so end-to-end QAT fits a single 24GB card on the 4B.
        opt = bnb.optim.PagedAdamW8bit([{"params": latent_p, "lr": args.lr},
                                        {"params": scale_p, "lr": args.scale_lr}])
        print("  optimizer: PagedAdamW8bit (CPU-offloaded states)")
    except Exception:
        opt = torch.optim.AdamW([{"params": latent_p, "lr": args.lr},
                                 {"params": scale_p, "lr": args.scale_lr}])
        print("  optimizer: AdamW (fp32; bitsandbytes unavailable)")

    cache = torch.load(args.teacher_cache)
    batches = load_calib_batches(args.calib, 1, args.seq, dev)
    nb = min(len(batches), len(cache["idx"]))
    if args.max_samples:
        nb = min(nb, args.max_samples)
    print(f"  {nb} calib seqs; {args.steps} steps")

    def lr_at(step, base):                                   # warmup then cosine decay to 5%
        if step < args.warmup:
            return base * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return base * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * prog)))

    t0, step, ema = time.time(), 0, None
    while step < args.steps:
        for bi in range(nb):
            if step >= args.steps:
                break
            ids = batches[bi].to(dev)
            t_idx = cache["idx"][bi].unsqueeze(0).to(dev)
            t_val = cache["val"][bi].unsqueeze(0).to(dev)
            out = model(ids).logits
            loss = loss_fn(out, t_idx, t_val, args.temperature)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(latent_p + scale_p, 1.0)
            opt.param_groups[0]["lr"] = lr_at(step, args.lr)
            opt.param_groups[1]["lr"] = lr_at(step, args.scale_lr)
            opt.step()
            step += 1
            kl = loss.item()
            ema = kl if ema is None else 0.9 * ema + 0.1 * kl
            if step % args.log_every == 0 or step == 1:
                print(f"   step {step}/{args.steps}  KL={kl:.4f} ema={ema:.4f} "
                      f"lr={lr_at(step,args.lr):.2e} ({(time.time()-t0)/60:.1f}m)", flush=True)

    print("folding latent -> ternary and saving...")
    fold_to_ternary(model)
    model.config.use_cache = True
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.half().save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.orig_config, trust_remote_code=True).save_pretrained(args.out)
    print(f"Done. ternary HF model -> {args.out}  (eval via eval_ternary --plain-hf, or export TQ2_0)")


if __name__ == "__main__":
    main()
