#!/usr/bin/env python3
"""
eval_ternary.py — Fidelity of the ternary student vs the FP teacher on held-out text.

This is the direct measure of "how much did quantization cost": it runs the same held-out
tokens through the original FP model and the ternary model and reports
  - perplexity of each (and the ratio / % increase),
  - mean top-k KL(teacher || student),
  - top-1 agreement (how often the ternary argmax matches the FP argmax).
Perplexity ratio near 1.0, low KL, and high top-1 agreement => the ternary model tracks the
original closely. Use text the calibration never saw (generate with a different --seed).

The FP teacher is loaded with device_map/offload (slow, like the precompute) then freed; the
ternary student is the packed build from e2e_qp_distill (~12 GB on GPU 0). They are never
resident at the same time.

Usage:
    python eval_ternary.py \
        --fp-path /path/to/Qwen3.6-27B/snapshots/<hash> \
        --ternary-path ./output_e2eqp_cont/modified_model \
        --orig-config-path /path/to/Qwen3.6-27B/snapshots/<hash> \
        --calib ./output_recovery/eval_data.json --seq 1024 --max-samples 48
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent))
from e2e_qp_distill import build_student, load_calib_batches, BLOCK_SIZE


@torch.no_grad()
def fp_pass(args, batches):
    from transformers import AutoModelForCausalLM
    n_gpu = torch.cuda.device_count()
    mm = {i: args.gpu_mem for i in range(n_gpu)}
    mm["cpu"] = args.cpu_mem
    off = Path(args.ternary_path).parent / "_eval_offload"
    off.mkdir(parents=True, exist_ok=True)
    print(f"[FP] loading teacher (device_map=auto, max_memory={mm})...")
    model = AutoModelForCausalLM.from_pretrained(
        args.fp_path, trust_remote_code=True, dtype=torch.bfloat16,
        device_map="auto", max_memory=mm, offload_folder=str(off), low_cpu_mem_usage=True)
    model.eval()
    nll, ntok, ref = 0.0, 0, []
    for i, b in enumerate(batches):
        ids = b.to("cuda:0")
        logits = model(ids).logits[0].float()                  # [T, V]
        lp = F.log_softmax(logits[:-1], dim=-1)
        tgt = ids[0, 1:]
        nll += -lp.gather(1, tgt.unsqueeze(1)).sum().item()
        ntok += tgt.numel()
        val, idx = torch.topk(logits, args.topk, dim=-1)
        ref.append((idx.to(torch.int32).cpu(), val.to(torch.float16).cpu()))
        if (i + 1) % 10 == 0:
            print(f"   [FP] {i + 1}/{len(batches)}")
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return nll, ntok, ref


@torch.no_grad()
def ternary_pass(args, batches, ref, device):
    if args.plain_hf:
        # load the model dir as an ordinary FP model (e.g. the rotation-only model, which is
        # NOT ternary so the packed loader can't read it). 27B -> device_map offload.
        from transformers import AutoModelForCausalLM
        n_gpu = torch.cuda.device_count()
        mm = {i: args.gpu_mem for i in range(n_gpu)}
        mm["cpu"] = args.cpu_mem
        off = Path(args.ternary_path).parent / "_eval_offload2"
        off.mkdir(parents=True, exist_ok=True)
        print(f"[plain-hf] loading {args.ternary_path} (device_map=auto)...")
        model = AutoModelForCausalLM.from_pretrained(
            args.ternary_path, trust_remote_code=True, dtype=torch.bfloat16,
            device_map="auto", max_memory=mm, offload_folder=str(off), low_cpu_mem_usage=True)
        in_dev = "cuda:0"
    else:
        print("[ternary] building packed student...")
        model, _ = build_student(args.ternary_path, args.orig_config_path, BLOCK_SIZE, device)
        in_dev = device
    model.eval()
    nll, ntok = 0.0, 0
    kl_sum, kl_n, agree, agree_n = 0.0, 0, 0, 0
    for bi, b in enumerate(batches):
        ids = b.to(in_dev)
        logits = model(ids).logits[0].float()                  # [T, V]
        dev = logits.device
        lp = F.log_softmax(logits[:-1], dim=-1)
        tgt = ids[0, 1:].to(dev)
        nll += -lp.gather(1, tgt.unsqueeze(1)).sum().item()
        ntok += tgt.numel()
        t_idx, t_val = ref[bi]
        t_idx, t_val = t_idx.to(dev), t_val.to(dev)
        # proper top-k KL: renormalize BOTH distributions over the teacher's top-k support,
        # so KL == 0 exactly when student == teacher.
        log_q = F.log_softmax(torch.gather(logits, -1, t_idx.long()), dim=-1)
        p = torch.softmax(t_val.float(), dim=-1)
        kl = (p * (torch.log(p + 1e-9) - log_q)).sum(-1)
        kl_sum += kl.sum().item()
        kl_n += kl.numel()
        agree += (logits.argmax(-1) == t_idx[:, 0]).sum().item()
        agree_n += logits.shape[0]
        if (bi + 1) % 10 == 0:
            print(f"   [{'plain-hf' if args.plain_hf else 'ternary'}] {bi + 1}/{len(batches)}")
    return nll, ntok, kl_sum / max(kl_n, 1), agree / max(agree_n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp-path", required=True)
    ap.add_argument("--ternary-path", required=True)
    ap.add_argument("--orig-config-path", required=True)
    ap.add_argument("--calib", required=True, help="HELD-OUT token file (different --seed).")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--max-samples", type=int, default=48)
    ap.add_argument("--gpu-mem", default="20GiB")
    ap.add_argument("--cpu-mem", default="120GiB")
    ap.add_argument("--ref-file", default=None, help="Where the FP-pass results are staged.")
    ap.add_argument("--reuse-ref", action="store_true",
                    help="If the ref-file already exists, skip the FP pass and reuse it "
                         "(share one FP-teacher pass across several per-stage evals).")
    ap.add_argument("--plain-hf", action="store_true",
                    help="Load --ternary-path as an ordinary FP model (for the rotation-only "
                         "model, which isn't ternary).")
    ap.add_argument("--_fp-only", action="store_true",
                    help="(internal) run only the FP pass and save results, then exit.")
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ref_file = args.ref_file or str(Path(args.ternary_path).parent / "_eval_ref.pt")

    # ---- child process: FP pass only, save to disk, exit (frees ALL GPU memory) ----
    if getattr(args, "_fp_only"):
        batches = load_calib_batches(args.calib, 1, args.seq, device)
        if args.max_samples:
            batches = batches[:args.max_samples]
        nll_fp, ntok_fp, ref = fp_pass(args, batches)
        torch.save({"nll": nll_fp, "ntok": ntok_fp, "ref": ref}, ref_file)
        print(f"[FP] staged results -> {ref_file}")
        return

    # ---- parent: get the FP teacher reference (reuse if asked and present, else compute) ----
    if args.reuse_ref and os.path.exists(ref_file):
        print(f"Reusing existing FP reference: {ref_file}")
    else:
        import subprocess
        cmd = [sys.executable, os.path.abspath(__file__), "--_fp-only",
               "--fp-path", args.fp_path, "--ternary-path", args.ternary_path,
               "--orig-config-path", args.orig_config_path, "--calib", args.calib,
               "--seq", str(args.seq), "--topk", str(args.topk),
               "--max-samples", str(args.max_samples), "--gpu-mem", args.gpu_mem,
               "--cpu-mem", args.cpu_mem, "--ref-file", ref_file]
        print("Running FP pass in a child process (guarantees GPU is freed before the student loads)...")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print("FP child process failed; aborting.")
            return
    d = torch.load(ref_file)
    nll_fp, ntok_fp, ref = d["nll"], d["ntok"], d["ref"]

    batches = load_calib_batches(args.calib, 1, args.seq, device)
    if args.max_samples:
        batches = batches[:args.max_samples]
    nll_t, ntok_t, kl, agree = ternary_pass(args, batches, ref, device)

    ppl_fp = math.exp(nll_fp / max(ntok_fp, 1))
    ppl_t = math.exp(nll_t / max(ntok_t, 1))
    print("\n" + "=" * 56)
    print(f"  FP teacher  perplexity : {ppl_fp:8.4f}")
    print(f"  ternary     perplexity : {ppl_t:8.4f}")
    print(f"  ppl ratio (ternary/FP) : {ppl_t / ppl_fp:8.4f}  "
          f"(+{100 * (ppl_t / ppl_fp - 1):.1f}%)")
    print(f"  mean top-{args.topk} KL(FP||tern): {kl:8.4f} nats")
    print(f"  top-1 agreement        : {100 * agree:7.2f}%")
    print("=" * 56)
    print("Guide: ppl ratio < ~1.10, KL < ~0.3, agreement > ~85% = tracks the original well.")


if __name__ == "__main__":
    main()