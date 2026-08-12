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
def greedy_generate(model, prompt_ids, gen_len, device, log_every=0, tag=""):
    """Greedy-decode `gen_len` new tokens after `prompt_ids` ([1, P] or [P]).

    No KV cache: recomputes the full forward each step so it works with any model
    forward (the packed ternary student does not expose a standard cache path). Fine
    for short offline-eval continuations; keep gen_len/prompts modest.
    Returns the FULL token sequence (prompt + generated) as a 1-D CPU LongTensor.
    """
    seq = prompt_ids.view(1, -1).to(device)
    P = seq.shape[1]
    for s in range(gen_len):
        nxt = model(seq).logits[0, -1].argmax().view(1, 1)
        seq = torch.cat([seq, nxt], dim=1)
        if log_every and (s + 1) % log_every == 0:
            print(f"   [{tag}] gen {s + 1}/{gen_len}")
    return seq[0].to(torch.int32).cpu()


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
    # --- teacher greedy rollouts: stage the continuations so the student can be
    #     compared on FREE generation (the signal teacher-forced KL misses) ---
    gen = []
    if args.gen_prompts > 0:
        print(f"[FP] greedy-generating {args.gen_len} tokens for "
              f"{min(args.gen_prompts, len(batches))} prompts...")
        for i, b in enumerate(batches[:args.gen_prompts]):
            prompt = b[0, :args.gen_prompt_len]
            full = greedy_generate(model, prompt, args.gen_len, "cuda:0",
                                   log_every=0, tag="FP")
            gen.append(full)                                    # [gen_prompt_len + gen_len]
            print(f"   [FP] rollout {i + 1}/{min(args.gen_prompts, len(batches))}")
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return nll, ntok, ref, gen


@torch.no_grad()
def ternary_pass(args, batches, ref, gen_ref, device):
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
    import contraction_clamp; contraction_clamp.maybe_install(model)  # C3: clamp eval-target only (not FP ref)
    nll, ntok = 0.0, 0
    kl_sum, kl_n, agree, agree_n = 0.0, 0, 0, 0
    cflip, cflip_n = 0, 0
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
        s_arg = logits.argmax(-1)
        agree += (s_arg == t_idx[:, 0]).sum().item()
        agree_n += logits.shape[0]
        # confident-flip rate: positions where the teacher was confident (top-1 prob over
        # its top-k support > conf_thresh) but the student's argmax disagrees. arXiv 2407.09141:
        # these flips track generative-quality loss even when perplexity is preserved.
        t_conf = p[:, 0]                                        # teacher top-1 prob (top-k support)
        conf_mask = t_conf > args.conf_thresh
        cflip += (conf_mask & (s_arg != t_idx[:, 0])).sum().item()
        cflip_n += conf_mask.sum().item()
        if (bi + 1) % 10 == 0:
            print(f"   [{'plain-hf' if args.plain_hf else 'ternary'}] {bi + 1}/{len(batches)}")

    # --- generation fidelity: independent greedy rollouts vs the teacher's ---
    gen_metrics = None
    if gen_ref:
        match_tok, match_n, firstdiv, gnll, gntok = 0, 0, [], 0.0, 0
        for gi, t_full in enumerate(gen_ref):
            P = args.gen_prompt_len
            prompt = t_full[:P]
            s_full = greedy_generate(model, prompt, args.gen_len, in_dev,
                                     log_every=0, tag="ternary")
            t_cont = t_full[P:P + args.gen_len].to(torch.long)
            s_cont = s_full[P:P + args.gen_len].to(torch.long)
            eq = (t_cont == s_cont)
            match_tok += eq.sum().item()
            match_n += eq.numel()
            # tokens until the FIRST divergence (capped at gen_len)
            fd = (~eq).nonzero()
            firstdiv.append(int(fd[0].item()) if fd.numel() else eq.numel())
            # student perplexity ON THE TEACHER'S OWN ROLLOUT (cross-distribution NLL)
            tf = t_full.to(torch.long).to(in_dev).view(1, -1)
            lg = model(tf).logits[0].float()
            dev = lg.device
            lpg = F.log_softmax(lg[P - 1:P - 1 + args.gen_len], dim=-1)
            tg = tf[0, P:P + args.gen_len].to(dev)
            gnll += -lpg.gather(1, tg.unsqueeze(1)).sum().item()
            gntok += tg.numel()
            print(f"   [gen] rollout {gi + 1}/{len(gen_ref)}  "
                  f"first-div@{firstdiv[-1]}/{args.gen_len}")
        gen_metrics = {
            "match": match_tok / max(match_n, 1),
            "first_div": sum(firstdiv) / max(len(firstdiv), 1),
            "gen_ppl": math.exp(gnll / max(gntok, 1)),
        }

    return (nll, ntok, kl_sum / max(kl_n, 1), agree / max(agree_n, 1),
            cflip / max(cflip_n, 1), gen_metrics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp-path", required=True)
    ap.add_argument("--ternary-path", required=True)
    ap.add_argument("--orig-config-path", required=True)
    ap.add_argument("--calib", required=True, help="HELD-OUT token file (different --seed).")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--max-samples", type=int, default=48)
    ap.add_argument("--conf-thresh", type=float, default=0.5,
                    help="Teacher top-1 prob above which a student argmax disagreement counts "
                         "as a 'confident flip'.")
    ap.add_argument("--gen-prompts", type=int, default=0,
                    help="Number of prompts for the free-generation fidelity check "
                         "(0 = skip generation, teacher-forced metrics only).")
    ap.add_argument("--gen-prompt-len", type=int, default=128,
                    help="Tokens taken from each eval sample as the generation prompt.")
    ap.add_argument("--gen-len", type=int, default=48,
                    help="New tokens greedy-generated per prompt (no KV cache; keep modest).")
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
        nll_fp, ntok_fp, ref, gen = fp_pass(args, batches)
        torch.save({"nll": nll_fp, "ntok": ntok_fp, "ref": ref, "gen": gen}, ref_file)
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
               "--cpu-mem", args.cpu_mem, "--ref-file", ref_file,
               "--gen-prompts", str(args.gen_prompts),
               "--gen-prompt-len", str(args.gen_prompt_len),
               "--gen-len", str(args.gen_len)]
        print("Running FP pass in a child process (guarantees GPU is freed before the student loads)...")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print("FP child process failed; aborting.")
            return
    d = torch.load(ref_file)
    nll_fp, ntok_fp, ref = d["nll"], d["ntok"], d["ref"]
    gen_ref = d.get("gen") or []
    if args.gen_prompts > 0 and not gen_ref:
        print("[warn] --gen-prompts set but the FP reference has no staged rollouts "
              "(stale --reuse-ref?); skipping the generation check.")

    batches = load_calib_batches(args.calib, 1, args.seq, device)
    if args.max_samples:
        batches = batches[:args.max_samples]
    nll_t, ntok_t, kl, agree, cflip, gen_m = ternary_pass(args, batches, ref, gen_ref, device)

    ppl_fp = math.exp(nll_fp / max(ntok_fp, 1))
    ppl_t = math.exp(nll_t / max(ntok_t, 1))
    print("\n" + "=" * 56)
    print("  -- teacher-forced (held-out text) --")
    print(f"  FP teacher  perplexity : {ppl_fp:8.4f}")
    print(f"  ternary     perplexity : {ppl_t:8.4f}")
    print(f"  ppl ratio (ternary/FP) : {ppl_t / ppl_fp:8.4f}  "
          f"(+{100 * (ppl_t / ppl_fp - 1):.1f}%)")
    print(f"  mean top-{args.topk} KL(FP||tern): {kl:8.4f} nats")
    print(f"  top-1 agreement        : {100 * agree:7.2f}%")
    print(f"  flip rate (1-agree)    : {100 * (1 - agree):7.2f}%")
    print(f"  confident-flip rate    : {100 * cflip:7.2f}%  (teacher top-1 prob > {args.conf_thresh})")
    if gen_m is not None:
        print("  -- free generation (greedy rollout vs teacher's) --")
        print(f"  token exact-match      : {100 * gen_m['match']:7.2f}%  over {args.gen_len} gen tokens")
        print(f"  tokens to 1st divergence: {gen_m['first_div']:7.2f} / {args.gen_len}")
        print(f"  student ppl on tchr roll: {gen_m['gen_ppl']:8.4f}")
    print("=" * 56)
    print("Guide: ppl ratio < ~1.10, KL < ~0.3, agreement > ~85% = tracks on held-out text.")
    print("       Generation dissociation shows up as LOW confident-flip-tolerance / EARLY")
    print("       first-divergence even when the teacher-forced numbers look fine.")


if __name__ == "__main__":
    main()