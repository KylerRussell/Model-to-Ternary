#!/usr/bin/env python3
"""
Component localization for the ternary retrieval collapse (follow-up to C2's negative result:
K-projection alone barely dents recall, so where does the 0.44 collapse come from?).

Splices the REAL ternary weights (from a fully-quantized checkpoint) into an otherwise-FP model,
ONE component group at a time, and measures multi-key associative recall. Whichever group drops
recall toward the full-ternary floor is the true bottleneck — and points at the next mechanism.
"""
import os, sys, glob, argparse
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoModelForCausalLM, AutoTokenizer
import recall_probe

DELTA = "Qwen3_5GatedDeltaNet"


def load_tern_weights(path):
    # AutoModelForCausalLM exposes the text stack as `model.layers.*`, but the checkpoint saves it
    # under the multimodal prefix `model.language_model.layers.*` — normalize so keys match params.
    d = {}
    for f in glob.glob(os.path.join(path, "*.safetensors")):
        with safe_open(f, framework="pt", device="cpu") as h:
            for k in h.keys():
                d[k.replace("model.language_model.", "model.")] = h.get_tensor(k)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp", default="output_4b/rot/modified_model")
    ap.add_argument("--tern", default="output_4b/expb_dtw_e2e/modified_model")
    ap.add_argument("--orig-config", default="output_4b/untied_4b")
    ap.add_argument("--pairs", default="16,32,48,64")
    ap.add_argument("--trials", type=int, default=100)
    args = ap.parse_args()
    dev = "cuda:0"
    tok = AutoTokenizer.from_pretrained(args.orig_config, trust_remote_code=True)
    n_list = [int(x) for x in args.pairs.split(",")]

    print(f"[abl] loading FP {args.fp}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(args.fp, trust_remote_code=True, dtype=torch.bfloat16).to(dev).eval()
    cfg = model.config.get_text_config()
    kd = cfg.linear_num_key_heads * cfg.linear_key_head_dim   # key_dim for qkv row slicing
    print(f"[abl] loading ternary weights {args.tern}", flush=True)
    tern = load_tern_weights(args.tern)
    params = dict(model.named_parameters())
    # sanity: keys line up
    miss = [k for k in params if k not in tern]
    print(f"[abl] params={len(params)}  tern-keys={len(tern)}  fp-only(unmatched)={len(miss)}", flush=True)

    def is_qkv(n):  return n.endswith(".linear_attn.in_proj_qkv.weight")
    def sel(*subs): return lambda n: any(s in n for s in subs)

    # group -> (name-predicate, row-slice or None)   row-slice only meaningful for in_proj_qkv
    groups = [
        ("qkv_Q",        is_qkv,                              slice(0, kd)),
        ("qkv_K",        is_qkv,                              slice(kd, 2 * kd)),
        ("qkv_V",        is_qkv,                              slice(2 * kd, None)),
        ("qkv_all",      is_qkv,                              None),
        ("in_proj_z",    sel(".linear_attn.in_proj_z.weight"), None),
        ("in_proj_a+b",  sel(".linear_attn.in_proj_a.weight", ".linear_attn.in_proj_b.weight"), None),
        ("deltanet_out", sel(".linear_attn.out_proj.weight"), None),
        ("deltanet_ALL", sel(".linear_attn.in_proj", ".linear_attn.out_proj"), None),
        ("mlp",          sel(".mlp.gate_proj.weight", ".mlp.up_proj.weight", ".mlp.down_proj.weight",
                             ".mlp.linear_fc1.weight", ".mlp.linear_fc2.weight"), None),
        ("fullattn",     sel(".self_attn.q_proj.weight", ".self_attn.k_proj.weight",
                             ".self_attn.v_proj.weight", ".self_attn.o_proj.weight"), None),
        ("ALL_quant",    sel(".linear_attn.in_proj", ".linear_attn.out_proj", ".mlp.gate_proj",
                             ".mlp.up_proj", ".mlp.down_proj", ".mlp.linear_fc1", ".mlp.linear_fc2",
                             ".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj",
                             ".self_attn.o_proj"), None),
    ]

    print(f"\n{'group':<15}{'#tensors':>9}{'meanRestr':>11}   per-N restr")
    print("-" * 70)
    print(f"{'FP (control)':<15}{0:>9}", end="", flush=True)
    res = recall_probe.run(model, tok, dev, n_list, args.trials, seed=0)
    mr = sum(res[N]['restr'] for N in n_list) / len(n_list)
    print(f"{mr:>11.4f}   " + " ".join(f'{res[N]["restr"]:.3f}' for N in n_list))

    for gname, pred, sl in groups:
        patched = []
        for n, p in params.items():
            if pred(n) and n in tern:
                t = tern[n].to(p.device, p.dtype)
                if sl is not None:
                    p.data[sl].copy_(t[sl])
                else:
                    p.data.copy_(t)
                patched.append((n, sl))
        res = recall_probe.run(model, tok, dev, n_list, args.trials, seed=0)
        mr = sum(res[N]['restr'] for N in n_list) / len(n_list)
        print(f"{gname:<15}{len(patched):>9}{mr:>11.4f}   " + " ".join(f'{res[N]["restr"]:.3f}' for N in n_list), flush=True)
        # restore FP weights for the patched tensors from the cached FP checkpoint
        for n, sl in patched:
            fp_t = fp_cache[n].to(params[n].device, params[n].dtype)
            if sl is not None:
                params[n].data[sl].copy_(fp_t[sl])
            else:
                params[n].data.copy_(fp_t)


if __name__ == "__main__":
    # snapshot FP weights (CPU) before patching so we can restore between groups
    import types
    _argv = sys.argv
    ap = argparse.ArgumentParser(); ap.add_argument("--fp", default="output_4b/rot/modified_model")
    fp_path = ap.parse_known_args()[0].fp
    print(f"[abl] caching FP weights for restore from {fp_path}", flush=True)
    fp_cache = load_tern_weights(fp_path)   # same loader, FP checkpoint
    main()
