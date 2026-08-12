#!/usr/bin/env python
"""Untie a tied-embedding qwen3_5 checkpoint so the QuaRot pipeline can rotate it transparently.

Tied models (tie_word_embeddings=True) store ONLY embed_tokens and reuse it as lm_head at runtime.
QuaRot must rotate embed (embedding role) and lm_head (input-rotate + final-norm absorption) DIFFERENTLY
— impossible with one shared tensor, which breaks rotation transparency (the 2B testbed gave 512 ppl /
91x on the rotation-sanity eval). This materializes a separate `lm_head.weight` = copy of embed_tokens
and sets tie_word_embeddings=False, producing a numerically-identical but UNTIED model that the pipeline
handles exactly like the (already-untied) 27B.

  python untie_embeddings.py --src <tied snapshot> --out <untied dir>
"""
import argparse, json, os, shutil, glob
import torch
from safetensors import safe_open
from safetensors.torch import save_file

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--embed-name", default="model.language_model.embed_tokens.weight")
ap.add_argument("--lmhead-name", default="lm_head.weight")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

# load every tensor from the source shard(s)
shards = sorted(glob.glob(os.path.join(args.src, "*.safetensors"))) or \
         sorted(glob.glob(os.path.join(args.src, "*.safetensors-*")))
assert shards, f"no safetensors in {args.src}"
tensors, meta = {}, {}
for sh in shards:
    with safe_open(sh, framework="pt") as f:
        m = f.metadata()
        if m: meta.update(m)
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
print(f"loaded {len(tensors)} tensors from {len(shards)} shard(s)")

assert args.embed_name in tensors, f"embed '{args.embed_name}' not found; keys e.g. {list(tensors)[:3]}"
if args.lmhead_name in tensors:
    print(f"⚠️ '{args.lmhead_name}' already present — model already untied; copying through unchanged.")
else:
    tensors[args.lmhead_name] = tensors[args.embed_name].clone().contiguous()
    print(f"materialized {args.lmhead_name} = clone({args.embed_name})  shape {tuple(tensors[args.lmhead_name].shape)}")

# write a single untied shard + index
out_shard = "model.safetensors"
save_file(tensors, os.path.join(args.out, out_shard), metadata=meta or {"format": "pt"})
total = sum(t.numel() * t.element_size() for t in tensors.values())
json.dump({"metadata": {"total_size": total},
           "weight_map": {k: out_shard for k in tensors}},
          open(os.path.join(args.out, "model.safetensors.index.json"), "w"), indent=2)

# copy aux files; patch config tie flag (top-level + text_config) to False
for fn in os.listdir(args.src):
    if fn.endswith(".safetensors") or fn.endswith(".safetensors.index.json") or fn.startswith("."):
        continue
    sp = os.path.join(args.src, fn)
    if not os.path.isfile(sp):
        continue
    if fn == "config.json":
        c = json.load(open(sp))
        c["tie_word_embeddings"] = False
        if isinstance(c.get("text_config"), dict):
            c["text_config"]["tie_word_embeddings"] = False
        json.dump(c, open(os.path.join(args.out, fn), "w"), indent=2)
        print("  patched config.json tie_word_embeddings -> False")
    else:
        shutil.copy2(sp, os.path.join(args.out, fn))
print(f"\nwrote untied model -> {args.out}")
