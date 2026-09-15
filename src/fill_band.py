import os, sys, glob, json, torch
sys.path.insert(0, os.path.dirname(__file__))
from e2e_qp_distill import _shard_map, _get_tensor
from family_sweep import LOG2_3, q_sym, q_vq, q_trellis, rel_err
SRC = glob.glob("/home/kasm-user/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/*")[0]
DEV = "cuda:1"
wm = _shard_map(SRC)
names = [sorted(k for k in wm if k.endswith(f"{x}.weight"))[16]
         for x in ("down_proj", "gate_proj", "up_proj")]
# candidates in 1.40-1.90, including cheaper SCALE encodings -- an 8-bit scale halves the scale
# overhead, which is the cheapest way to move ternary down toward its 1.585 information limit.
ARMS = [
    ("A ternary g256 s16", LOG2_3 + 16/256, lambda W: q_sym(W, LOG2_3, 256)),
    ("A ternary g256 s8",  LOG2_3 + 8/256,  lambda W: q_sym(W, LOG2_3, 256)),
    ("A ternary g128 s8",  LOG2_3 + 8/128,  lambda W: q_sym(W, LOG2_3, 128)),
    ("A ternary g64 s8",   LOG2_3 + 8/64,   lambda W: q_sym(W, LOG2_3, 64)),
    ("E vq k2048 d8 g128", 11/8 + 16/128,   lambda W: q_vq(W, 2048, 8, 128, device=DEV)),
    ("E vq k4096 d8 g256", 12/8 + 16/256,   lambda W: q_vq(W, 4096, 8, 256, device=DEV)),
    ("E vq k4096 d8 g128", 12/8 + 16/128,   lambda W: q_vq(W, 4096, 8, 128, device=DEV)),
    ("E vq k8192 d8 g256", 13/8 + 16/256,   lambda W: q_vq(W, 8192, 8, 256, device=DEV)),
    ("E vq k1024 d8 g256", 10/8 + 16/256,   lambda W: q_vq(W, 1024, 8, 256, device=DEV)),
]
rows = []
for n in names:
    W = _get_tensor(SRC, wm, n).float().to(DEV)
    print(f"--- {n.split('.')[-2]} {tuple(W.shape)}", flush=True)
    for lab, bpw, fn in ARMS:
        e = rel_err(fn(W), W)
        rows.append({"tensor": n, "arm": lab, "bpw": float(bpw), "err": e})
        print(f"   {lab:22s} {bpw:6.3f} bpw  {e:.5f}", flush=True)
    del W; torch.cuda.empty_cache()
json.dump(rows, open("output_sweep/band_fill.json", "w"), indent=1)
