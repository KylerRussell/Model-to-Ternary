"""head_swap.py — replace the ternary lm_head with an FP or int4 head, post-hoc.

QUESTION. embed+head is 24.0% of the 4B against 9.2% of the 27B, so the 4B testbed is the WORST case
for an output-head pathology. If ternarizing the head is what destroyed multi-step arithmetic
(7.1% acc|closed vs the FP teacher's 93.3%), restoring it should recover some of that.

ORDER OF ARMS. `fp` is run FIRST and is not a deployable config -- it is the UPPER BOUND. int4 is
strictly worse than bf16, so if the fp head does not recover arithmetic, no q4 head can, and the
hypothesis is dead for the price of one arm instead of two. Only a positive fp result licenses `q4`,
which then says how much of the gain survives quantization (int4+g64 = 2.0072 bpw, +12.7% at 27B).

BASIS. The head MUST come from the ROTATED FP model. QuaRot rotates the head's input side, so the
student's head and `output_4bpipe/rotbase` agree at median per-row cosine 0.8554 while the unrotated
`output_4b/untied_4b` sits at -0.0012 -- orthogonal. Sourcing from the wrong basis yields a model
that emits noise, so `swap_head` re-checks the cosine at load time and refuses below a floor.

CONFOUND, stated up front: the body was TRAINED against a ternary head and its distillation target
was the logits, so it has had every opportunity to compensate for head error. A null result is
therefore genuinely two-sided -- it can mean "the head is not the fault" or "body and head
co-adapted". Report Gate-A agreement alongside accuracy so the two can be told apart: a co-adaptation
break should show up as agreement DROPPING when the head improves.
"""
import os
import torch
import torch.nn as nn


def _find_head(model):
    for n, m in model.named_modules():
        if n.endswith("lm_head") and (hasattr(m, "packed") or isinstance(m, nn.Linear)):
            return n, m
    raise RuntimeError("no lm_head found")


@torch.no_grad()
def quant_int4_g64(w, group=64):
    """Symmetric int4, one fp16 scale per (row, 64-block). No zero-point -- same discipline as
    TQ1_64, and the configuration the repo's own cost table prices at 2.0072 bpw (+12.7% at 27B)."""
    out, inp = w.shape
    wg = w.float().reshape(out, inp // group, group)
    amax = wg.abs().amax(-1, keepdim=True)
    scale = (amax / 7.0).clamp_min(1e-8).half().float()      # fp16 scale, as the cost table assumes
    q = (wg / scale).round().clamp(-7, 7)
    return (q * scale).reshape(out, inp).to(w.dtype), q, scale


@torch.no_grad()
def swap_head(model, src_path, mode, cos_floor=0.5):
    """mode: 'fp' (bf16 upper bound) or 'q4' (int4 + per-g64 fp16 scales)."""
    from e2e_qp_distill import _shard_map, _get_tensor
    name, old = _find_head(model)
    dev = (old.scale.device if hasattr(old, "scale") else old.weight.device)
    # on CPU: this is a [248320, 2560] fp32 tensor (~2.5 GB) and it is only used for the guard
    ref = (old.dequant().float() if hasattr(old, "dequant") else old.weight.float()).cpu()
    w = _get_tensor(src_path, _shard_map(src_path), f"{name}.weight").float()
    assert w.shape == ref.shape, f"head shape {tuple(w.shape)} != model's {tuple(ref.shape)}"

    a = ref / ref.norm(dim=1, keepdim=True).clamp_min(1e-9)
    b = w / w.norm(dim=1, keepdim=True).clamp_min(1e-9)
    cos = float((a * b).sum(1).median())
    if cos < cos_floor:                                       # wrong-basis guard, not a formality:
        raise RuntimeError(                                   # the unrotated head scores -0.0012
            f"head from {src_path} has median per-row cosine {cos:.4f} vs the model's own head "
            f"(floor {cos_floor}). Almost certainly the WRONG BASIS — QuaRot rotates the head's "
            f"input side, so the source must be the ROTATED FP model.")

    if mode == "q4":
        w, _, _ = quant_int4_g64(w)
        err = (w.float() - _get_tensor(src_path, _shard_map(src_path), f"{name}.weight").float())
        print(f"head_swap: int4+g64 rel-err {err.norm()/ref.norm():.5f}", flush=True)
    elif mode != "fp":
        raise ValueError(f"mode must be 'fp' or 'q4', got {mode!r}")

    lin = nn.Linear(old.in_features, old.out_features, bias=False, device=dev, dtype=torch.bfloat16)
    lin.weight.data = w.to(dev, torch.bfloat16)
    lin.requires_grad_(False)
    parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
    setattr(parent, name.rsplit(".", 1)[-1], lin)
    print(f"head_swap: {name} -> {mode} (median row-cosine to the ternary head {cos:.4f})", flush=True)
    return lin


@torch.no_grad()
def apply_think_gain(model, c, row):
    """Raise P(</think>) by gain c. On the ternary head this scales that row's BLOCK SCALES; on a
    dense head it scales the row's WEIGHTS. fold_think_scale.py documents these as identical for an
    on-grid head (assignments unchanged, only scales move), so the arms stay comparable."""
    if c == 1.0:
        return
    _, m = _find_head(model)
    if hasattr(m, "scale") and hasattr(m, "block_size"):
        bpr = m.in_features // m.block_size
        m.scale[row * bpr:(row + 1) * bpr] *= c
        print(f"think gain c={c} applied to ternary head block-scales", flush=True)
    else:
        m.weight[row] *= c
        print(f"think gain c={c} applied to dense head row weights", flush=True)


@torch.no_grad()
def swap_embed(model, src_path, mode, cos_floor=0.5):
    """Same idea for embed_tokens, the other half of the 24%. It is an nn.Embedding, so
    build_student never ternarized it as a Linear -- but the CHECKPOINT's values are on-grid ternary,
    so restoring it needs the same rotated-basis source and the same guard."""
    from e2e_qp_distill import _shard_map, _get_tensor
    name = next(n for n, m in model.named_modules()
                if n.endswith("embed_tokens") and isinstance(m, nn.Embedding))
    emb = model.get_submodule(name)
    ref = emb.weight.float().cpu()
    w = _get_tensor(src_path, _shard_map(src_path), f"{name}.weight").float()
    assert w.shape == ref.shape, f"embed shape {tuple(w.shape)} != {tuple(ref.shape)}"
    a = ref / ref.norm(dim=1, keepdim=True).clamp_min(1e-9)
    b = w / w.norm(dim=1, keepdim=True).clamp_min(1e-9)
    cos = float((a * b).sum(1).median())
    if cos < cos_floor:
        raise RuntimeError(f"embed from {src_path} median row-cosine {cos:.4f} < {cos_floor}: "
                           f"wrong basis (QuaRot rotates the embedding too)")
    if mode == "q4":
        w, _, _ = quant_int4_g64(w)
    elif mode != "fp":
        raise ValueError(mode)
    emb.weight.data = w.to(emb.weight.device, torch.bfloat16)
    print(f"embed_swap: {name} -> {mode} (median row-cosine {cos:.4f})", flush=True)
