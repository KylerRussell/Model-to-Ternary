"""densify.py — hoist the ternary dequant out of the generation loop.

TernaryScaleLinear.forward is, verbatim:

    w = self.dequant().to(x.dtype)
    return F.linear(x, w, ...)

so the packed 2-bit codes are unpacked and rescaled into a full dense weight on EVERY forward call.
During training that is the point (the scale carries gradient). During autoregressive decoding it is
pure waste: the weight is constant across all 2048 steps, but we rebuild it once per layer per token.
The GEMV itself is memory-bound, so prepending an unpack+scale+cast roughly triples its memory
traffic -- which is why the ternary student takes 143 min for 48 GSM8K problems while the bf16 FP
teacher takes 60 min for the SAME problems despite generating 40% MORE tokens per chain.

densify() calls dequant() ONCE per module, under no_grad, and replaces the module with a plain
nn.Linear holding the result. This is not an approximation and not a re-quantization: it is exactly
the tensor forward() would have computed, cast to exactly the dtype forward() would have cast it to,
just hoisted out of the loop. Bit-identity is asserted by densify_selftest() and must be checked
before any run whose numbers are compared against a packed-path baseline.

INFERENCE ONLY. The dense weight is off the ternary grid the instant anything touches it, and no
gradient reaches `scale` any more. Never densify a model that is about to be trained.

ORDER MATTERS: apply THINK_ROW_SCALE (or any other scale edit) BEFORE densifying, or the edit is
silently discarded -- densify snapshots the scales as they stand.
"""
import torch
import torch.nn as nn


def _is_tsl(m):
    return hasattr(m, "packed") and hasattr(m, "scale") and hasattr(m, "dequant")


@torch.no_grad()
def densify(model, dtype=torch.bfloat16, verbose=True):
    """Replace every TernaryScaleLinear with an equivalent dense nn.Linear. Returns (n, bytes)."""
    targets = [(n, m) for n, m in model.named_modules() if _is_tsl(m)]
    n_done = n_bytes = 0
    for name, m in targets:
        w = m.dequant().to(dtype).contiguous()
        lin = nn.Linear(m.in_features, m.out_features, bias=m.bias is not None,
                        device=w.device, dtype=dtype)
        lin.weight.data = w
        if m.bias is not None:
            lin.bias.data = m.bias.to(dtype)
        lin.requires_grad_(False)
        parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
        setattr(parent, name.rsplit(".", 1)[-1], lin)
        n_done += 1
        n_bytes += w.numel() * w.element_size()
        del m
    torch.cuda.empty_cache()
    if verbose:
        print(f"densify: {n_done} linears -> dense {dtype}, {n_bytes/2**30:.2f} GiB", flush=True)
    return n_done, n_bytes


@torch.no_grad()
def densify_selftest(model, n_test=3, tol=0):
    """Assert the swap is EXACT on real modules from this model, before it is relied on.

    tol=0 means bit-identical outputs. That is the correct bar: densify claims to be an algebraic
    hoist, not an approximation, so any nonzero difference means a branch of dequant() was not
    reproduced and the run must not proceed."""
    targets = [(n, m) for n, m in model.named_modules() if _is_tsl(m)][:n_test]
    assert targets, "no TernaryScaleLinear modules found — wrong model kind?"
    for name, m in targets:
        dev = m.scale.device
        x = torch.randn(4, m.in_features, device=dev, dtype=torch.bfloat16)
        ref = m(x)
        w = m.dequant().to(torch.bfloat16).contiguous()
        got = torch.nn.functional.linear(x, w, m.bias.to(torch.bfloat16)
                                         if m.bias is not None else None)
        d = (ref.float() - got.float()).abs().max().item()
        assert d <= tol, f"densify NOT exact on {name}: max|delta|={d:.3e} (tol={tol})"
    print(f"densify selftest: {len(targets)} modules bit-identical", flush=True)
