#!/usr/bin/env python3
"""
C3 — Contraction clamp on the Gated-DeltaNet decay gate (checklist cluster C3;
reports R1-A10 / R2-A4 / R3-2 JSCE).

Gated DeltaNet transition is  A_t = alpha_t (I - beta_t k_t k_t^T),  alpha_t = exp(g_t) in (0,1),
where the off-axis eigenvalue is exactly alpha_t. Quantization error injected off the key axis is
damped ONLY by alpha_t; on channels/heads where alpha_t ~= 1 (near-lossless memory) the error
persists across the whole state and compounds over the 24 DeltaNet layers (FM3).

The clamp multiplies alpha_t by (1-eps) so the *quantized* recurrence is a strict contraction and
injected error decays geometrically, trading a little long-memory for error robustness.
In log space that is exactly  g_t -> g_t + log(1-eps)  (elementwise, before the kernel).

This installs a runtime wrapper on each DeltaNet's kernel callables (both the chunk/prefill kernel
and the single-step recurrent/decode kernel), intercepting the `g=` argument. Backend-agnostic
(works whether the fla fused CUDA kernels or the torch fallback are active). Reversible via remove().

Foldability note: this is the DIAGNOSTIC/inference form (the cheap decisive sweep the reports ask
for). An additive constant on g is not cleanly foldable into the current
g = -exp(A_log)*softplus(a+dt_bias) parametrization, so if a value of eps wins we fold it later
(e.g. as a per-head decay bias); the sweep decides whether that is worth doing at all.
"""
import os, math, torch

_CLASS = "Qwen3_5GatedDeltaNet"
_KERNEL_ATTRS = ("chunk_gated_delta_rule", "recurrent_gated_delta_rule")


def _wrap(fn, log_factor):
    def wrapped(*args, **kwargs):
        if "g" in kwargs and kwargs["g"] is not None:
            kwargs = dict(kwargs)
            kwargs["g"] = kwargs["g"] + log_factor
        else:
            # g is the 4th positional arg (query, key, value, g, ...) in the torch impls
            a = list(args)
            if len(a) >= 4 and torch.is_tensor(a[3]):
                a[3] = a[3] + log_factor
            args = tuple(a)
        return fn(*args, **kwargs)
    wrapped._cc_orig = fn
    return wrapped


def install(model, eps: float):
    """Clamp alpha_t -> alpha_t*(1-eps) on every DeltaNet layer. eps<=0 is a no-op."""
    if eps is None or eps <= 0:
        return 0
    log_factor = math.log(1.0 - eps)
    n = 0
    for m in model.modules():
        if type(m).__name__ != _CLASS:
            continue
        for attr in _KERNEL_ATTRS:
            fn = getattr(m, attr, None)
            if fn is not None and not hasattr(fn, "_cc_orig"):
                setattr(m, attr, _wrap(fn, log_factor))
        n += 1
    print(f"[contraction-clamp] eps={eps} (alpha*={1-eps:.4f}, +log={log_factor:.5f}) "
          f"applied to {n} DeltaNet layers", flush=True)
    return n


def remove(model):
    for m in model.modules():
        if type(m).__name__ != _CLASS:
            continue
        for attr in _KERNEL_ATTRS:
            fn = getattr(m, attr, None)
            if fn is not None and hasattr(fn, "_cc_orig"):
                setattr(m, attr, fn._cc_orig)


def maybe_install(model):
    """Env hook: set CONTRACTION_EPS=0.01 before running any eval to enable the clamp."""
    eps = os.environ.get("CONTRACTION_EPS", "").strip()
    if eps:
        try:
            return install(model, float(eps))
        except ValueError:
            print(f"[contraction-clamp] bad CONTRACTION_EPS={eps!r}, skipping", flush=True)
    return 0
