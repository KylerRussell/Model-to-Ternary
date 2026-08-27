# Research prompt — the offloaded-latent hot path: where arm B's time actually goes, and how to cut it

**Ask:** concrete, implementable methods to cut wall-clock in the specific code paths measured below.
Follow-up to `armb_throughput_prompt.md` (2026-08-22); read this one for the hot-path detail, that one
for scope/RAM inventory. Since then we implemented multi-process pipeline parallelism and profiled the
step properly, which **moved the diagnosis**: the cost is neither bandwidth nor arithmetic.

**Fixed constraints — proposals that violate these are not usable:**
* **Sequence length stays 2560.** Set by where the teacher emits end-of-thought tokens in generated
  rollouts, not by a coverage tradeoff.
* **Scope stays full-latent (arm B, all linears).** We are not reducing scope for speed.
* **No NVMe on this box.** Do not propose disk offload.
* **fp32 latent master is required.** The Adam update is ~2.5e-4 of latent magnitude vs bf16's 3.9e-3
  resolution, so a bf16 master is swamped. (A bf16 *compute* copy was tested: +3.8%, i.e. worse.)
* **Scales are frozen during assignment training** (V/P alternation, `--lr 0`); this is required, not
  incidental.

**Hardware:** 2x RTX 3090 (24 GB each), 629 GB DDR RAM, no NVMe. Measured PCIe, explicit-sync,
191 MB fp32 buffers: **H2D pinned 4.89 GB/s, D2H pinned 5.09 GB/s, D2H pageable 3.67 GB/s**
(pageable is only 28% slower than pinned here — this matters, see 5).

---

## 1. The computation

Ternary weights (`{-1,0,+1}` x group-64 8-bit scale). The assignment stage learns *which trit* each
weight takes via a per-weight fp32 latent `L` and a straight-through estimator. Exact code:

```python
Lb = latent_gpu.reshape(n_blocks, block_size)         # block_size = 64
s  = scale.unsqueeze(1).clamp_min(1e-8).to(Lb.dtype)  # frozen during assignment training
q    = torch.clamp(torch.round(Lb / s), -1, 1)        # hard trit
mask = (Lb.abs() < 1.5 * s).to(Lb.dtype)              # STE gate
deq  = q.detach() * s + (Lb - Lb.detach()) * mask     # fwd = q*s; grad flows to L through mask
return deq.reshape(out_features, in_features)
```

`forward()` is then `F.linear(x, deq.to(x.dtype))`. The latents do **not** fit in VRAM (rank 0 alone
holds 7.664B latents = 30.7 GB fp32, and Adam state doubles that to ~61 GB against a 24 GB card), so
they live in **pinned CPU RAM** and stream in per layer:

```python
latent_gpu = self.latent.to(self.scale.device, dtype=torch.float32, non_blocking=True)
```

Autograd routes the gradient back through that `.to()`, i.e. a full-size **fp32 D2H per latent per
microbatch**, accumulating into the CPU leaf's `.grad`.

**Gradient checkpointing (non-reentrant) is on**, so every latent is transferred H2D **twice** per
microbatch — once in the forward, once in the recompute.

`--latent-grad-release` runs a per-latent Adam step inside a post-accumulate-grad hook and frees the
grad immediately (saves 30.7 GB of fp32 grad buffers). The optimizer is Adam with a **block-256-shared
second moment** (1.48x faster than per-element on this CPU; per-element/int8/bf16/Adafactor variants all
lost):

```python
st["exp_avg"].mul_(b1).add_(gr, alpha=1-b1)
gb = gr.view(-1, 256); st["v_blk"].mul_(b2).add_(gb.pow(2).mean(dim=1), alpha=1-b2)
denom = (st["v_blk"]/bc2).sqrt_().add_(eps).unsqueeze(1)
param.view(-1,256).addcdiv_(st["exp_avg"].view(-1,256), denom.expand(-1,256), value=-lr/bc1)
```

**Parallelism:** 2-rank multi-process pipeline (NCCL), rank 0 = layers 0-39, rank 1 = 40-63 + norm +
lm_head + chunked loss; 2 microbatches in flight, 1F1B with a non-blocking activation handoff. Only the
boundary activation (26.2 MB) crosses; the ranks share no parameters, so there is no all-reduce.

---

## 2. THE MEASUREMENT THAT MATTERS — per-latent stage breakdown

One linear, 47.8M latents (= rank 0's per-linear average), seq 2560, explicit `cuda.synchronize()`:

| stage | ms |
|---|---|
| H2D latent fp32 (pinned) | 63.5 |
| STE dequant (the 6-temporary expression above) | 4.7 |
| GEMM `F.linear` | 3.1 |
| **backward, latent GPU-RESIDENT** | **13.9** |
| **backward, latent OFFLOADED** | **433.0** |
| — of which D2H 72.4 + CPU grad accumulate 26.4 | 98.8 |
| — **unexplained per-tensor overhead** | **320.3** |

**Arithmetic is ~8 ms. PCIe is ~136 ms. ~320 ms/latent is neither.** A GPU-resident latent runs the
same backward in 13.9 ms — a **31x** gap that transfer volume does not explain.

Corroborating whole-model numbers (rank 0, 40 layers, 160 latents, 2 microbatches, step = **225.0 s**):

* Bytes the step MUST move: H2D 122.8 GB (fwd + recompute, x2 mb) + D2H 61.4 GB = **184.2 GB**.
  At measured PCIe rates that is a **37.2 s floor = 17% of the step.**
* Op-level trace: `aten::mm` = **2.56%** of self-CUDA. The step is not compute.
* Same trace: `cudaMemcpyAsync` = **89.4% of self-CPU time** (283.7 s of 317.4 s) while actual transfer
  is only ~17% of the step. **The calling thread is blocked in memcpy far longer than the copies take**
  — i.e. ~960 forced sync points per step (640 H2D + 320 D2H) serialise CPU and GPU instead of
  overlapping them.
* Consistent with an older result: step time is **linear in latent COUNT** (64/249/497 latent tensors
  -> 102.6/165.7/327.1 s) while halving the BYTES moved did nothing. Cost tracks the number of
  per-tensor operations, not bandwidth.

**The central question: what is the ~320 ms/latent, and how do we remove it?** Our leading suspects,
untested: allocation + first-touch page faults on a fresh ~191 MB pageable CPU grad buffer every
backward; `AccumulateGrad` on a CPU leaf; implicit stream synchronisation forced by each transfer.

---

## 3. Structural facts a proposal can exploit

* Only **~3.55% of assignments ever move**, and motion saturates by ~step 60. A `--latent-candidate-tau`
  flag exists (update only latents within tau of a decision boundary); measured coverage at 4B:
  tau=0.001 -> 14.9%, 0.01 -> 15.7%, 0.05 -> 19.6%.
* The STE gate `|L| < 1.5s` means latents far from a boundary receive **zero gradient** — they are dead
  weight in every transfer, every optimizer step, and every byte of state.
* `q` and `mask` depend on `L` only through its position relative to per-block boundaries.
* Scales are FROZEN during this stage, so bin boundaries are static within a V/P phase.
* 629 GB of host RAM is available and mostly idle; VRAM is the scarce resource.
* The two ranks' host-side work is independent (no shared parameters), and aggregate H2D bandwidth
  scales cleanly across the two GPUs (7.80 -> 7.94 GB/s solo vs concurrent).

---

## 4. What we want

Ranked, concrete proposals targeting the ~320 ms/latent overhead and/or the 2x-per-microbatch H2D.
For each: the mechanism, expected effect on the numbers in section 2, what it costs in VRAM/RAM/quality,
and how to falsify it cheaply. Specific directions we would like assessed:

1. **Eliminating the per-tensor overhead** — CUDA host-register / `cudaHostAlloc` on the grad
   destination, persistent preallocated pinned grad buffers, a custom autograd Function that fuses
   D2H + Adam + free, or moving the whole latent path off the autograd graph.
2. **Batching across latents** — one fused transfer/optimizer op over many latents instead of ~960
   per-tensor operations, e.g. a flat arena per layer, `torch._foreach_*`, or CUDA graphs.
3. **Killing the recompute transfer** — keeping the dequantized bf16 weight (not the fp32 latent)
   resident for the layer's backward, or recomputing `deq` from a compact GPU-side representation.
4. **Shrinking what crosses PCIe** — transferring `deq` (bf16, 2 B/elem) instead of `L` (fp32, 4 B/elem)
   and doing the STE/optimizer host-side; or transferring only boundary-adjacent latents (see 3).
5. **Sparse/candidate latents** — exploiting the 3.55% mobility to hold only a candidate subset in fp32
   with state, and a compact form for the rest. What is the right data structure, and how do we avoid
   biasing which assignments *can* move?
6. **Overlap** — multiple CUDA streams and double-buffering so transfers hide behind compute. Our
   prefetch attempt exists but is not currently a win; we want to know whether the ~960 sync points
   can be removed at all given autograd's CPU-leaf semantics.

---

## 5. Already tested — do NOT re-propose without new argument

Every one of these was proposed from a plausible indirect signal and **measured neutral or worse**:

| idea | result |
|---|---|
| Pinning the gradient D2H (`--latent-pin-grad`) | **6.2% WORSE** (239.3 vs 225.0 s/step). Pageable D2H is only 28% slower than pinned here (3.67 vs 5.09 GB/s), so it saves ~4.6 s of a 225 s step and the staging copy costs more. |
| Partial GPU residency (`--latent-gpu-budget 8`, 35% of latents) | Unsettled/worse: 243.8 and 282.1 s/step in two runs of the same config. Bounded to ~35% anyway (needs ~61 GB/rank). |
| Rebalancing the pipeline split (40 -> 36) | **15.6% WORSE** (260.0 vs 225.0). Rank 1 owns the loss and crosses over to critical within 4 layers. |
| bf16 latent *compute* copy | +3.8% (worse) |
| Reusing the grad buffer (`set_to_none=False`) | **26% slower** — turns an assignment into a read-modify-write |
| Per-element / int8 / bf16 / Adafactor second moment | +10.3% / +47% / +14% (all worse) than block-256 |
| SGD+momentum instead of Adam | rejected on quality |
| In-process (single-interpreter) pipelining | impossible: one python thread cannot feed both GPUs |

The only thing that has worked: fixing a 1F1B bubble where rank 0 released the next activation *after*
its backward, leaving rank 1 idle for the whole of stage-0's backward — **247.6 -> 225.0 s/step, -9.1%**.

**Methodological warning for anyone reading our numbers:** within-run variance is <1%, but two runs of
the *same* config have differed by 16%. Compare only back-to-back runs, one variable at a time, and
never a profiled run against a clean one (profiler overhead here is +17.6%).
