# Research prompt — a 503 s training step that is 97.8% latent plumbing and 2.2% GPU compute

**Ask:** concrete, implementable ways to cut wall-clock per optimizer step, targeting the two costs
that dominate it. Both currently run 3.5-4x slower than their own measured baselines on this machine,
so the question is not "is this physics" but "what is making them miss their own numbers".

Third prompt in a series. `armb_throughput_prompt.md` has the scope/RAM inventory;
`armb_hotspot_prompt.md` has the offloaded-latent hot path. **This one supersedes their timing
numbers**: those were measured at `--lr 0`, where the latent optimizer step is skipped entirely, so
the optimizer never ran and Adam state was never allocated. Everything below is at a real learning
rate.

---

## 1. The measurement

27B ternary student, 2-rank multi-process pipeline (NCCL), `--train-weights all --tw-layer-stride 1`,
`--latent-lr 5e-7`, `--pipe-parallel-mb 2`, seq 2560, split 40/24, NUMA co-located, gradient
checkpointing on, latents fp32 in host RAM (`--latent-offload`), Adam with a block-256 shared second
moment stepped inside the backward by `--latent-grad-release`.

Rank 0 holds **311 latents = 16.49B elements** (53.0M each, 202 MiB fp32). Steady-state step times
502, 498, 505, 507, 498 s.

| bucket | s/step | share | count/step |
|---|---|---|---|
| **CPU Adam** (`_lat_step_one`, block-256) | **236.6** | **47.0%** | 620 = 311 x 2 microbatches |
| **latent H2D** (`latent.to(cuda)`) | **214.1** | **42.5%** | 1240 = 311 x 2 mb x 2 (fwd + checkpoint recompute) |
| gradient D2H (into a persistent host arena) | 41.6 | 8.3% | 620 |
| **everything else** (GPU compute + autograd graph) | **~11** | **2.2%** | — |
| total step | 503.3 | | |

Stage level (`PP_PROF`), same steps:

```
r0  fwd 21.1   BLOCKED   0.0   bwd 454.7      <- never blocks: owns the critical path
r1  fwd 80.2   BLOCKED 187.8   bwd 223.6      <- idle 37% of wall clock
```

## 2. THE TWO ANOMALIES — both costs miss their own measured baseline

**H2D runs at 1.23 GB/s on a link measured at 4.89 GB/s.**
172.7 ms to move 202 MiB. Raw `copy_` benchmarks on this exact machine, with explicit synchronisation:
H2D pinned 4.89 GB/s, H2D pageable 4.88, D2H pinned 5.09, D2H pageable 3.67. So the transfer itself is
**4.0x off the link**. The call is `self.latent.to(self.scale.device, dtype=fp32, non_blocking=True)`,
which allocates a fresh GPU tensor every time and is immediately consumed by the STE, so nothing
overlaps. `--latent-pin` was REMOVED (it measured ~2%: 3.31 vs 3.23 GB/s, and mlocked 106 GB that the
kernel then could not reclaim -- swap is 0 on this box).

**CPU Adam runs 3.5x slower than its own benchmark.**
381.6 ms per 53.0M-element latent. The same block-256 kernel was benchmarked at 48.2 ms per 23.6M
latents = 2.04 ns/element, which predicts 108 ms. The step is:

```python
st["exp_avg"].mul_(b1).add_(gr, alpha=1-b1)
gb = gr.view(-1, 256)
st["v_blk"].mul_(b2).add_(gb.pow(2).mean(dim=1), alpha=1-b2)   # gb.pow(2) is a FULL-SIZE temporary
denom = (st["v_blk"]/bc2).sqrt_().add_(eps).unsqueeze(1)
param.view(-1,256).addcdiv_(st["exp_avg"].view(-1,256), denom.expand(-1,256), value=-lr/bc1)
```

Touches ~3 full-size fp32 arrays per latent (param, exp_avg, grad) plus a full-size `pow(2)`
temporary. At 53M elements that is ~636 MB read+written per latent per step, x620 calls = ~394 GB of
host memory traffic per step. Single-threaded (`OMP_NUM_THREADS=1` is set by torchrun by default) on a
64-core box where only 16 cores are on the rank's NUMA node.

## 3. Fixed constraints — proposals violating these are unusable

* **Sequence length stays 2560.** Set by where the teacher emits end-of-thought tokens.
* **Scope stays full-latent (all linears, stride 1).** Not negotiable for speed.
* **NO NVMe.** Disk offload of latents/optimizer state is out; the only storage is a slow overlay fs.
  (`--latent-state-nvme` and `--snap-nvme` existed and were REMOVED for this reason.)
* **fp32 latent master required.** The Adam update is ~2.5e-4 of latent magnitude vs bf16's 3.9e-3
  resolution. A bf16 *compute* copy was tested: +3.8% (worse).
* **Hardware:** 2x RTX 3090 24 GB (GPU0 on PCIe gen3 x8, GPU1 gen3 x16, both on NUMA node 1 of 4),
  64 cores / 4 NUMA nodes, 629 GiB RAM, **swap disabled**, container cgroup `memory.max` 620 GiB
  (peak usage now 274 GiB, so ~350 GiB is free), `memory.high` not writable.
* **Latents do not fit in VRAM**: 16.49B on rank 0 alone = 66 GB fp32 against a 24 GB card.

## 4. Where we want ideas, in order of measured payoff

1. **CPU Adam, 47% of the step.** Why is it 3.5x its own benchmark, and what closes that? Candidates
   we have NOT tested: multi-threading it (currently 1 thread on 16 node-local cores), eliminating the
   full-size `gb.pow(2)` temporary, fusing the four passes into one over the array, a C++/Numba kernel,
   `torch._foreach_*` batched across latents, or doing the moment update on GPU while the master stays
   on CPU. Which of these actually attacks a memory-bandwidth-bound loop, and what is the realistic
   ceiling given ~394 GB/step of host traffic?
2. **Latent H2D, 42.5%.** Why 1.23 GB/s on a 4.89 GB/s link, and how do we get the transfer to overlap
   compute instead of stalling in front of it? Note the 2x factor from gradient checkpointing
   (every latent crosses twice per microbatch). Is there a way to make the checkpoint recompute reuse
   the forward's transfer, or to keep a compact GPU-side representation and reconstruct?
3. **Rank 1 idles 187.8 s (37%).** Rank 0 never blocks. Moving layers to rank 1 was measured 15.6%
   WORSE at stride 2 under uncontrolled placement, but re-measured at 2.2% once placement was
   controlled -- so the split is close to optimal *for the current cost structure*. If Adam and H2D
   shrink, that changes. Is there a better decomposition than a layer split (e.g. splitting the
   OPTIMIZER across ranks, or having rank 1 run some of rank 0's Adam)?
4. **Anything that reduces the 1240 transfers + 620 Adam calls per step**, given only ~3.55% of
   assignments ever move and motion saturates by ~step 60.

## 5. Already tested — do NOT re-propose without new argument

| idea | result |
|---|---|
| `--latent-pin` (pinned host latents) | ~2% (3.31 vs 3.23 GB/s), and mlocks 106 GB that cannot be reclaimed with swap=0. Removed. |
| `--latent-pin-grad` (pinned grad destination) | 6.2% WORSE. Pageable D2H is only 28% slower than pinned here. |
| Partial GPU residency (`--latent-gpu-budget 8`) | OOMs the 24 GB card; bounded to ~35% of latents anyway. |
| bf16 latent compute copy | +3.8% |
| per-element / int8 / bf16 / Adafactor second moment | +10.3% / +47% / +14% vs block-256 |
| SGD+momentum instead of Adam | rejected on quality |
| Grad-buffer reuse (`set_to_none=False`) | 26% slower (turns an assignment into read-modify-write) |
| NVMe/memmap offload of Adam state | no NVMe on this box; removed from the code |
| Rebalancing the pipeline split (40 -> 36) | +2.2% under controlled placement |
| P1 "GPU holds discrete state, CPU holds continuous" | ceiling measured at -17.3% via a timing spike; not implemented |

## 6. Methodological notes for anyone reading our numbers

* Everything before this prompt was measured at `--lr 0`, which SKIPS the optimizer step. Those runs
  compare configurations validly but understate absolute step time by a large factor.
* This box ran at 7-51% run-to-run variance until NUMA placement was controlled
  (`numactl --cpunodebind=N --membind=N` per rank). Uncontrolled, memory could land on a node 3 hops
  from its GPU, costing 55%. With it controlled, within-config spread is <1%.
* Polling-based memory instrumentation is useless here: with swap disabled, kernel reclaim starves
  userspace samplers exactly when memory spikes. Three memory bugs (146 GB, 211 GB, 106 GB) were found
  only by phase probes plus a live-tensor inventory deduped by storage, not by sampling.
