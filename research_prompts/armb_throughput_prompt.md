# Research prompt — making full-latent (arm B) assignment training tractable on a bandwidth-bound CPU

**Ask:** propose concrete, implementable methods to cut the wall-clock of a 26.0B-latent STE
assignment-training run. We have already decided to run arm B (all latents) rather than the cheaper
arm A (`down` only); we are **not** looking for advice to reduce scope. We want speed at fixed scope.

Everything below is measured on this project unless marked as an estimate. Numbers in **bold** are the
ones a proposal has to beat or respect.

---

## 1. What the computation is

We convert a dense LLM to ternary weights (`{-1,0,+1}` × a group-64, 8-bit scale). After GPTQ +
block-wise adaptive-precision (block-AP) recovery produces a ternary "skeleton", an **assignment stage**
improves *which trit* each weight takes, by gradient descent on a continuous latent `L` per weight with
a straight-through estimator (STE). A weight's trit is decided by `L`'s position relative to
quantization-bin boundaries set by the scale `s`; the STE gate is `|L| < 1.5·s`.

Per optimizer step, for each latent, we currently hold and touch in **CPU RAM**:

| tensor | dtype | bytes/latent |
|---|---|---|
| latent `L` | fp32 | 4 |
| grad | fp32 | 4 |
| Adam `exp_avg` | fp32 | 4 |
| Adam `exp_avg_sq` | fp32 | 4 |
| best-checkpoint snapshot | fp32 | 4 |
| overhead | — | ~2.9 |

Measured total: **18.9 bytes/latent + 6.92 GB base** (least-squares over three scopes, predicts within
0.2 GB). The forward/backward runs on GPU; the latents and their optimizer state do **not** fit in VRAM,
so they are CPU-resident (`--latent-offload`) and the optimizer step is a pure CPU memory-bound loop.

**Scope inventory for the 27B target** (64 layers, hidden 5120, intermediate 17408, vocab 248320,
48 linear-attn + 16 full-attn), counted from the actual checkpoint:

| scope | latents | RAM @18.9 B |
|---|---|---|
| `down` only (arm A) | 5.704B | 115 GB |
| all MLP | 17.11B | 330 GB |
| attention (incl. DeltaNet) | 5.54B | 111 GB |
| `lm_head` | 1.271B | 31 GB |
| **EVERYTHING (arm B)** | **26.049B** | **499 GB** |

## 2. The hardware — this is the crux

- **CPU: 4× Intel Xeon E5-4650 @ 2.70 GHz (Sandy Bridge-EP, 2012).** 32 physical cores / 64 threads,
  4 NUMA nodes, DDR3. **AVX only — no AVX2, no FMA, no AVX-512.**
- **RAM: 629 GB.** Memory is NOT the binding constraint; arm B's 499 GB fits.
- **GPU: 2× RTX 3090 (24 GB), connected PHB (PCIe host bridge), no NVLink**, both affine to NUMA node 1.
- Storage: 1.7 TB free.

The optimizer step is **memory-bandwidth bound**, not FLOP bound. Evidence: thread count barely moves it
(8/16/32/64 threads span only 90.7 / 96.6 / 79.0 / 88.5 ms per 23.6M latents), and `numactl
--interleave=all` vs `--cpunodebind=0 --membind=0` differ by ~5%. Step time scales exactly linearly with
latent count (4 layers → 8 layers doubles it).

## 3. Measured baseline and what we have already tried

Current production config is `torch.optim.Adam(per_layer_groups, foreach=False)`, fp32 throughout.
Measured on 27B-shaped `down_proj` tensors (5120 × 17408 = 89.1M latents each):

| variant | ms / 23.6M latents | arm B s/step (optimizer only) | 6244-step run | vs current |
|---|---|---|---|---|
| **Adam fp32 `foreach=False` (CURRENT)** | **81.1** | **89.5 s** | **155.2 h (6.5 d)** | 1.00× |
| Adam fp32 `foreach=True`, one group | 62.9 | 69.5 s | 120.5 h | 1.29× |
| SGD + momentum, `foreach=True` | 20.7 | 22.9 s | 39.7 h | **3.91×** |
| Adam, bf16 moments (hand-rolled) | 118.5 | 130.8 s | 226.8 h | 0.68× (**slower**) |
| 8-bit block-wise Adam (`src/adam8bit_cpu.py`) | — | — | — | 0.91× (**slower**, +10.3%/step) |

**Two negative results worth internalising:** every attempt to shrink optimizer-state *dtype* has made
this machine SLOWER, because without AVX2/FMA the dequant/convert cost exceeds the bandwidth saved.
int8 moments: +10.3% per step. bf16 moments: +47%. Do not propose quantized optimizer state as a speed
lever here (it remains a valid *memory* lever, which we do not need).

`foreach=True` is a free 1.29× that we are not currently taking — it was disabled because on the old
60 GB box the fused path's same-size temporaries pushed RSS to 43.8 GB and wedged the process. At 629 GB
that constraint is gone, **but** temporaries scale with param-group size: one group of 26.0B latents
would add ~104 GB on top of 499 GB. Group-size tuning is an open design question.

## 4. Hard constraints — a proposal that violates these is not usable

1. **The latent must stay fp32.** bf16 latents were measured to destroy the model (held-out KL 0.9495 vs
   0.6624). The STE gate decides flips from *distance to a bin boundary*; a floating exponent gives
   resolution proportional to `|L|` instead of a uniform step. Optimizer *moments* carry no boundary
   information and may be altered freely — that axis is just slow here (§3).
2. **Scales must stay frozen during assignment training (`--lr 0`).** Co-training scales and assignments
   is the least stable configuration we have found: at matched step 20, scales training → KL 3.30
   (destroyed) vs scales frozen → KL 0.2612 (intact), reproduced twice. Training the scale moves the
   quantization boundary *underneath* the latents and flips assignments wholesale. The V/P alternation
   (assignments move with `s` frozen; `s` refit with assignments frozen) is required, not optional.
3. **A transition-rate servo (TALR) governs the latent lr.** It measures the fraction of assignments
   flipping per step and adjusts lr toward an annealed target (clamps ×0.6 on overshoot, opens ×1.3 when
   under). Flip *rate*, not lr, is the controlled variable — raw lr cannot control flip count. Any
   proposal changing the optimizer must say how TALR still gets a measurable, controllable flip rate.
4. **A cold-start lr ramp is mandatory.** Un-servoed post-warmup steps did all the damage in one run
   (assign-moved burst to 10.09% at step 40, then froze; ~40% of that damage was permanent). Start the
   latent lr low and let TALR ramp up.
5. **Layer-sequential passes stack; scope-sequential passes do not.** Splitting work along the *layer*
   axis is safe (each group dips ~4%, recovers, finishes ahead). Splitting along the *scope* axis
   (e.g. `gate` then `up`) always damages and never recovers — SwiGLU's `down(silu(gate(x))·up(x))`
   makes gate/up multiplicative, so tuning one co-adapts the other's current assignments. Any
   decomposition must cut on layers, never on projections within an MLP.

## 4b. MEASURED: the CPU optimizer is NOT the dominant cost — read this before proposing anything

We ran a paired 4B smoke test (240 steps, 8 `down_proj` layers = 188.7M latents, seq 2560, 1×3090)
swapping ONLY the latent optimizer. Wall clock:

| arm | wall | max RSS |
|---|---|---|
| Adam | 6670.45 s | 21.0 GB |
| SGD+momentum (3.91× faster optimizer) | **6669.21 s** | 20.9 GB |

**Identical to 0.02%.** At 27.8 s/step, the optimizer over 188.7M latents is ~0.65 s = **2.3% of the
step**; GPU forward/backward is everything else. A 3.91× optimizer win bought nothing measurable.

Extrapolating to 27B arm B: the optimizer term is 89.5 s/step (26.0B latents) while forward/backward
scales roughly with parameter count (27B/4B ≈ 6.5×, and 2 GPUs instead of 1). That puts the optimizer at
an ESTIMATED 30–50% of the step — material, but capping any optimizer-only speedup at ~1.5–2× overall.

**Consequence: proposals targeting only the CPU optimizer have bounded upside.** We still want them (see
§5) but a proposal that also attacks the GPU forward/backward — or that reduces the NUMBER of steps —
has a much larger ceiling. Measuring the true 27B split is our next step; treat the 30–50% as an
estimate, not a measurement.

## 4c. MEASURED: SGD+momentum is REJECTED on quality — do not re-propose it

Same paired smoke test, same 240 steps, SGD lr calibrated from the measured latent grad RMS
(1.529e-04) for a matched effective step size (`lr_sgd = lr_adam / rms = 3.27e-02`):

| | Adam | SGD+momentum |
|---|---|---|
| entry held-out KL | 1.0114 | 1.0114 |
| final held-out KL | **0.6335 (−37.4%)** | 0.7550 (−25.4%) |
| assign-moved | **3.551%** | 11.069% |
| assign-moved @ step 40 | 2.257% | **10.548%** (burst) |
| per-layer assign-moved | min 3.29 med 3.52 max 4.08 (**ratio 1×**) | min 7.39 med 7.89 max **32.44** (ratio 4×) |

SGD moved 3.1× as many assignments for two-thirds of the KL benefit, front-loaded the movement into a
burst (the §8r damage mode), and let one layer run to 32.4% against a 7.9% median. Adam's per-coordinate
normalisation is doing real work: it equalises motion across latents and layers, which a single global
lr structurally cannot. TALR throttled to gain 0.03–0.16 but the damage lands in the un-servoed window
before its first measurement.

Caveat: only ONE SGD lr was tested. A lower lr would move less, but the per-layer imbalance is
structural, not a tuning artifact. Any proposal replacing Adam must preserve **per-coordinate step
normalisation** — that is the property being paid for, not the moment estimates as such.

## 4d. MEASURED: assignment motion is sparse, structurally identifiable, and finishes early

Two facts that we think are the most promising opening, both measured on the real 4B skeleton.

**(i) ~15% of latents sit EXACTLY on a decision boundary at init, and they are free to identify.**
Distance to the nearest boundary is `d = | |L/s| - 0.5 |`. Over 188.7M latents (8 layers):

| d < | 0.001 | 0.002 | 0.005 | 0.01 | 0.05 | 0.2 | 0.5 |
|---|---|---|---|---|---|---|---|
| fraction | 14.91% | 14.91% | 15.24% | 15.69% | 19.63% | 38.89% | 85.02% |

Flat from 0.001→0.002 ⇒ this is a SPIKE at d≈0, not a density tail. Mechanism confirmed: `fp-spread`
init sets `L = clamp(w_fp, (t-0.5)s, (t+0.5)s)`, so wherever the GPTQ/block-AP trit DISAGREES with naive
FP rounding, `w_fp` falls outside the bin and the clamp pins `L` to the bin edge. Per layer: clamp fires
16.6–17.5%, FP-rounding disagreement 13.2–14.1%, and the disagreement set is a strict SUBSET of the
clamped set. **The candidate set is therefore known at initialisation for free — it is exactly "where the
assignment disagrees with FP rounding" — and needs no distance scan to construct.**

**(ii) Motion saturates early.** In the Adam arm, `assign-moved` reached 3.511% by step 60 and was then
flat for the remaining 180 steps (3.542 → 3.539 → 3.541 → … → 3.551) while held-out KL kept improving
0.7122 → 0.6335. Flips finish in the first ~40 post-warmup steps; the rest of the run is refinement of
latents that never cross a boundary.

Cost model if optimizer state were maintained for a candidate fraction α only (dense Adam = 16 B/latent
touched per step; rescan reads L only, 4 B):

| α | rescan every 50 steps | vs dense |
|---|---|---|
| 5% | 0.88 B | 18.2× |
| 10% | 1.68 B | 9.5× |
| 20% | 3.28 B | 4.9× |

Open risks we have NOT resolved: scales are frozen, so a latent excluded from updates has a static `d`
and can never re-enter the candidate set on its own — how should rescan work, and does (ii) mean it
barely matters? Does restricting updates break TALR's global rate measurement? And per §4b, an
optimizer-side win of even 10× is capped at ~1.5–2× end-to-end unless paired with a forward/backward win.

## 5. What we are asking for

Concrete, implementable proposals to reduce arm B **end-to-end wall clock**, ranked by expected speedup
× confidence. Given §4b, we are now MOST interested in the first two:

- **The GPU forward/backward — the actual dominant term.** 27B, seq 2560, 2× RTX 3090 (24 GB, PHB, no
  NVLink), bf16, with a cached top-k teacher (no teacher forward at train time). The student is a packed
  ternary model with fp32 CPU latents shuttled per layer. What are the real levers — activation
  checkpointing policy, microbatch/accum shape, keeping more of the packed model resident, avoiding
  latent↔GPU transfer per step, FSDP vs DDP for a 27B student on 2×24 GB, sequence packing, or lowering
  `--seq` (currently 2560, chosen to exceed the STEM close length: median 1786, p75 2514)? Which of
  these interact badly with STE, whose backward must see the same boundary geometry as the forward?
- **Exploiting the sparse, pre-identifiable candidate set (§4d).** ~15% of latents are pinned exactly on
  a boundary at init and are identifiable for free as "assignment disagrees with FP rounding"; motion
  saturates by ~step 60. Design the algorithm: how to maintain optimizer state for a subset, what
  rescan cadence (if any) is needed given scales are frozen so excluded latents have static `d`, how
  TALR's rate measurement should be computed over a subset, and how to avoid biasing the flip
  population. Note the end-to-end ceiling from §4b.
- **Doing fewer steps.** Data scaling measures +1.67 pt eval2k per doubling of unique tokens with no
  saturation observed, and fresh tokens are worth ~3× repeats. Our unique-token ceiling is 1.67B (bound
  by one scarce source in a fixed mixture). Is there a principled way to spend a fixed latent-update
  budget on more unique data and fewer passes? This attacks the step COUNT, so it multiplies with any
  per-step win.
- **Overlap.** The GPU forward/backward and the CPU optimizer step are currently serialized, though
  `--latent-grad-release` already steps each latent inside backward via a post-accumulate-grad hook.
  How much more overlap is available, and what synchronisation does TALR's global rate measurement
  actually require?
- **Optimizer, if it can keep per-coordinate normalisation.** SGD is rejected (§4c) because a single
  global lr cannot equalise motion across latents and layers. Are there optimizers with Adam-like
  per-parameter scaling at lower memory traffic — Adafactor's factored second moment, Lion (sign-based,
  one state tensor, but is a sign update compatible with distance-to-boundary dynamics?), or a
  second moment shared per block of 256 (the existing scale-block granularity)? Note that shrinking
  state DTYPE is already refuted on this CPU (§3): the lever must reduce state COUNT or FOOTPRINT
  without a per-element convert.
- **Kernel-level.** Is a hand-written AVX(1)/OpenMP Adam meaningfully better than PyTorch's dispatch on
  Sandy Bridge? Would `torch.compile` help a CPU memory-bound loop? Does the 4-socket topology permit
  useful thread/memory pinning per layer group given each group is one tensor?

## 6. What we will do with the answer

Validate the top one or two proposals on a 4B testbed (32 layers, same pipeline, frozen 1946-sequence
eval2k referee) as a paired comparison against the Adam baseline at matched steps and data, checking
both teacher-forced agreement and a free-generation gate (commit ≥68%, loop ≤30%, compression ≤3.1) —
teacher-forced metrics alone are proven blind here, having read 79–84% on a model that scored below
random on free generation. Then run the 27B arm B once.

## 7. Open measurements

1. **The 27B step split.** §4b measured the 4B split (optimizer 2.3% of step) and extrapolates the 27B
   optimizer share to 30–50%, but that is an ESTIMATE from parameter-count scaling. Measuring ~50 real
   27B arm B steps and reporting the optimizer/forward-backward split is our next action, and it decides
   how much any §5 proposal is worth. Say which regime your proposal helps in.
2. **Whether the §4d candidate set holds at 64 layers.** All boundary statistics are from the 4B
   (32 layers). Per-layer clamp rates were stable (16.6–17.5%) across depth there, but the 27B has
   twice the depth and a different attention mix (48 linear-attn + 16 full-attn).
3. **Whether motion still saturates at 6244 steps.** §4d(ii) observed saturation by step 60 in a
   240-step run. A 6244-step production run may keep recruiting new flips slowly; that would change the
   rescan requirement for any candidate-set design.
