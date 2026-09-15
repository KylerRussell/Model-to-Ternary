# Deep research prompt — the (format × method) cross product at ≤2.1 bpw

**What we need:** for each *deployment format family*, the best published result achieved by **any
method paired with that format**, normalised to whole-model bits/weight, so we can choose a format
per deployment band on evidence rather than on naive-rounding capacity.

**Why this is not the obvious question.** We measured ten format families under round-to-nearest with
no recovery. That systematically penalises exactly the formats whose *method* does the heavy lifting:
AQLM, QuIP# and QTIP all report numbers from bespoke optimisation, so scoring their formats under
naive rounding measures our encoder, not their representation. The unit of comparison has to be the
**(format, method) pair**, and we need the cross product, not one axis of it.

---

## 0. What we have already measured — start from here, don't re-derive it

Stock Qwen3.5-4B and Qwen3.5-27B, round-to-nearest, no calibration or recovery, every arm given a
real search over its own parameters. Relative reconstruction error, and end-to-end perplexity with
every quantizable 2D tensor replaced (49,104 held-out tokens; fp16 baseline **4.34**):

| bpw | format | recon err | ppl (RTN) | % of Shannon bound |
|---|---|---|---|---|
| 3.062 | int3 scalar g256 | 0.218 | 7.06 | 55.1% |
| **2.062** | **trellis k=2, L=12** | **0.258** | **11.83** | **92.8%** |
| 2.062 | VQ k=256 d=4 | 0.319 | — | 75.0% |
| 2.252 | lattice E₈ g256 | 0.337 | — | 62.3% |
| 1.835 | ternary scalar g64 | 0.433 | 2962.68 | 64.8% |
| 1.688 | VQ k=8192 d=8 | 0.388 | 657.80 | 80.1% |
| 1.562 | VQ k=4096 d=8 | 0.421 | 1677.12 | 80.5% |
| 1.062 | trellis k=1, L=10 | 0.528 | 20060.11 | 90.7% |

Two facts from this that shape the question:

1. **Under RTN the entire sub-2-bpw regime is unusable** (ppl 658–20,060). Whatever the right format
   is down there, a *method* is doing the work, and that is what we need the literature for.
2. **Distance from the Shannon bound `D(R) = σ²2^(−2R)`** (valid here: measured excess kurtosis is
   0.18–1.24, so weights are near-Gaussian) separates structural headroom from encoder headroom.
   Trellis at **92.8%** has little left to give — Viterbi is already an optimal encoder for it.
   Scalar ternary at **64.8%** and int3 at **55.1%** are far from the bound, so most of their gap is
   *encoder* inefficiency that a good method should be able to close. **Tell us how much of it
   published methods actually close.**

## 1. The three bands we must choose for

Bands are set by deployable size for a 27B at 4k context, not by round numbers. The 8 GB consumer
card is where they separate.

| band | budget | fits | current pick (RTN evidence only) |
|---|---|---|---|
| **A** | ~2.0–2.1 bpw | 12 GB | trellis k=2 |
| **B** | ~1.65–1.80 bpw | 8 GB, tight | VQ k=8192 d=8 |
| **C** | ~1.50–1.60 bpw | 8 GB, headroom | VQ k=4096 d=8 |

For each band we want the **best (format, method) pair in the literature**, with its measured cost.

## 2. The cross product — this is the core deliverable

Rows are format families, columns are method classes. Fill every cell you can with the **best
published result**, and mark cells nobody has tried.

**Format families (rows):** uniform scalar symmetric · uniform scalar asymmetric · non-uniform scalar
(NF4/companded) · micro-float (MXFP4/NVFP4) · vector quantization · lattice (E₈, Barnes-Wall) ·
trellis-coded (QTIP) · additive multi-plane · low-rank + quantized residual · sparse–dense hybrid ·
mixed precision.

**Method classes (columns):** round-to-nearest · Hessian/second-order PTQ (GPTQ, OBQ) · activation-
aware scaling (AWQ, SmoothQuant) · learned rounding (AdaRound) · learned clipping/transform
(OmniQuant) · incoherence processing / rotation (QuaRot, QuIP, SpinQuant) · block-wise reconstruction
QAT (EfficientQAT, block-AP) · full QAT / from-scratch (BitNet) · knowledge distillation · codebook
fine-tuning (PV-Tuning) · RL / verifiable-answer post-training.

For each filled cell: **citation · model + size · whole-model bpw · metric and value · whether the
method is specific to that format or general.**

## 3. Questions we most need answered

1. **How much of the gap to the Shannon bound does a good method close, per family?** Our RTN
   measurements put scalar ternary at 64.8% of optimal and trellis at 92.8%. If GPTQ-class methods
   recover most of scalar ternary's gap, the band-B and band-C picks change. If they do not, the
   near-bound families win on structure and the question is settled.
2. **Is there a published (format, method) pair that is usable — not merely reported — below
   1.8 bpw whole-model?** We are looking for a working model, so state the metric and say plainly
   whether it is perplexity, a multiple-choice benchmark, or free-generation capability.
3. **Which cells of §2 are genuinely empty?** We suspect trellis and lattice have been paired with
   almost nothing except their own authors' methods, and that the IQ formats — which carry most
   local inference in the world — have essentially no academic method work at all. If so, the empty
   cells are a contribution in themselves.
4. **Does any theory give a tighter bound than the memoryless Gaussian D(R)?** Weights have structure
   (row correlation, outlier channels, layer-dependent variance) that a memoryless bound ignores. Is
   there a published rate-distortion analysis for LLM weights specifically, and does it change which
   family should win at ~1.6 bpw?
5. **What does each format cost to *encode*?** Viterbi over a trellis, k-means over a codebook and
   GPTQ's Hessian inverse are very different one-time costs. A format that needs a week of encoding
   for a 27B is a different proposition from one that needs an hour, and this is rarely reported.

## 4. Hard constraints — screen every candidate against these

* **Weight-only.** Activations stay bf16; we have no activation quantizer. Methods requiring coupled
  low-bit activations (TWLA, QuEST, DBellQuant) are out of scope for deployment — catalogue them,
  but say so.
* **Whole-model accounting, always.** Report the bpw including embeddings, LM head, norms and every
  side structure — codebooks, outlier indices, low-rank factors, bit-allocation maps. Headline
  figures in this literature routinely cover linear projections only: a nominal 1.58 bpw method
  leaving vocabulary tensors in FP16 is **3.43 bpw** whole-model on an 8B, and one published 1.64 bpw
  method ships an artifact at ~8.8 bits/parameter. If a paper does not permit the whole-model figure
  to be computed, say so — that is itself a finding.
* **Hardware:** 2× RTX 3090 (24 GB, no NVLink), AVX-only host, no NVMe. A format with no working
  kernel is a research result, not a deployment option; distinguish the two.
* **Deployability is separate from rate–distortion.** Entropy-coded formats dominate on pure R-D and
  are excluded by every production runtime, for the structural reason that symbol *k+1*'s bit offset
  depends on decoding symbol *k*. Keep those two frontiers distinct.

## 5. What a good answer looks like

The filled cross product, with empty cells marked as empty rather than quietly omitted. We would
rather learn that our band-B and band-C picks are wrong than have them confirmed — the RTN evidence
behind them is weak by construction, and a well-evidenced *"scalar ternary plus method X beats VQ at
1.6 bpw, here is the number"* is the single most useful thing you can return.

Prefer free-generation capability evidence over perplexity, and perplexity over reconstruction error;
say which you are reporting. Do not repeat hardware folklore about codebook lookups being infeasible
on consumer GPUs — `IQ2_XXS` has shipped in llama.cpp for two years.
