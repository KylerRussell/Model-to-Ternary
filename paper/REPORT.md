# Choosing a sub-2-bit weight format: a measured survey

**Rough draft.** Everything here is measured in this repository; provenance in `paper/FINDINGS.md`,
taxonomy in `paper/format_taxonomy.md`, code in `src/family_sweep.py`. Limits are §6 and are not
optional reading — several results are narrower than they look.

---

## 1. The question

Published sub-2-bit quantization work almost universally evaluates a method against a representation
of its own choosing. A method that "fails at 1.6 bits" may have failed against a *format*, not
against a bit budget, and the literature cannot separate the two. Neither could we until we measured
more than one format under one protocol.

We therefore ask a narrower question than most of the field: **holding the model and the protocol
fixed, what does each format family buy per bit?** — and then, because bits compound into deployable
size, **which format should be used at each size a real GPU admits?**

## 2. What counts as a family

A taxonomy assembled from what tooling makes easy will reliably conclude that the easy thing is best.
Our first attempt produced three families — symmetric scalar, asymmetric scalar, vector quantization
— which are precisely the three implementable in an afternoon. Enumerating from structural axes
instead (level geometry · quantization unit · decomposition · code length · decode dependency · bit
allocation) yields **thirteen** populated families; the ten omitted included both eventual winners.

| | family | instances | measured |
|---|---|---|---|
| A | uniform scalar, symmetric | `Q4_0`, `TQ1_0`, BitNet | yes |
| B | uniform scalar, asymmetric | `Q2_K`, GPTQ, AWQ, EXL2 | yes |
| C | non-uniform scalar | `IQ4_NL`, NF4 | yes |
| D | micro-float / block FP | `MXFP4`, `NVFP4` | yes |
| E | vector quantization | `IQ1_S/M`, AQLM, VPTQ | yes |
| F | lattice | QuIP# (E₈) | yes |
| G | trellis-coded | QTIP | yes |
| H | additive multi-plane | PTQTP, BiLLM | yes |
| I | low-rank + residual | OneBit (SVID) | yes |
| J | sparse–dense hybrid | SpQR, SqueezeLLM | yes |
| K | entropy-coded | SoftWater, ECASQ | rate only |
| L | mixed precision | EXL2, PTQ1.61 | yes |
| M | weight sharing | pre-LLM | excluded, relevance |

## 3. Method

**Stock weights only.** Qwen3.5-4B and Qwen3.5-27B as published — no untying, no rotation, nothing
from this project's pipeline. Rotation in particular reshapes the local variance that fine-grained
scales exist to capture, so measuring formats on our own checkpoints would let our pipeline pick the
winner.

**Each family at its ceiling, not at one implementation.** Codebooks are k-means-fitted to the actual
weight distribution rather than taken from a published grid; lattices are decoded exactly; trellises
are solved by exact Viterbi; multi-plane is fitted by alternating least squares; mixed-precision
ranks channels by the error they *actually* incur rather than by a proxy. A method requiring a family
should be screened against what the family can do.

**Rate is computed, not quoted.** Every side structure is charged: both planes and both scale sets for
multi-plane, the factors for low-rank, value *and* index for sparse-hybrid, the allocation map for
mixed precision. For lattices the rate is *measured* — `log₂(distinct points used)/8` — because
rounding to the infinite E₈ lattice has no bounded index and a nominal figure would credit the family
with error achievable only at unbounded rate. This mattered: lattice rate rose from 1.65–1.88 to
2.25–2.55 bpw when measured on representative tensors.

**Two metrics.** Relative reconstruction error per tensor (cheap, 57 configs), and end-to-end
perplexity with every quantizable 2D tensor replaced by its dequantized form (expensive, 6 configs,
49,104 held-out tokens). All arms are round-to-nearest with **no recovery training** — see §6.

## 4. Results

### 4.1 Reconstruction error predicts perplexity exponentially

Six formats spanning four families, 1.06–3.06 bpw:

| bpw | format | recon err | perplexity |
|---|---|---|---|
| 16.000 | fp16 | — | **4.34** |
| 3.062 | int3 g256 | 0.220 | **7.06** |
| **2.062** | **trellis k2** | **0.258** | **11.83** |
| 1.688 | vq k8192 | 0.388 | 657.80 |
| 1.562 | vq k4096 | 0.420 | 1677.12 |
| 1.835 | ternary g64 | 0.435 | 2962.68 |
| 1.062 | trellis k1 | 0.528 | 20060.11 |

    log(ppl) = 27.31 · recon_err − 4.20        r = 0.99610

A 0.01 absolute change in reconstruction error multiplies perplexity by **1.31×**. This is the result
that makes a reconstruction-only sweep worth running: 57 cheap measurements become predicted
end-to-end quality.

It also corrects a belief we held for most of this project — that local reconstruction metrics never
predict anything. They predict very well. What misled us is that the mapping is *exponential* and
every format we had compared sat at recon 0.43–0.46, where the model is already destroyed;
reconstruction was correctly answering "still broken?" every time. (The metric we were right to
distrust, block-MSE on hidden states, is a different quantity — we over-generalised from it.)

### 4.2 The frontier is scale-invariant

The same 57 configurations on Qwen3.5-4B and Qwen3.5-27B MLP tensors differ by **+0.00003 to +0.009
relative error**, mostly under +0.002, with rank order unchanged:

| bpw | config | 4B | 27B | Δ |
|---|---|---|---|---|
| 2.062 | trellis k2 L12 | 0.25798 | 0.25801 | +0.00003 |
| 2.062 | trellis k2 L10 | 0.26441 | 0.26448 | +0.00007 |
| 2.062 | vq k256 d4 | 0.31934 | 0.31973 | +0.00040 |
| 1.835 | ternary g64 | 0.43256 | 0.43363 | +0.00107 |
| 2.252 | lattice E8 | 0.33679 | 0.33940 | +0.00261 |

**A format decision made at 4B transfers to 27B** — at the reconstruction level. This is the
opposite of this project's experience with *methods*, where the 4B is a known-catastrophic testbed
(−62% relative MMLU against the 27B's −18%). Formats and methods do not share that sensitivity.

### 4.3 The frontier is role-invariant

MLP, attention and embedding tensors give identical rankings within 1–5%, at both scales. The
mechanism was checked rather than assumed: excess kurtosis is 0.18–1.24 across all roles at both
scales — every role is mildly heavy-tailed and close to Gaussian, so the frontier is the same
**because the source is effectively the same**. The claim is scoped accordingly: a model whose roles
have genuinely divergent distributions should break this, and ours does not.

Consequence: applying one format uniformly across the model is justified. That is the implicit
assumption in every published sub-2-bit method and, as far as we can tell, had not been checked.

### 4.4 Which families reach the frontier

Below ~2 bpw the frontier is **trellis (G)** and **vector quantization (E)**. Scalar families reappear
from ~3 bpw; non-uniform scalar (C) owns the 4-bit band, beating uniform int4 by ~15% at identical
bits. Three families never reach the frontier at any rate or role: **D (micro-float), H (multi-plane),
L (mixed precision)**. Lattice (F) reaches it only once its rate is measured honestly, and then only
above 2.2 bpw.

Two incidental results worth stating because they are cheap and commonly got wrong:

* **`TQ2_0` is strictly dominated by `TQ1_0`** — identical alphabet and identical g256 scale, so
  identical reconstruction *by construction*, at 22% more bits. It is a throughput format, never a
  compression point.
* **Symmetric int2 *is* ternary** with one of four codes unused, and measures identically. Ternary
  needs log₂3 = 1.585 bits, so packing reclaims 0.415 bpw for free — which is the entire reason the
  TQ formats exist.

## 5. Decisions

### 5.1 Why three formats and not one

Bits compound into deployable size, and size is a step function against real hardware. For a 27B at
4k context:

| bpw | weights | + KV + runtime | 8 GB | 12 GB | 16 GB |
|---|---|---|---|---|---|
| 1.562 | 4.80 G | 6.85 G | **YES** | YES | YES |
| 1.688 | 5.18 G | 7.23 G | **YES** | YES | YES |
| 1.750 | 5.37 G | 7.42 G | **YES** | YES | YES |
| 1.835 | 5.63 G | 7.68 G | **no** | YES | YES |
| 2.062 | 6.33 G | 8.38 G | **no** | YES | YES |
| 3.062 | 9.40 G | 11.45 G | no | no | YES |

**The 8 GB card is where the bands separate.** It is what makes 1.75 versus 1.84 bpw a deployment
decision rather than a rounding difference, and it is why a single recommended format would be the
wrong output for this survey. Each band is the best answer to a different hardware question.

### 5.2 The three picks

| band | pick | bpw | recon | ppl (RTN) | rationale |
|---|---|---|---|---|---|
| **~2.0** | **trellis k2 L12, g256** | 2.062 | 0.258 | **11.83** | the only sub-3-bpw format that survives RTN at all; 250× better perplexity than ternary for 12% more bits |
| **~1.7** | **VQ k8192 d8, g256** | 1.688 | 0.388 | 657.80 | best in band; beats ternary g64 on **both** axes (4.5× perplexity at 8% fewer bits) |
| **~1.56** | **VQ k4096 d8, g256** | 1.562 | 0.420 | 1677.12 | best at the ternary-equivalent rate; beats ternary g256-s8 (1.616 bpw, 0.4389) on both axes |

**None of the three is scalar ternary**, which was not the expected outcome and is the survey's
sharpest result. At every rate where ternary competes, a codebook format reaches lower error at equal
or lower bpw. Ternary's appeal is that log₂3 = 1.585 is a natural-looking target and that packing is
simple; neither is an argument about quality.

### 5.3 What the bands look like with a real encoder

An earlier draft of this section claimed that under RTN every sub-2-bpw format lands on the destroyed
side, and that bands B and C are therefore starting points for a recovery pipeline rather than
deployable configurations. **Model-wide GPTQ shows that framing understated what a post-training
encoder achieves on its own.**

| format | bpw | RTN ppl | GPTQ ppl | vs fp16 (4.34) |
|---|---|---|---|---|
| **trellis k2** | 2.062 | 11.83 | **6.53** | **1.51×** |
| ternary g64 | 1.835 | 2962.68 | **57.84** | 13.34× |

**Band A is deployable today.** Trellis + GPTQ at 2.062 bpw reaches 1.51× fp16 perplexity, and beats
int3 + RTN at 3.062 bpw (6.53 vs 7.06) — **33% fewer bits and better quality**.

**GPTQ narrows the format gap 28× (250× → 8.9×) without closing it.** Reconstruction error rose by a
similar proportion in both formats under GPTQ (+39% trellis, +42% ternary), so the encoder favours
neither; the residual 8.9× is the sphere-packing deficit of §4.4, structural and not recoverable by
encoding. This is the measurement that decides the band picks, and it decides them for the non-scalar
formats.

**Bands B and C remain provisional for a different reason than before.** Not because sub-2 bpw is
unusable — ternary at 57.84 is a working if degraded model — but because **the band B and C picks
(VQ k8192, VQ k4096) have no end-to-end GPTQ number**. They were selected on RTN reconstruction, and
VQ sits structurally between the two formats that were measured. That is the gap to close next.

### 5.4 Design decisions in this survey, and what was wrong with them

Recorded because the same failures are what the survey documents in the literature.

* **The bit budget was inherited, not derived.** This project targeted 1.78 bpw to match a released
  ternary model's operating point. That is a competitive anchor and a fine thing to cite; adopting it
  as *the* evaluation point made every comparison ask "which format is best at the bpw we already
  picked". §5.1 derives the bands from hardware instead.
* **Family-bounded search.** Our own format, TQ1_64, was designed by optimising scale granularity
  within the scalar-ternary family (g256 → g128 → g64), exhaustively and correctly. §4.4 shows that
  axis is movement *along* the rate–distortion line; the axis that moves a format *off* it sat behind
  an assumption ("no zero-point") that was never itself tested.
* **Convenience-chosen taxonomy** (§2) and **convenience-chosen tensors** — an early sweep ran on
  three `(32, 2560)` tensors picked by alphabetical position, which reversed the verdict for two
  families and made the low-rank arm reconstruct an identity.

The generalisation, stated once: **whenever a choice is made by what is nearest to hand rather than
by what the question requires, it decides the answer.** A survey that applies a structural screen to
other people's work and exempts its own is not worth reading.

## 6. Limits

* **RTN only, and this is the survey's weakest joint.** No arm receives GPTQ, calibration, or
  recovery training, so this measures formats at *our encoder's* competence rather than at their
  ceiling. The distortion is not uniform across families: measured against the Shannon bound
  `D(R) = sigma^2 2^(-2R)`, trellis reaches **92.8% of optimal** while scalar ternary reaches
  **64.8%** and int3 **55.1%**. For trellis, Viterbi is already a near-optimal encoder, so little is
  left on the table; for the scalar families most of the gap is plausibly *encoder* inefficiency that
  a Hessian-aware or learned-rounding method could close. **The families ranked lowest here are
  exactly those with the most to gain from a better method**, so §5.2's band B and C picks are
  provisional pending the (format × method) literature review.
* **Perplexity is a screen, not capability.** This project has separately shown a model can pass
  teacher-forced metrics at 78.5% agreement while solving 0/18 arithmetic problems its teacher
  solves. No capability benchmark was run on these formats.
* **The exponential fit is six points on one model**, and the 0.258–0.388 interval is unsampled, so
  the usable-error boundary is a range (~0.26–0.39), not the number 0.30. The slope has not been
  checked at 27B — 64 layers compound error through twice the depth, and it could steepen.
* **Scale-invariance is reconstruction-only.** §4.2 does not license the claim that *end-to-end*
  behaviour is scale-invariant; that would need the 27B end-to-end run, which needs all 11 shards.
* **One model family.** Qwen3.5 only. Every claim is scoped to near-Gaussian weight distributions
  (§4.3) and should be re-checked before transfer.
* **G is measured with our own trellis**, not QTIP's; **K (entropy coding) is rate-only**; **M is
  excluded**.

## 7. Open

1. **End-to-end at 27B** — the one measurement that would confirm the band selection transfers.
2. **Fill the 0.258–0.388 reconstruction gap** with two or three arms, to locate the usable boundary
   rather than bracket it.
3. **Post-recovery ranking.** If recovery training reorders the families, §5.2 changes. This is the
   single largest open risk to the recommendations.
4. **A capability benchmark** on the ~2.06 pick, since perplexity 11.83 against fp16's 4.34 is a
   large relative gap whose task-level meaning is unmeasured.
