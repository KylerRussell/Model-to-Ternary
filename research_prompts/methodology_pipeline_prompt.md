# Deep research prompt — best-known METHOD PIPELINES for four specific encoders

**The encoders are now fixed.** We are no longer asking which format to use; we have chosen four and
measured them. What we need is: **for each of these four, what is the best-known way to encode a model
into it**, including multi-stage pipelines, at a training budget a two-GPU lab can afford.

---

## 0. The four encoders

Chosen by measurement (stock Qwen3.5-4B and -27B, whole-model bpw, every side structure charged):

| # | encoder | structure | bpw | role |
|---|---|---|---|---|
| **E1** | **Trellis-coded** (k=2, L=12, g256) | bitshift trellis, Viterbi encode, value = deterministic pseudorandom function of state, no stored codebook | **2.062** | band A (~2 bpw, 12 GB card) |
| **E2** | **Vector quantization** (k=8192, d=8, g256) | k-means codebook over 8-dim sub-vectors, one codebook per tensor, per-group fp16 scale | **1.688** | band B (~1.7 bpw, 8 GB tight) |
| **E3** | **Vector quantization** (k=4096, d=8, g256) | same family as E2, smaller codebook | **1.562** | band C (~1.56 bpw, 8 GB headroom) |
| **E4** | **Bonsai Q2** (2-bit codes, g64, fp16 scale) | the shipped `prism-ml/Ternary-Bonsai-27B` format, ggml custom type 42 | **2.125** (`Q2_0`) / **2.250** (`Q2_g64`) | external production baseline |

**E4 is the anchor and it is not ours.** It is a real, widely downloaded production model. We include
it so that our encoders are compared against something deployed rather than only against each other.

We also carry **scalar ternary (g64, 1.835 bpw / g256, 1.647 bpw)** as the incumbent reference,
because it is what most of the sub-2-bit literature uses — though our measurements place it off the
Pareto frontier under both encoders we have tried.

## 1. What we have already measured — do not re-derive this

Stock Qwen3.5-4B, whole model quantized, 49,104 held-out tokens, fp16 baseline **ppl 4.34**:

| encoder | bpw | RTN ppl | GPTQ ppl |
|---|---|---|---|
| E1 trellis k2 | 2.062 | 11.83 | **6.53** |
| E2 vq k8192 | 1.688 | 657.80 | **37.04** |
| E3 vq k4096 | 1.562 | 1677.12 | **73.93** |
| *scalar ternary g64* | 1.835 | 2962.68 | *57.84* |

Plus: GPTQ lifts every family by a similar proportion (35–43% on proxy loss) and narrows the
trellis-to-ternary gap 28× without closing it; distance from the Shannon bound `D(R)=σ²2^(−2R)` under
RTN is **92.8% (trellis) / 80% (VQ) / 65% (ternary)**, which is the sphere-packing shaping gain
(1.53 dB, 0.254 bits/dim) showing up as measured error.

**E4 has no measurement from us yet**, and the published Bonsai model's quality reflects **30B tokens
of QAT**, which we cannot match. That is exactly why we need the method literature: to separate what
their *encoder* contributes from what their *training budget* contributes.

## 2. The core ask — PIPELINES, not single methods

**Multi-stage pipelines are in scope and are probably the answer.** Stage ordering changes outcomes:
a GPTQ-initialised QAT starts from a far better point than a randomly-rounded one, and our own
production pipeline is five stages (rotate → calibrate → GPTQ + block-wise QAT → assignment training →
end-to-end distillation). We want the best-known *sequence*, not the best-known *step*.

For each encoder E1–E4, report the best published pipeline you can find, as an ordered list:

```
stage 1: <method>   purpose   cost (tokens / GPU-hours)   format-specific or general?
stage 2: ...
...
result:  model, whole-model bpw, metric + value, and which stage contributed most
```

Pipelines we already know are worth asking about, non-exhaustively:

* incoherence rotation → Hessian PTQ → block-wise QAT (the QuIP#/QTIP shape)
* Hessian PTQ as initialisation → QAT → end-to-end distillation (our shape)
* calibration-aware init → codebook fine-tuning (PV-Tuning, for E2/E3)
* from-scratch QAT with a straight-through estimator (BitNet, for E4-like scalar grids)
* any pipeline where a later stage is reported to *undo* an earlier one — negative interactions are
  as useful to us as positive ones

## 3. Questions that decide what we build

1. **What is the marginal value of each stage?** We can afford perhaps three stages, not six. If
   rotation contributes 80% of a pipeline's gain and codebook fine-tuning 5%, we need to know that.
   Papers usually report the full pipeline; ablations are what we actually need.
2. **Does stage ordering matter, and where?** Specifically: is GPTQ-before-QAT reliably better than
   QAT alone, and by how much? Our repo has a `--qat-gptq-init` flag built on that assumption and it
   has never been ablated here.
3. **Which stages are structurally format-specific?** PV-Tuning means nothing without a codebook;
   a soft-rounding schedule means nothing without a scalar grid. We want the matrix cells that are
   *empty by construction* marked as such rather than left blank — a distinction the previous report
   we commissioned failed to make.
4. **What token budget does each stage need before it stops paying?** Bonsai used 30B. This project
   has used 2.6M–64M. Is there a published budget/quality curve for block-wise QAT? If the curve is
   flat after 100M tokens, that changes what we can claim.
5. **For E4 specifically:** what does `prism-ml` document about the Bonsai pipeline — QAT recipe,
   token count, calibration corpus, whether rotation or Hessian PTQ was used before QAT? Anything
   from model cards, technical reports, or code counts.
6. **Is there a published method that targets trellis or lattice encoders other than their own
   authors' pipelines?** We suspect not, and if so the empty column is itself a finding.

## 4. Hard constraints

* **Weight-only.** Activations stay bf16. Methods requiring low-bit activations are out of scope for
  deployment; catalogue them but say so.
* **Budget:** 2× RTX 3090 (24 GB, no NVLink), AVX-only host, no NVMe. A pipeline needing 180,000
  GPU-hours is a fact about the field, not an option for us — label it that way.
* **Whole-model bpw, always**, including embeddings, head, norms, codebooks, and any side tensors.
  Headline rates in this literature routinely cover linear projections only.
* **Say which metric you are quoting** — free-generation capability, multiple-choice benchmark,
  perplexity, or reconstruction — and prefer them in that order.

## 5. What a good answer looks like

Ordered pipelines with per-stage ablations and costs, for four named encoders. We would rather learn
that our band picks need a different pipeline than have the ones we guessed confirmed.

The single most valuable thing you can return is **a stage-level ablation table**: which stage buys
what, for which encoder, at what budget. That is what lets us spend three stages well instead of six
badly.

Do not repeat the claim that codebook or lattice decode is infeasible on consumer GPUs — `IQ2_XXS`
has shipped in llama.cpp for two years, and we have measured these encoders ourselves.
