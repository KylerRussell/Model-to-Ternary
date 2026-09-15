# Review of the (format × method) cross-product report

Screened before any of it enters `REPORT.md`. **The theory is excellent and directly explains our
measurements. Several headline numbers are wrong, and two are wrong in ways that would change the
band selections.** Treat the mechanism as citable; treat the tabulated results as unverified.

---

## 1. What is right, and genuinely valuable

### 1a. Sphere-packing explains our ranking — this is the answer to F17's open question

Scalar quantizers tile space with hypercubes; lattices and trellises tile it with sphere-like Voronoi
cells. Zador's bound puts the asymptotic **shaping gain at 1.53 dB = 0.254 bits/dimension**, which is
exactly the structural deficit we measured as scalar ternary sitting at 64.8% of the Shannon bound
against trellis's 92.8%. **Our empirical ordering is a known geometric result, not an artifact.**

### 1b. Why second-order PTQ cannot rescue scalar ternary

GPTQ compensates a rounding error by pushing it into not-yet-quantized weights,
`δw = −(w_q − w)/[H⁻¹]_qq · H⁻¹_{:,q}`. At ≥3 bpw the residual is small relative to the grid spacing
and Cholesky propagation is stable. **At ~1.8 bpw the rounding error routinely exceeds the distance
between adjacent grid points**, so compensation accumulates and drives later weights outside the
representable range. This is a mechanism, and it predicts the failure we observed rather than
describing it.

### 1c. Incoherence processing restores the Gaussian assumption — the key insight

The Hessian proxy `H = E[xxᵀ]` is ill-conditioned (κ > 10⁵) from activation outliers. A randomized
Hadamard transform `H̃ = VHVᵀ` disperses that energy, driving the incoherence parameter toward unity
and making the loss approximately isotropic:
`Tr(ΔW̃ H̃ ΔW̃ᵀ) ≈ Tr(H)/n · ‖ΔW̃‖²_F`.

**This is why the memoryless-Gaussian bound is the right reference *after* rotation, and why
lattice/trellis dominate there** — their Voronoi cells maximise spherical packing efficiency in high
dimension. It retroactively justifies F17's use of the Gaussian bound and explains why QuaRot exists.

### 1d. Correlated-source bound is strictly lower than memoryless

Reverse water-filling over the weight covariance eigenspectrum,
`R(D) = ½ Σ max(0, log₂(λᵢ/θ))`, with transformer eigenspectra decaying as `λᵢ ∝ i^-α`, α ∈ [0.8,1.4].
So `R_correlated(D) < R_memoryless(D)`: **our F17 efficiencies are upper bounds on true efficiency**,
and the real headroom is larger than we reported. This also explains why low-rank + residual methods
remain coherent at ~1.1 bpw.

### 1e. Confirmations of our own measurements

* Micro-float floors at **4.25 bpw** — matches our measured `MXFP4` exactly; structurally excluded
  from ≤2.1 bpw, as we found.
* The empty cells match our prediction: **trellis and lattice appear only with their originators'
  pipelines**, and the IQ formats — carrying most local inference worldwide — have essentially no
  academic rate-distortion literature.

## 2. Errors that would change decisions

### 2a. CRITICAL — the 27B parameter accounting is wrong, and it propagates into every band

The report claims 31.21B linear + **1.56B** embed/head for a 27B. Read from the actual
`Qwen3.5-27B` config: **23.82B body + 2.54B embed/head = 26.37B**, so embed/head is **9.64%**, not
4.76% — understated by **1.63×**.

Its derived vocabulary penalty `Δ = 0.760 bpw` is therefore too small:

| linear bpw, FP16 vocab | our whole-model | report's rule | gap |
|---|---|---|---|
| 1.58 | **2.971** | 2.340 | **+0.63** |
| 2.00 | **3.350** | 2.760 | **+0.59** |

Every band budget in the report is optimistic by ~0.6 bpw. Its Band B claim of "1.58 bpw linear +
4-bit vocab = 1.72 bpw whole-model" does not reproduce on our numbers.

### 2b. The Shannon section's formula contradicts its own number

It states `η_min = 2^(-2R)`, computes `2^-4.124 = 0.0573` for R = 2.062, then reports **92.8%**
efficiency. But `0.0573 / 0.258 = 22.2%`. The 92.8% is **our** figure, which uses relative error as an
**L2 ratio** (bound `2^-R = 0.2395`); pasted into the squared-error convention it no longer follows.
Both conventions are defensible; mixing them is not. **Fix the convention before citing either.**

### 2c. GLVQ claims are internally inconsistent and would invert the recommendations

The matrix lists **GLVQ at 2.00 bpw with WikiText-2 PPL 3.36** — better than the Band A pick (QTIP,
3.78) and near FP16 (~3.12). If true, GLVQ should *be* Band A. Instead GLVQ is recommended for Band C
at PPL 4.80, and the conflict is never reconciled. "GLVQ" is also not a work I can place in the
literature. **Do not cite until a paper is produced.**

### 2d. Minor but disqualifying for a table

* **BitNet b1.58 whole-model given as both 2.27 and 2.35 bpw** in different sections.
* **NanoQuant at 1.10 bpw / PPL 11.2** — an extraordinary claim, unverifiable, unfamiliar.
* Several RTN cells report our own measurements back to us as citations
  ("Tseng et al.; Qwen3.5-27B; PPL 2962.7"). Those are ours, unpublished; attributing them to a
  third party is a citation error we must not inherit.

## 3. The single most decision-relevant claim — and it is unsourced

> "Applying second-order post-training updates to scalar ternary improves its fraction of the Shannon
> bound from 64.8% to approximately 71.5% before diverging."

This directly answers the question the prompt was written to ask: **how much of the gap does a good
method close?** If GPTQ moves scalar ternary only 64.8% → 71.5%, it cannot approach trellis's 92.8%,
and the band picks should stay non-scalar. No citation is given.

**It is testable here, cheaply.** We already have the formats, the stock weights, and the Shannon
reference; adding a Hessian-aware (GPTQ-style) encoder and re-measuring ternary and trellis would
settle it with our own numbers instead of an unsourced one. That is the highest-value next
experiment, and it is the one that decides bands B and C.

## 4. Verdict

**Cite §1 (mechanism) freely — it is standard theory, correct, and explains our results.** Treat §2
as errata requiring correction before use. Treat every tabulated perplexity as unverified pending the
primary paper, and re-derive all whole-model bitrates from actual configs rather than the report's
rule. The report's own most important claim (§3) should be replaced by our own measurement.
