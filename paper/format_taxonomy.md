# The weight-format design space — complete taxonomy

**Why this document exists.** A first pass at "compare the families" produced three: symmetric scalar,
asymmetric scalar, vector quantization. Those are the three easiest things to implement in an
afternoon, not the three families that exist — a taxonomy chosen by implementation convenience, which
would have published a survey of our own tooling. This enumerates the space from structural axes
first, then records what we can measure, so that omissions are *declared* rather than silently
inherited.

## Generative axes

Any weight format is a choice on six independent axes. The families below are the populated cells.

| axis | options |
|---|---|
| **1. level geometry** | uniform spacing · non-uniform (companded/quantile) · floating-point |
| **2. quantization unit** | scalar (1 weight) · vector (d weights jointly) |
| **3. decomposition** | single term · additive (multi-plane / low-rank / sparse-hybrid) |
| **4. code length** | fixed-stride · variable-length (entropy-coded) |
| **5. decode dependency** | independent per unit · sequential (trellis / Viterbi) |
| **6. bit allocation** | uniform across tensor · mixed per channel/layer |

## The families

| # | family | defining structure | real instances | measurable here | shipping kernel |
|---|---|---|---|---|---|
| **A** | uniform scalar, symmetric | `x = s·q`, q ∈ {−L..L} | `Q4_0`, `Q8_0`, `TQ1_0`, `TQ2_0`, TQ1_64, BitNet | **yes** | yes |
| **B** | uniform scalar, asymmetric | `x = s·q + z` | `Q2_K`…`Q6_K`, GPTQ, AWQ, EXL2, MLX | **yes** | yes |
| **C** | non-uniform scalar | fixed companded level set | **`IQ4_NL`**, **NF4** (QLoRA) | **yes** | yes |
| **D** | micro-float / block FP | exponent+mantissa per weight, shared block exponent | **`MXFP4`**, **`NVFP4`**, FP8, OCP MX | **yes** | yes |
| **E** | vector quantization | k centroids over d-dim sub-vectors | `IQ1_S/M`, `IQ2_*`, AQLM, VPTQ | **yes** | yes |
| **F** | lattice quantization | structured, algorithmically decodable | **QuIP# (E₈)**, Barnes-Wall | **yes** | yes (bespoke) |
| **G** | trellis-coded | sequential dependency, Viterbi decode | **QTIP** | *hard* | bespoke |
| **H** | additive multi-plane | `W ≈ α₁T₁ + α₂T₂` | PTQTP, DB-LLM, BiLLM, E2M-ATQ | **yes** | no |
| **I** | low-rank + quantized residual | `W ≈ AB + Q` | OneBit (SVID), ExTernD, Recover-LoRA | **yes** | no |
| **J** | sparse–dense hybrid | low-bit dense + fp16 outlier set | SpQR, SqueezeLLM, PB-LLM, FlashQuant | **yes** | research only |
| **K** | entropy-coded | variable-length codes over quantized symbols | SoftWater, ECASQ | **yes (rate only)** | **none** |
| **L** | mixed-precision | per-channel/layer bit allocation | EXL2, PTQ1.61, AWSRC | **yes** | yes (EXL2) |
| **M** | weight sharing / hashing | codes shared across tensors | HashedNets (pre-LLM) | yes | no |

Three were measured in the first pass: **A, B, E**. Ten were not.

## Two structural notes that change how the results must be read

### K is a pure rate transform, not a distortion trade

Entropy coding does not change the reconstruction at all — it re-encodes the *same* symbols in fewer
bits. On a rate–distortion plot every family therefore has an entropy-coded twin **directly to its
left at identical error**, at the empirical symbol entropy `H(q)` rather than `log₂(levels)`.

Consequence: **the rate–distortion frontier and the deployable frontier are different frontiers.**
Family K dominates its uncoded parent on pure R-D and is excluded by deployability alone — no
production runtime implements variable-length decode, for the structural reason that symbol *k+1*'s
offset depends on decoding symbol *k*. The gap between the two frontiers is precisely what the
deployment constraint costs, and quantifying it is a result in its own right rather than a caveat.

### Families H, I, J, L are not "formats" in the same sense

A–G specify how a *single* tensor is stored. H–L add structure *on top of* a base format: a second
plane, a low-rank term, an outlier set, a bit-allocation policy. They must be measured as
**modifiers**, applied to a base family, or the comparison is category-confused. This is the error
made in the earlier format census, which read `IQ2_XXS`'s magnitude+sign planes as multi-plane
superposition — a sign plane is not an independent additive term with its own scale.

## Measurement plan

**Measure at family ceiling, not at one implementation.** Codebooks fitted by k-means to the actual
weight distribution; lattices decoded exactly; multi-plane fitted by alternating least squares. A
method requiring a family should be screened against what the family can do, not against llama.cpp's
particular grid.

**Stock weights only.** Our checkpoints are untied and QuaRot-rotated, and rotation reshapes exactly
the local variance that fine-grained scales exist to capture.

**Report the bpw honestly, whole-tensor**, including every side structure a family requires —
codebooks, outlier indices, low-rank factors, per-channel bit maps. A family that hides its overhead
in a side tensor is the accounting failure this paper documents elsewhere.

### Declared omissions

* **G (trellis)** — Viterbi decode over a learned trellis is a substantial build. Screened on paper,
  not measured. Its absence is a hole in the sub-2-bit frontier, since QTIP reports strong results.
* **M (weight sharing)** — no modern LLM-PTQ instance; excluded on relevance, not difficulty.
* Anything requiring activation-domain quantization is out of scope by the weight-only constraint,
  which is a property of our runtime and is declared as such rather than presented as a limit of the
  method.
