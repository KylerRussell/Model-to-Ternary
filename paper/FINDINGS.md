# Findings register — paper testing only

**Scope: findings produced by the paper's own testing of methodologies and formats.** Nothing from
the project's prior experimental history belongs here — that lives in `logs/RESULTS_SUMMARY.md` and
is *referenced* where a paper finding depends on it, never copied. If a line here cannot be traced to
a test we ran in service of the paper, it should be deleted.

Status: **HOLDS** — measured, survived challenge · **PROVISIONAL** — measured once · **OPEN** — test
designed or running.

---

## F1. ggml block sizes verified against the installed package · HOLDS

Read from `gguf.GGML_QUANT_SIZES` rather than taken from the census:

| type | block / bytes | bpw |
|---|---|---|
| `Q1_0` | 128 / 18 | 1.1250 |
| **`IQ1_S`** | 256 / 50 | **1.5625** |
| `TQ1_0` | 256 / 54 | 1.6875 |
| `IQ1_M` | 256 / **56** | 1.7500 |
| *TQ1_64 (ours)* | *512 / 114* | *1.7812* |
| `TQ2_0`, `IQ2_XXS` | 256 / 66 | 2.0625 |
| `IQ2_XS` | 256 / 74 | 2.3125 |
| `IQ2_S` | 256 / 82 | 2.5625 |
| `Q2_K` | 256 / 84 | 2.6250 |

**8 of 9 census claims exact.** The one error (`IQ1_M` given as 64 B) is internally inconsistent
rather than wrong in conclusion — 56 B *is* 1.75 bpw.

*Method: direct package enumeration, 2026-09-14. Establishes the format census as citable, unlike the
method census (F7).*

## F2. Codebooks and per-block zero-points are deployable on commodity GPUs · HOLDS

The method census justified most FAIL verdicts with hardware claims — vector quantization "breaks
down… due to indirect memory dereferencing and cache thrashing", asymmetric offsets "destroy the
addition-only GEMV pipeline". Both are falsified by shipping code: llama.cpp ships `IQ1_S`, `IQ1_M`,
`IQ2_XXS`, `IQ2_XS`, `IQ2_S` (codebook lookups) and `Q2_K` (scales **and** mins) with CUDA kernels.

The zero-point argument fails algebraically, not just empirically:

    Σ(s·qᵢ − m)·yᵢ = s·Σqᵢyᵢ − m·Σyᵢ

At batch 1 the activation sums are computed once per token and reused across rows, so an offset costs
**one scalar MAC per block**, not a per-weight subtraction.

**Consequence for the paper:** verdicts of the form "method class X is undeployable" are, on present
evidence, facts about TQ1_64 rather than about hardware. The claim must be stated against a format
class, with the class named.

## F3. IQ1_S is smaller and finer-grained than TQ1_64 — and loses on reconstruction · PROVISIONAL

| | bpw | scale granularity | alphabet |
|---|---|---|---|
| **TQ1_64 (ours)** | 1.7812 | g64, 8-bit sub-scale | full ternary |
| `TQ1_0` | 1.6875 | g256 | full ternary |
| **`IQ1_S`** | **1.5625** | **g32**, 3-bit sub-scale + fp16 super | **2048 of 6561** 8-D vectors |

`IQ1_S` is 12.3% smaller with finer granularity, so the census predicted it would win. **Measured, it
loses** — matched tensor sets, same weights, same objective, RTN all arms:

| mean relative reconstruction error | rotated | unrotated |
|---|---|---|
| `TQ1_0` | 0.43800 | 0.44160 |
| **TQ1_64** | **0.43228** (3/3) | **0.43527** (3/3) |
| `IQ1_S` | 0.44631 | 0.44826 |

Ordering is identical in both regimes: **TQ1_64 < TQ1_0 < IQ1_S.**

**What this does and does not establish.** It does *not* show TQ1_64 is better end-to-end —
reconstruction error is the metric this project has documented as never once catching a real failure
(block-MSE improved 99.8% on a collapsed residual stream). Claiming victory on it would be the exact
error we criticise in others. What it does establish is narrower and still useful: **the census's
claim that IQ1_S delivers "superior perplexity recovery" is unsupported**, it attached no
measurement, and the one measurement that now exists points the other way.

**A caveat that cuts against us.** TQ1_64 beats TQ1_0 by **1.3% relative error for 5.6% more bits**.
Per bit, that is not a clear win. The real case for g64 rests on the end-to-end agreement measurement
(82.06% vs 79.31%, `TQ1_64_SPEC.md`), not on reconstruction — and that measurement has no IQ1_S
counterpart yet.

**Lower bound for IQ1_S, not a verdict.** All arms are unweighted RTN; IQ1_S is designed around
`imatrix` importance weighting and is understated without it. F3a is the part that survives weighting.

*Test: `src/format_sim.py`, `src/fmt_recon.py`, `experiments/sweep/fmt_matched.sh`.*

## F3a. IQ1_S cannot represent exact zero · HOLDS

Reconstruction is `x = dl·(g + δ)` with `δ = ±0.125`, so a grid **zero** dequantizes to `±0.125·dl`,
never 0. Our ternary weights are **~45.6% exact zeros**, so nearly half of all weights carry a
systematic error the format cannot avoid at any scale setting.

This is an alphabet-level property, independent of search quality or importance weighting, and it is
the structural explanation for F3's ordering.

## F3b. RETRACTED — "rotation flips the IQ1_S/TQ1_0 ordering"

I hypothesised that QuaRot, by removing channel outliers, would remove the local variance that fine
scales exist to capture, and so make coarse-scale ternary artificially competitive. An initial run
appeared to confirm it: IQ1_S beat TQ1_0 unrotated (0.45673 vs 0.46140) and lost rotated.

**Matched tensor sets falsify it.** The ordering is identical in both regimes. The apparent flip was a
sampling artifact — the unrotated draw was 3/4 tiny `(32, 2560)` projections while the rotated draw
included a `9216×2560`. *A format comparison across different tensors measures the tensors.*
`fmt_recon.py` now takes an explicit tensor list so both arms are scored on the same weights.

## F3c. IQ1_M STRICTLY DOMINATES TQ1_64 — smaller *and* lower error · HOLDS

The IQ1_S comparison asked the wrong question. Comparing error without holding bits fixed is
meaningless, and correcting for that changes the answer completely.

**First: the three scalar-ternary-ish formats sit on one rate-distortion line.** Fitting error
against bpw across IQ1_S / TQ1_0 / TQ1_64 gives `err = -0.0643·bpw + 0.5467` with **r = -0.99972**
and residuals of ±0.0002. None of them dominates any other — each buys error reduction at ~0.064 per
bit. **TQ1_64 is not better than TQ1_0 or IQ1_S; it is further along the same curve.**

**Then `IQ1_M` breaks the line.** At **1.75 bpw** it is the matched-rate comparison for our 1.7812:

| format | bpw | err (rotated) | err (unrotated) | residual vs line |
|---|---|---|---|---|
| `IQ1_S` | 1.5625 | 0.44631 | 0.44826 | +0.00008 |
| `TQ1_0` | 1.6875 | 0.43800 | 0.44160 | −0.00019 |
| **`IQ1_M`** | **1.7500** | **0.41350** (3/3) | **0.41483** (3/3) | **−0.02068** |
| TQ1_64 (ours) | 1.7812 | 0.43228 | 0.43527 | +0.00011 |

**IQ1_M is 1.8% smaller AND 4.3% lower error than TQ1_64**, in both rotation regimes, on matched
tensors. That is strict domination — not a trade. Its residual from the line is **100× the other
three's**, so it is on a genuinely better rate-distortion curve rather than further along the same one.

**Why.** IQ1_M carries g16 3-bit sub-scales (against our g64) *and* a delta sign per 8 weights — a
one-bit offset at g8 granularity. Our format has g64 scales and no offset at all. At ~1.75 bpw,
spending bits on fine scales plus a per-group sign beats spending them on a full ternary alphabet
with coarse scales.

**Strength of the conclusion.** Reconstruction error is a metric this project distrusts (F-series
preamble), but **strict domination needs much less from the metric than a margin does**: when one
option is better on both size and error, the conclusion survives unless the metric is
*anti*-correlated with quality, not merely noisy. And the comparison is unweighted RTN, which
*understates* IQ1_M — it is the arm designed around `imatrix` weighting.

**Consequence: TQ1_64 is not defensible on present evidence.** The paper cannot claim it as a
considered choice without either an end-to-end result overturning this, or a statement that we chose
it before measuring. `IQ1_M` also ships in stock llama.cpp, while TQ1_64 lives in our fork
(`TQ1_64_SPEC.md` still lists `LLAMA_FTYPE_MOSTLY_TQ1_64` as incomplete) — so it wins on
deployability too.

*Verified: `verify_iq1m()` packs our parameters into real 56-byte blocks — including the fp16 super
split across four scale-word top nibbles — and matches gguf's dequantizer exactly (max |Δ| = 0).*

## F3d. Offset capability, not alphabet or granularity, sets position on the R-D line · HOLDS

Seven formats measured on matched tensors. Residual is against `err = -0.0643·bpw + 0.5467`, fitted
through the offset-free scalar formats (r = -0.99972, residuals ±0.0002):

| format | bpw | offset | scale gran. | err (rot) | residual | Pareto |
|---|---|---|---|---|---|---|
| `Q1_0` | 1.1250 | none | g128 | 0.61838 | **+0.1440** | frontier |
| `IQ1_S` | 1.5625 | 1-bit / g32 | g32+super | 0.44631 | +0.0001 | frontier |
| `TQ1_0` | 1.6875 | none | g256 | 0.43800 | −0.0002 | frontier |
| **`IQ1_M`** | 1.7500 | 1-bit / **g8** | g16+super | **0.41350** | **−0.0207** | frontier |
| **TQ1_64** | 1.7812 | none | g64+super | 0.43228 | +0.0001 | **DOMINATED** |
| `TQ2_0` | 2.0625 | none | g256 | 0.43800 | +0.0239 | **DOMINATED** |
| `Q2_K` | 2.6250 | **4-bit / g16** | g16+super | **0.32970** | **−0.0482** | frontier |

**Of seven formats, exactly two are Pareto-dominated — and one is ours.** Ordering is identical on
the unrotated checkpoint.

**The mechanism is offset granularity, and it is monotone:**

| offset | residual |
|---|---|
| none (`TQ1_0`, TQ1_64, `TQ2_0`) | ≈ 0 |
| 1 bit per 32 weights (`IQ1_S`) | ≈ 0 — too coarse to help |
| 1 bit per 8 weights (`IQ1_M`) | −0.021 |
| 4 bits per 16 weights (`Q2_K`) | −0.048 |

Scale granularity alone does **not** move a format off the line: `TQ1_0` (g256) and TQ1_64 (g64) sit
on the *same* line and differ only in position along it. **Our "no zero-point" format axiom — which
disqualifies a large share of published methods in the screen — is the most expensive constraint in
the design space, and F2 established it is a choice rather than a hardware limit.**

Two corollaries: `TQ2_0` is the same alphabet and scale as `TQ1_0`, so its reconstruction is identical
by construction at 22% more bits — it is a throughput format, never a compression point. And `Q1_0`
(binary, +0.144, the largest residual measured) has no zero state and cannot express the ~46% of
weights that round to zero, so the third ternary symbol is worth far more than the 0.43 bpw it costs.

*Draft section: `paper/sec_format_choice.md`.*

## F11. RETRACTED and replaced — the tensor sample drove the conclusion

**F11 originally claimed "no scalar-ternary configuration is Pareto-optimal at any bit rate."** That
is false. It was measured on three `(32, 2560)` `in_proj_a` tensors — a strided slice of
alphabetically sorted names, which is the *same* sampling error already caught once in the earlier
format comparison and repeated here in a new place. Re-run with a **stratified sample by tensor kind**
(`down_proj`, `gate_proj`, `up_proj` — where the parameters actually live), two families change
frontier status.

### F11a. Corrected frontier, stratified MLP tensors, stock Qwen3.5-4B · HOLDS

| bpw | family / config | rel_err |
|---|---|---|
| 1.062 | G trellis k1 L10 g256 | 0.52779 |
| 1.438 | E vq k2048 d8 g256 | 0.45611 |
| 1.625 | E vq k2048 d8 g64 | 0.45290 |
| **1.647** | **A sym ternary g256** | **0.43894** |
| **1.710** | **A sym ternary g128** | **0.43663** |
| **1.835** | **A sym ternary g64** | **0.43256** |
| 1.963 | I lowrank r16 tern g64 | 0.42541 |
| 1.995 | J sparse 0.5% tern g64 | 0.40783 |
| 2.062 | E vq k256 d4 g256 | 0.31934 |
| **2.062** | **G trellis k2 L12 g256** | **0.25798** |
| 3.062 | G trellis k3 L10 g256 | 0.13753 |
| 4.125 | C nonunif NF4 g128 | 0.09034 |
| 5.000 | B asym int4 g32 | 0.07593 |

**Ternary IS on the frontier**, at 1.647–1.835 bpw. The original claim was an artifact.

**But the frontier has a sharp knee at ~2.06 bpw.** Trellis k2 gives **0.258 against ternary g64's
0.433 — a 40% error reduction for 12% more bits.** The right question is therefore not "is ternary
Pareto-optimal" (it is, narrowly) but "is 0.23 bpw worth 40% of the error", and on this evidence it
plainly is. Families reaching the frontier: **A(7), C(5), E(4), G(4), B(1), I(1), J(1)**. Never
reaching it: **D (micro-float), F (lattice), H (multi-plane)**.

### F11b. Lattice rate is data-dependent, and a small sample understates it · HOLDS

`F lattice E8` was 2 frontier points on the tiny tensors and is **absent** from the corrected
frontier. Its error *improved* on real tensors (−0.016 to −0.034) while its **measured rate rose from
1.65–1.88 to 2.25–2.55 bpw**, pushing it out of the sub-2 band entirely.

The cause is the honest-rate accounting working as intended: rate is `log₂(distinct lattice points
used)/8`, and a `(32, 2560)` tensor exercises far fewer points than a `(9216, 2560)` one. **Any family
whose codebook adapts to the data has a data-dependent rate, and measuring it on unrepresentative
tensors understates it.** Had the rate been quoted nominally, this error would have been invisible.

### F11c. The sampling error changed two families' verdicts · HOLDS (method)

Three `(32, 2560)` tensors versus three real MLP tensors flipped **ternary** (never-on-frontier →
on-frontier) and **lattice** (frontier-owner → never-on-frontier). Neither direction was predictable
from the other sample.

Third instance of the same class in this project: family-bounded search (F3b), convenience-chosen
taxonomy (F12), and now convenience-chosen tensors. **The generalisation: whenever a choice is made
by what is nearest to hand rather than by what the question requires, it decides the answer.** For the
format work specifically, the rule is now: stratify by tensor kind, never by name order, and never
score a compression family on tensors small enough for its side structures to be degenerate — the
low-rank arm at rank 32 on `min_dim` 32 was reconstructing an identity and reporting it as a result.

## F13. The format frontier is ROLE-INVARIANT · HOLDS

F11c showed the tensor sample decides the answer, so the sweep was re-run split by tensor role rather
than pooled. Sub-2.3 bpw, stock Qwen3.5-4B:

| rank | MLP | attention | embedding |
|---|---|---|---|
| 1 | **G trellis k2 L12** 0.25798 | **G trellis k2 L12** 0.25886 | **G trellis k2 L12** 0.25872 |
| 2 | G trellis k2 L10 0.26441 | G trellis k2 L10 0.26551 | G trellis k2 L10 0.26548 |
| 3 | E vq k256 d4 0.31934 | E vq k256 d4 0.32173 | E vq k256 d4 0.32218 |
| 4 | F lattice E8 0.33679 | F lattice E8 0.34534 | F lattice E8 0.35649 |
| 5 | H plane x2 binary 0.34786 | H plane x2 binary 0.35180 | H plane x2 binary 0.35662 |

**The ordering is identical in all three roles and the values agree to within 1–5%.** Frontier family
membership is identical too — A(7), B(1), C(5), E(4), G(4), I(1), J(1) — with embedding differing only
by dropping J.

**Mechanism, checked rather than assumed.** Identical rankings across roles are suspicious, so the
distributions were measured: excess kurtosis **0.803** (mlp.down_proj), **0.845** (attn.in_proj_qkv),
**1.335** (attn.out_proj), **0.647** (embed_tokens). All four roles are mildly heavy-tailed and close
to Gaussian. The frontier is role-invariant **because the source is effectively the same**, not
because format choice is magically independent of what it encodes.

**Consequences.** Applying one format uniformly across the model is justified — per-role format
selection buys essentially nothing, which is worth stating because it is the implicit assumption in
every published sub-2-bit method and had never been checked here. It also yields a falsifiable
prediction: a model whose roles have *genuinely* different distributions (extreme outlier channels, a
MoE router, a quantization-hostile embedding) should break role-invariance. Our 4B does not, so the
claim is scoped to near-Gaussian weight distributions and must be re-checked before transfer.

## F14. Mixed-precision (family L) never reaches the frontier · HOLDS

Per-channel bit allocation, with sensitivity **measured** rather than proxied — quantize at the low
rate, rank channels by the error they actually incur, promote the worst fraction, which is
greedy-optimal for this objective and is what EXL2 does in spirit. Rate charges the allocation map
(1 bit/channel, amortised), because omitting it is the same error as sparse-hybrid omitting indices.

Result: **0.373 (MLP) / 0.379 (attn) / 0.384 (embed) at 2.189 bpw**, against trellis's **0.258 at
2.062 bpw** — dominated on both axes in every role. Mixed precision spends bits moving channels
between two coarse grids; trellis spends the same bits buying a *continuous* effective codebook.

Together with F13 this closes the family sweep at **ten families measured, three never reaching the
frontier in any role: D (micro-float), H (multi-plane), L (mixed-precision)**, plus F (lattice) which
reaches it only on MLP.

## F15. Weight reconstruction error predicts perplexity EXPONENTIALLY (r = 0.996) · HOLDS

Six formats quantized end-to-end on the stock 4B (every quantizable 2D tensor replaced by its
dequantized form, RTN, no recovery), evaluated on 49,104 held-out tokens:

| bpw | format | recon err | perplexity |
|---|---|---|---|
| 16.000 | fp16 | — | **4.34** |
| 3.062 | int3 g256 | 0.220 | **7.06** |
| **2.062** | **trellis k2** | **0.258** | **11.83** |
| 1.835 | ternary g64 | 0.435 | 2962.68 |
| 1.688 | vq k8192 | 0.388 | 657.80 |
| 1.562 | vq k4096 | 0.420 | 1677.12 |
| 1.062 | trellis k1 | 0.528 | 20060.11 |

    log(ppl) = 27.31 · recon_err − 4.20        r = 0.99610

**A 0.01 absolute change in reconstruction error multiplies perplexity by 1.31×.**

### This forces a correction to how we have read our own history

The project's standing position (`RESULTS_SUMMARY` §13 batch, F-series preamble) is that local
reconstruction metrics "never once caught a real failure". Reconstruction error is in fact an
excellent predictor — r = 0.996 across formats spanning 1.06–3.06 bpw and four orders of magnitude of
perplexity. Two things reconcile that, and both are needed:

1. **The mapping is exponential, and we were comparing inside the destroyed regime.** Every ternary
   variant this project ever compared sits at recon 0.43–0.46, where perplexity is already 2000–5000.
   Differences there are real and predictive — and irrelevant, because the model is gone either way.
   Reconstruction was answering "still broken?" correctly every time.
2. **The metric we distrusted was not this one.** §13's block-MSE is *hidden-state* MSE per block
   during block-AP, a different quantity from final *weight* reconstruction error. The old finding
   stands for block-MSE; it does not license distrusting weight reconstruction, and we generalised
   it too far.

### Consequence

The 57-config family sweep can now be read as *predicted perplexity*, not just relative error, which
is what makes a reconstruction-only sweep worth running at all. The practical rule: **~0.30 recon
error is the usable boundary** under RTN (0.258 → ppl 11.8; 0.388 → ppl 658), and no sub-2-bpw format
measured reaches it.

**Limits.** Six points, one model, RTN only, and perplexity is itself a screen for capability rather
than a measure of it. The 0.258–0.388 interval is unsampled, so the boundary's *position* is a range,
not a number. The slope is fitted on formats from four different families, which is what makes it
interesting — but it has not been checked at another scale, and F16 is that check.

## F16. The format frontier is SCALE-invariant (4B -> 27B) · HOLDS

The same 57 configurations on stock Qwen3.5-4B and Qwen3.5-27B MLP tensors:

| bpw | config | 4B err | 27B err | delta |
|---|---|---|---|---|
| 2.062 | trellis k2 L12 | 0.25798 | 0.25801 | **+0.00003** |
| 2.062 | trellis k2 L10 | 0.26441 | 0.26448 | +0.00007 |
| 2.062 | vq k256 d4 | 0.31934 | 0.31973 | +0.00040 |
| 1.835 | ternary g64 | 0.43256 | 0.43363 | +0.00107 |
| 2.252 | lattice E8 g256 | 0.33679 | 0.33940 | +0.00261 |

All deltas are **+0.00003 to +0.009**, mostly under +0.002, and rank order is unchanged. **A format
decision made at 4B transfers to 27B.**

This is the *opposite* of this project's experience with METHODS, where the 4B is a
known-catastrophic testbed (-62% relative MMLU against the 27B's -18%, §13am-i). Formats and methods
do not share that scale sensitivity, and conflating them would have led us to distrust a
transferable measurement.

**Role-invariance also survives at 27B**: MLP and embedding give identical rankings, and excess
kurtosis is 0.18-1.24 across all roles at both scales — every role remains mildly heavy-tailed and
close to Gaussian, which is the mechanism F13 identified.

**Limit that matters.** This is reconstruction-only. It does NOT establish that *end-to-end*
behaviour is scale-invariant: the 27B has 64 layers against the 4B's 32, so error compounds through
twice the depth and the F15 slope could steepen. The 27B end-to-end run needs all 11 shards and has
not been done.

## F17. Distance from the Shannon bound explains the ranking · HOLDS

For a memoryless Gaussian source, `D(R) = sigma^2 * 2^(-2R)`, so the minimum achievable RELATIVE
error at rate R is `2^-R`. Weights are near-Gaussian here (excess kurtosis 0.18-1.24 across roles and
scales, F13/F16), making this the right reference. Measured against it, under RTN:

| format | bpw | measured | bound | % of optimal |
|---|---|---|---|---|
| **trellis k2 L12** | 2.062 | 0.25798 | 0.23948 | **92.8%** |
| trellis k1 L10 | 1.062 | 0.52779 | 0.47897 | 90.7% |
| VQ k4096 d8 | 1.562 | 0.42082 | 0.33868 | 80.5% |
| VQ k8192 d8 | 1.688 | 0.38767 | 0.31036 | 80.1% |
| VQ k256 d4 | 2.062 | 0.31934 | 0.23948 | 75.0% |
| ternary g256 | 1.647 | 0.43894 | 0.31930 | 72.7% |
| **ternary g64** | 1.835 | 0.43256 | 0.28029 | **64.8%** |
| lattice E8 g256 | 2.252 | 0.33679 | 0.20993 | 62.3% |
| int3 g256 | 3.062 | 0.21751 | 0.11974 | 55.1% |

**This explains F11/F13's ranking rather than merely restating it.** Trellis wins because
trellis-coded quantization is *designed* to approach the rate-distortion bound — Viterbi over a
long-constraint trellis is a near-optimal encoder — and scalar ternary's ~65% is the classic granular
gap of scalar quantization. The ordering is not an empirical accident; it is the theory.

**The load-bearing caveat.** A format's gap to the bound under RTN conflates two things: structural
inefficiency (irreducible for that representation) and *encoder* inefficiency (RTN is a poor encoder).
For trellis the two are nearly the same, since Viterbi is already optimal for it — 92.8% is close to
that format's ceiling. For int3 at 55.1% and ternary at 64.8%, most of the gap is plausibly encoder
inefficiency that GPTQ-class methods could close.

**Therefore RTN cannot settle the band choices.** It measures formats at *our* encoder's competence,
and the families furthest from the bound are exactly those with the most to gain from a better one.
Driving prompt for the literature: `research_prompts/format_method_crossproduct_prompt.md`.

## F18. Hessian-aware encoding helps every family ~equally; it does NOT close the structural gap · HOLDS

Answers the cross-product report's unsourced claim (that second-order PTQ moves scalar ternary from
64.8% to ~71.5% of the Shannon bound) with our own measurement. Real calibration Hessians
`H = E[xx^T]` from 16 sequences, block-wise GPTQ generalised so a format encodes a column block
jointly (reduces exactly to textbook GPTQ at block size 1). Mean over 3 stratified MLP tensors:

| format | bpw | recon RTN | recon GPTQ | **proxy RTN** | **proxy GPTQ** | proxy gain |
|---|---|---|---|---|---|---|
| ternary g64 | 1.835 | 0.4326 | 0.6296 | 0.3760 | 0.2153 | **42.7%** |
| ternary g256 | 1.647 | 0.4389 | 0.6304 | 0.3907 | 0.2522 | 35.4% |
| vq k4096 d8 | 1.562 | 0.4208 | 0.5683 | 0.3551 | 0.2131 | 40.0% |
| **trellis k2** | 2.062 | 0.2655 | 0.3874 | **0.2221** | **0.1362** | 38.7% |

**GPTQ raises plain reconstruction error while cutting proxy loss by 35–43%.** That is not a bug: GPTQ
minimises `Tr(dW H dW^T)`, the second-order term of the task loss, *not* `||dW||_F`. It deliberately
accepts larger weight error where the activations do not care. Two consequences:

* **Shannon efficiency is not defined for a GPTQ-encoded weight.** The bound is a statement about
  reconstruction error; GPTQ optimises a different objective. The report's "64.8% -> 71.5%" compares
  quantities that are not comparable, and our own F17 efficiencies apply to RTN only.
* **The report's conclusion survives its broken framing.** On the metric GPTQ actually optimises, the
  ternary-to-trellis gap narrows only from **1.69x to 1.58x**. Every family gains ~35–43%; the
  ordering is untouched.

**The cleanest way to state it:** ternary+GPTQ (proxy 0.2153) lands almost exactly on trellis+RTN
(0.2221). **A good encoder buys scalar ternary roughly what the better format gave away for free** —
and trellis+GPTQ then moves on to 0.1362, so the format advantage compounds rather than being
absorbed. This is the sphere-packing deficit behaving as theory predicts: it is structural, and
encoding cannot recover it.

**Decision consequence:** bands B and C stay non-scalar. The result that would have overturned them —
GPTQ closing most of ternary's gap — did not occur.

**Limits.** Proxy loss is a better predictor than reconstruction but is still not end-to-end; the
verdict needs perplexity with GPTQ applied model-wide, which is not yet run. One-shot Hessians from
the FP model, not sequentially propagated as production GPTQ does. Three tensors, one model.

## F19. End-to-end with GPTQ: trellis at 2.06 bpw reaches ppl 6.53 · HOLDS

Model-wide sequential GPTQ (Hessians captured from the partially quantized model, so each layer
compensates for its predecessors; embeddings RTN in both arms since a lookup has no input covariance):

| format | bpw | RTN ppl | **GPTQ ppl** | gain | vs fp16 (4.34) |
|---|---|---|---|---|---|
| **trellis k2** | 2.062 | 11.83 | **6.53** | 1.8x | **1.51x** |
| ternary g64 | 1.835 | 2962.68 | **57.84** | **51.2x** | 13.34x |

**Headline: trellis + GPTQ at 2.062 bpw beats int3 + RTN at 3.062 bpw (6.53 vs 7.06)** — 33% fewer
bits *and* better quality. At 1.51x fp16 perplexity this is a genuinely deployable model, not a
survivor.

**GPTQ narrows the format gap 28x — from 250x to 8.9x — and does not close it.** Both formats moved
the same direction and by similar proportions in reconstruction error (+39% trellis, +42% ternary),
so the encoder is not advantaging either; the residual 8.9x is the structural sphere-packing deficit
(F17), exactly as theory predicts.

**Correction to my own prediction, stated in advance.** I predicted from F18's proxy-loss gap (1.58x)
that "trellis should retain roughly its 1.5x advantage". The actual perplexity gap is **8.9x** —
direction right, magnitude badly wrong. Proxy loss is a squared-error surrogate and perplexity is
exponential in error, so a 1.58x proxy gap maps to a far larger perplexity gap. **Proxy loss ranks
formats; it does not size the difference between them.**

### F19a. RETRACTION — the sub-2-bpw regime is not "unusable"

`REPORT.md` §5.3 stated that under RTN every sub-2-bpw format lands on the destroyed side, and called
bands B and C "starting points for a recovery pipeline, not deployable configurations". Ternary at
1.835 bpw with **GPTQ alone** — no QAT, no distillation, no assignment training — reaches **57.84**,
four orders of magnitude from "destroyed". The RTN-only framing understated what a post-training
encoder achieves by itself. Corrected in the report.

### F19b. SCOPING — F15 is encoder-specific, not a law

`log(ppl) = 27.31·recon − 4.20` (r = 0.996) was fitted on RTN points, where reconstruction error is
the quantity being minimised. Applied to GPTQ weights it **over-predicts by 4,794x** (ternary GPTQ:
predicts 277,295, actual 57.84), because GPTQ deliberately buys lower activation-weighted error with
*higher* weight error.

**Correct statement: reconstruction error predicts perplexity WITHIN a fixed encoder. Comparisons
across encoders must use the encoder's own objective.** This is the second finding this session I
recorded too broadly — the first being "reconstruction metrics never predict anything". Both errors
have the same shape: a relationship measured under one condition, written down as general.

**Limits.** One model, one calibration set (16 sequences), perplexity not capability. VQ — the band B
and C picks — has no end-to-end GPTQ number yet, which is now the gap that matters most.

## F12. The first taxonomy was chosen by implementation convenience · HOLDS (method)

The first family sweep contained exactly three families — symmetric scalar, asymmetric scalar, vector
quantization — which are the three things implementable in an afternoon. Enumerating the space from
structural axes instead (level geometry · quantization unit · decomposition · code length · decode
dependency · bit allocation) yields **thirteen** populated families.

The omitted ten included **lattice quantization, which turns out to own the 1.7–1.9 bpw frontier**,
and **non-uniform scalar, which wins the 4-bit band** — so the omission was not cosmetic; it excluded
both winners. A taxonomy assembled from what tooling makes easy will reliably conclude that the
easy thing is best.

*Recorded as a method finding because it is the same failure as F3b's family-bounded search, one
level up: there, searching within a family; here, choosing the families by convenience.*

## F4. Our IQ1_S simulation is faithful, not an approximation · HOLDS

`verify_iq1s()` packs our chosen `(d, s, δ, grid index)` into real 50-byte IQ1_S blocks and
dequantizes them with **gguf's own dequantizer**. Match is **exact** (max |Δ| = 0 over 64 blocks).

Two fidelity bugs were caught by that check rather than by inspection:

* the grid is stored `(1, 1, 2048, 8)`, not `(2048, 8)`;
* **the format stores the super-scale as fp16 and the simulation used fp32** — which would have
  credited IQ1_S with precision the format does not have. Residual was 1.59e-05 until fixed.

*Without this check the comparison would have measured something that merely resembles IQ1_S.*

## F5. The format comparison must not be rigged, in either direction · HOLDS (design)

Fairness constraints written into `format_sim.py`:

* every arm gets the same weights, the same optional importance weighting, and a genuine search over
  its own format's free parameters — a 24-point scale sweep for the ternary arms, a joint
  (super-scale × 3-bit sub-scale × δ sign × 2048 grid entries) search for IQ1_S;
* **all arms are RTN.** Our deployed TQ1_64 model is assignment-trained and IQ1_S has no trained
  counterpart here; scoring a trained arm against an RTN arm is the mismatched-baseline error we
  criticise in others. This measures *format capacity*. A format that wins at RTN is the one worth
  building a trainer for.

**Known residual risk:** IQ1_S quality in llama.cpp depends on `imatrix` importance weighting
(`H_ii ≈ Σ X_ik²`). An unweighted comparison understates it. The first result is therefore a
*matched-unweighted lower bound for both arms*, not a final verdict.

## F6. Variable-length entropy coding fails under every shipping runtime · HOLDS

Structural rather than incidental: symbol *k+1*'s bit offset depends on decoding symbol *k*, so
parallel strided reads require multi-pass prefix sums. No production engine implements it.

**This is the only FAIL class that is a property of parallel hardware rather than of a format
choice** — and therefore the only one the paper may state unconditionally.

*Source: format census, consistent across every runtime it surveyed.*

## F7. The method census is not citable at row level · HOLDS

Two of **our own internal acronyms** were bound to unrelated published papers, both marked FAIL while
shipping in our pipeline:

| our name | what it is here | census claim |
|---|---|---|
| `CAKLD` | Confidence-Aware KL Divergence — our distillation loss | "Cross-Attention Knowledge & Layer Distillation", arXiv:2402.10631 |
| `TALR` | our flip-rate servo | "Ternary Adaptation via Low-Rank Sidebranches", arXiv:2608.24469 |

Also: `QuEST` cited as arXiv:2411.04330, which the project's *earlier* report correctly cited as
Kumar et al. *Scaling Laws for Precision*; CAKLD and BitDistiller double-counted under one ID; a
PRISMA flow whose stage-4 exclusions total 54 against a claimed 64; and our own unpublished internal
methods inside the literature denominator, contaminating every percentage.

**Consequence:** every row must be opened personally before it enters the paper, and all percentages
recomputed over published work only. Contrast F1 — the format census survived checking.

## F8. Category error: sign planes are not multi-plane superposition · HOLDS

The format census marks `IQ2_XXS`/`IQ2_XS` as "Multiple Planes: Yes (magnitude + sign)" and concludes
that multi-plane methods such as PTQTP "map directly onto" them. They do not: a sign plane carries
the sign *of the same magnitude*, whereas superposition is `W ≈ α₁T₁ + α₂T₂` — two independent
discrete terms with independent scales.

The valid mapping in that same paragraph is **AQLM's additive VQ**, which genuinely is a sum of
codebook lookups. Keep that one; drop the IQ2 claim.

## F9. Bits-accounting claims must be recomputed, not quoted · HOLDS

Two errors found by recomputation:

* **AQLM "1.00 bpw (1×16)"** — a 16-bit index over an 8-dimensional group is **2 bpw**.
* **`IQ1_S` listed as having a zero-point** — its one-bit ±0.125 displacement is not the same
  capability as `Q2_K`'s 4-bit `d_min`, and sharing a matrix column would mislead any screen run
  against it.

Independently confirmed and worth keeping: leaving embeddings and head in FP16 costs **+2.096 bpw**
on an 8B (13.1% of parameters × 16 bits), taking a nominal 1.58 bpw method to **3.43 bpw
whole-model**; `BitNet-b1.58-2B-4T` ships at **3.73 bpw tied / 5.36 bpw untied**.

## F10. The most-deployed sub-2-bit formats have the least academic literature · PROVISIONAL

The IQ family drives most local consumer inference through llama.cpp and was developed empirically —
greedy coordinate descent against a diagonal-Hessian `imatrix` proxy — while academic work (AQLM,
QuIP#, VPTQ) ships bespoke CUDA barely represented in the ecosystem people run.

**Not yet assertable.** Needs a count of published methods targeting IQ formats; expected ≈ 0. If it
holds it is the paper's thesis one level up — the literature optimises away from deployment.

---

## Open

* **F3 end-to-end** — the reconstruction screen is done and does not settle the question. An
  end-to-end agreement comparison (the metric that produced 82.06 vs 79.31 for g64 vs g256) is what
  would, and IQ1_S has no such number yet.
* **F3 with imatrix** — the present comparison is a matched *unweighted* lower bound. IQ1_S is
  designed around importance weighting; re-run with it before the margin is quoted anywhere.
* **F10 count** — published methods targeting IQ formats.
* Re-derivation of every method-census percentage over published work only (F7).
