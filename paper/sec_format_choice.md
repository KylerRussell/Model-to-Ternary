# §2 — Choosing a deployment format

> **Draft.** Measurements are reconstruction-only, unweighted RTN, 3 matched tensors, one 4B model.
> Every limit is stated in §2.5; the conclusion is not yet safe to generalise.

## 2.1 Why the format is a first-class question

A method that "fails at 1.6 bits" may have failed against a *format*, not against a bit budget.
Published sub-2-bit work almost universally evaluates against a representation of its own choosing, so
the literature cannot distinguish the two — and neither could we, until we measured more than one
format under one protocol.

This section fixes a deployment format by measurement rather than by assumption. It also corrects the
choice this project had already made.

## 2.2 Protocol

Seven formats, all round-to-nearest, on the **same three tensors** from the same checkpoint, scored
with the same weighted relative reconstruction error. Matched tensor sets matter: an earlier run drew
tensors independently per arm and produced a spurious result, because *a format comparison across
different tensors measures the tensors* (§2.5).

Each arm gets a genuine search over its own free parameters — a 24-point scale sweep for the scalar
formats, a joint search over (super-scale × sub-scale × offset × 2048-entry codebook) for the IQ
arms. Handicapping an arm's search would manufacture the result.

**Fidelity.** The `IQ1_S` and `IQ1_M` arms use llama.cpp's real grids and dequantization formulae via
the `gguf` package. Our chosen parameters are packed into real 50- and 56-byte blocks and checked
against **gguf's own dequantizer**: exact, max |Δ| = 0. That check caught two bugs inspection missed,
including the format storing its super-scale in fp16 where our simulation used fp32 — which would
have credited `IQ1_S` with precision the format does not have.

## 2.3 Results

Mean relative reconstruction error, rotated checkpoint (unrotated in parentheses; ordering identical):

| format | bpw | offset capability | scale granularity | error | vs. line | Pareto |
|---|---|---|---|---|---|---|
| `Q1_0` | 1.1250 | none | g128 | 0.61838 (0.62879) | **+0.1440** | frontier |
| `IQ1_S` | 1.5625 | 1-bit per **g32** | g32 + super | 0.44631 (0.44826) | +0.0001 | frontier |
| `TQ1_0` | 1.6875 | **none** | g256 | 0.43800 (0.44160) | −0.0002 | frontier |
| **`IQ1_M`** | **1.7500** | 1-bit per **g8** | **g16** + super | **0.41350 (0.41483)** | **−0.0207** | **frontier** |
| **TQ1_64** *(ours)* | 1.7812 | **none** | g64 + super | 0.43228 (0.43527) | +0.0001 | **DOMINATED** |
| `TQ2_0` | 2.0625 | none | g256 | 0.43800 (0.44160) | +0.0239 | **DOMINATED** |
| `Q2_K` | 2.6250 | **4-bit per g16** | g16 + super | 0.32970 (0.33096) | **−0.0482** | frontier |

"vs. line" is the residual from `err = −0.0643·bpw + 0.5467`, fitted through the three *offset-free or
coarse-offset* scalar formats (`IQ1_S`, `TQ1_0`, TQ1_64), which lie on it with **r = −0.99972** and
residuals of ±0.0002.

### 2.3.1 Two formats are Pareto-dominated, and one of them is ours

`TQ1_64` is dominated by **`IQ1_M`, which is simultaneously 1.8% smaller and 4.3% lower error.** This
is not a trade-off to be weighed; it is domination on both axes, reproduced in both rotation regimes.
`TQ2_0` is likewise dominated — it is the *same alphabet and the same g256 scale* as `TQ1_0`, so its
reconstruction is identical by construction, at 22% more bits. It exists for kernel throughput, not
for compression, and should never be presented as a compression point.

### 2.3.2 What determines position relative to the line

Ranking the residuals against each format's offset capability gives a clean monotone ordering:

| offset capability | example | residual |
|---|---|---|
| none | `TQ1_0`, TQ1_64, `TQ2_0` | ≈ 0 (on the line) |
| 1 bit per 32 weights | `IQ1_S` | ≈ 0 (too coarse to help) |
| 1 bit per 8 weights | `IQ1_M` | **−0.021** |
| 4 bits per 16 weights | `Q2_K` | **−0.048** |

**The ability to represent a fine-grained offset — not alphabet size, not scale granularity — is what
moves a format off the rate-distortion line.** Scale granularity alone does not: `TQ1_0` (g256) and
TQ1_64 (g64) sit on the *same* line, differing only in where along it they land. And a coarse offset
does not either: `IQ1_S` carries a ±0.125 displacement but shares it across 32 weights, and lands on
the line with the offset-free formats.

This has a direct consequence for the screen used throughout this paper. The constraint "no
zero-point", which we adopted as a format axiom and which disqualifies a large fraction of published
methods, is **the single most expensive constraint in the design space** — and it is a choice, not a
hardware limit (§3).

### 2.3.3 The third symbol earns its bit

`Q1_0` is binary and sits **+0.144 above the line** — by far the largest residual measured. Binary has
no zero state, so it cannot express the ~46% of weights that round to zero under ternary. The ternary
alphabet's extra symbol is worth considerably more than the 0.43 bpw it costs over binary.

## 2.4 Decision, and how it was reached

**On this evidence TQ1_64 is not the right deployment target.** It is Pareto-dominated by a format
that ships in stock llama.cpp, whereas TQ1_64 requires the fork we maintain.

### 2.4.1 The decision record

TQ1_64 was specified on **2026-08-12**. Its stated rationale, quoted from the spec as written and
unchanged since:

> `TQ2_0` (2.0625 bpw) and `TQ1_0` (1.6875 bpw) both use **QK_K = 256 with a single scale per block
> (g256)**. Our model is trained at **g64** because finer scale granularity measurably wins […]
> Exporting a g64 model to TQ2_0 **requantizes it to g256 and destroys it**: MMLU-Pro 25.7% → 10.6%
> (random) […] Hence a format that can hold g64.

with the supporting measurement: eval2k agreement **79.31% (g256) → 80.62% (g128) → 82.06% (g64)**.

The comparison in §2.3 was run on **2026-09-14/15**. `IQ1_S` and `IQ1_M` are not mentioned anywhere in
this project's history until 2026-09-14 — **33 days after the format was fixed.** Both dates are
recoverable from version control, which is why this account can be checked rather than taken on
trust.

### 2.4.2 What was wrong with it

The reasoning was not sloppy and its measurement was not wrong. Scale granularity does improve
agreement, by 2.75 points, and stock `TQ1_0`/`TQ2_0` genuinely cannot express g64. Every step
follows.

The error was the **scope of the comparison**: the search ran exhaustively *within* the scalar-ternary
family — g256 → g128 → g64 → g32, four points on one axis — and never crossed to another family. §2.3
shows that axis is movement *along* the rate-distortion line rather than off it, and that the axis
which does move a format off the line, offset granularity, was excluded by an assumption ("no
zero-point") never itself tested.

One further fact belongs here rather than in a footnote, because it removes the most convenient
excuse: **the information was available.** `IQ2_XXS` had been shipping in llama.cpp since early 2024
and `IQ1_S`/`IQ1_M` well before this project began. Nothing had to be invented or awaited. We did not
look outside the family we had chosen.

### 2.4.3 Why this is reported rather than repaired quietly

We name the error class as **family-bounded search**: optimising exhaustively within a representation
family while treating the family boundary as given. It is the same failure this paper documents in
the wider literature — methods evaluated against a format of the author's own choosing, so that a
result about *a* format is reported as a result about a *bit budget* (§2.1).

Reporting it has three concrete consequences for the rest of the paper:

1. The screen in §3 must be stated against a **format class**, naming the class, never against
   TQ1_64. A verdict of "method X is undeployable" is otherwise a statement about our design choice.
2. The screen itself inherits the same risk. "No zero-point" disqualifies a large share of published
   methods, and §2.3.2 shows it is the most expensive constraint in the design space while §3 shows
   it is a choice, not a hardware limit. Both facts are required for the screen to be honest.
3. Every format claim in this paper is now checked across families before it is made, and the
   remaining unmeasured formats are listed as unmeasured (§2.5) rather than assumed comparable.

A paper that applies a structural screen to other people's work and exempts its own is not worth
reading. Being our own worked example is not a concession made reluctantly; it is the strongest
available evidence that the screen finds real errors, since it found one that cost us the artifact we
had built.

## 2.5 Limits

**Reconstruction error is a screen, not a verdict.** This project has documented that layer-wise
metrics never once caught a real end-to-end failure — block-MSE improved 99.8% on a model whose
residual stream had collapsed. A *margin* on this metric would prove little. **Domination on both
axes is a stronger claim**: it survives unless the metric is *anti*-correlated with quality, not
merely noisy. That is a much weaker assumption, but it is still an assumption, and the end-to-end
comparison at matched bits has not been run.

**Unweighted RTN understates the IQ arms specifically.** llama.cpp quantizes the IQ formats against an
activation importance matrix (`H_ii ≈ Σ X_ik²`). Running unweighted removes an advantage designed
into those formats, so the measured gap for `IQ1_M` is a **lower bound**.

**Our arms are stronger than stock.** Giving every arm a 24-point scale search makes `TQ1_0` here
better than llama.cpp's own `TQ1_0` quantizer would produce. This measures *format capacity*, which is
the right quantity for the screen, but it is not a claim about shipped quantizer quality.

**Three tensors, one 4B model, one seed.** Enough for domination reproduced across two rotation
regimes; not enough for the margin to be quoted.

**A prior hypothesis of ours was falsified here.** We predicted that QuaRot's outlier removal would
strip the local variance that fine scales exist to capture, making coarse-scale ternary artificially
competitive. Matched tensor sets show the ordering is identical rotated and unrotated; the apparent
effect was a sampling artifact.

### Open

1. **End-to-end agreement, `IQ1_M` vs TQ1_64 at matched bits** — the measurement that would settle
   §2.4, and the one neither format has against the other.
2. **Re-run with importance weighting** before any margin here is quoted.
3. **`IQ2_XXS` / `IQ2_XS` / `IQ2_S` unmeasured** — their sign-plane machinery is not yet implemented,
   so the 2.0–2.6 bpw region rests on `Q2_K` alone.
4. **Retargeting cost** — our pipeline trains ternary *assignments* by STE with per-block scales.
   Targeting `IQ1_M` turns that into selecting among 2048 codebook entries plus a per-group sign: a
   different optimisation problem, not a re-parameterisation. That cost belongs in the decision.
