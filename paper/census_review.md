# Review of the methodology census (2026-09-14)

Screening of the returned deep-research census before any of it enters the paper. The census is
genuinely useful — it is the first thing we have with a real denominator — but **it must not be
ingested as-is.** Errors below are ordered by how much damage they would do in print.

---

## 1. The finding that matters most: the census is INVARIANT to TQ1_0 vs TQ1_64

Every failure category the census identifies is a property of the **format class**, not of our
particular format:

| failure category | fails TQ1_64? | fails stock TQ1_0? |
|---|---|---|
| asymmetric offsets / zero-points | yes | **yes** |
| multi-plane superposition | yes | **yes** |
| VQ codebooks / lattice decoders | yes | **yes** |
| sparse FP16 outlier side-tensors | yes | **yes** |
| variable-length entropy codes | yes | **yes** |
| coupled activation quantizers | yes | **yes** |
| tunable norm affines | yes (QuaRot folds them) | **yes** (same pipeline) |

The **only** thing that differs between TQ1_0 and TQ1_64 is scale granularity — g256 vs g64 — and
that changes **none** of the 30 FAIL verdicts. So the paper's central claim does not rest on TQ1_64
at all. State it against *the class of formats with one scale per block and no side tensors, of which
stock TQ1_0 is the canonical member*, and TQ1_64 drops from a premise to an implementation detail.

This is strictly stronger: "methods fail against the established format" is a far better paper than
"methods fail against our bespoke format."

## 2. Errors in the register — do not propagate these

### 2a. CRITICAL: internal acronyms matched to unrelated published papers

Two of our **internal** method names have been bound to real but unrelated arXiv entries:

| our name | what it actually is (this repo) | what the census claims |
|---|---|---|
| **CAKLD** | Confidence-Aware KL Divergence — our distillation *loss*, `L = CAKLD_all + β·mean(KL over commit window)` | "Cross-Attention Knowledge & Layer Distillation", arXiv:2402.10631, verdict **FAIL** |
| **TALR** | our **flip-rate servo** in assignment training | "Ternary Adaptation via Low-Rank Sidebranches", arXiv:2608.24469, verdict **FAIL** |

Both are **in our production pipeline**. A register that marks our own shipping components as FAIL,
under invented expansions of their acronyms, is not reliable at the row level. Treat every row whose
citation we have not personally opened as unverified.

### 2b. Three rows contradict our own "already in the pipeline" list

`QEP` (FAIL), `CAKLD` (FAIL), `EfficientQAT`/`block-AP` (NEEDS-ADAPTATION) are all recorded in
`research_prompts/next_methods_prompt.md` §5 as already shipping here. Block-AP **is** our skeleton.

### 2c. Citation collisions

* **QuEST cited as arXiv:2411.04330.** That ID is Kumar et al., *Scaling Laws for Precision* — and
  the **previous** research report in this same project cited it correctly as such. Two reports, same
  ID, different papers. QuEST is a different work entirely.
* **CAKLD and BitDistiller share arXiv:2402.10631** and are listed as separate register rows (#21,
  #41). CAKLD is a *component of* BitDistiller, not a peer method. That is a double-count.
* **OPSA cited as arXiv:2605.09240**; our log records **2608.31046**.

### 2d. The PRISMA flow does not balance

Stage 4 claims **64 excluded** but the stated reasons total **54** (38 superseded + 16
irrecoverable). Ten exclusions are unaccounted for. In a systematic review the flow diagram is the
one thing that must reconcile exactly.

### 2e. Our own unpublished internal methods are inside the literature denominator

`NAP`, `QUASAR`, `SoftWater`, `ECASQ`, `SQuaT`, `LCD` appear with no citation, attributed to
"Primary literature / Internal Evaluation Batch". **A literature census cannot include unpublished
internal work** — it inflates n=48 and contaminates every percentage derived from it (62.5% FAIL,
16.7% free-gen, 25.0% cross-order). Those headline fractions must be recomputed over published work
only, with our internal batch reported separately.

### 2f. "Position 0.27" is our own measurement, fed back as a literature finding

The census presents confident divergence at position 0.27 as an established property of sub-2-bit
deployment, with a figure. It is **§13an of this log** — one 4B model, one seed, and measured with
DRY active, a confound we recorded at the time. It is also over-read: it was the *mean index of the
first confident flip across rollouts*, not a universal horizon. Citing our own unpublished number
back to ourselves as external corroboration is circular and would be caught.

### 2g. The E2M-ATQ "misrepresentation" charge is unfair and would cost us credibility

The census accuses arXiv:2609.09240 of systematically misrepresenting its footprint. That paper
**explicitly warns against the exact reading** — its guardrail list says to avoid "the whole model is
1.64 bits/parameter", and it states the 8.24 GiB artifact size plainly. The accounting *gap* is real
and worth tabulating; the accusation of concealment is not, and it is aimed at the one paper in the
census that was scrupulous about it.

## 3. What to keep

* **The funnel discipline** — 384 → 262 → 118 → 54 → 48, once the arithmetic is fixed and our
  internal methods are removed from the denominator.
* **The multi-scale refinement, which corrects our own assumption.** We assumed <20% evaluate at more
  than one scale; the census says **60.4% do, but only 25.0% span orders of magnitude** and 35.4%
  compare architecturally adjacent pairs (7B/13B). That distinction is better than our hunch and is
  the right gap statement for the introduction — *if* it survives re-derivation over published-only
  rows.
* **Free-generation evaluation at 16.7%** — directly supports our §8 thesis, and is the single most
  useful number in the report.
* **The vocabulary-penalty arithmetic.** Leaving embed+head in FP16 costs **+2.096 bpw** on an 8B
  (13.1% of parameters × 16 bits), taking a nominal 1.58 bpw method to **3.43 bpw whole-model**. The
  algebra checks out and it is the cleanest quantitative statement of the accounting problem we have.
* The structural failure taxonomy itself, which independently reproduces our screen.

## 4. Required action

1. Recompute every percentage over **published methods only**.
2. Personally open every citation before a row enters the paper. Two acronym collisions in 48 rows
   implies more we have not caught.
3. Re-run or drop rows contradicting our pipeline list (QEP, CAKLD, block-AP, TALR).
4. Restate the paper's claim against the **format class**, not TQ1_64 (§1 above).
5. Settle the TQ1_0 question empirically — see `PLAN.md` §B.0.
