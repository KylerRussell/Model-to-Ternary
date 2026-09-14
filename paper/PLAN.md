# The Sub-2-Bit Protocol — full work plan

**Status:** draft for discussion, 2026-09-14. Supersedes nothing; `logs/RESULTS_SUMMARY.md` remains
the record of what was actually measured.

A matched-protocol study of what survives extreme low-bit quantization, run across three model
scales, scored against real benchmarks, with enough seeds to mean something.

---

## 0. The decision

Three different papers hide under "systematic overview of ternary quantization":

| | what it is | verdict |
|---|---|---|
| **A** | literature survey + taxonomy | **no** — no barrier to entry, no experiments, and we would be reviewing papers we cannot run |
| **B** | matched-protocol evaluation under a fixed deployment format | **yes** — this is what the project uniquely has |
| **C** | measurement/reproducibility paper | fold into B as §7 |

### The claim

> Most reported sub-2-bit gains do not transfer to a fixed deployment format, and the failures are
> predictable in advance from a single structural property — whether the method assumes a richer
> parameterisation than the format provides.

The contribution is **the decision rule, not the catalogue**. A survey ages in a year; a screen that
other groups can apply to their own format does not.

### The circularity that must be fixed

The screen was *derived from* five observed failures. Validating it on those same five is circular
and a reviewer will say so on page one. The fix is the spine of the whole program:

> **Pre-register screen verdicts for methods not used to derive it, then test them.**

An out-of-sample hit rate turns a lab notebook into a result. Everything in §A exists to build the
candidate pool that makes this possible.

---

## A. Systematic search protocol

**This section was added after review and it changes the paper's central claim, not just its
thoroughness.** Without a pre-specified search, the ~13 methods in the log are a *convenience
sample* — the papers we happened to encounter. "Systematic" is a claim about method, and a
convenience sample cannot support it.

### What it upgrades

| | denominator | strength |
|---|---|---|
| before | "13 methods we tried failed" | anecdote with good measurement |
| **after** | "N methods met inclusion criteria; the screen predicts M fail; we tested K; it was right in J" | **a result about the literature** |

That is the difference between a lab report and a systematic review, and it costs search time rather
than GPU time.

### A.1 Sources

* arXiv (`cs.LG`, `cs.CL`, `cs.AI`) — primary, since this literature moves faster than proceedings
* ACL / EMNLP / NeurIPS / ICLR / ICML proceedings, for peer-reviewed versions and anything not on arXiv
* Semantic Scholar / Google Scholar for citation graph
* **Forward and backward citation chasing** from anchor papers (BitNet, BitNet b1.58, GPTQ, QuaRot,
  ParetoQ, TWLA, PTQTP, CAT-Q) — this catches work that uses none of our query terms
* Vendor/industry technical reports (e.g. arXiv:2609.09240), which are often where deployment-realistic
  numbers live

### A.2 Query terms

Run each independently; union the results.

```
ternary quantization LLM          1.58-bit                  sub-2-bit
binary LLM / 1-bit LLM            extreme low-bit           W2A16 / W1.58A16
trit / trit-plane                 post-training quantization ternary
quantization-aware training low-bit   weight-only 2-bit      ternary weights transformer
sub-1-bit / fractional bit        vector quantization LLM weights
```

### A.3 Inclusion criteria

Include if **all** hold:

1. operates on the **weights** of a transformer LLM
2. targets an effective weight budget of **≤ 2 bits/weight** (state the paper's own accounting; see A.5)
3. reports *some* empirical result — capability, perplexity, or reconstruction
4. is described in enough detail to apply the screen

### A.4 Exclusion criteria

Exclude, recording the reason:

* activation-only or KV-cache-only quantization (we are weight-only, bf16 activations)
* ≥ 3-bit targets
* method described only by results, with no recoverable procedure
* superseded by a later version from the same authors (keep the latest, note the chain)

### A.5 The bits-accounting trap — screen for it explicitly

Papers report bpw on wildly different bases. arXiv:2609.09240 reports **1.64 bpw** but ships an
**8.24 GiB artifact for an 8B model (~8.85 bits/parameter)**, because its budget covers linear
projections only and leaves embeddings, LM head, norms and KV at FP16. Our 1.7812 bpw counts
everything including a ternarised head.

**For every included method, record three numbers:** the paper's claimed bpw, what that figure covers,
and the implied whole-checkpoint bits/parameter. A table of those three columns across N papers is a
publishable contribution by itself, and it is pure desk work.

### A.6 Screening and recording

Two-stage: title/abstract → full text. Record counts at each stage (PRISMA-style flow: identified →
screened → full-text assessed → included → screened-out-with-reason). Dual-screen where feasible and
record disagreements rather than silently resolving them.

Output is a **method register** — one row per included method — in `paper/method_register.md`:

| field | notes |
|---|---|
| id, citation, date | |
| what it varies | the §0 screen question |
| **screen verdict** | PASS / FAIL / NEEDS-ADAPTATION, **with reasoning** |
| claimed bpw / coverage / implied whole-model | A.5 |
| evidence type | free-gen capability, MC benchmark, PPL, or reconstruction only |
| scales evaluated | in the original paper |
| tested here? | and at which scales |

### A.7 Pre-registration

**Screen verdicts must be committed to the repository with a timestamp before the corresponding runs
start.** A git commit hash is sufficient and is the cheapest credible pre-registration available.
Without this, §6 of the paper is worth nothing — and it is the section carrying the contribution.

### A.8 Deliverable

A deep-research prompt to drive the search lives at
`research_prompts/methodology_search_prompt.md`. It must carry the format constraint, the screen,
the exclusion list of already-tested methods, and the bits-accounting requirement, or it will return
the same PTQ papers we have already resolved.

---

## B. Scope

### B.0 Why TQ1_64 and not stock TQ1_0 — the premise a reviewer attacks first

If the paper screens methods against a format we invented, the result reads as "methods fail against
our bespoke format," which is a much weaker claim than it looks. This has to be settled before §3 is
written.

#### What we already have (`TQ1_64_SPEC.md`)

Stock `TQ1_0` (1.6875 bpw) and `TQ2_0` (2.0625 bpw) both use QK_K=256 with **one scale per 256-weight
block (g256)**. Our model is trained at **g64**, and granularity measurably matters:

| granularity | eval2k agreement |
|---|---|
| g256 (= what TQ1_0 / TQ2_0 can express) | 79.31% |
| g128 @ 8-bit scales | 80.62% |
| **g64 @ 8-bit scales** | **82.06%** |
| g32 @ 4-bit scales | 73.56% |

So TQ1_64 buys **+2.75 pp agreement for +0.0937 bpw (+5.6% size)** over TQ1_0. That is a real and
defensible trade — and it is the argument the paper should make.

#### The gap in that argument — state it ourselves

The spec also records that exporting the g64 model to TQ2_0 destroys it (MMLU-Pro 25.7% → 10.6%,
i.e. random; GPQA 30.3% → 16.8%). **That number is not evidence for TQ1_64 over TQ1_0.** It measures
a *mismatched export*: a model trained at g64 re-quantized to g256. A model trained at g256 and
exported to TQ1_0 would not be destroyed. Using it as a format comparison would be the same error as
scoring a method against a baseline it was never fitted to.

The honest comparison is **train-to-format, end to end**:

| arm | pipeline target | export | bpw |
|---|---|---|---|
| A | g256 throughout | stock `TQ1_0` | 1.6875 |
| B | g64 throughout | `TQ1_64` | 1.7812 |

Same model, same data, same seeds, scored on the real benchmark suite (§D). Three outcomes, all
publishable:

* **A ≈ B** → use stock TQ1_0. The paper gets *stronger*: we screen against the established format.
* **B > A by more than +5.6% size justifies** → we have an empirical answer, and "scale granularity
  is worth more than bits at 1.7 bpw" is a finding in its own right.
* **B < A** → we have been paying for a worse format, which we would need to know regardless.

#### The claim does not actually depend on the outcome

Every failure category in the census — zero-points, multi-plane superposition, VQ codebooks, sparse
outlier tensors, entropy codes, coupled activation quantizers, tunable norm affines — fails **stock
TQ1_0 identically**. Granularity changes none of the 30 FAIL verdicts (see `paper/census_review.md`
§1). So state the paper's claim against **the class of formats with one scale per block and no side
tensors, of which stock TQ1_0 is the canonical member**, and TQ1_64 becomes an implementation detail
rather than a premise. The head-to-head above then supports a secondary, narrower claim about
granularity.

### B.1 The scale ladder

Methods that help at 4B and vanish at 27B would be a genuinely novel finding; the literature has no
answer because almost nobody runs the same method at more than one scale. The closest prior work
(arXiv:2609.09240) ran two points and correctly declined to call it a scaling law. **Three points can
at least separate monotone from non-monotone.**

| rung | checkpoint | layers | hidden | intermediate | vocab | ~params | embed+head | status |
|---|---|---|---|---|---|---|---|---|
| 4B | `Qwen3.5-4B` | 32 | 2560 | 9216 | 248,320 | 4.4B | **29.1%** | cached, pipeline validated |
| 9B | `Qwen3.5-9B` | 32 | 4096 | 12288 | 248,320 | 9.0B | **22.6%** | to download (4 shards) |
| 27B | `Qwen3.5-27B` | 64 | 5120 | 17408 | 248,320 | 26.4B | **9.6%** | to download (11 shards) |

Geometry above is **read from the real `config.json` of each repo** (2026-09-14), not estimated.
All three are `model_type: qwen3_5` with identical vocabulary.

#### RESOLVED: the generation confound

The original pair confounded scale with model generation — the cached 4B is **Qwen3.5** and the
cached 27B is **Qwen3.6**. Resolution: **run the ladder entirely within Qwen3.5**. All three rungs
exist and no re-anchoring of the 4B is needed, so every 4B result already in the log survives as the
ladder's bottom rung.

#### The generation control: `Qwen3.8-27B`

If we spend an arm outside the 3.5 generation it should be the **newest** available, because the
control's job is to bound how much *generation* moves results at fixed scale — and the widest gap is
the stronger probe. A single-step 3.5→3.6 comparison could read small and prove nothing either way.

This turns out to be unusually clean. The three 27B checkpoints are **architecturally identical**:

| | layers | hidden | intermediate | vocab | ~params | embed+head |
|---|---|---|---|---|---|---|
| `Qwen3.5-27B` | 64 | 5120 | 17408 | 248,320 | 26.4B | 9.6% |
| `Qwen3.6-27B` | 64 | 5120 | 17408 | 248,320 | 26.4B | 9.6% |
| `Qwen3.8-27B` | 64 | 5120 | 17408 | 248,320 | 26.4B | 9.6% |

Differing shard counts (11 / 15 / 18) are packaging, not architecture. **Training is therefore the
only thing that varies**, which is exactly what a generation control requires — anything the arm
measures is attributable to the checkpoint's training, not to its shape. `Qwen3.7-27B`, `Qwen3.5-14B`
and `Qwen3.5-32B` do not exist.

#### The ladder's two steps are not the same kind of step

State this before a reviewer does:

* **4B → 9B** is a *width* increase at constant depth (H 2560→4096, L=32 both)
* **9B → 27B** changes *both* (L 32→64, H 4096→5120)

So "scale" is not a single axis here. This is usable rather than merely awkward: an effect that
appears at 4B→9B is width/superposition-related, while one appearing only at 9B→27B implicates depth.
It does mean no result may be reported as a function of parameter count alone.

#### A free experiment hiding in the vocabulary

Vocabulary is fixed at 248,320 across the family, so the embed+head share falls **29.1% → 22.6% →
9.6%** as pure arithmetic in width and depth — three clean points. **Part of any "scale helps" effect
is the embedding fraction shrinking, not representation superposition.** The two are separable with
machinery that already exists (`src/head_swap.py`, under an hour per rung): run the head/embed swap
arms at each rung and decompose the scale benefit into *fixed-vocabulary dilution* vs *everything
else*. No prior ternary paper has done this.

---|---|---|---|---|---|---|
| 4B | `Qwen3.5-4B` | 32 | 2560 | 248,320 | **29.1%** | cached, pipeline validated |
| 9B | `Qwen3.5-9B` | — | — | — | ~18% (est.) | **to download** |
| 27B | `Qwen3.5-27B` | — | — | — | ~10% (est.) | **to download** |

#### RESOLVED: the generation confound

The original pair confounded scale with model generation — the cached 4B is **Qwen3.5** and the
cached 27B is **Qwen3.6**. Resolution (2026-09-14): **run the ladder entirely within Qwen3.5**, using
4B / 9B / 27B from that one generation. No re-anchoring of the 4B is needed, which preserves every
4B result already in the log as the ladder's bottom rung.

Consequences to carry:

* `Qwen3.5-9B` and `Qwen3.5-27B` must be downloaded; only `Qwen3.5-4B` and `Qwen3.6-27B` are cached.
* The geometry above for 9B/27B is **estimated** and must be re-derived from the real configs before
  any claim rests on it — the 29.1%/9.6% figures for the original pair were computed, not assumed.
* `Qwen3.6-27B` remains useful as a **generation control**: running one arm on both 3.5-27B and
  3.6-27B measures the generation effect directly, which is the cheapest way to show the confound
  we avoided was real rather than hypothetical. Worth one arm.

#### A free experiment hiding in the vocabulary

Vocabulary is fixed at 248,320 across the family, so the embed+head share falls from 29.1% to 9.6%
as pure arithmetic in width and depth. **Part of any "scale helps" effect is the embedding fraction
shrinking, not representation superposition.** The two are separable with machinery that already
exists (`src/head_swap.py`, runs in under an hour): run the head/embed swap arms at each rung and
decompose the scale benefit into *fixed-vocabulary dilution* vs *everything else*. No prior ternary
paper has done this.

### B.2 Bit regimes — spine vs test set

Widening to 1-bit and sub-1-bit is right in instinct and a trap in naive execution:

* methods are mostly **not format-portable** — a ternary method assumes three levels, a binary method
  two-with-a-scale. Porting one to the other means evaluating *our reimplementation*, not the
  published method. That is how survey papers become uncitable.
* 13 methods × 3 regimes × 3 scales × a real seed budget is not affordable on 2× RTX 3090.

**Recommendation.** Keep **ternary × three scales** as the spine. Use the other bit regimes as the
**out-of-sample validation set for the screen** (§0). The screen is about parameterisation richness
versus format capacity, so it *should* generalise across bit depths — and if it does not, that is a
real result about its limits.

**Caveat to write down early:** sub-1-bit is barely a format category. Below one bit/weight you need
structured sparsity, vector quantization or shared codebooks — all of which need side tensors or
variable-length codes and fail the deployment screen on contact. Stated plainly that is a
contribution; padded out it is the weakest section in the paper.

---

## C. Statistical design

The project has measured its own noise floors, so this is a calculation, not an argument.
Two-sided, α=0.05, 80% power, `n ≈ 7.84σ²/δ²`:

| quantity | SD | effect to resolve | seeds/arm |
|---|---|---|---|
| skeleton agreement | 0.309 pp | 0.5 pp | **3** |
| skeleton agreement | 0.309 pp | 0.3 pp | **9** |
| skeleton agreement | 0.309 pp | 0.2 pp | **19** |
| gate commit rate, **unpaired** | 0.103 | 0.05 | **34** |
| gate commit rate, **paired** | 0.012 | 0.05 | **2–3** |

### C.1 Pairing beats seeds by ~16×

Hold prompts, seed and harness fixed; vary only the method. Every comparison is then paired by
construction. Spend seeds on absolute rates that cannot be paired, and on quantities whose cross-seed
behaviour is unknown.

### C.2 Rare-event statistics need their own budget

The row-gain sweep (§13at) demonstrated the failure mode: `commit_correct_rate` replicated exactly
across seeds, while `think_len/teacher` — computed over **three** closing rollouts — moved from
**1.279 (PASS)** to **0.646 (FAIL)**. Any statistic conditioned on a rare event must be sized by how
rare the event is, not by the headline metric.

### C.3 Sequential, not uniform

Screen wide at n=2 to find anything that moves; promote survivors to the full budget.
**Pre-register the promotion rule** so the sequential design cannot become p-hacking after the fact.

---

## D. The benchmark gate

Teacher-forced agreement read **78.50% PASS** on a model scoring **0/18** on the arithmetic its own
teacher could solve. Across 13 methods, perplexity, KL and layer-wise MSE were worse than useless:
block-MSE improved 99.8% on a model whose residual stream had collapsed, and pointed the *wrong way*
on a method costing −10.3 pp end-to-end. **Real benchmarks are the only gate that has never given a
false pass here.**

### D.1 Four design rules

1. **Both task shapes.** Multiple-choice benchmarks are comparable to the literature but structurally
   blind to our actual failure — a model that cannot commit still scores on MMLU. Free-generation
   tasks expose it. Report both, separately, never averaged.
2. **Chance-correct everything.** `R = (A_s − B)/(A_t − B)`, B = max(chance, majority-class floor).
   Raw retention ratios are not comparable across tasks with different floors.
3. **Score against what the teacher can solve.** Every apparent win in the head experiments evaporated
   under this denominator — the ternary model's "correct" answers were on problems the FP teacher also
   fails. Raw accuracy systematically overstates capability at these bit depths.
4. **n ≥ 500 per task.** At n=24 a paired comparison produced 3 discordant pairs and p=0.25 — an
   unfalsifiable result. arXiv:2609.09240 used n=500; that is the floor.

### D.2 Also in the protocol

Contamination screening; a pinned harness version; a fixed decoding config stated once and applied to
every arm. A benchmark suite that drifts between arms silently destroys pairing.

---

## E. Paper outline

Numbered because the argument depends on the order — the screen must be stated before it is tested.

| § | section | carries | status |
|---|---|---|---|
| 1 | Introduction | positive-result bias in sub-2-bit work; nobody re-runs under a fixed format | HAVE |
| 2 | The deployment format as constraint | TQ1_64 as a contract: one scale per (row, 64-block), no zero-point, no side tensors, no variable-length codes | HAVE |
| 3 | The screen | decision rule stated falsifiably, *with its in-sample derivation labelled as such* | HAVE |
| 4 | **Systematic search** | protocol, PRISMA counts, method register, bits-accounting table | **NOT STARTED (§A)** |
| 5 | Protocol | noise floors, pairing, seed budgets, benchmark suite, reproduction gate | PARTIAL |
| 6 | Methods at three scales | the core matrix; method×scale interaction; dilution-vs-superposition decomposition | NEEDS 9B + 27B |
| 7 | Out-of-sample screen test | pre-registered verdicts → results → hit rate | NOT STARTED |
| 8 | What the metrics missed | agreement, KL and block-MSE each endorsing broken models; the measurement traps | HAVE |
| 9 | Limits | one family; bit regimes not fully crossed; sub-1-bit unrunnable under the format | HAVE |

**Protect §8 from the cutting-room floor.** A catalogue of metrics that confidently endorsed broken
models — each with symptom, diagnostic, and the invariant that now prevents it — is the part other
groups will reuse, and it costs nothing extra to write because the incidents are already logged.

---

## F. Work plan

### Tier 0 — before any of the above. Every item is worth doing whether or not a paper happens.

| id | task | depends on | why |
|---|---|---|---|
| **T0.0** | **Systematic search + method register** (§A) | — | defines the candidate pool; everything downstream inherits its denominator. Desk work, no GPU. |
| ~~**T0.1**~~ | ~~Track the experiment drivers~~ **DONE 2026-09-14** | — | 69 drivers moved to `experiments/sweep/` (tracked) with cross-references rewritten; outputs stay in the ignored `output_sweep/`. `tools/capture_env.sh` snapshots repo commit, host, GPU, package versions, and **HF dataset/model revision hashes** to `paper/env/`. |
| **T0.2** | External reproduction gate | T0.1 | a reviewer cannot otherwise distinguish "these methods don't work" from "your pipeline is broken" — and this project has shipped a silently zeroed FP teacher that passed three sanity checks. Target: within ~0.05 pts of a published number. |
| **T0.8** | **TQ1_0 vs TQ1_64 head-to-head** (§B.0) | T0.4 | the paper's premise. One extra 4B pipeline run targeting g256, exported to stock TQ1_0, scored against the g64/TQ1_64 arm on the real suite. Until this exists, "our format" is an assumption. |
| **T0.3** | Download `Qwen3.5-9B` and `Qwen3.5-27B` | — | ladder decided and geometry verified (§B.1). ~18 GB + ~54 GB bf16 against 1.5 TB free. Optional 4th: `Qwen3.8-27B` as the generation control. |
| **T0.4** | Stand up the benchmark suite | T0.1 | §D; every existing result is scored at n=24–48, which cannot resolve anything |
| **T0.5** | Pre-register screen verdicts | T0.0 | §A.7; without a timestamped commit, §7 carries no weight |
| **T0.6** | Re-baseline headline claims | T0.1, T0.4 | the harness changed materially (a discarded generation sweep was removed from the import path, moving the RNG stream). Pre-fix and post-fix numbers are not comparable by construction. |
| **T0.7** | Measure the 27B once | T0.3 | the 4B is on record as a "mechanism testbed only" (−62% rel. MMLU vs the 27B's −18%). Until this exists we do not know which paper we are writing. |

### Tier 1 — needed for the systematic-overview framing

* mid-size rung acquired and full pipeline run
* method × scale matrix at the promoted seed budget
* dilution-vs-superposition decomposition (B.1) at all three rungs
* out-of-sample screen test against the register (§7)
* matched-bpw baselines — "method X doesn't help" is much weaker than "X doesn't beat spending the
  same bits uniformly"

### Tier 2 — optional

* second model family (roughly doubles tier 1; alternative is to narrow the title explicitly)
* additional bit regimes beyond the screen test set

---

## G. Risks and open questions

* **Which mid-size rung.** Driven by T0.3 — availability within a single generation matters more than
  hitting exactly 9B.
* **Full matrix at 27B, or survivors only?** Cost says survivors; the method×scale claim weakens if
  the 27B column is sparse. Compromise: run at 27B only the methods that span the screen's decision
  boundary.
* **Is one family enough?** If yes, the title narrows to name it. That is a real cost to generality
  and should be paid consciously, not discovered in review.
* **Search reproducibility.** Record query strings, dates run, and result counts. A search nobody can
  re-run is the same defect as an experiment nobody can re-run.
* **Format contract for other bit depths.** The screen is stated against TQ1_64; generalising it needs
  the constraint restated in format-independent terms.
* **Negative-results venue risk.** A mostly-null paper needs a venue that will take it. The screen's
  out-of-sample hit rate is the positive result that makes it publishable — which is another reason
  §A and §7 are load-bearing rather than decorative.

---

## H. Provenance of the numbers used above

Every figure in this plan traces to `logs/RESULTS_SUMMARY.md` or to a check run on 2026-09-14:

| number | source |
|---|---|
| 0.309 pp skeleton SD; 0.103 / 0.012 commit SD | §13ac, §13ah |
| 78.50% Gate A pass vs 0/18 arithmetic | §13am, §13ao |
| block-MSE failures (99.8%, wrong-way) | §13 batch summary, SESSION STATUS §D |
| head upper bound 0/18, McNemar p=1.000 | §13ao |
| think_len 1.279 → 0.646 across seeds | §13at |
| 29.1% / 9.6% embed+head; vocab 248,320 | computed from configs, 2026-09-14 |
| Qwen3.5 vs Qwen3.6 generation mismatch | HF cache inspection, 2026-09-14 |
| 0 tracked drivers vs 54 tracked src files | `git ls-files`, 2026-09-14 |
| arXiv:2609.09240 figures (1.64 bpw, 8.24 GiB, n=500, +8.9 retention) | §13ar |
