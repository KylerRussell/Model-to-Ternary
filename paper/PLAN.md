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

### B.1 The scale ladder

Methods that help at 4B and vanish at 27B would be a genuinely novel finding; the literature has no
answer because almost nobody runs the same method at more than one scale. The closest prior work
(arXiv:2609.09240) ran two points and correctly declined to call it a scaling law. **Three points can
at least separate monotone from non-monotone.**

| rung | checkpoint | layers | hidden | vocab | embed+head | status |
|---|---|---|---|---|---|---|
| 4B | `Qwen3.5-4B` | 32 | 2560 | 248,320 | **29.1%** | cached |
| ~9B | *to be chosen* | — | — | — | ~18% | **missing** |
| 27B | `Qwen3.6-27B` | 64 | 5120 | 248,320 | **9.6%** | cached, never run |

#### BLOCKER: the generation confound

**The cached 4B is Qwen3.5; the 27B is Qwen3.6.** Any scale claim from that pair confounds scale with
model generation. The mid-size rung is the opportunity to fix it — choose all three rungs from one
generation, re-anchoring the 4B if necessary. Discovering this after the runs would invalidate the
paper's central matrix.

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
| **T0.1** | **Track the experiment drivers** | — | `output_sweep/` is caught by the `output*/` ignore rule, so **0 driver scripts are versioned** against 54 tracked source files. Every experiment is currently unreproducible by anyone, including us after a disk failure. |
| **T0.2** | External reproduction gate | T0.1 | a reviewer cannot otherwise distinguish "these methods don't work" from "your pipeline is broken" — and this project has shipped a silently zeroed FP teacher that passed three sanity checks. Target: within ~0.05 pts of a published number. |
| **T0.3** | Fix the ladder's generation confound | — | blocks §6; discovering it later invalidates the central matrix |
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
