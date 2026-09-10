# Research prompt — recovering multi-step REASONING at 1.78 bpw, not reconstruction error

**Ask:** how do we close a *capability* gap — multi-step arithmetic that survives teacher-forced
evaluation but collapses in free generation — on a ternary LLM at a fixed 1.78 bpw deployment format?

We have just closed a 13-paper batch in which **none of 13 methods produced a resolvable
improvement**, and a decoding-time campaign that fixed the *behaviour* (looping, compression,
commitment) without touching the *capability*. Do not send us more layer-wise reconstruction work.
Read §0 and §1 before proposing anything.

Numbers in **bold** are measured on this project and are what a proposal must beat or respect.

---

## 0. THE SCREEN — apply to every candidate before proposing it

Five of six method failures last batch shared one cause: **the method assumes a richer
parameterisation than our deployment format provides.**

| method | assumed | our reality |
|---|---|---|
| SchurQuant | a continuous suffix absorbing chunk error | the suffix is ternarised too |
| NAP | tunable normalization affines | QuaRot **folds them away**; stored `w == 0` |
| CAT-Q | a stably learnable scale inside a soft map | ill-conditioned here (7.7e6 gradient gap) |
| QUASAR | codes and dequantizer stored separately | **one** scale does both jobs |
| SQuaT | a student feature/activation lattice | weight-only; activations are bf16 |

**State explicitly for each proposal: what quantity does it need to vary, and does TQ1_64 allow it?**
Four of those five were diagnosable on paper or by a short unit test. If a method needs a zero-point,
a per-weight code stored apart from its scale, an activation quantizer, an unfolded norm affine, or
variable-length codes, say so and either drop it or price the adaptation.

## 1. THE PHENOMENON — this is the whole question

A ternary model can be **simultaneously**:

| measurement | 4B ternary | FP teacher |
|---|---|---|
| eval2k top-1 agreement (teacher-forced) | **~70%** | — |
| Gate A (teacher-forced, deploy bar >=77%) | **78.50% PASS** | — |
| GSM8K accuracy **given it finished reasoning** | **7.1%** (2/28) | **93.3%** (14/15) |

Same model, same decoding config, same 48 problems. **Teacher-forced next-token agreement of 70%
coexists with ~7% multi-step arithmetic.** Per-token agreement does not compose across a 600-token
reasoning chain, and arithmetic is unforgiving — one wrong intermediate value destroys the answer.

**Explain and attack THIS.** Not perplexity, not layer-wise MSE, not reconstruction error. Across our
last batch, local reconstruction metrics **never once** caught a real failure: block-MSE improved
99.8% on a model whose residual stream had collapsed, improved while another method cost -50.5 pp
end-to-end, and pointed the *wrong way* on a method that cost -10.3 pp.

### Scale caveat you must respect

The numbers above are the **4B mechanism testbed, which is known-catastrophic**:

| | ternary vs FP (MMLU-Pro) | relative |
|---|---|---|
| 4B | 18.2 vs 47.7 | **-62%** |
| **27B (unoptimised)** | 50.3 vs 61.7 | **-18%** |

**27B absorbs ternary ~3.4x better and is the actual target.** The 4B is also the worst case for the
output-head pathology: embed+head is **24.0%** of the 4B vs **9.2%** of the 27B. A proposal whose
value is specific to small models is not useful; a proposal that explains the 3.4x scale difference
is very useful.

## 2. What we deploy (hard constraints)

* **TQ1_64**: ternary `{-1,0,+1}` x **one positive scale per (row, 64-block)**, **1.7812 bpw**,
  self-contained 512-weight superblocks, strided GEMV in `llama.cpp`. **No zero-point, no per-weight
  side tensors, no variable-length/entropy codes** (this is why ECASQ and SoftWater were rejected).
* **Weight-only.** Activations bf16. **No activation quantizer exists anywhere in the codebase.**
* **QuaRot Hadamard rotation** in Phase 1 folds the RMSNorm affine into the following linear (stored
  norm weights are exactly 0) and removes channel outliers.
* **embed_tokens and lm_head are BOTH ternarised** for footprint parity — each is [248320, 2560].
  Known cost table for un-ternarising the head: `q4_K head+embed` = 2.0300 bpw (+14.0% size at 27B);
  `int4 + g64 fp16 scales` = 2.0072 bpw (+12.7%). **A proposal may spend bpw, but must say how much
  and justify it against simply using more bits everywhere.**
* **Hardware**: 2x RTX 3090 (24 GB, PHB, no NVLink), Sandy Bridge-EP host (AVX only, no AVX2/FMA),
  ~200 GB RAM, **and NO NVMe** — do not propose disk offload.
* **Sequence length is fixed at 2560** (set by where the teacher emits end-of-thought).

## 3. The pipeline and what each stage is worth

`QuaRot rotate -> CoT-aware calibrate -> GPTQ + block-AP QAT skeleton -> assignment training (STE,
full-latent) -> E2E distillation -> TQ1_64 GGUF export`

| stage | eval2k agreement | mean KL |
|---|---|---|
| skeleton (block-AP), n=3 | **56.273 +/- 0.309 %** | 1.1897 +/- 0.0155 |
| + E2E distillation | **~70.7 %** | 0.626 |

**E2E is by far the largest lever ever measured here** — bigger than all 13 paper methods combined,
every one of which was a wash or a negative. A proposal that improves the E2E stage is worth more
than one that improves the skeleton.

Calibration is already CoT-aware: **50.0% of calibration sequences carry `<think>`** (self-generated
teacher rollouts, 50/50 with generic replay). Block-AP's objective already **is** hidden-state MSE at
every layer; adding feature-KD at E2E is a measured **wash** (70.89% -> 70.50%).

## 4. Measurement bar

* Skeleton A/B noise floor: **SD 0.309 pp**; a single run resolves ~0.93 pp and nothing finer.
* Free-generation gate rates are far worse: **commit_rate SD 0.103 across seeds**, ~9x the SD of the
  same quantity measured as a PAIRED delta (0.012). "Helps by X" survives on pairing; "passes the bar"
  needs seeds.
* **State the expected effect size.** Anything under ~1 pp on a 4B skeleton is not testable here
  without a multi-seed campaign.
* Beware headline gains that came from rescuing a BROKEN baseline (e.g. a sequential-CBQ baseline at
  >1e4 PPL). Ours is not broken. Discount such gains explicitly.

## 5. Already tried — do NOT re-propose

**13-paper batch, all resolved:** OPSA (adopted) · AYOT (**already implemented here independently**)
· ICBQ (wash) · CAT-Q (wash) · NAP (**-7.6 SD**) · QUASAR (**-33 SD**) · SchurQuant (diverged) ·
SoftWater, ECASQ (entropy coding — format-incompatible) · FlashQuant (outliers already removed by
QuaRot) · ExTernD (needs >=5.2 bpw) · SQuaT (**null by construction** — needs a feature lattice we do
not have) · LCD, AWSRC (parked — custom GEMV).

**Already in the pipeline:** GPTQ + act-order · block-AP / EfficientQAT block reconstruction ·
AdaRound · CDQuant · QuaRot · QEP error propagation · per-input-channel `--col-scale` folded into the
norm gain · learned per-block scales with LSQ gradient scaling · 8-bit scale QAT · ternary embed +
GPTQ lm_head · CAKLD / top-k logit KD · commit-weighted loss · on-policy student rollouts · TALR
flip-rate servo · full-latent STE assignment training · 1F1B pipeline parallelism.

**Decoding-time, already done:** DRY sampler (loop -0.43, 3/3 seeds; stock settings optimal — gentler
settings measured WORSE on both loop and commit) · `</think>`-row logit gain (commit +0.097, 3/3
seeds). Both **0 bpw**, both adopted, and **neither changes accuracy**: acc|closed is flat at
0.00 / 0.059 / 0.071 across the gain sweep. **Decoding control cannot fix this — do not propose more
samplers.**

## 6. Questions we most want answered

1. **Why does 70% teacher-forced agreement give ~7% multi-step accuracy, and what predicts it?** Is
   there a published measure of *reasoning-chain* survival under quantization better than per-token
   agreement or KL? We need a metric that correlates with free-gen accuracy — ours provably do not.
2. **What explains the 3.4x scale advantage (27B -18% vs 4B -62%)**, and does it imply the 27B will
   not show this pathology, or only show it later?
3. **Where should bpw be spent, if anywhere?** The head is 24% of the 4B / 9.2% of the 27B and
   un-ternarising it costs +14% size. Is there evidence that selective precision on specific
   components (head, embed, first/last blocks, attention vs MLP) buys back *reasoning* specifically —
   as opposed to perplexity?
4. **Is arithmetic/multi-step reasoning recoverable by TRAINING at fixed format** — RL on verifiable
   answers, self-consistency distillation, process supervision, or long-CoT-specific objectives —
   given we already run on-policy rollouts and CAKLD logit KD in E2E?
5. **Is there a quantization-aware approach that targets error ACCUMULATION over a generated sequence**
   rather than per-token or per-layer error? That is the mechanism we think is at fault and the one
   nothing we tried addresses.

## 7. What we want back, per candidate

1. **Citation** (arXiv id + title + date), 2025-2026 preferred, not in §5.
2. **The §0 screen**: what must it vary; does TQ1_64 permit it; if adaptation is needed, what does the
   paper's reported gain depend on that we would be dropping?
3. **Which stage** it attaches to (§3), and whether it can be screened at the **skeleton** (~2.4 h),
   needs the **assignment stage** (~28 h), or **E2E** (~1 h/arm at 1000 seqs).
4. **Expected effect size on free-generation accuracy**, against §4's bars — and say plainly if the
   paper's evidence is only perplexity/reconstruction, because that has never predicted anything here.
5. **A falsifier**: the cheapest measurement that shows it is NOT working. Prefer a structural
   invariant (sparsity histogram, activation-norm profile, gradient magnitudes, think-length vs the
   teacher) over a loss curve — in this project loss curves have caught **zero** of the failures and
   invariants caught **all** of them.
6. **bpw cost**, and its justification against simply using more bits.

A well-argued "the literature does not address sequence-level error accumulation at sub-2-bit, here is
the closest adjacent work and why it falls short" is more valuable than a list of PTQ papers.
