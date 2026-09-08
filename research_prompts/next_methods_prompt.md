# Research prompt — what is left that could actually move a 1.78 bpw ternary reasoning LLM

**Ask:** find recent methods (2025–2026) that would plausibly improve THIS pipeline, and screen them
against the constraints below **before** proposing them. We have just run a 13-paper batch in which
**none produced a resolvable improvement**, and the failures were not random — they share one cause.
A proposal that repeats that cause is worse than no proposal.

Everything below is measured on this project unless marked as an estimate. Numbers in **bold** are
what a proposal must beat or respect.

---

## 0. THE SCREEN — apply this first, to every candidate

Five of the last six failures had the same root cause: **the method assumes a richer parameterisation
than our deployment format provides.**

| method | assumed | our reality |
|---|---|---|
| SchurQuant (2608.15567) | a *continuous* suffix that absorbs each chunk's error | the suffix is ternarised too |
| NAP (2608.03919) | tunable normalization affines | QuaRot **folds them away**; stored `w ≡ 0` |
| CAT-Q (2608.01078) | a stably learnable scale inside a soft map | our scale is ill-conditioned there (7.7e6 gradient gap) |
| QUASAR (2608.13966) | codes and dequantizer stored **separately** | **one** scale sets both |
| SQuaT (2608.10709) | a student **feature/activation** lattice | weight-only; activations are bf16 |

**For every method you propose, state explicitly: what quantity does it need to vary, and does
TQ1_64 (§1) let us vary it?** Four of those five were diagnosable on paper or by a 10-line unit test.
If a method needs a zero-point, a per-weight code stored apart from its scale, an activation
quantizer, an unfolded norm affine, or anything other than *one positive scale per (row, 64-block)*,
say so and either drop it or give the exact adaptation and what the adaptation costs.

---

## 1. What we deploy (hard constraints)

* **Format: TQ1_64.** Ternary `{-1,0,+1}` × **one positive fp scale per (row, 64-weight block)**.
  **1.7812 bpw**, uniform, self-contained, fixed 512-weight superblock, strided GEMV in `llama.cpp`.
  **No zero-point. No per-weight side tensors. No variable-length codes.** Any method requiring
  entropy coding is dead on arrival — that is why ECASQ (2608.18147) and SoftWater (2608.12026) were
  rejected: variable-length codes break the strided GEMV.
* **Weight-only.** Activations stay bf16. There is **no activation quantizer anywhere in the repo.**
* **Phase 1 is QuaRot** (Hadamard rotation). This **folds the RMSNorm affine into the following
  linear**, so stored norm weights are exactly `0` (the norm is zero-centered: `y = x̂·(1+w)`).
  Rotation exists to remove channel heterogeneity — so methods that feed on per-channel structure
  have had that structure deliberately removed upstream.
* **Sequence length is fixed at 2560.** Set by where the teacher emits end-of-thought tokens. Not
  negotiable; do not propose shortening it.
* **Hardware: 2× RTX 3090 (24 GB, PHB, no NVLink), Sandy Bridge-EP host (AVX only — no AVX2/FMA),
  ~200 GB RAM, and NO NVMe.** Do not propose disk offload of latents, optimizer state, or snapshots.

## 2. The pipeline, and where each stage lands

`QuaRot rotate → calibrate → GPTQ + block-AP QAT skeleton → assignment training (STE) → E2E
distillation → TQ1_64 GGUF export`

Measured on the 4B testbed (eval2k, 1946 sequences, teacher-forced, deterministic):

| stage | top-1 agreement | mean KL(fp‖tern) |
|---|---|---|
| skeleton (block-AP), **n=3** | **56.273 ± 0.309 %** | **1.1897 ± 0.0155** |
| + E2E distillation | **~70.7 %** | **0.626** |

**Calibration is already CoT-aware**: 50.0 % of calibration sequences carry `<think>`, 47.5 % carry
`</think>` (self-generated teacher rollouts mixed 50/50 with generic replay). This was derived here
independently and is the same idea as AYOT (2608.01078) — do not re-propose it.

**block-AP's objective already IS hidden-state MSE against the FP block, at all 32/64 layers.**
Feature-level matching is structurally present before E2E runs; adding it again at E2E is a measured
wash (70.89 % → 70.50 %, KL 0.6262 → 0.6259).

## 3. THE ACTUAL PROBLEM — read this before proposing anything

The full pipeline **passes the teacher-forced gate and fails the free-generation gate.**

* Gate A (eval2k top-1 agreement, teacher-forced, deterministic): **78.50 %** vs a ≥77 % bar → **PASS**
* Gate B (free generation, sampled, seeded): **FAIL**

| Gate B metric | bar | best measured |
|---|---|---|
| loop rate | ≤ 0.30 | **0.5625** |
| commit rate | ≥ 0.68 | **0.4250** |
| compression ratio | ≤ 3.1 | **3.8287** |

Those best numbers are **with** OPSA (2608.31046) already applied — a confirmed win over 5/5 seeds
(loop 0.7083→0.5625, commit 0.3417→0.4250, comp 4.8567→3.8287, all p<0.05) and still not enough.

**Teacher-forced metrics are proven blind here.** They have read 79–84 % on models that scored below
random in free generation. So:

> **The highest-value thing you can find is a method that improves FREE-GENERATION behaviour at
> extreme low bit — looping, early commitment, degenerate repetition — not one that lowers layer-wise
> reconstruction error.**

We have strong evidence that reconstruction error is the wrong target: across this batch, local
metrics **never once** caught a real failure. Block-MSE *improved* while one method destroyed the
model end-to-end (−50.5 pp), *improved 99.8 %* on a model whose residual stream had collapsed, and
improved 10–15 % at depth on a method that cost −10.3 pp. Do not propose methods whose evidence is
purely layer-wise reconstruction MSE or perplexity.

## 4. Measurement bar (why small claims are unusable)

Re-running the identical skeleton config with only a different RNG seed gives **SD 0.309 pp** on
agreement (n=3: 56.11 / 56.63 / 56.08). **A single-run A/B here cannot resolve anything below
~0.93 pp / 0.0465 KL.** A method whose reported gain is a fraction of a point is not testable here
without multi-seed runs. **State the expected effect size**; if it is under ~1 pp on a 4B skeleton,
say so, because that changes whether it is worth a 2.4 h screen or a 3-seed 7 h campaign.

Note also that the paper batch's reported gains mostly came from **rescuing a broken baseline**
(e.g. sequential-CBQ Qwen3-8B at >1e4 PPL). Our baseline is not broken. Gains conditioned on a
catastrophic baseline will not transfer; discount them explicitly.

## 5. Already tried — do NOT re-propose

**Tested, resolved (this batch):** OPSA 2608.31046 (**adopted**, still short of Gate B) · AYOT
2608.01078 (**already implemented independently**) · ICBQ 2608.09595 (wash, +0.89 SD) · CAT-Q
2608.01078 (wash once its scale is frozen) · NAP 2608.03919 (**−7.6 SD**) · QUASAR 2608.13966
(**−33 SD**) · SchurQuant 2608.15567 (negative / diverged) · SQuaT 2608.10709 (null by construction)
· SoftWater 2608.12026, ECASQ 2608.18147 (entropy coding — format-incompatible) · FlashQuant
2608.15531 (outliers already removed by QuaRot) · ExTernD 2607.13511 (needs ≥5.2 bpw) · LCD
2608.11786, AWSRC 2608.23144 (**parked** — both need custom runtime GEMV paths that break the
standalone TQ1_64 kernel).

**Also already in the pipeline (do not propose):** GPTQ + act-order · block-AP / EfficientQAT block
reconstruction · AdaRound · CDQuant coordinate descent · QuaRot/Hadamard rotation · QEP error
propagation · per-input-channel `--col-scale` folded into the norm gain · learned per-block scales
with LSQ gradient scaling · 8-bit scale QAT · ternary embed + GPTQ lm_head · CAKLD / top-k logit KD ·
commit-weighted loss · on-policy student rollouts · TALR flip-rate servo · 1F1B pipeline parallelism.

## 6. What we want back

For each candidate, in priority order by **expected effect on Gate B**:

1. **Citation** (arXiv id + title + date). Prefer 2025–2026. Must not be in §5.
2. **The screen (§0)**: what it needs to vary, and whether TQ1_64/weight-only/QuaRot allows it. If it
   needs adaptation, give the adaptation and say what the paper's reported gain depends on that we
   would be dropping.
3. **Which stage it attaches to** in §2, and whether it can be screened at the **skeleton** (2.4 h) or
   requires the assignment stage (~28 h) or E2E (~1 h/arm at 1000 seqs).
4. **Expected effect size** against §4's 0.93 pp resolution limit, and on which metric — say plainly
   if the evidence is only reconstruction/perplexity (§3).
5. **Falsifier**: the cheapest measurement that would show it is NOT working — ideally a structural
   invariant (sparsity histogram, activation-norm profile, gradient magnitude), not a loss curve.
   In this project those caught every failure and the training loss caught none.
6. **bpw cost.** Anything above 1.7812 bpw must justify itself against the alternative of simply
   using more bits.

Breadth is welcome beyond PTQ/QAT: decoding-time methods, sampling/repetition control, self-distillation
or RL-style objectives for free-generation quality, and anything targeting long-CoT degradation under
compression are all in scope **provided they survive §0 and need no format change.** A well-argued
"nothing in the literature addresses this; here is why" is a more useful answer than a list of
reconstruction-error papers.
