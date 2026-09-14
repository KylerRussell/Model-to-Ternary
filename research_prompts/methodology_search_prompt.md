# Deep research prompt — systematic census of sub-2-bit weight quantization methods

**This is a CENSUS, not a recommendation request.** We are not asking "what should we try next." We
are building the denominator for a systematic review: *how many published sub-2-bit weight
quantization methods exist, and what fraction of them can even be run under a fixed deployment
format?*

So: **completeness beats relevance.** A method you are confident will not work for us still belongs
in the register, with the reason recorded. Do not pre-filter to things you think we would like.

---

## 0. What we are doing, so you can judge what matters

We ran ~13 published low-bit methods against **one fixed deployment format** and almost none produced
a resolvable improvement. The failures were not random — five of six shared a single cause:

> **the method assumes a richer parameterisation than the deployment format provides.**

We want to know whether that generalises across the literature. That requires a complete-as-possible
census, a screen verdict recorded for each entry **before** we run anything, and then a measured
out-of-sample hit rate.

## 1. The format (the constraint every method is screened against)

* **TQ1_64**: ternary `{-1, 0, +1}` × **one positive scale per (row, 64-weight block)**, **1.7812
  bpw**, self-contained 512-weight superblocks, strided GEMV.
* **No zero-point. No per-weight side tensors. No variable-length / entropy codes.**
* **Weight-only.** Activations are bf16. There is no activation quantizer anywhere in the codebase.
* QuaRot Hadamard rotation folds the RMSNorm affine into the following linear (stored norm weights
  are exactly 0).
* **embed_tokens and lm_head are BOTH ternarised** for footprint parity.
* Hardware: 2× RTX 3090 (24 GB, no NVLink), ~200 GB RAM, **no NVMe** — do not propose disk offload.

## 2. Scope — what to include

Include a method if **all** of these hold:

1. it operates on the **weights** of a transformer LLM
2. it targets an effective weight budget of **≤ 2 bits/weight** (by the paper's own accounting — see §4)
3. it reports some empirical result (capability, perplexity, or reconstruction)
4. it is described in enough detail to apply the screen in §3

Exclude, **recording the reason**:

* activation-only or KV-cache-only quantization
* ≥ 3-bit targets
* results with no recoverable procedure
* superseded versions (keep the latest; note the chain)

### Sources

arXiv (`cs.LG`, `cs.CL`, `cs.AI`); ACL/EMNLP/NeurIPS/ICLR/ICML proceedings; Semantic Scholar.
**Do forward and backward citation chasing** from these anchors — it catches work that uses none of
our query terms: BitNet, BitNet b1.58, GPTQ, QuaRot, AWQ, OmniQuant, ParetoQ, AQLM, PTQTP, CAT-Q,
TWLA. Include vendor/industry technical reports; that is often where deployment-realistic numbers
live.

### Query terms (run independently, union the results)

```
ternary quantization LLM       1.58-bit                    sub-2-bit LLM
1-bit LLM / binary LLM         extreme low-bit quantization W2A16 / W1.58A16
trit-plane                     post-training quantization ternary
low-bit quantization-aware training                        weight-only 2-bit
sub-1-bit LLM                  vector quantization LLM weights
ternary weights transformer    binarized language model
```

## 3. The screen — apply to every entry

**State explicitly: what quantity does the method need to vary, and does TQ1_64 allow it?**

Verdict one of: **PASS** (runnable as published) / **NEEDS-ADAPTATION** (give the adaptation and say
what the paper's reported gain depends on that we would be dropping) / **FAIL** (say which constraint
it violates).

Real examples of FAIL from our own batch, so you can calibrate:

| method | assumed | our reality |
|---|---|---|
| NAP | tunable normalization affines | QuaRot folds them away; stored `w ≡ 0` |
| QUASAR | codes and dequantizer stored separately | **one** scale does both jobs |
| SQuaT | a student feature/activation lattice | weight-only; activations are bf16 |
| SoftWater, ECASQ | entropy coding | variable-length codes break the strided GEMV |
| E2M-ATQ (2609.09240) | a nonzero per-row offset `μ`, a second ternary plane, salience masks | no zero-point, one scale, uniform superblock layout |

A second screen, which killed three candidates from a previous search: **does the method assume a
localizable first error** (restart-from-first-wrong-step, mixed-policy distillation)? We measured our
first confident divergence at **position 0.27** of the generated sequence — there is nothing to
localize, so those methods degenerate to ordinary SFT here.

## 4. The bits-accounting requirement — do this for EVERY entry

Papers report bits/weight on wildly different bases and it is the single most misleading number in
this literature. arXiv:2609.09240 reports **1.64 bpw** but ships an **8.24 GiB artifact for an 8B
model (~8.85 bits/parameter)**, because its budget covers linear projections only and leaves
embeddings, LM head, norms and KV cache at FP16. Ours counts everything, including a ternarised head.

**Record three numbers per method:**

1. the paper's **claimed** bits/weight
2. **what that figure covers** (which tensors are excluded — embeddings? head? norms? KV?)
3. the **implied whole-checkpoint bits/parameter**, computed if not stated

If the paper does not give enough information to compute (3), say so — that is itself a finding.

## 5. Already resolved — include in the register, but mark as TESTED

Do not spend search effort rediscovering these; do record them so the census is complete.

OPSA (adopted) · AYOT (independently implemented here) · ICBQ (wash) · CAT-Q (wash) · NAP (−7.6 SD) ·
QUASAR (−33 SD) · SchurQuant (diverged) · SoftWater · ECASQ · FlashQuant · ExTernD · SQuaT · LCD ·
AWSRC · GPTQ · AdaRound · CDQuant · QuaRot · QEP · EfficientQAT/block-AP · CAKLD · TALR ·
E2M-ATQ/KOTMS (2609.09240).

## 6. Output format — one row per method

```
ID / citation (arXiv id + title + date) / venue
Bit regime:            ternary | 1-bit | sub-1-bit | other ≤2-bit
What it varies:        (the screen question)
Screen verdict:        PASS | NEEDS-ADAPTATION | FAIL  + reasoning
Bits accounting:       claimed / coverage / implied whole-model
Evidence type:         free-gen capability | MC benchmark | perplexity | reconstruction only
Scales evaluated:      (model sizes in the original paper)
Multi-scale?           does the paper run the SAME method at >1 size?
Notes:                 anything a re-implementer would need
```

## 7. Two questions we especially want answered from the census

1. **How many papers evaluate the same method at more than one model scale?** We suspect it is very
   few. If so, that is a quantified gap statement for our introduction, not a hunch.
2. **What fraction of methods report free-generation capability** versus perplexity/reconstruction
   only? Our experience is that reconstruction metrics never once caught a real failure — block-MSE
   improved 99.8% on a model whose residual stream had collapsed. A census-level number would turn
   that anecdote into a claim about the field.

## 8. What a good answer looks like

Completeness with reasons, not a shortlist. A register of 40 methods with 30 marked FAIL and the
constraint named for each is far more valuable to us than 5 promising candidates — because the
fraction that fails **is the paper's result**.

Also report your search process: queries run, dates, result counts at each screening stage
(identified → screened → full-text → included). A search nobody can re-run has the same defect as an
experiment nobody can re-run.
