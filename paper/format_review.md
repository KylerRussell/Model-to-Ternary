# Review of the format-space census (2026-09-14)

Screening of the returned format census. **This one is well grounded** — unlike the method census, its
central claims survive independent checking, and its conclusion is uncomfortable for us.

---

## 1. Verified independently

Block sizes read from the `gguf` package installed in this repo, against the report's claims:

| type | actual blk/bytes | actual bpw | report | |
|---|---|---|---|---|
| `TQ1_0` | 256 / 54 | 1.6875 | 54 B, 1.6875 | ok |
| `TQ2_0` | 256 / 66 | 2.0625 | 66 B, 2.0625 | ok |
| `IQ1_S` | 256 / 50 | **1.5625** | 50 B, 1.5625 | ok |
| `IQ1_M` | 256 / **56** | 1.7500 | **64 B**, 1.7500 | **byte count wrong** (bpw right) |
| `IQ2_XXS` | 256 / 66 | 2.0625 | 66 B, 2.0625 | ok |
| `IQ2_XS` | 256 / 74 | 2.3125 | 74 B, 2.3125 | ok |
| `IQ2_S` | 256 / 82 | 2.5625 | 82 B, 2.5625 | ok |
| `Q2_K` | 256 / 84 | 2.6250 | 84 B, 2.625 | ok |
| `Q1_0` | 128 / 18 | 1.1250 | 18 B, 1.125 | ok — **and it exists** |

8 of 9 exact. The one error is internally inconsistent rather than wrong in conclusion (56 B *is*
1.75 bpw). Compare the method census, where two of our own internal acronyms were bound to unrelated
papers. **Treat this register as usable; treat that one as unverified.**

## 2. The finding that hurts: TQ1_64 may be indefensible

| | bpw | scale granularity | alphabet |
|---|---|---|---|
| **TQ1_64 (ours)** | **1.7812** | g64, 8-bit sub-scales | full ternary |
| `TQ1_0` | 1.6875 | g256 | full ternary |
| **`IQ1_S`** | **1.5625** | **g32** (3-bit) + g256 super | **2048 of 6561** 8-D ternary vectors |

**`IQ1_S` is 12.3% smaller than TQ1_64 and has finer scale granularity than ours.** Our entire
published justification for TQ1_64 is that granularity matters — 79.31% (g256) → 80.62% (g128) →
82.06% (g64) eval2k agreement. IQ1_S goes *further* in the direction we argued for, at *lower* cost.

The trade it makes is on the alphabet: IQ1_S keeps only 2048 of the 6561 possible 8-dimensional
ternary vectors, discarding 68.8% of the space, and compensates with a ±0.125 grid displacement. So
the real question is empirical: **does finer scaling on a restricted alphabet beat coarser scaling on
the full alphabet?** We have argued half of that question and never tested the other half.

The report asserts IQ1_S delivers "superior perplexity recovery" but **attaches no measurement**. That
is exactly the kind of claim this project does not accept on faith — from others or from itself.

### The required experiment

Simulate IQ1_S in PyTorch the way `src/sim_tq164.py` simulates our own format, quantize the same
model, and score on the same harness. Direct, affordable, decisive.

**One design trap to avoid:** IQ1_S quality depends heavily on llama.cpp's **imatrix** (activation
importance) calibration, `H_ii ≈ Σ_k X_ik²`. A naive PyTorch simulation without importance weighting
would understate IQ1_S and hand us a flattering, false result. The comparison must either give IQ1_S
its imatrix or give both formats the same importance weighting. Rigging this in our favour would be
worse than not running it.

### If IQ1_S wins

Non-trivial consequence: our pipeline trains ternary **assignments** with STE and per-block scales.
Targeting IQ1_S means the assignment problem becomes "choose one of 2048 codebook entries per 8
weights" — a different optimisation, not a re-parameterisation. Cost that honestly before committing.

## 3. Errors and overstatements — do not propagate

* **`IQ1_M` block size**: 56 bytes, not 64 (the report's own bpw figure implies 56).
* **AQLM "1.00 bpw (1×16)"**: a 16-bit index over an 8-dimensional group is **2 bpw**, not 1. The
  headline AQLM configurations are ~2 bpw. Treat as wrong unless a group size >8 is specified.
* **IQ1_S "zero-point: Yes"** overstates. The ±0.125 δ is a one-bit grid displacement, not a general
  per-block offset, and it sits in the same matrix column as `Q2_K`'s genuine 4-bit `d_min`. Those
  are different capabilities and screening on the column as written would mislead.
* **CATEGORY ERROR — multi-plane.** The report marks `IQ2_XXS`/`IQ2_XS` as "Multiple Planes: Yes
  (magnitude + sign)" and concludes multi-plane methods like PTQTP "map directly onto" them. They do
  not. A sign plane carries the sign *of the same magnitude*; multi-plane superposition is
  `W ≈ α₁T₁ + α₂T₂`, two independent discrete terms with independent scales. The valid mapping in
  that paragraph is **AQLM's additive VQ**, which genuinely is a sum of codebook lookups. Keep the
  AQLM claim, drop the IQ2 one.

## 4. Genuine results worth keeping

* **Variable-length entropy coding fails under every shipping runtime.** Not one production engine
  implements it, and the reason is structural — the bit offset of symbol *k+1* depends on decoding
  symbol *k*, so parallel strided reads are impossible without multi-pass prefix sums. This is the
  one FAIL class that is a property of parallel hardware rather than of a format choice, and the
  paper can state it as such.
* **Sparse outlier side-tensors are runtime-confined.** Research kernels exist (`spqr_cuda`), but
  GGUF, MLX, Marlin and vLLM have not adopted them. Production handles outliers by Hadamard
  equalisation or mixed-bitrate sub-blocks instead. A weaker claim than "impossible", and the honest
  one.
* **Coupled activation quantization is infeasible *on this hardware*** — Ampere GA102 has no sub-INT4
  tensor units, and our Sandy Bridge-EP host lacks the AVX2 that `bitnet.cpp`'s TL2 kernel needs.
  Note the scope: a property of our machine, not of the method class.
* **The zero-point algebra**, which is the cleanest technical point in the report:
  `Σ(s·qᵢ − m)yᵢ = s·Σqᵢyᵢ − m·Σyᵢ`. At batch 1 the activation sums are computed once per token and
  reused across rows, so the offset costs one scalar MAC per block, not a per-weight subtraction.
  That single identity dismantles the "zero-points destroy addition-only GEMV" claim.
* **BitNet's real deployment footprint**: 1.58 bpw covers BitLinear layers only. `BitNet-b1.58-2B-4T`
  ships at **3.73 bpw tied / 5.36 bpw untied** whole-model, because a 128256×2560 embedding is 656 MB
  in FP16. Independent confirmation of our own vocabulary-penalty finding, on the flagship native
  ternary model.

## 5. The strongest paper finding in either census

**The most-deployed sub-2-bit formats have almost no academic literature.** The IQ family drives the
majority of local consumer LLM inference through llama.cpp, and was developed empirically — greedy
coordinate descent against a diagonal-Hessian `imatrix` proxy — rather than through the block-wise
second-order machinery the literature favours. Meanwhile the academic work (AQLM, QuIP#, VPTQ,
SpinQuant) ships its own bespoke CUDA and is barely represented in the ecosystem people actually run.

That is the same thesis as our §8, one level up: **the literature optimises away from deployment.**
It is also directly checkable — count published methods targeting IQ formats. We expect ~zero.

## 6. Required actions

1. **Run the IQ1_S vs TQ1_64 comparison** (§2), with matched importance weighting. It may end the
   case for our format, and we should want to know.
2. Re-derive the format claim in `PLAN.md` §B.0 once that number exists.
3. Drop the IQ2/multi-plane mapping; keep the AQLM one.
4. Fix `IQ1_M` to 56 B and the AQLM bpw before anything is cited.
5. Add "published methods targeting IQ formats" as a census count (§5).
