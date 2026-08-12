# TQ1_64 — ternary weight format, 1.7812 bpw

Reference implementation: `src/tq164.py` (`encode_row` / `decode_row`). **The ggml/CUDA kernels must reproduce
`decode_row` bit-for-bit.** The Python packer is verified bit-exact (`max|diff| = 0`) against
`src/sim_tq164.py`, which was validated end-to-end on the real model (see "Validation" below).

## Why this format exists

`TQ2_0` (2.0625 bpw) and `TQ1_0` (1.6875 bpw) both use **QK_K = 256 with a single scale per block (g256)**.
Our model is trained at **g64** because finer scale granularity measurably wins:

| granularity | eval2k agreement | note |
|---|---|---|
| g256 (= TQ1_0 / TQ2_0) | 79.31% | what the stock formats can express |
| g128 @ 8-bit scales | 80.62% | |
| **g64 @ 8-bit scales** | **82.06%** | what the model is trained at |
| g32 @ 4-bit scales | 73.56% (chat recipe) | finer grid cancelled by coarser scales |

Exporting a g64 model to TQ2_0 **requantizes it to g256 and destroys it**: MMLU-Pro 25.7% → 10.6% (random),
GPQA 30.3% → 16.8%. Hence a format that can hold g64.

## Layout

**SELF-CONTAINED** superblock = **512 weights**, **114 bytes**. No per-row side data.

```
offset  size  field  meaning
  0     96 B  qs     480 trits, 5 per byte, base-3:  b = t0 + 3*t1 + 9*t2 + 27*t3 + 81*t4
 96      8 B  qh      32 trits, 4 per byte, base-3:  b = t0 + 3*t1 + 9*t2 + 27*t3
104      8 B  sc       8 uint8 sub-scales, one per 64-weight group
                       actual_scale[g] = (sc[g] / 255.0) * d
112      2 B  d        fp16 super = max over this superblock's 8 g64 scales
```

* trit encoding: stored value `t ∈ {0,1,2}` represents weight sign `t - 1 ∈ {-1,0,+1}`.
* `3^5 = 243 ≤ 256`, so 5 trits fit one byte. 512 is not divisible by 5, so the superblock uses the TQ1_0
  split (480 @5/byte + 32 @4/byte) — the tight packing. Weights cost **1.625 bpw**, scales **0.1562 bpw**.
* **bpw = 114*8/512 = 1.7812**, uniform (self-contained, no row term).
* Decode order is little-digit-first within each byte: element `5*i + k` uses digit `k` of `qs[i]`;
  for the remainder, element `480 + 4*j + k` uses digit `k` of `qh[j]`. Digits are **plain base-3**
  (`digit_k = (b / 3^k) % 3`).

  **Deliberate divergence from TQ1_0.** Upstream TQ1_0 instead (a) orders elements digit-major within
  32-byte chunks (`x[m + n*32]` lives in digit `n` of byte `m`) and (b) rescales each packed byte by
  `(q*256 + 242)/243` so decode can extract digit `n` with `((q * 3^n) * 3) >> 8` — a multiply+shift that
  vectorises well. We keep the simpler element-major plain-base-3 order because:
    - the SIMD win does not transfer cleanly anyway: TQ1_0's vec_dot accumulates a whole 256-block into one
      int32 and scales once, whereas TQ1_64 needs **8 independently-scaled 64-element partial sums**, so the
      hot loop has to be restructured regardless;
    - it is already verified bit-exact end-to-end (encoder ↔ simulator ↔ written GGUF), and changing it now
      would mean re-exporting;
    - correctness first: the scalar decoder is trivially auditable against `decode_row`.
  If profiling later shows the unpack is the bottleneck, adopting TQ1_0's byte ordering is a contained change
  (encoder + decoder + one re-export) — but do it as a measured optimisation, not on spec.

### Why the super is INSIDE the superblock (and why 512)
ggml requires a **uniform block layout** (`nbytes = nelems/blck_size * type_size`) with no per-row side data,
so a per-row super would need ggml core surgery. Keeping it in-block costs +0.025 bpw (80 MB at 27B) and is
actually **more precise** (0.090% vs 0.116% median scale error) because the super spans 8 groups, not a whole row.

512 rather than 256 or 1024:

| superblock | bytes | bpw | divides our row widths (2560/4096/9216/248320) |
|---|---|---|---|
| 256 | 58 | 1.8125 | 4/4 |
| **512** | **114** | **1.7812** | **4/4** |
| 1024 | 226 | 1.7656 | 2/4 — 2560 % 1024 ≠ 0 ✗ |

### Scale encoding — decided by measurement, not assumption
Sub-scale error on our actual weights:

| encoding | median err | max err | bpw | note |
|---|---|---|---|---|
| fp8 e4m3 sub-scales | **4.23%** ✗ | 19.9% | 1.750 | rejected |
| per-row fp16 + u8 | 0.116–0.135% | ~1% | 1.756 | needs ggml core surgery |
| **in-block fp16 (512) + u8** | **0.090–0.104%** | 0.80% | **1.7812** | **chosen — standard layout, better precision** |
| in-block fp16 (256) + u8 | 0.070% | 0.80% | 1.8125 | more precise but larger |

fp8 fails because its 3-bit mantissa is the wrong 8 bits. A **linear** uint8 fraction works because
`--scale-qat-bits 8` already put the trained scales on a 256-value grid (log encoding measured worse: 0.36%).

## Validation (already done, in fp)

`src/sim_tq164.py` applies this exact encode→decode to every ternary tensor and writes a normal HF checkpoint,
so the standard evals can measure the format's true cost.

| metric | unencoded g64 | **TQ1_64 (final 512 layout)** | per-row variant (earlier) |
|---|---|---|---|
| eval2k top-1 agreement (n=1946) | 80.46% | **80.47%** | 80.46% |
| eval2k KL | 0.2916 | 0.3162 | 0.3162 |
| free-gen commit @2048 (n=48) | 75.0% | 79.2% | 75.0% |
| free-gen trunc | 20.8% | 18.8% | 20.8% |
| free-gen loop | 29.2% | 22.9% | 27.1% |
| comp-ratio | 3.08 | 2.85 | 2.84 |
| weight error | — | 0.18–0.19% max, **0.0587% rel-RMS** | 0.195% max, 0.0710% rel-RMS |

**Lossless.** The load-bearing number is **eval2k agreement 80.47 vs 80.46 on 1946 sequences** — a tight
measurement showing the encoding changes essentially no top-1 decisions.

⚠️ **Do NOT read the free-gen deltas as improvements.** That gate is n=48, so ±12 pt CI: commit 75.0→79.2% is
36/48 → 38/48 (2 prompts) and loop 29.2→22.9% is 14/48 → 11/48 (3 prompts). A 0.06% weight perturbation
reshuffles sampling trajectories, so which prompts happen to loop changes. The honest conclusion is
*statistically indistinguishable from baseline on free-gen, and lossless on the tight metric*.

## Size

| | 4B (ours) | 27B |
|---|---|---|
| TQ2_0 2.0625 bpw | 1.19 GB | 6.96 GB |
| TQ1_0 1.6875 bpw | 0.98 GB | 5.70 GB |
| **TQ1_64 1.7812 bpw** | **1.03 GB** | **6.01 GB** |

Ternary Bonsai 27B ships 5.9 GB (at g128) — TQ1_64 is within 2% of that while holding the finer **g64**
granularity, which is worth +2.75 pt agreement over g256.

**Embed and lm_head must also be ternary-packed.** Our first export stored them at `q4_K`/`q6_K` and wasted
875 MB (1719 MiB → 977 MiB when packed ternary).

## Implementation checklist

- [x] `src/tq164.py` reference encoder/decoder (the SPEC)
- [x] `tools/check_tq164_consistency.py` — guards reference vs vectorised sim (bit-identical) + trit exactness
- [x] **GGUF writer** `tools/gguf_tq164.py` — rewrites an f16 GGUF into TQ1_64, lossless by construction
      (packs the trits that are already there; does NOT round-trip through `llama-quantize`, which would
      re-derive the quantization — that is what destroyed the g64 model when exported to TQ2_0).
      Tensor selection is EMPIRICAL (2-D, cols%512==0, every g64 group on-grid), which correctly catches
      `token_embd.weight` — ternary here but NOT matched by config QUANTIZE_PATTERNS, so a name-based rule
      would have silently left 358 MB in q4_K.
      Verified on the 4B: 427/427 tensors preserved, 250 packed, **0 one-dimensional tensors packed**
      (norms copied), shapes correct, decode-back error 0.196% (= the uint8 scale, as designed).
      **eval_gguf/final4b-TQ1_64.gguf = 1093 MB** vs 1719 MB for the (broken) TQ2_0 export.
      embed and lm_head: 1271 MB f16 -> 141.5 MB each (9x).
- [x] **ggml core** (branch `tq1_64` in llama.cpp): `GGML_TYPE_TQ1_64 = 43`, `block_tq1_64` + static_assert
      (114 B), `type_traits` entry with `to_float`/`from_float_ref`.
      ⚠️ **id 43, NOT 42** — 42 is already `GGML_TYPE_Q2_0` in this fork; using it would have made llama.cpp
      silently misread our tensors as Q2_0.
- [x] **ggml-cpu**: `quantize_row_tq1_64_ref`, `dequantize_row_tq1_64`, `ggml_vec_dot_tq1_64_q8_K`
      (8 independently-scaled 64-element partial sums per 512-block; a superblock spans exactly two Q8_K
      blocks, and each 64-group lies wholly inside one of them), `type_traits_cpu` entry, and TQ1_64 added to
      all 7 `ops.cpp` type-dispatch switches (`get_rows` was the first crash).
      Base-3 digits come from a 256x5 LUT — the naive `(b / 3^n) % 3` division per element made prompt-eval
      unusably slow (didn't finish in 300 s).
- [x] **verification**: `tests/test-tq1-64.c` — sizeof, C round-trip trit exactness, all-zero blocks, and a
      cross-check against a binary dump from the Python reference: **`max|C - Python| = 0`, 0 mismatching of
      9216 weights**. Re-run after any codec change.
- [x] llama.cpp loader: ftype mapping so the model reports as ternary
- [ ] llama.cpp: `LLAMA_FTYPE_MOSTLY_TQ1_64` proper (currently reported as TQ1_0), quantize-tool string
- [x] **CUDA (dequantize → cuBLAS)**: `dequantize_block_tq1_64` in `convert.cu` (one CUDA block per 512-weight
      superblock, 64 threads x 8 elements so no thread straddles a 64-group boundary), registered in the
      to_fp16/to_fp32 getters, and `GGML_TYPE_TQ1_64` added to the MUL_MAT `supports_op` list.
      **TQ1_64 is explicitly EXCLUDED from the fused MMVQ/MMQ paths** — both the direct dispatch and
      `ggml_cuda_should_fuse_mul_mat_vec_q` — because base-3 unpack + 8 per-64 sub-scales do not fit those
      templates; without the exclusion batch-1 hits `mul_mat_vec_q`'s `default: GGML_ABORT`.
      `get_rows` is left on CPU (its CUDA template needs the 2-values-per-call `dequantize_q*` interface;
      the embedding lookup is tiny).
      **Measured on one RTX 3090, 4B model: pp 3.55 → 462 t/s (130x), tg 2.94 → 16.7 t/s (5.7x) vs CPU.**
      Output verified identical to the CPU path (same three prompts, same answers).
      *(Note: stock TQ1_0/TQ2_0 still have NO CUDA kernel at all — that is why the earlier TQ2_0 benchmark ran
      4.5x slower than Q8_0: its matmuls fell back to CPU.)*
- [x] **CUDA fused GEMV** (`ggml-cuda/tq1-64.cu`) for the batch-1 generation path — reads packed trits
      directly, never materialises fp16 weights. Dispatched from `ggml_cuda_mul_mat` when `src1->ne[1]==1`;
      prompt processing still uses dequant→cuBLAS (higher arithmetic intensity, cuBLAS wins there).
      **tg 16.72 → 53.87 t/s (3.2x), pp 462 → 530 t/s.** Output verified identical to the dequant path.

      ⚠️ **COALESCING IS THE WHOLE STORY — and the naive version was 2x SLOWER than cuBLAS.**
      First attempt: one superblock per thread. At a given byte index the warp's 32 lanes then touch
      addresses 114 B apart, so every 32-byte transaction delivers **one useful byte** → 7.99 t/s.
      Fix: one superblock per WARP, lane L reads `qs[L]`, `qs[L+32]`, `qs[L+64]` → 32 consecutive bytes per
      load → 53.87 t/s. **6.7x from memory layout alone, zero algorithmic change.**
      Two things that were *not* the problem, checked rather than assumed: `ptxas -v` showed 0 bytes spill
      (the `gacc[e>>6]` register array was fine), and the base-3 digit extraction is cheap *provided* the
      divisors are compile-time constants (`/3, /9, /27, /81` → multiply-shift; a runtime `for(t<nn) b/=3`
      loop emits real integer division and is catastrophic).
### CUDA GEMV optimisation log (RTX 3090, 4B model, tg t/s)

| step | tg | note |
|---|---|---|
| dequant → cuBLAS | 16.7 | materialises fp16 weights every matmul |
| fused, one superblock per **thread** | **8.0** | *slower than cuBLAS* — lanes 114 B apart, 1 useful byte per 32 B transaction |
| fused, one superblock per **warp** | 53.9 | coalesced weight reads — **6.7x from layout alone** |
| + shared trit staging, `float4` y loads | 74.2 | 15 scalar y LDGs → 4 vectorised (+38%) |
| + y staged once per block, 8 rows/block | 76.9 | only +3.7% — see below |

**Lesson worth keeping:** the two big wins were both *memory access shape*, not arithmetic. The two
hypotheses that turned out wrong were worth testing cheaply before rewriting: register spilling (`ptxas -v`
showed **0 bytes spill**) and the y re-read (traffic really is ~18x the weights, but y is 36 KB so it already
lived in L2 — reducing *cache* traffic bought little, and the required `__syncthreads()` per superblock ate
part of it).

At 76.9 t/s we are at ~8% of the 927 t/s memory-bound ceiling (1.01 GiB model / 936 GB/s), so the kernel is
still **instruction bound**. Remaining ideas, roughly in expected-value order:
- [ ] Cheaper unpack: phase 1 does 15 scattered byte-stores to shared per lane per superblock. A LUT giving
      5 trits as a packed word, or writing 4-byte groups, would cut that.
- [ ] Skip the ~45% of trits that are zero (sparsity is real but branchy — needs measurement, not assumption).
- [x] ~~A batched/MMQ path for prompt processing~~ — **TRIED AND REVERTED, it lost to cuBLAS.**
      Traffic analysis predicted ~19x less data (5.3 vs 100 MB per 2560x9216 layer at batch 32) and the op is
      memory bound (~22 us tensor-core math vs ~105 us memory). Measured: **pp 547 → 286 (NB=4) → 234 (NB=8)**.
      Why the analysis was wrong: (a) registers cap columns-per-pass at NB, so batch 32 becomes 32/NB
      SEQUENTIAL passes each re-reading *and re-unpacking* the whole weight matrix — the amortisation never
      happens; (b) the NB columns are `ncols` floats apart, so their float4 loads hit unrelated cache lines
      (raising NB made it *worse*). A real win needs a **tiled** MMQ (weights staged in shared, reused across
      a column tile), not a widened GEMV. Prompt processing stays on dequant→cuBLAS.
- [x] ~~Tiled MMQ (attempt 2: lane-per-column)~~ — **ALSO REVERTED, pp 547 → 129.**
      Inverted the mapping so lane L owns output column L: accumulators drop to 1/lane (no register cap, one
      pass over the weights) and trit reads become perfect shared broadcasts. But it trades the register wall
      for a **memory wall** — each lane then streams its *own* activation column, so the warp's loads are
      `ncols` floats apart and every `float4` is a separate transaction, ~4096 scattered loads per warp per
      superblock. 4x worse than cuBLAS.

      **What the two failures actually say.** The unpack can be amortised over columns (attempt 1) or the
      accumulators can be made cheap (attempt 2), but not both — unless *both operands* are staged in shared
      memory, which is what a real tiled GEMM does and what neither attempt did. A genuine win needs:
      weight tile [BM×BK] unpacked into shared, activation tile [BK×BN] loaded coalesced into shared, each
      thread computing a register micro-tile (e.g. 4×4) from shared. With BM=64/BN=64/BK=128 that is ~40 KB
      of shared and a substantially larger kernel. Given cuBLAS here is tensor-core-backed and already at
      547 t/s, that is a real project, not an increment — and **prompt processing is not the deployment
      bottleneck anyway** (generation is what a served model spends its time on, and that path is ours).
- [ ] `__ldg`/`cp.async` for the weight stream.
- [ ] Round-trip test in C: pack with the Python encoder, unpack in C, assert bit-equality with `decode_row`.
