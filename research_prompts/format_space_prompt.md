# Deep research prompt — the sub-2-bit DEPLOYMENT FORMAT space

**This is a census of FORMATS, not methods.** A previous census screened ~48 quantization methods
against one format (TQ1_64) and concluded that 62.5% fail. That conclusion is only as interesting as
the format is representative — and we have just established it is **not** representative. We need the
format axis before the method verdicts mean anything.

---

## 0. Why this prompt exists — the error that triggered it

The previous census justified most of its FAIL verdicts with hardware arguments of this shape:

> *"Vector quantization breaks down on consumer GPUs due to indirect memory dereferencing and cache
> thrashing."*
> *"Asymmetric offsets destroy the addition-only GEMV pipeline."*

**Both claims are falsified by shipping code.** The `gguf` package installed in this repo enumerates
these ggml types, all with working CUDA kernels in llama.cpp:

| type | what it proves |
|---|---|
| `IQ1_S`, `IQ1_M` | **sub-2-bit formats built on codebook/grid lookups** — deployed, fast |
| `IQ2_XXS`, `IQ2_XS`, `IQ2_S` | more codebook-based quantization at ~2–2.5 bpw |
| `Q2_K` | 2-bit k-quant carrying **per-sub-block scales AND mins** — i.e. zero-points |
| `TQ1_0`, `TQ2_0` | scalar ternary, one scale per 256-weight block |
| `Q1_0` | present in the enumeration; characterise it |

So codebooks and zero-points are **not** unexecutable on commodity hardware. They are unexecutable in
*one particular format*. The real screen is therefore not "does this method fit TQ1_64" but:

> **Which format class does this method require, and does a deployable format in that class exist?**

That is the question this prompt must answer. Do not repeat the previous census's hardware
generalisations; check them against formats that actually ship.

## 1. What to enumerate

Every weight-storage format at **≤ 2.5 bits/weight** that has a **real inference kernel** someone can
run — not a paper appendix. Include at minimum:

* **llama.cpp / ggml**: `TQ1_0`, `TQ2_0`, `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `Q2_K`,
  `Q1_0`, and anything newer
* **BitNet native / W1.58**: what is the *deployment* format for BitNet b1.58 and its descendants?
  (`bitnet.cpp` kernels, `TL1`/`TL2` lookup-table formats, `I2_S`.) How does a native-ternary
  checkpoint actually get stored and executed, and at what true whole-model bpw?
* **1-bit**: BitNet b1.0, AF1, and any binary format with shipping kernels
* **VQ runtimes**: AQLM, QuIP#, QTIP, VPTQ — each ships its own CUDA kernel. Characterise them as
  formats, not just as algorithms
* **GPU-native 2-bit**: Marlin, Machete, exllamav2 `EXL2`/`EXL3`, AWQ/GPTQ packed layouts, TensorRT-LLM
* **MLX, Apple/ANE, and mobile NPU** ternary or 2-bit formats, if any

## 2. The capability axes — the core deliverable

For every format, answer each axis **yes/no/with what cost**. This table *is* the contribution; the
method verdicts fall out of it mechanically.

| axis | question |
|---|---|
| alphabet | binary / ternary / 2-bit int / codebook indices / lattice points |
| **zero-point or offset** | can it store a per-block `μ ≠ 0`? |
| **scale granularity** | weights per scale (32 / 64 / 128 / 256 / per-tensor), and scale precision |
| **hierarchical scales** | does it carry a super-scale plus sub-scales (k-quant style)? |
| **multiple planes** | can it store two or more discrete planes per weight? |
| **codebook / LUT** | does dequantization index a table? How large, and where does it live? |
| **side tensors** | can it carry per-row or per-channel auxiliary tensors? |
| **sparse/outlier path** | does it support a mixed-precision outlier set? |
| **variable-length codes** | fixed-stride or entropy-coded? |
| **self-contained block** | is a block decodable without per-row side data? |
| **true bpw** | headline, and whole-model with embeddings/head included |
| **kernel maturity** | CUDA? CPU SIMD? measured throughput vs FP16 on a comparable shape? |
| **who can produce it** | which quantizers emit this format today? |

## 3. Questions we specifically need answered

1. **What is the lowest-bpw format with a production kernel, and what does it give up?** `IQ1_S` is
   ~1.56 bpw against `TQ1_0`'s 1.6875 — if a codebook format is both *smaller* and *shipping*, the
   case for scalar ternary has to be made on quality, not size.
2. **Do any published methods target the IQ-family formats?** These formats have quantizers inside
   llama.cpp but we are not aware of an academic literature aimed at them. If that gap is real it is
   a notable finding: the most-deployed sub-2-bit formats may have the least published method work.
3. **What does a BitNet b1.58 / W1.58 model actually deploy as?** Native-ternary training is widely
   reported; the *storage and kernel* story is much less clear. True whole-model bpw, please, not the
   headline.
4. **Does anything support per-block zero-points at ≤2 bpw with a fast kernel?** `Q2_K` suggests yes.
   If so, the entire "no zero-point" constraint is a choice rather than a hardware limit, and every
   method we failed on that basis needs re-screening.
5. **Which formats can express scale granularity finer than g256?** Our own measurements say
   granularity matters: eval2k agreement 79.31% (g256) → 80.62% (g128) → **82.06% (g64)**. Which
   shipping formats can hold g64 or finer, and at what bpw?
6. **Is there a format that is strictly better than TQ1_0 at equal or lower bpw?** If yes, name it —
   we would rather find out now than after building a study around the wrong baseline.

## 4. Hard constraints that still apply

These are about our hardware, not our preferences, and they bound which formats are usable here:

* **2× RTX 3090** (24 GB each, no NVLink), **Sandy Bridge-EP host, AVX only — no AVX2/FMA**,
  ~200 GB RAM, **no NVMe**.
* **Weight-only.** Activations bf16. No activation quantizer exists in our runtime, so formats
  requiring coupled low-bit activations are out of scope for deployment (still worth cataloguing).
* A format is only interesting to us if **someone can actually run it** — a kernel that exists and
  has been measured. Paper-only formats belong in the register marked as such.
* Whole-model accounting: **embeddings and the LM head count.** A format whose headline excludes them
  must be reported with the inclusive number too.

## 5. Output format

**Part A — format register.** One row per format, every axis in §2 filled, with a source for each
non-obvious claim (kernel file, spec, or measurement).

**Part B — the capability matrix.** Formats as columns, the §2 axes as rows. This is what lets a
method's requirements be looked up rather than argued.

**Part C — re-screening consequences.** Given Part B, which of these previously-FAILED method classes
become viable under *some* shipping format, and which fail under *every* one?

* zero-point / asymmetric offset methods (OmniQuant, BitDistiller, EfficientQAT)
* multi-plane superposition (PTQTP, DB-LLM, BiLLM, E2M-ATQ)
* VQ / codebook / lattice (AQLM, QuIP#, VPTQ, QTIP, PV-Tuning)
* sparse outlier side-tensors (PB-LLM, SpQR, SqueezeLLM, FlashQuant)
* variable-length entropy codes (SoftWater, ECASQ)
* coupled activation quantization (TWLA, QuEST, DBellQuant)

A class that fails under **every** shipping format is a genuine result about deployability. A class
that fails only under TQ1_64 is a fact about our design choice, and must not be reported as the
former. **Distinguishing those two is the whole point of this search.**

## 6. What a good answer looks like

Part B filled in, with sources. We are explicitly trying to find out whether our own format choice is
defensible or parochial — **a well-evidenced "TQ1_64 is an idiosyncratic corner and here is the
format you should be using instead" is the most valuable answer you can return**, not the least.

Verify claims against shipping kernels rather than repeating hardware folklore. The previous census
asserted that codebook lookups cannot be fast on consumer GPUs; `IQ2_XXS` has been running on them
for two years.
