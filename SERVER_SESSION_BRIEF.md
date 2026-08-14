# Server session brief — full-latent assignment training at 27B on 640 GB RAM

**Read this first, then `logs/RESULTS_SUMMARY.md` §9–§11.** Everything below is measured on this project,
not assumed. Numbers in **bold** are the ones you will be compared against.

---

## 1. Why this machine exists

Every assignment-training result so far was **memory-capped to one weight scope at a time**. At
18.9 bytes/latent the 60 GB dev box could hold `down` across 32 layers (755M latents ≈ 25 GB) and nothing
more. The one question we could never answer:

> **Does training ALL latents jointly beat training `down` alone?**

At 4B we could only approximate it — all-MLP in 8-layer groups, data-starved to 4M unique tokens — and got
**+0.5 pt**, which was too small to justify the memory engineering *on that box*. With 640 GB the real
experiment is finally runnable at 27B, where the 4B approximation may not generalise.

## 2. The memory model (measured, least-squares over three points)

> **18.9 bytes/latent + 6.92 GB base** — predicts attn@8L, gateup@8L and down@32L within 0.2 GB.

Decomposes as 4 (fp32 latent) + 4 (Adam `exp_avg`) + 4 (`exp_avg_sq`) + 4 (best-checkpoint snapshot)
+ ~2.9 overhead. An earlier **27.3** figure appears in older notes — it came from only two points 40.6M
apart and was **44% too high**. Use 18.9.

**27B inventory, counted from the actual checkpoint** (`Qwen3.6-27B`, 64 layers, hidden 5120, inter 17408,
vocab 248320, 48 linear-attn + 16 full-attn): **26.049B quantizable latents** excluding `embed_tokens`.

| scope | latents | RAM @18.9 | RAM @8.9 (patched) |
|---|---|---|---|
| `down` only, 64L | 5.704B | **115 GB** | 58 GB |
| all MLP (gate+up+down) | 17.11B | **330 GB** | 159 GB |
| attention (all, incl. DeltaNet) | 5.54B | 111 GB | 56 GB |
| `lm_head` | 1.271B | 31 GB | 18 GB |
| **EVERYTHING** | **26.049B** | **499 GB** | **239 GB** |

All of these fit in 640 GB. On the 60 GB box, only the first row's *half* did.

`tools/apply_mem_wins.py` (committed, `--check` passes, **not yet applied**) drops 18.9 → ~8.9 via 8-bit
Adam moments (`--adam8bit`, `src/adam8bit_cpu.py`) and memmapped snapshots (`--snap-nvme`). Measured:
**99.85% identical flip decisions** vs fp32 moments, +10.3% step time. Apply it if you want headroom;
you do not need it to fit.

## 3. THROUGHPUT IS NOW THE BINDING CONSTRAINT — measure it before committing

Memory stops being the limit; wall-clock becomes it. Measured CPU Adam: **50.3 ms per 23.6M latents**.

| scope | optimizer-only s/step | a 6244-step run |
|---|---|---|
| `down` 64L (5.7B) | ~12 s | ~21 h |
| all MLP (17.1B) | ~36 s | ~63 h |
| everything (26.0B) | ~56 s | ~97 h (4 days) |

…**before** the 27B forward/backward, which is ~7× the 4B's. **First task on the new box: run ~50 steps of
`down`@64L and measure real s/step.** If the box is DDR3/multi-socket, expect NUMA penalties — use
`numactl --interleave=all`, and check whether the Adam update parallelises across sockets. If full-latent
lands at weeks rather than days, fall back to all-MLP or layer groups.

## 4. Hard-won constraints — do not re-derive these

1. **A second MLP projection trained SEQUENTIALLY always damages and never recovers.** Confirmed 4×,
   including 3000 steps warm-started from a 16M `down` (entry 0.2341 → 0.3261, never beat entry). SwiGLU is
   `down(silu(gate(x))·up(x))`: gate and up multiply, so tuning one co-adapts the others' current
   assignments. **If you train >1 MLP projection, train them JOINTLY** (`--train-weights mlp` or `gateup`).
2. **Layer-sequential passes DO stack** (`--tw-layer-stride`/`--tw-layer-offset`): each group dips ~4%,
   recovers, finishes ahead. Cross-layer coupling is additive via the residual stream; within-layer MLP
   coupling is multiplicative. Put sequential boundaries on the layer axis, never the scope axis.
3. **Assignments scale with data**: +1.67 pt eval2k agreement per doubling, log-linear across 17×, no
   saturation at 16M. Fresh tokens ≈ **3× repeats**. Unique calib caps at 15.0M tok on the old box.
4. **Diminishing returns law**: every mechanism gains ~15–20 pt on a weak model and ~0–3 pt on a strong
   one. Expect the full-latent gain to be *small* on top of a well-trained `down` — size the experiment to
   detect ~1 pt, and run a paired control.
5. **bf16 latents FAIL** (held-out KL 0.9495 vs 0.6624). The STE gate `|L| < 1.5·s` decides flips from
   distance-to-boundary; a floating exponent gives resolution proportional to |L| instead of a uniform
   step. Quantise Adam *moments*, never the latent.

## 5. Traps that cost us hours — all fixed in the committed code, keep them on

- **`--abort-patience` is measured in STEPS**, not evals (`ho_worse × eval_every ≥ 100 × patience` ⇒ 300 at
  the default 3). At a coarse `--eval-every` that is ONE eval. It killed E2E at step 3000/12488 **while it
  was improving monotonically**. **Use `--abort-patience 30`+ on any long stage**, or 1000000 to disable.
- **Step-0 ENTRY BASELINE** now seeds `best_ho_kl`, so a stage that never beats its entry restores that
  entry. Without it a stage silently shipped a model 33% worse than its input under a clean-looking `best=`.
- **Cold-start lr ramp + TALR gating** (`--latent-lr-ramp-steps`): the raw base lr used to land full-strength
  before TALR's first measurement and burst the assignments; ~40% of that damage is permanent. Keep it.
  Consequence: **base latent-lr only needs to be within ~1 order of magnitude** — do NOT extrapolate the old
  "1e-8 for 27B" figure. Calibrate from the first post-ramp `[talr]` line (want the rate within ~2× target).
- `--ckpt-every 0` — the default 100 triggers a full `save_student()` mid-training (a ~20 GB spike).
- `foreach=False` for the CPU Adam group; the fused path allocates same-size temporaries across the whole
  group (43.8 GB RSS, D-state wedge).
- **`pgrep -f` self-matches** any shell whose command text contains the pattern. Match on `comm=="python"`
  via `ps`/`awk` (the run scripts' single-instance guard already does).
- **Per-run held-out KL is NOT comparable across runs** — `held_idx = last 4 of the LOADED subset`, so
  changing `--max-samples` changes the metric. Only **eval2k** (frozen 1946-seq referee) is comparable.

## 6. Reference numbers (4B testbed, eval2k top-1 agreement vs FP, frozen 1946-seq referee)

| model | eval2k | loop | commit | comp | Gate B |
|---|---|---|---|---|---|
| block-AP skeleton | 63.09% | 97.9% | 2.1% | 20.02 | catastrophic |
| + `down` (16M tok) | 78.35% | 47.9% | 72.9% | 3.34 | FAIL |
| **+ E2E → best model** | **81.32%** | **27.1%** | **75.0%** | **2.90** | **PASS** |
| scale-only E2E (no assignments) | 80.86% | 29.2% | 75.0% | 3.08 | PASS |
| full pipeline, 4M cold start | 81.00% | 37.5% | 64.6% | 3.52 | FAIL |
| FP teacher | — | 25.0% | 75.0% | 2.40 | ceiling |

**Both stages matter, for different reasons.** Assignments do the bulk of free-gen repair (commit
2.1 → 72.9%); E2E does the final approach on loop rate (47.9 → 27.1) and that is what crosses the gate.
On teacher-forced agreement they are near-substitutes; on free-generation they are not.

**Gate B (free-gen) is THE deploy gate and is now fail-closed** in `run_full_pipeline.sh`
(commit ≥68% · loop ≤30% · comp ≤3.1; `GATE_ADVISORY=1` overrides). A 63.09%-agreement model loops on 98%
of prompts — teacher-forced metrics are blind to this by construction.

## 7. Setup on the new box

```bash
git clone https://github.com/KylerRussell/Model-to-Ternary.git
cd Model-to-Ternary
python -m venv .venv && .venv/bin/pip install -r requirements.txt
ENV_VERBOSE=1 source ./env.sh          # prints resolved / MISSING for every path
```

`env.sh` derives `REPO_ROOT` from its own location and resolves `MODEL_4B`, `MODEL_27B` (globs the newest
HF-cache snapshot — no pinned hash), `CTM_DATA`, `EVAL2K_JSON`, llama.cpp binaries, `HOST_RAM_GB`.
Per-host overrides go in `env.local.sh` (gitignored).

**COPY THESE FROM THE OLD BOX — do not regenerate:**

| what | why |
|---|---|
| `output_4b/eval2k.json` | **the frozen referee.** Every number above is against this exact draw. Regenerating breaks all comparability. |
| `output_4bpipe/calibration_data.json` + `chat_pool.json` + `teacher_topk.pt` (5.8 GB) | the 16M calib — ~14 h of chat rollouts to rebuild |
| `output_4bpipe/e2e_on_assign/` | the 81.32% reference model |
| `output_4b/untied_4b/` | 4B testbed (untied embeddings) |

The 27B needs its own rotation/calib/teacher — that is a cold `run_full_pipeline.sh` run.

## 8. Suggested plan

**Step 0 — port sanity (cheap).** Score the copied `e2e_on_assign` on eval2k; it must read **81.32%**. If it
does not, the port is wrong and nothing downstream is comparable.

**Step 1 — measure throughput.** ~50 steps of `down`@64L on the 27B. Record s/step and peak RSS; confirm
the 18.9 B/latent model holds at this scale (it was fit on the 4B). This decides whether full-latent is
days or weeks.

**Step 2 — THE experiment, paired.** Same skeleton, same calib, same total steps, differing only in scope:

- **arm A**: `--train-weights down --tw-layer-stride 1` (5.7B latents, ~115 GB)
- **arm B**: `--train-weights all --tw-layer-stride 1` (26.0B latents, ~499 GB)

Score both on eval2k **and** Gate B. Use `--abort-patience 30`, `--ckpt-every 0`, and the default cold-start
ramp. Expect the difference to be small (see §4.5) — if arm B does not clear arm A by more than ~1 pt, the
scope axis is closed at 27B too and the answer is "spend it on data instead".

**Step 3 — if arm B wins**, fold `--train-weights all` into `run_full_pipeline.sh` Phase 4.5 (currently
`down` then `attn`; `ASSIGN=0` disables the phase entirely) and do a full cold run with Gate B enforced.

**Open questions worth attacking with the extra RAM:** whether the 4B's +0.5 pt group-joint result was
memory- and data-limited rather than a real ceiling; and whether more unique calibration data (>15M, the old
cap) keeps paying at +1.67 pt/doubling — that curve had no bend at 16M and data was the one lever we never
exhausted.

## 9. Where the record lives

- `logs/RESULTS_SUMMARY.md` — §9 assignment program, §10 condensed raw artifacts, §11 pipeline validation.
- `research_prompts/joint_scope_memory_prompt.md` — the memory problem (note: quotes the stale 27.3 figure).
- `research_prompts/e2e_diminishing_returns_prompt.md` — why the 2nd mechanism adds so little.
- `experiments/` — finished campaign scripts, unmaintained, absolute paths, referenced models deleted.
