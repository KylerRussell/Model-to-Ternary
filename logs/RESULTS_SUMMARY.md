# Condensed Experiment Results (distilled from logs/ before cleanup)

Metrics: **in-domain ppl ratio (ternary/FP) / OOD ppl ratio / free-gen degeneration rep4% at bin7**
(lower = better; teacher degen ≈ 40–46%). "recall" = multi-key associative-recall probe MEAN restr
(FP≈0.9975). All 4B unless noted. Raw logs deleted after this summary; the mechanism narrative lives in
`research_reports/PRIORITIZED_MECHANISM_CHECKLIST.md` + memory.

## 1. Scale runs & GPTQ-vs-QAT crossover (the key scaling data)
Post-E2E in-domain / OOD / degen (deployable). Both arms = same E2E (ema + γ2 + diverse calib; feat=0).

| model | GPTQ+E2E | QAT+E2E | winner |
|---|---|---|---|
| **4B** (pipeline_test_4b) | ~1.399 | **1.3264** | QAT (Δ+0.073) |
| **9B** (run_pipeline_test_9b, 4M tok) | **1.3194 / 1.7186 / 64.8%** | 1.8579 / 2.8085 / 83.1% | **GPTQ (Δ+0.539)** |
| **27B** (4M tok) | **1.0791 / 1.4507 / 70.6%** | (worse) | **GPTQ (Δ+0.386)** |

→ **Crossover confirmed: QAT>GPTQ at 4B, but GPTQ>QAT at 9B AND 27B (large margin).** GPTQ scales, QAT doesn't.
**Mechanism (9B block-AP stage, pre-E2E):** QAT block-AP collapses to **14.08×** in-domain (OOD 24.6×) vs GPTQ's
**2.64×** (OOD 5.08×) — the latent-weight scale-only QAT reconstruction breaks down at scale, and E2E can't
rescue such a broken init. (GPTQ-vs-QAT-at-scale gets its own research prompt now that we have this data.)
Note: 9B GPTQ deployable (1.3194/1.7186/64.8%) beats the 4B deployable on all 3 → good scaling.

**Setup gotcha (fixed):** feature distillation (`--feat-weight>0 --cache-hidden`) makes a ~34GB hidden
teacher cache; E2E `torch.load`s the WHOLE cache in EACH DDP rank → 2×34GB > 60GB RAM → swap-death + DDP
socket-timeout hang. Dropped feat (unvalidated anyway; 4B baselines used feat=0). Re-enable only with a
memory-safe (mmap/streamed/single-rank) cache loader.

## 2. 4B mechanism campaign (block-AP=single QAT → E2E variants; run_4b_dg / exp_4b)
| config | evals |
|---|---|
| baseline (single QAT, γ=1 E2E) | 1.3264 / 1.8846 / 88.5% |
| **(b) γ=2 decision-weighted E2E — the deployable** | **1.3429 / 1.9305 / 68.8%** ✅ (degen win) |
| (a) multi-layer N=4 QAT + γ2 | 1.3691 / 2.0481 / 72.1% ✗ |
| γ=3 E2E | 1.3646 / 1.9805 / 84.0% ✗ |
| (d) Tequila deadzone→bias + γ2 | 1.6317 / 2.4158 / 92.7% ✗✗ |
Verdict: only **γ=2 decision-token weighting** helped (degen 88.5→68.8 at tiny ppl cost).

## 3. Moonshot mechanism sweep (2026-07-01/02; C-clusters from the checklist)
| mechanism | test | result | verdict |
|---|---|---|---|
| **C4 super-outlier** | per-256-block domination pre/post QuaRot | post-rot blocks ~Gaussian (exkurt≈0); 50–60× super-weights → 13–18× | **DEAD** — rotation already solved FM1 |
| **C2 angular key** | ternarize only K, MSE vs angular, rest FP | recall: K-MSE 0.9875, K-angular 0.9675 (FP 0.9975) | **DEAD** — key not the bottleneck, no headroom |
| **C3 contraction clamp** | α→α(1−ε) sweep on decay gate | recall 0.44→0.74 (peak ε≈.15–.2) BUT degen 68.8→99.2% @ε.05, 91% @ε.01; ppl worse | **REJECTED** — recall gain = forgetting artifact, destroys fluency |
| **component ablation** | splice real ternary weights per-component into FP | recall collapse: **MLP ~79%**, whole DeltaNet recurrence ~27% (key least), full-attn ~18% | MLP is the source, recurrence the amplifier |
| **C1 decision-weighted recon** | decision-token-weighted MLP Hessian | rel-err 0.673≈baseline; no recall gain | no cheap benefit (payoff downstream-only) |
| **C6 joint MLP** (down absorbs gate/up err) | probe: −41% MLP out-err (0.673→0.399); pipeline GPTQ+C6→γ2 E2E | **1.4113 / 2.1302 / 79.6%** — worse than baseline | didn't beat baseline; α=0.5 retry was a wash (block-AP 2.46>2.35). Helps GPTQ path but GPTQ<QAT |
Overall: of the reports' foldable mechanisms, **none beat the QAT+E2E baseline** → E2E is the real lever
(see `research_reports/e2e_deep_research_prompt.md`).

## 3b. E2E-QP improvement tests (4B, GPTQ block-AP init; fast metrics only, no MMLU/GPQA)
Clean isolation: SAME GPTQ block-AP → baseline γ2 E2E vs mechanism γ2 E2E (`run_onpolicy_4b.sh` etc.).
GPTQ→γ2-E2E baseline @4B = **1.4278 / 2.0738 / 82.5% degen / recall .705** (note: worse degen than the
QAT deployable's 68.8% — consistent with QAT>GPTQ@4B — but better recall .705 vs .44).

| mechanism | in / OOD / degen-b7 / recall | vs baseline |
|---|---|---|
| **A1+B2 on-policy** (25%, rollout96+96, unlik0.5) | 1.4363 / 2.0778 / **78.3%** / .7425 | degen −4.2pp, ~4.7× cost, under-dosed — **PARKED** |
| **undertrain 4000 steps** (=2 epochs) | **1.3977 / 2.0714 / 71.7% / .830** | **🏆 BIG WIN — E2E under-converged** (degen −10.8pp, recall +.125) |
| LR 1e-5 | 1.4575 / 2.1025 / 85.2% / .703 | worse (4B wants higher LR) |
| A2 skew-KL α.1 | 1.5670 / 2.4557 / 86.9% / .745 | worse everywhere |
| B4 Fisher-EMA | 1.4284 / 2.0756 / 91.5% / .748 | neutral ppl, worse degen (Adam already preconditions) |
| A3(ii) MLP-input scale (norm-fold) | 1.4276 / 2.0699 / 87.9% / .708 | ran OK (fold works), no gain |
| A4 down_proj moves (STE, 8-bit Adam) | 1.4262 / 2.0732 / 87.5% / .765 | neutral-to-worse (flat ppl, degen +5pp, recall +.06) |

**E2E campaign verdict:** the report's mechanisms (A2/B4/A3) were all neutral-to-worse; the ONLY win was
**"E2E was under-converged" → 2× steps (2000→4000, 1→2 epochs over the same 4M calib): degen 82.5→71.7%,
recall .705→.830.** 2000 steps already = ~1 full epoch. **Decision (user): 2 epochs is the compute
sweet-spot; don't chase 4 epochs (8000 steps) — real headroom is MORE DATA (bigger calib), not more passes.**

**On-policy verdict (2026-07-03):** mechanism WORKS directionally (degen/slope/recall all improve, ppl
flat) but the win is SMALL and it costs ~4.7× baseline E2E wall-clock (rollout generation). **Diagnosed
UNDER-DOSED: rollout len 96 ≪ the 480-token degen-eval regime**, so the student is never exposed to its
own drift at the LONG contexts where looping actually happens. **PARKED for now due to time cost** —
future work: re-test with rollout→256-480, higher on-policy frac, more steps to see how far it pushes
degen; run it LAST after the cheaper mechanisms. Impl lives in `src/e2e_qp_distill.py` (`--on-policy-*`).

## 4. Reference baselines
- **4B deployable:** QAT→γ2 E2E = **1.3429 / 1.9305 / 68.8%** (teacher free-gen degen 46%).
- **27B deployable:** GPTQ→scale-only E2E, 4M tok = **1.0791 / 1.4507 / 70.6%**.
- Recall probe: FP ≈ 0.9975, full-ternary ≈ 0.4425.
- The real bar: beat 9B-q8 (MMLU-Pro 0.542 / GPQA 0.428) on downstream — not yet measured on our model.

---
## 5. GPTQ-init QAT strengthening + downstream + data-scaling (2026-07, post-cleanup condense)

**Mechanism recap:** QAT was never broken — the old arm was a strawman (cold FP-init, hot 1e-4 LR). The win is
layer-type-conditional: attention/DeltaNet are OPTIMIZER-bound (STE polish helps), MLP is near-Babai-optimal
(GPTQ≡CVP, keep-best reverts every MLP polish). Recipe = GPTQ-init + freeze MLP + STE-polish attn-only + keep-best.

**4B strengthening sweep (block-AP ppl, in-domain):** gptq_op 2.7851 → decider full-QAT (GPTQ-init, name-route,
polish-then-revert-MLP) 2.5173 → best A1 attn-only STE **a1_lr1e-4_ep4 = 1.8315** (freeze-MLP-up-front beats
polish-then-revert; LR/epoch-hungry, reverts=0). A4 saliency helps at moderate LR (3e-5/ep4 = 1.8932 vs matched MSE 1.9170). A2 AdaRound
= NO-OP (reverts all, step-starved: needs ~2-20k steps not ~64-256). A3 coupling-route underperforms plain A1.

**4B post-E2E (NP=24 ID/OOD/KL/flips):** plain-GPTQ→E2E 1.4278/2.0738/0.4535/24.07 · decider qatgi_lr1e-5 1.4227/
2.0126/0.4447/23.73 · **A6 (a1_lr1e-4_ep4 ×2E2E) 1.3566/1.9798/0.3904/22.05** (overall carry) · saliency×hiLR
1.3499/2.0224/0.3875/21.87 (best ID/KL/flips but WORST OOD = overfit; contradicts researcher Pareto prediction).
No ppl↔KL/flips decoupling at matched NP. Carry A6 to 27B; pick polish-LR by OOD (high-LR overfits OOD).

**4B DOWNSTREAM (MMLU-Pro / GPQA, samples=4):** tern4b-A6(1.81GB) 18.2/17.4 · q0.8b(0.81GB) 17.8/13.3 ·
q2b(2.01GB) 27.4/25.3 · q4b-FP(4.48GB) 47.7/36.0. → 4B ternary CATASTROPHIC (−62% rel MMLU vs FP), loses to
same-mem 2B-q8. BUT prior UNOPT 27B ternary: MMLU 61.7/**50.3**/54.2(9B) · GPQA 48.4/**38.8**/42.8(9B) — 27B
absorbs ternary ~3.4× better (−18% not −62%), only ~4pts short of 9B-q8. 27B is the right regime; 4B = mechanism
testbed only. TARGETS: beat 9B-q8 54.2/42.8; improve unopt-27B 50.3/38.8; FP ceiling 61.7/48.4.

**4B E2E DATA-SCALING (fixed A6 skeleton, --epochs 2; tokens→ID/OOD/KL/flips):** 0.5M 1.4357/2.1237/0.4396/23.97 ·
4M 1.3541/1.9848/0.3910/22.02 · **16M 1.2421/1.9469/0.2901/18.28** · 64M 1.2784/1.8997/0.3264/19.85. KEY: OOD
monotonic↓ (unsaturated) BUT KL/flips/ID form a U — best @16M, REGRESS @64M (broader mix → better OOD, less
in-domain-specialized). Since KL/flips=downstream proxy, **27B E2E sweet spot = ~16M tokens, NOT more** (also ~4×
cheaper). ~1807 sec/Mtok (linear at fixed epochs). Block-AP weights SATURATE ~few-k samples; only E2E scales w/ data.
Plot: calib_sweep_curve.png.

**Tooling added:** src/block_ap_recovery.py --qat-gptq-init/--qat-keep-best/--qat-attn-only/--qat-adaround/
--qat-route-coupling/--qat-loss saliency; src/build_saliency.py; src/kl_flips_eval.py; e2e_qp_distill.py --epochs +
mmap teacher-load; run_4b_strengthen_sweep.sh, run_4b_saliency_himlr.sh, run_4b_calib_sweep.sh, export_4b_gguf.sh,
eval_harness/run_evals_4b.sh; converter qwen2-tokenizer fix for 0.8B.

---
## 6. Placement thesis (CDQuant, margin-Jacobian) — ALL NEGATIVE (2026-07-04) + NP=48 degen ref

Thesis "place the error, don't fight it" — better ASSIGNMENT placement (beyond greedy GPTQ) survives E2E?
Verdict: exhausted/negative — GPTQ is already near-optimal placement; scale-only E2E redoes any placement gain.
- **CDQuant** (coordinate-descent assignment refinement, same on-grid objective): block-AP byte-identical to
  plain GPTQ (no-op); GPTQ+CDQuant→E2E = **1.4274 / 2.0748** ≡ baseline 1.4278/2.0738. No survival.
- **#4 Margin-Jacobian** (re-quantize MLP with a margin-sensitivity-weighted Hessian): FAILED badly at every
  strength — raw λ=1 → **1.6966 / 2.7795**; shrinkage λ=0.2 → **1.6796 / 2.7371** (both ≫ baseline; recall UP
  = the recall-probe-gaming pattern). Metric-aligned placement HURTS (ill-conditioned).
- Together with CDQuant no-op → placement thesis fully refuted. Levers are convergence × DATA, not placement.

**NP=48 degen reference (the low-noise degen metric; NP=8 was ~7pp noisy):** plain-GPTQ→E2E b7=**90.6%**
(slope +70.8pp); gptq_cdqE2E b7=89.1%; teacher b0=9.9%→b7=58.4%. (Strengthening post-E2E degen was measured
at NP=48/24 — see §5; e.g. A6 b7≈90.3% comparable to baseline 90.6%, i.e. degen flat while KL/flips improved.)

## 7. Assignment-QAT retirement → fresh-token scaling saga → scale-axis tests (2026-07-09→18)

**Arm-B assignment-QAT (PV-Tuning-style sparse proximal flips), CORRECTED + GATE-HONEST → RETIRED.**
Naive STE diverged; corrected arm (fresh-grad probe, proximal gate Δ<0, trust-region ρ for η, disjoint
256-seq PAIRED-ΔKL generalisation gate) is stable and the gate works (caught winner's-curse events with
in-sample realΔ<0 but out-of-sample ΔKL_G>0 at 4σ). Gate-honest result: mlp-scope flips == scale-only
step-for-step (0.2889 vs 0.2890 held-out; 151 flips of 2.3B). Root cause (researcher-confirmed): ~60%
skeleton saturation (GPTQ≈Babai/CVP-optimal) + ~30% scale/assignment threshold coupling. Runs: armb_smoke /
armb_mlp / armb_mlp_gate / scaleonly_4m (models deleted).

**Fresh-token scale-only scaling (Stage 0, canonical tables kept: scaleonly_fresh_results.txt +
eval2k_scores.txt).** 1-epoch fresh curve FLOORS on fixed eval: 4M .4110 · 8M .3438 · 16M .3301 · 32M .3254
· 64M .3250 (K∞=.3247, α≈2.1, floor beats power by 22 AIC) while OOD falls monotonically (2.01→1.7923 best
ever) and per-run own-held-out (top-64 metric) keeps falling. Resolution of the paradox (researcher rounds
6-7, predictions committed then SCORED): **EPOCHS DOMINANT** — 16M_2ep .2922 (+2nd epoch −.0379) vs recipe
(constant-LR+EMA-select) only −.0109; researcher's B-dominant weighting INVERTED by data (1 hit, 3 range
misses across (a)-(d)). Muennighoff equivalence VIOLATED: 16M×2ep (.2922) ≫ 32M×1ep (.3254) at identical
steps/schedule/LR-integral. Controls: constlr-2ep .3058 → epoch gain ≈ 64% plain second-pass + 36% LR-tail
consolidation; constlr REGRESSES OOD (1.9945) → annealing protects robustness. Fresh-draw falsifier
16Md2_2ep .3002 fixKL / .2988 eval2k (≈16M_2ep's .2981 → draw-variance nil, mechanism real). 32M_2ep .2988
fixKL / .3022 eval2k → epoch gain SHRINKS toward a common in-mixture floor (~.298-.302 eval2k for ALL 2-ep
runs); tokens buy OOD only (32M_2ep OOD 1.8516). **SETTLED PRODUCTION RECIPE: 2 epochs × linear-decay-to-
zero × final-select (no EMA); token count chosen by OOD/benchmark budget, not KL. Endgame (researcher,
accepted): E2E protocol ≤0.2-0.4 MMLU-Pro pts — remaining 3.9/4.0-pt gap lives in the frozen skeleton
(upstream axes) + mixture. User decision: protocol program CLOSED; no (g) 64M_2ep, no mixture enrichment.**

**eval2k referee:** frozen 1946-seq same-mixture draw (seed 9001, output_4b/eval2k.json) + per-seq PAIRED
dumps (output_4b/eval2k_perseq/, kl_flips_eval PER_SEQ_OUT). Old 24-seq fixed eval has ~±0.008 draw noise;
eval2k paired resolves 0.0004 at 5σ. kl_flips_eval now streams (no FP-logit cache; NP~2000 OK).

**Scale-axis tests (user Q, 2026-07-15→18).** TQ2_0 scale cost: 0.0625 bpw = 3.03% (~190MB @27B).
(1a) Post-hoc scale rounding (eval2k, base .2981): 8-bit +0.0002 FREE · 6-bit +0.0013 · 5-bit +0.0051 ·
4-bit +0.0260 cliff. (1b) 4-bit scale-QAT (STE log-grid): .3099 = recovers ~55% of cliff, +0.0118 residual.
OPTION TABLE (all require ggml fork — TQ2_0 kernel reads fp16 d): 8-bit ~95MB free; 6-bit ~120MB near-free;
4-bit ~140MB not worth it. (2) Perpendicular col-scales (per-input-channel, foldable: MLP-in→norm γ,
down-in→up rows; o_proj not foldable under GQA): **v1 "exact null" was a FOLD BUG (user caught it)** —
Qwen3_5RMSNorm is ZERO-CENTERED (fwd = x·(1+w)) and rotation leaves stored w≡0, so the w·a3 fold DISCARDED
the trained a3 (also invalidates the historical §3b A3(ii) "no gain" verdict). Fixed fold: w'=(1+w)·a3−1.
**v2 (fold verified, a3 range [.994,1.006]): real but tiny — paired ΔKL −0.00038±0.00007 (t=−5.5, 57.6%
seqs improved), all 7 metrics up.** Downstream-of-block-AP saturation CONFIRMED on valid footing (3 probes:
gate-honest flips null, col-scale ~1% of an epoch-gain, half the scale bits informationally dead).
--col-scale kept in the final 27B recipe (free at deploy). KEPT MODELS: scaling/16M_2ep (settled-recipe
baseline for paired comparisons) + scaling/16M_2ep_colscale_v2 (current-best incl. micro-win); KEPT DATA:
test1_data/{calib_16M.json, teacher_16M.pt} (standard 16M×2ep probe pair for the axes program).

## 8. Bonsai pivot → ternary embed+head → free-gen collapse → granularity → commit fix (2026-07-21→30)

**TARGET PIVOT.** New target = Ternary Bonsai 27B (94.6% of FP16, full-QAT to 30B tokens). Thesis = TOKEN-
EFFICIENT conversion (match within 5% at 100M-1B tok). Consequence: **ternarize embed_tokens AND lm_head** for
footprint parity (each [248320,2560] = 1.27B params ≈ 29% of the 4B). g128/g64 allowed (not TQ2_0-servable;
needs Bonsai-style packing).

**8a. THE FREE-GEN COLLAPSE (ternary lm_head) — diagnosed + FIXED.** Ternarizing embed+head scored 6% MMLU-Pro
/ 1% GPQA (BELOW random 10%/25%) while teacher-forced KL-agreement was 79-84% — i.e. **teacher-forced KL is
BLIND to free-gen collapse**. RAW completion was coherent; CHAT looped `<think>\n<think>…`. Diagnostic ladder
(3 researcher rounds): (1) plumbing ruled out; (2) lm_head special-row fp16 protection FAILED; (3) logit-gap
scan on FP hidden → ternary head ranks `<think>` LOWER than FP (head innocent); (4) ternary-BODY hidden makes
`<think>` top through BOTH heads; (5) embed deviation UNIFORM (special .435 ≈ content .434). ⇒ **context-
specific DISTRIBUTION COLLAPSE at the generic-calib-ABSENT `assistant\n<think>\n` position**, not a row defect.
FIX = chat/reasoning-format E2E + chat-context block-AP/GPTQ Hessians (+ trained lm_head assignments). Result:
below-random → functional. Gate built (src/freegen_gate.py, M1-M4 at induced boundaries): start_think argmax-
agreement **0.000 → 0.850**.

**8b. GRANULARITY SWEEP — g64@Q8 WINS (settled).** Size arithmetic: g64+8-bit scales == g128+fp16 == ~1.71 bpw.
8-bit scales are ~free even POST-HOC (g128: 80.65% fp16 → 80.62% post-hoc-8bit); QAT-8bit is tighter still.
Anchors (old generic pipeline, eval2k agreement / KL / OOD): **g256@fp16 79.31 / .3619 / 2.2171 (= TQ2_0) ·
g128@fp16 80.65 / .3244 / 2.1414 · g64@QAT-8bit 82.06 / .2793 / 2.0255**. g64@Q8 beats g128@fp16 by +1.4pt at
IDENTICAL size, +2.75 over TQ2_0, and beats the fp16-embed/head reference (80.85/.2981) — finer body
granularity outweighs ternarizing embed+head. Reasoning proxy (g64 anchor): late-gen 84.02%, answer 84.51%.
Also g128@Q8 (~1.65bpw = TQ2_0 size) = 80.62% ⇒ +1.3 free upgrade over g256 IF packable. On the CHAT recipe:
g256 72.77 → **g64@Q8 73.65** (+0.88, free-gen unchanged); g32@Q4 73.56 = tied/worse (4-bit scales cancel the
finer grid) ⇒ **trend flattens at g32; g64@Q8 is the deploy point.**

**8c. THE COMMIT / OVER-THINKING PROBLEM (4 runs; 3 failed) and its RESOLUTION.** After the collapse fix the
model still over-thought (wouldn't emit `</think>`) and looped. **Free-gen ledger @2048-token budget (48
prompts, temp .6; the 768 budget was CONFOUNDING "won't stop" with "budget too small" — FP itself only committed
48% @768): FP 75.0 commit / 25.0 trunc / 25.0 loop / 2.40 compR · V1(50% chat) 62.5/37.5/29.2/3.34 ·
combined-2560(10% chat) 56.2/43.8/39.6/3.15 · StageB 50.0/41.7/56.2/3.91.**
- FAILED #1 — density-within-chat: seq 1024 packing SPLITS `<think>`(pos0) from `</think>`(pos~1786) into
  different chunks; max_new alone moved corpus density only 0.6%→1.9%. Fixing seq to 2560 gave 4.3% but commit
  stayed 25%. (max-new close-rate sweep, src/sweep_maxnew_close.py: 5%@768 · 31%@1792 · 49%@2560 · 62%@4096;
  median close position 1786, 75th pct 2514.)
- FAILED #2 — Stage A commit-token REWEIGHT (α=3, require-close corpus 9.6% density): **exactly 25.0% = no-op.**
  Root cause: commit-window tokens are **0.135% of corpus**, and in a NORMALIZED weighted mean `Σ(kl·c·w)/Σ(c·w)`
  they got 0.40% of the weight (per-token multiplier 1.004×). Reweighting a rare class inside a normalized mean
  cannot move the objective.
- FAILED #3 — Stage B offline semi-on-policy (1000 student rollouts, teacher-scored, 30% own-trajectories):
  **WORSE on every axis** (commit 25→20.8 @768; @2048 50.0 commit / 56.2 loop). Cause: 58% of rollouts never
  closed, so uniform forward-KL over them = **self-distillation on the pathology** (model collapse,
  arXiv:2305.17493) + "KL Agreement Trap" (teacher/student stay close on degraded prefixes → weakest signal
  exactly where needed).
- ★ **SUCCEEDED — the ONE RUN (`output_4b/final_g64q8`): 50% require-close chat + 50% generic, `</think>`
  density 47.2%, 16M tok, seq 2560, g64@Q8, from combined-2560, lr 1e-5, + ADDITIVE commit objective
  `L = CAKLD_all + β·mean(KL over commit window)`, β=1.5.** The additive form is un-diluted by rarity
  (per-token weight 133× vs the reweight's 1.004×) and is SAFE at large β because the term is KL-TO-TEACHER
  (self-limiting), not a CE/reward close-bonus (which causes length attractors, arXiv:2010.07174).
  **Gate @2048: eval2k 80.46% · commit 75.0% · trunc 20.8% · loop 29.2% · compR 3.08 — ALL FOUR TARGETS PASSED
  (≥77 / ≥68 / — / ≤30 / ≤3.1). Commit is EXACT FP parity (both 36/48); trunc NOMINALLY BEATS FP (20.8 vs 25.0).**
  eval2k 80.46% is the highest of any full-recipe model. **The Pareto trade-off is broken** — previously only
  one of {high agreement, good commit} was attainable (V1 72.8/62.5 vs combined 79.8/56.2). Held-out fell
  monotonically 0.1841→0.1691 over all 20 evals ⇒ β=1.5 stable. Residual gap: compR 3.08 vs FP 2.40 (max 18.52
  ⇒ ≥1 severe loop persists); loop 29.2 vs 25.0 (within n=48 noise, ±12pt).

**8d. METHODOLOGY CORRECTIONS (each invalidated earlier numbers).**
- **Loop-gate bugs:** pad_token == `<|im_end|>` == EOS == 248046 and `config.eos_token_id` is None ⇒ generate
  never stopped and filled the budget repeating `<|im_end|>`; batch padding was also scored. Counting that as
  generation produced FAKE ~90% "loop" rates (FP included). Fix: pass `eos_token_id` AND truncate each gen at
  its first EOS. Second bug: 5-gram×3 / sentence×3 n-gram detector over-flags legitimate reasoning that
  restates drafts (flagged clean FP at 42%) → raised to ×5/×4; **zlib comp-ratio is the primary metric**
  (FP≈2.4, loopy >4). These two bugs inflated every loop number reported before 2026-07-24.
- **Held-out KL/flips are NOT comparable ACROSS runs** (user caught): the held-out slice is the last
  `--heldout-n` seqs of *that run's own corpus*. comb2560 held-out was 4% chat → KL .2497/flips 21.3%; stageB
  42% chat → **.1538/17.8%**; FINAL 46% → .1849/19.7%. FP-authored thinking traces are low-entropy and
  teacher-authored ⇒ trivially predictable. **StageB had the BEST held-out KL of any run and was the WORST
  model** (its held-out contained 30% of its own rollouts). Only eval2k (fixed 1946-seq set) and the loop-gate
  (fixed prompts) are cross-run valid.
- **False-positive abort:** the divergence guard `ho_worse*eval_every >= 300*abort_patience/3` reduces to
  `ho_worse>=1` at eval_every=300 ⇒ aborted the FINAL run at step 600/6244 on a +1% wobble, while also measuring
  PLAIN CAKLD during a `--commit-beta` run (the intended trade looks like divergence). Fixed: `--abort-patience 30`.
- **Perf:** ternary decode is ~18× slower than FP (packed TernaryScaleLinear dequantizes every step); batch 8 on
  1 GPU projected 36h for a rollout phase → data-parallel 2 GPUs + batch 24 = **5.7× faster**.

**8e. NEW TOOLING.** src/{freegen_gate,loop_gate,commit_diag,sweep_maxnew_close,gen_student_rollouts,
augment_corpus,fold_think_scale}.py; build_diverse_calib gained `--chat-frac/--chat-src` (chat as a SOURCE in
the diverse split, scales with `--tokens`); build_chat_calib gained `--require-close/--easy-frac/--seed/--device`;
e2e_qp_distill gained `--commit-beta/--commit-pre/--commit-post` (additive), `--scale-qat-bits`,
`--onpolicy-teacher-cpu` (accelerate cpu_offload — a PURE-CPU teacher fails: fla Gated-DeltaNet Triton kernels
are GPU-only), `--onpolicy-calib`, `--abort-patience`. Post-hoc `</think>`-row lm_head calibration built
(run_thinkcal.sh + fold_think_scale.py): lm_head IS a TernaryScaleLinear so scaling `scale[row*40:(row+1)*40]`
by c leaves assignments untouched = on-grid/TQ2_0-exact. **NOT applied — FINAL is already at FP-parity commit,
so the smallest c reaching parity is c=1.0.**

**8f. GGUF BENCHMARK — recipe VALIDATED, TQ2_0 packing FATAL for g64 (2026-07-30).**
MMLU-Pro (300q) / GPQA (198q), 4 samples, non-think MCQ, llama.cpp served:

model                          size    MMLU-Pro   GPQA     runtime(mmlu/gpqa)
FP-4B (Q8_0, ceiling)          4.2GB   47.7%      36.0%    2.5 / 1.7 min
** FINAL Q8_0 (faithful g64)   4.9GB   25.7%      30.3%    6.3 / 4.9 min
Qwen-2B fp8 (Q8_0)             2.1GB   27.4%      25.3%    2.0 / 1.3 min
Qwen-0.8B fp8 (Q8_0)           0.85GB  17.8%      13.3%    2.3 / 2.0 min
OLD tern4b (A6, g256 TQ2_0)    1.0GB   18.2%      17.4%    34.8 / 24.1 min
** FINAL TQ2_0 (g256 requant)  1.8GB   10.6%      16.8%    28.6 / 21.4 min
(the ternary-embed+head COLLAPSE we fixed: 6% / 1% — below random 10% / 25%)

TWO CONCLUSIONS:
(1) **THE RECIPE IS VALIDATED.** At faithful precision the FINAL model scores 25.7 / 30.3 vs the old ternary's
    18.2 / 17.4 (+7.5 / +12.9 pt). **On GPQA it BEATS Qwen-2B fp8 (30.3 vs 25.3) — a 2x-larger model** — and
    reaches 84% of the FP-4B ceiling; on MMLU-Pro it is just under the 2B (25.7 vs 27.4) at 54% of FP.
    Runtime 34.8→6.3 min is independent confirmation that it now STOPS instead of rambling (the commit fix).
(2) **TQ2_0 CANNOT HOLD A g64 MODEL.** Requantizing g64→g256 costs ~15pt MMLU-Pro / ~13.5pt GPQA and lands at
    random. Verified mechanically BEFORE exporting: the weights are on-grid at block 64 and NOT at block 256,
    so TQ2_0 (inherently g256) remaps the ternary assignments. The 28.6/21.4-min runtimes are the tell.
    Exporting Q8_0 alongside is what made this diagnosable — a TQ2_0-only run would have read as
    "the whole recipe failed".
CAVEAT ON THE SIZE THESIS: the validated quality is currently only reachable at Q8_0 (4.9GB), which defeats the
footprint argument. The weights ARE ternary at ~1.71 bpw — only the llama.cpp container cannot express g64.
DEPLOYMENT OPTIONS: (a) retrain BLK=256 → TQ2_0-exact, costs ~2.75pt agreement (pipeline supports it, one env
var); (b) custom g64 packing (Bonsai-style, needs a ggml fork) → full quality at ~1.71 bpw; (c) g128@Q8
(~1.65 bpw = TQ2_0 size, +1.3pt over g256, still not TQ2_0-exact).
RULE: always verify on-grid-ness at the TARGET block size before trusting a GGUF export, and always export a
faithful reference (Q8_0) alongside to separate model quality from packing loss.

**8g. TQ1_64 CUSTOM FORMAT — end-to-end benchmark (2026-07-31).** llama.cpp fork branch `tq1_64`;
format spec TQ1_64_SPEC.md; 512-weight superblock, 114 B, **1.7812 bpw** (27B projects to 6.01 GB).

model                        size    MMLU-Pro        GPQA
FP-4B Q8_0 (ceiling)         4.20 G  47.7% / 2.5m    36.0% / 1.7m
our model @ Q8_0             4.91 G  25.7% / 6.3m    30.3% / 4.9m
** our model @ TQ1_64 **     1.09 G  24.4% / 13.3m   28.8% / 11.2m
our model @ TQ2_0 (broken)   1.81 G  10.6% / 28.6m   16.8% / 21.4m
Qwen-2B fp8                  2.10 G  27.4% / 2.0m    25.3% / 1.3m
Qwen-0.8B fp8                0.85 G  17.8% / 2.3m    13.3% / 2.0m
old ternary A6 (g256 TQ2_0)  1.00 G  18.2% / 34.8m   17.4% / 24.1m

CONCLUSIONS.
(1) **TQ1_64 matches Q8_0 within noise** (-1.3 MMLU-Pro, -1.5 GPQA; n=300/198 x4 samples => ~+-5 pt CI) at
    **4.5x smaller**. Confirms the lossless prediction from eval2k (80.47 vs 80.46) and the 0.06% weight RMS.
(2) **vs the broken TQ2_0 export: +13.8 MMLU-Pro, +12.0 GPQA at 40% smaller.** Same weights, same recipe —
    the ONLY difference is g64 vs g256 scale granularity. This single row justifies the whole fork.
(3) **Same-memory claim holds**: vs Qwen-0.8B fp8 (0.85 G, the honest same-size competitor) +6.6 MMLU-Pro /
    +15.5 GPQA. vs Qwen-2B fp8 at 2x our size we win GPQA (28.8 vs 25.3), lose MMLU-Pro (24.4 vs 27.4).
(4) vs the old ternary at the same 1 GB: +6.2 MMLU-Pro / +11.4 GPQA (recipe + format together).

CUDA KERNEL PERF (RTX 3090, tg t/s): dequant->cuBLAS 16.7 -> superblock-per-thread 8.0 (SLOWER: uncoalesced)
-> superblock-per-warp 53.9 -> +shared trit staging & float4 y loads 74.2 -> +y amortised over 8 rows 76.9.
pp 462 -> 547. Both big wins were memory ACCESS SHAPE, not arithmetic.

**NEGATIVE RESULT — batched/MMQ kernel for prompt processing (REVERTED).** Traffic analysis said a fused
batched kernel should move ~19x less data than dequant->cuBLAS (5.3 vs 100 MB per 2560x9216 layer at batch 32)
and that the op is memory bound (~22 us of tensor-core math vs ~105 us of memory). Built it anyway and it LOST:
**pp 547 -> 286 (NB=4) -> 234 (NB=8)**. Two reasons the analysis was wrong: (a) register limits force NB
columns per pass, so batch 32 means 32/NB SEQUENTIAL passes that each re-read AND re-unpack the whole weight
matrix — the amortisation never materialises; (b) the NB inner columns are `ncols` floats apart, so their
float4 loads land on unrelated cache lines (raising NB made it worse, not better). Beating cuBLAS+tensor-cores
here needs a properly TILED MMQ (weights staged in shared and reused across a column tile), not a widened GEMV.
Reverted; prompt processing stays on dequant->cuBLAS. Generation keeps the fused GEMV.

**8h. TILED MMQ — SECOND ATTEMPT, ALSO REVERTED (2026-07-31).** After the widened-GEMV failure (§8g), tried a
lane-per-column mapping: lane L owns output column L, so accumulators are 1/lane (no register cap, ONE pass
over the weights) and trit reads become perfect shared-memory broadcasts. **pp 547 -> 129 t/s (4x worse).**
Cause: it trades the register wall for a MEMORY wall — each lane then streams its own activation column, so the
warp's float4 loads are `ncols` floats apart and each becomes a separate transaction (~4096 scattered loads per
warp per superblock). VERDICT ACROSS BOTH ATTEMPTS: you can amortise the unpack over columns OR make the
accumulators cheap, but not both, unless BOTH operands are staged in shared — i.e. a real tiled GEMM
(weight tile [BM x BK] unpacked to shared + activation tile [BK x BN] loaded coalesced to shared + per-thread
register micro-tile; BM=64/BN=64/BK=128 ~ 40 KB shared). That is a project, not an increment, against a
tensor-core cuBLAS already at 547 t/s. Prompt processing is also NOT the deployment bottleneck — generation is,
and that path is ours (fused GEMV, 77 t/s vs 16.7 for dequant->cuBLAS). Both attempts reverted; tree clean at
commit 3cbe66492.

**8i. FIRST END-TO-END `run_full_pipeline.sh` RUN — PLUMBING VALIDATED, GATES FAILED (2026-08-01).**
4B cold start (WORK=output_4bpipe, ORIG=untied_4b), TOTAL 19:59:20. Phase timings: rot 0:00:16 · chat pool
10:12:39 · calib 0:00:18 · Block-AP 1:15:56 · teacher 0:29:44 · E2E 6:44:04 · gates 1:16:39.
WHAT WORKED: corpus `</think>` density 47.5% (ref 47.2%); skeleton verified exactly on the g64 ternary grid
incl. embed_tokens + lm_head (sparsity .455); E2E held-out fell MONOTONICALLY over all 20 evals
(0.6828 -> 0.4265, flips 31.26% -> 25.79%), no divergence, no abort; commit term confirmed firing
(_add_commit_term has an empty-mask guard, so NOT a repeat of the Stage-A no-op).
**GATES FAILED: commit 41.7% (target >=68, ref 75.0) · trunc 58.3% (ref 20.8) · loop 33.3% (<=30) ·
compR 3.12 (<=3.1) · GateA top-1 68.86%.** Failure signature = the original over-thinking pathology.
**ROOT CAUSE (hypothesis): E2E UNDERTRAINING, not a recipe bug.** The reference FINAL was warm-started
(skeleton -> E2E on combined-2560 -> SECOND E2E at lr 1e-5) ~ 32M tokens of E2E reaching held-out 0.1691;
this cold run did ONE pass at lr 2e-5 (16M) reaching 0.4265 with the curve STILL DESCENDING at step 6000.
=> the script encodes a SINGLE-pass Phase 5 but the validated model needed TWO. Fix under test: second pass
at lr 1e-5 warm-started from output_4bpipe/e2eqp (reuses the same calib + teacher cache; no regeneration).
ALSO FIXED THIS RUN: Phase 2a was single-GPU (GPU1 idle ~11h) -> now shards one model per GPU inside ONE
memguard scope; and build_chat_calib.py gained --device-map/--gpu-mem/--cpu-mem because a plain .to(device)
load CANNOT hold the 27B (51.8 GiB) on a 24 GiB card -> Phase 2a would have hard-OOM'd at 27B.
NOTE: sharding gave only ~6% (not 2x) — see the thermal-coupling finding.

**8j. E2E PASS 2 (warm-start refinement) — NULL on the deploy gate (2026-08-02).** Testing the §8i
undertraining hypothesis: warm-started from the §8i cold-run E2E, lr 1e-5, 2ep, SAME calib + teacher cache
(no regeneration), 06:29:20. Held-out fell 0.4270 -> 0.3907 but PLATEAUED (last 300 steps moved 0.0004).
**Gate B: commit 41.7% (20/48) — IDENTICAL to pass 1's 20/48 · trunc 58.3->54.2 · loop 33.3->31.2 ·
compR 3.12->3.13 (max 22.16->12.75) · GateA 68.86->69.00%.** => a second E2E pass is NOT the missing
ingredient; ~2x E2E exposure moves commit by exactly zero.
**METHOD ERROR CORRECTED:** the hypothesis rested on comparing this run's held-out KL (0.4265) to the
reference's (0.1691). INVALID — different held-out sets (ours 47.5% chat = model-generated reasoning traces,
the reference's ~10% chat = mostly generic text; traces are intrinsically harder to predict). Only Gate B is
comparable across runs.
**ALSO ELIMINATED:** chat-pool quality. Ours vs the reference's chat_pool_final.json are statistically
indistinguishable — close-rate 95.1 vs 94.5%, close pos 1120 vs 1076 tok (frac 0.438 vs 0.420),
repeated-5gram .0305 vs .0299, 100% unique 64-tok prefixes in both.
SURVIVING HYPOTHESIS = corpus PATH (curriculum): reference did skeleton+pass1 on the 4.3% corpus and
introduced 47.2% chat only at pass 2; the pipeline uses 47.5% everywhere incl. the Block-AP Hessians.
Test running: run_curriculum.sh.

**8k. ROOT CAUSE — THE ROTATION WAS BROKEN; §8i/§8j/CURRICULUM ALL INVALID (2026-08-02).**
Chasing why C4 scored eval2k 70.53% (below V1's 72.77) with an absurd mean KL of 3.4764 nats and only
49/1,990,758 positions at FP p_max>0.5, the FP *reference* turned out to be broken:
  OLD rot (output_4b/rot, 2026-07-01): KL 0.0009 | agreement 99.06% | FP-confident 79.08%   <- lossless
  NEW rot (every rotation this campaign): KL 3.9949 | agreement 91.49% | FP-confident  0.00%  <- BROKEN
Diff of the two rotated models: 738/739 tensors IDENTICAL; only `lm_head.weight` differs. rms shows why —
original .013029, GOOD .040807 (x3.13), BROKEN .013029 (x1.00): the broken run ZEROED the final norm without
absorbing its gain. The final norm's effective gain is (1+w), mean 3.195, so FP logits shrank ~3.1x, softmax
went flat, and the model became confident essentially nowhere.
**MECHANISM:** the Bonsai pivot (2026-07-21) added 'lm_head.weight' to config.QUANTIZE_PATTERNS so block_ap
could ternarize the head. `is_rotatable_projection()` was defined as "matches QUANTIZE_PATTERNS", so this
silently flipped lm_head False->True, routing it into convert.py's FIRST branch (line ~203), which only
absorbs .self_attn./.linear_attn. layer norms. convert.py's dedicated lm_head handler — the one doing
`norm_w + 1.0` — became UNREACHABLE DEAD CODE. Timeline: good rot 07-01 (pre-change) · config rewrite 07-21 ·
all campaign rotations 07-31+ (broken). The reference final_g64q8 used the pre-pivot output_4b/rot, so ALL
REFERENCE NUMBERS STAND.
**BLAST RADIUS:** the teacher cache is built from $ROT, so every student in §8i/§8j/curriculum was distilled
toward a teacher that is never confident — which IS the "won't commit to </think>" pathology (commit 41.7%,
trunc 58.3%). Every gate/agreement number from 2026-07-31 onward is void, and the curriculum hypothesis was
explaining a symptom; it remains UNTESTED.
**FIX (verified):** is_rotatable_projection() now returns False for lm_head/embed_tokens, decoupled from
QUANTIZE_PATTERNS (which still returns True for lm_head so block_ap keeps ternarizing the head). Re-run
rotation is BIT-IDENTICAL to the 07-01 good one (max|diff| 0.000e+00) and reproduces KL 0.0009 / 99.06% /
79.08% confident.
**REUSABLE ACROSS THE REDO:** chat_pool.json and calibration_data.json/calib_eval.json are generated from the
UNROTATED untied_4b + CTM data, so they are UNAFFECTED — the 10.2h chat-pool generation does NOT repeat.
Regenerate: rotation (16s) + skeleton + teacher cache + E2E + gates ~= 9.8h.

**8l. RERUN WITH FIXED ROTATION — ROOT CAUSE CONFIRMED, 3/4 GATES PASS (2026-08-02).** run_full_pipeline.sh
cold, WORK=output_4bpipe, TOTAL 10:55:44 (chat pool + calib REUSED — rotation-independent — so the 10.2h
generation did not repeat). New fail-closed Phase-1 guard PASSED: rot-check KL 0.0006 / agreement 98.66% /
FP-confident orig 75.96% vs rot 76.13% (broken run: 0.00%).
**GATE B: commit 75.0% (36/48) = EXACT FP PARITY and matches the reference exactly · trunc 16.7% (BEATS both
the reference's 20.8 and FP's 25.0) · compR mean 3.04 (< ref 3.08) max 13.70 (< ref 18.52) · loop 37.5% (target
<=30, ref 29.2) · think_len 630 over 40 closers.**
**GATE A (now TRUE eval2k, frozen 1946-seq referee): 77.53% — PASSES the >=77 gate** (ref 80.46). Sanity
restored: mean KL 0.4012 nats (was 3.4218) and FP-confident>0.5 at 1,315,619/1,990,758 positions (was 49).
E2E held-out fell 0.3243 -> 0.2160 monotonically (v1 broken: 0.6828 -> 0.4265).
=> **The ENTIRE commit pathology (41.7% -> 75.0%) was the broken FP teacher of §8k**, not undertraining, not
warm-restart, not chat-pool quality, not curriculum. §8i/§8j conclusions are void; the curriculum hypothesis
is UNTESTED and now live again.
OUTSTANDING: loop 37.5% vs <=30. CAUTION — n=48 gives a +/-12pt binomial CI, so 37.5 vs ref 29.2 is WITHIN
NOISE, and both continuous measures of the same pathology IMPROVED (compR mean and max). Tighten n before
spending a 17h curriculum run on it. Residual eval2k gap 77.53 vs 80.46 (-2.93) is consistent with the
chat-throughout cost seen historically (V1 chat-throughout 72.77 vs combined-2560 79.76 vs FINAL 80.46).

**8m. QAT DATA-SCALING SWEEP — skeleton scales with DATA not epochs, saturates ~512 samples (2026-08-04).**
Block-AP skeleton only (NO E2E; absolute eval2k 61-64% is far below the E2E'd 77.53% — comparable only within
the sweep, per eval-stage-comparability). Corpus = the fixed 47.5% calib; frozen eval2k referee, NP=512.
RAM solved via ACT_SPILL_GB=4 (see §8-RAM / [[nvme-activation-spill]]).
  point  samples  epochs  tokens   updates  eval2k-agree  KL
  A      128      4       0.33M    512      61.88         0.9954
  B      256      4       0.66M    1024     63.18         0.9158
  C      512      4       1.31M    2048     64.01         0.8627
  D      1024     4       2.62M    4096     63.80         0.8739
  E      256      16      0.66M    4096     61.89         0.9676
VERDICT:
 1) SCALES WITH DATA, NOT OPTIMISATION. The update-matched control (D vs E, both 4096 updates): D (4x the
    distinct data) = 63.80 beats E (4x the epochs, same data) = 61.89 by +1.91 agree / -0.094 KL. And the
    epochs axis at fixed data (B 256x4=63.18 vs E 256x16=61.89) shows MORE EPOCHS ON THE SAME DATA slightly
    HURT (overfit the calib). So the skeleton wants distinct tokens, not more passes.
 2) DATA SATURATES FAST. A->B->C rises 61.88->63.18->64.01, then D (2x C's data) = 63.80 ~ C (TIED, 0.21pt =
    noise; KL likewise C .8627 ~ D .8739). Knee at ~512 samples ~ 1.3M tokens.
 3) IN-SAMPLE KEEPS FALLING, HELD-OUT DOESN'T. Block-MSE(L31) 4.00->3.80->3.56->3.40e-2 (A->D) monotone, but
    held-out agreement peaks at C — beyond ~512 samples extra data cuts training error without generalising
    (mild overfit regime).
IMPLICATION FOR 27B: the block-AP skeleton needs only ~512-1024 samples (~1.3-2.6M tok) x 4 epochs to
saturate — cheap. Do NOT pour the token budget into skeleton calibration or extra QAT epochs. Reserve tokens
for E2E, which DOES scale with tokens (OOD) at 2 epochs (§7). Skeleton calib can stay modest at 27B.

**8n. STAGE-0 DIAGNOSTIC — CAPACITY CEILING CONFIRMED (2026-08-04, tools/gram_diagnostic.py).** Retraining-free
test of the report's capacity-vs-estimation question, from the FP model's activation Gram H=XᵀX (layers 8/16/24,
1024 calib + 256 held-out seqs @2560). TWO clean results:
 (1) H-SHAPE SATURATES AT THE SWEEP KNEE. Trace-normalized covariance-shape distance ‖Ĥ_n−Ĥ_1024‖_F/‖·‖:
     L16 128:0.057 256:0.034 512:0.016 → within ~3% of converged by 512 samples (=1.3M tok), exactly where
     eval2k agreement plateaus (§8m). r_eff=tr/λmax ≈ 2.3/2.7/4.7 (L8/16/24) — Gram dominated by 1-2 massive
     directions; stable rank 5-21; top-256 eigvecs hold 62-73% energy. (NOTE: the raw run's 'spectral_conv'
     0.88/0.75/0.50 was a SCALE ARTIFACT — H is an unnormalized SUM so H_1024≈2·H_512; trace-normalize first.)
 (2) CALIB H SPANS HELD-OUT. Held-out energy captured by CALIB top-k eigenbasis vs held-out's OWN top-k:
     ratio 0.94-0.985 across layers/ranks (L16 top-256: calib .686 vs own .700 = 0.98). Only ~1.5-6% of
     recoverable second-moment energy is missing ⇒ more/better calibration data cannot help.
VERDICT: the §8m plateau is a CAPACITY/ASSIGNMENT gap on the ternary grid, NOT a data/estimation problem —
confirms [[fixed-teacher-ceiling]] directly. By the report's threshold (>2pt from shrinkage+held-out-selection
⇒ estimation), we can predict shrinkage WON'T lift it (≤6% Gram headroom), so we SKIP that experiment. The lever
is the report's Stage 2: bounded CE + MOBILE trit assignments (warm-start 77.5%), the only objective that
reintroduces data-scaling. Raw Grams saved logs/gram_diag_grams.npz (re-analyzable without re-running forwards).

**8o. ORACLE (ASSIGNMENT LEVER) — FIRST-ORDER FLIP RANKING IS INVALID; explains the cascade (2026-08-05).**
`tools/oracle_assignment.py`: freeze the 77.5% model, fresh gradient of held-out CE+KL wrt down_proj effective
weights (Arm-B mutable trits + grad accumulator), rank candidate trit changes by first-order Δ̂=ḡ·s·(t'−t),
commit top-K UNCONSTRAINED, measure actual held-out loss + FP-agreement, revert, sweep K. Baseline KL .2949 /
CE 1.4403 / agreement 77.72% (FP gap 22.28 pt; matches the deployed 77.53% — harness sane).
**RESULT: every budget makes it WORSE, monotonically —** K=1e-6 (736 flips) agree −0.15pt · 1e-5 −2.18 ·
1e-4 −25.5 · 1e-3 −71.0 · 1e-2 −77.7 (0% agreement). Best "recovery" −0.7%.
**INTERPRETATION: the test measured the SEARCH, not the lever.** We commit only flips with predicted Δloss<0
yet actual loss RISES every time ⇒ first-order ranking cannot identify beneficial trit flips. Reason: a flip
moves a weight by s·d where s ≈ that weight's own magnitude ⇒ a ~100% perturbation, so the linear Taylor term
is meaningless at that step size.
**⇒ THIS EXPLAINS THE §8-CE CASCADE (bimodality).** STE-latent flipping uses the SAME first-order signal, so
flips systematically damage the model → loss ↑ → grads ↑ → more boundary crossings → runaway. Bimodality and
the oracle failure are ONE root cause. Also retro-explains Arm-B: its accept/reject gate was empirically
screening out these bad predictions, hence stable-but-only-0.0001%-committed.
**NOT evidence of capacity-limited** — a null from a broken search says nothing about whether good assignments
exist. To actually measure the assignment lever, need a SECOND-ORDER method: OBQ/GPTQ Hessian saliency
Δ=(Q(w)−w)²/[H⁻¹]_ii WITH the compensating update (which is exactly what block-AP's GPTQ init already does
per-layer), or AdaRound-style relaxation, or direct held-out screening of small flip batches.

**8p. SECOND-ORDER ORACLE (GPTQ re-solve) — ALSO WORSE; current model is a strong JOINT optimum (2026-08-05).**
`tools/oracle_gptq.py`: collect H=XᵀX at each down_proj input at the CURRENT operating point (post-E2E, 24
seqs), re-run `_gptq_ternary(W_fp, H, g64)` (Hessian-weighted rounding WITH H⁻¹ error compensation — the thing
first-order lacks) from the rotated FP weights, install, measure.
  baseline (current)                    agreement 77.72%  CE+KL 1.0151
  GPTQ re-solve, own absmax scales      62.48% (−15.24)   1.9776   [19.66% of assignments changed]
  GPTQ re-solve, E2E scales KEPT        67.61% (−10.11)   1.5873   [same 19.66%]
Keeping the trained scales recovers ~5pt of the damage (scales matter) but the ASSIGNMENT change still costs
10pt ⇒ NOT a scale confound: the 19.66% of assignments GPTQ prefers are genuinely WORSE for the end loss.
**CONCLUSION (with §8o): two independent principled searches both fail to beat the current assignment.**
(a) first-order end-loss ranking is invalid (flip = ~100% weight perturbation); (b) second-order GPTQ is
optimal for the LAYER-WISE proxy ‖X(W_fp−Q)‖², which is NOT the end loss — E2E spent 16M tokens co-adapting
scales to the EXISTING trits, and a fresh layer-wise solve discards that joint optimum. The deployed model sits
in a strong JOINT optimum of (assignments × scales).
**STILL NOT proof no better assignment exists** — both failed methods optimize the wrong thing (invalid step
model / wrong objective). Untested: a method optimizing the END loss over discrete assignments (AdaRound-style
relaxation, or V/P alternation with MANDATORY held-out screening). But the cheap routes are closed.

**8q. STAGE 0 (assignment-mobility fix) — BUILT; init repaired but NOT sufficient alone (2026-08-05, partial).**
Implemented in e2e_qp_distill.py per the data-appetite report: (1) `--latent-init fp-spread --fp-model <rot>` =
bin-clamped FP-spread latent init `L=clamp(w_fp,(t∓0.5)s)` (asserts no assignment changes ⇒ byte-identical
function at init; logs the near-boundary fraction); (2) `--latent-warmup-steps` = hold latent lr at 0 while Adam
accumulates v̂ (verified on GPU that the group-wise lr mask freezes/releases latents correctly); (3) TALR
`--target-tr/--tr-every/--tr-final-frac` = servo the latent lr to a target TRANSITION RATE (lr alone can't
control flip count), annealed coarse→fine, clamping harder on overshoot (x0.6) than opening up (x1.3). Also
lowered CE weight to 0.1 per the report (keep the bounded KL dominant).
**PARTIAL RESULT (run stopped early, init variable only — NO warmup, NO TALR):** near-boundary fraction
**0.000% → 26.06%** (real FP ≈9.8%) ⇒ the degenerate init IS repaired. But at the same latent-lr 5e-3 that
cascaded before, it still cascaded — and FASTER (assign-moved 15.37% and KL 15.59 by step 5, vs 0.015% at step
5 previously). Consistent: 26% near-boundary is ~2.7x FP density, so an even larger poised population crosses at
once when the step is that large.
**⇒ init is NECESSARY BUT NOT SUFFICIENT (as the report predicted).**
**FOLLOW-UP RUN (init + warmup + TALR, latent-lr 1e-4, warmup 50, target-tr 5e-4) — DECISIVE NEGATIVE, and it
indicts the DESIGN not the init: at step 20 the LATENTS ARE FROZEN (warmup holds latent lr = 0) yet
assign-moved is already 2.85% and KL has exploded 0.21 → 3.30.** Latents cannot move, so those flips come from
the SCALES (still training at lr 1e-5): with fp-spread latents sitting NEAR boundaries, ordinary scale training
moves the quantization boundary UNDER them and flips assignments wholesale. Reproduced twice (KL 3.47 / 3.30).
**⇒ THE WARMUP FROZE THE WRONG THING.** Bin-centre init was accidentally "protecting" assignments from scale
motion (latents 0.5s from a boundary); fp-spread removes that protection, so scales and assignments become
tightly coupled — which is exactly the (assignment x scale) coupling of §8p's joint optimum, now observed
dynamically. **This is direct evidence for the report's Q4: co-training scales and assignments is the LEAST
stable option; the V/P alternation (V-phase = assignments move with s FROZEN; P-phase = s refit with
assignments frozen) is REQUIRED, not optional.** Next: freeze scales entirely during the V-phase (lr 0 on the
scale group, not the latent group) and only then apply TALR to the latents.
(Also fixed en route: the diagnostics themselves OOM'd — a float32 755M-trit snapshot is 3GB; assign-moved% and
TALR now share ONE fixed 1M-weight sample. Runs still OOM near the end at 22.7GB: fp32 latents 3GB + paged Adam
+ activations is simply at the edge on 24GB for down_proj scope.)

**8r. V-PHASE (scales FROZEN) — GRADED REGIME ACHIEVED; kill-criterion-1 does NOT fire (2026-08-06).**
Config: fp-spread init + `--lr 0` (scales FROZEN = the V-phase) + latent-lr 1e-4 + warmup 20 + TALR target
5e-4→1e-4, 8 down_proj layers (`--tw-layer-stride 4`, added because 32-layer fp32 latents ≈3GB sits at the very
edge of 24GB and OOM'd repeatedly).
**THE DECISIVE CONTRAST (same setup, step 20): scales TRAINING → KL 3.30 (destroyed); scales FROZEN → KL 0.2612
(intact).** Scale motion was driving the cascade: with fp-spread latents sitting NEAR boundaries, training the
scales moves the quantisation boundary UNDER them and flips assignments wholesale. V/P separation is REQUIRED.
**TALR works as a controller:** measured 1.81e-3/step overshoot, cut latent-lr 2.16e-5→1.30e-5, rate fell to
6.89e-4 → 5.7e-4 → ... → 1.4e-4, tracking the annealing target. Flip rate CONVERGED instead of running away
(contrast: every earlier run went 0.015% → 18% → 37%).
**BUT the net effect on a POST-E2E model was NEGATIVE:** KL 0.2612 → 0.4122 (burst at step 40) → recovered
monotonically to 0.3277 by step 240; flips 21.90% → 24.71%. `assign-moved` burst to 10.09% at step 40 then
FROZE (10.09→10.18 over the next 200 steps) ⇒ **the un-servoed first post-warmup steps did ALL the movement AND
all the damage**; the controlled phase could only partially undo it. TALR has no measurement to act on until
tr_every steps after warmup, so the base latent-lr is applied raw. FIX: start latent-lr LOW (~5e-6) and let TALR
ramp UP (it opens x1.3 when below target) instead of starting hot and clamping down.
**INTERPRETATION (user's point, correct): testing on the post-E2E model is the WRONG subject.** That model has
16M tokens of scale co-adaptation to its existing trits — the §8p joint optimum — so moving assignments with
scales frozen can only hurt. **In implementation the order is block-AP → assignment stage → E2E**, so the
assignment stage should be tested on the RAW block-AP skeleton, whose scales have NOT been co-adapted. Next test
does exactly that (and per user: NO E2E afterwards until the assignment stage is optimised — E2E is 6+h).

**8s. ASSIGNMENT STAGE ON THE RAW BLOCK-AP SKELETON — PRODUCTIVE (2026-08-06). The lever works.**
Same V-phase machinery as §8r (fp-spread init + scales FROZEN `--lr 0` + TALR), but applied to the RAW
block-AP skeleton (`output_4bpipe/modified_model`, pre-E2E) instead of the post-E2E model — the actual
implementation order (block-AP → assignment stage → E2E). Base latent-lr lowered 1e-4 → 5e-6 so TALR ramps UP
from below instead of bursting. 8 down_proj layers (stride 4), 240 steps, scales frozen throughout.
  step  20 (baseline, latents still frozen)  KL 0.7358  flips 35.54%  assign-moved 0.000%
  step  60                                   KL 0.5118  flips 30.27%  assign-moved 3.350%
  step 120                                   KL 0.4788  flips 29.35%  assign-moved 3.556%
  step 240 (final)                           KL 0.4625  flips 28.97%  assign-moved 3.655%
**KL −37% (0.7358→0.4625); agreement 64.5% → 71.0% (+6.5 pt); MONOTONE at every eval; no cascade, no OOM.**
Only 3.66% of assignments moved — sparse targeted flips, not wholesale churn. TALR tracked its annealing target
the whole way (8.96e-4 → 2.3e-5 as the target annealed 4.3e-4 → 1.0e-4). The low starting lr fully fixed §8r's
burst (3.21→3.66% gradual vs 10.09→10.18% front-loaded).
**THE CONTRAST IS THE FINDING — same machinery, opposite sign:**
  post-E2E model (77.5%): KL 0.2612 → 0.3277  ⇒ HURTS
  raw block-AP skeleton : KL 0.7358 → 0.4625  ⇒ HELPS (−37%)
The post-E2E model's scales are co-adapted to its trits over 16M tokens (§8p joint optimum), so assignment moves
can only break it; the raw skeleton's assignments are genuinely suboptimal and improvable. **The assignment
stage belongs BETWEEN block-AP and E2E, never after E2E.**
CAVEATS: 8/32 layers, 4-seq held-out slice, 240 steps — trend is strong and monotone but magnitude needs a
fuller run. NO E2E run yet (deliberate, per user: optimise the assignment stage first; E2E is 6+h).

**8t. TEST 1 — LAYER COVERAGE (2026-08-06). Full 32 needs offload; more coverage is NOT free.**
INFRA ADDED: `--latent-offload` — assignment latents (and their grads) live in CPU RAM, streamed to GPU inside
each layer's CHECKPOINTED forward, so only one layer's latent is GPU-resident; autograd routes the grad back to
the CPU leaf. Frees 6.04GB ⇒ full 32-layer coverage fits. Forces `torch.optim.Adam` (bitsandbytes CANNOT step
CPU params — verified) and **single-GPU (DDP rejects mixed cpu/cuda module params — ValueError)**. Also added
`--tw-layer-stride` (subset of layers). bf16 latents are NOT an option: the Adam update is ~2.5e-4 of the latent
magnitude vs bf16's 3.9e-3 resolution ⇒ swamped entirely.
**METHODOLOGICAL CATCH: `--target-tr` is a FRACTION of all latents, so holding TR fixed while raising coverage
raises ABSOLUTE flips/step proportionally** (32 layers @5e-4 = 377k flips/step vs 8 layers @5e-4 = 94k). The
first 32-layer run was therefore 4x more aggressive, not a coverage test — it degraded (0.7355 → 0.7925) and was
CPU-bound at ~46s/step (~6h), so it was killed. Matching ABSOLUTE flips is the right control for perturbation
size (matching the FRACTION would instead hold per-layer optimisation constant — the two answer different
questions; neither is uniquely "fair").
**COVERAGE AT MATCHED ABSOLUTE FLIPS (baseline 0.7355):**
  step:        +20     +40     +60     +80    +100    +120
  8L @5e-4   0.5804  0.5118  0.4942  0.4840  0.4788  (final 0.4625, −37%)
  16L @2.5e-4 0.7535  0.6173  0.5725  0.5548  0.5392  0.5261 (still descending at step 150/240)
16L dips first then recovers monotonically — NOT the flat degradation of the over-aggressive 32L run, which
supports "the 32L result was flip-rate, not a coverage ceiling". But 16L tracks ~0.06 BEHIND 8L at equal step
count, i.e. at matched perturbation, spreading the same flips over 2x the layers gives each layer half the
optimisation. **Provisional: more coverage is not free; 8 layers (stride 4) is the better cost/benefit so far.**
16L also ran ~35s/step vs 8L's ~10s/step. Final 16L number pending.

**8u. DEPTH IS A GENUINE PROBLEM — inter-layer error COMPOUNDING, not per-layer imbalance (2026-08-06).**
Checked explicitly because the 27B is 64 layers, so anything broken at 32 is worse at 64. Added a PER-LAYER
flip diagnostic (sampled, logged as per-layer[min/med/max/ratio] each eval).
**Per-layer flip rates are UNIFORM** — 32L run shows min 2.63 / med 3.10 / max 4.05 (ratio ~2x, tightening to
~1x later) ⇒ the single global latent-lr and global TR target are NOT producing per-layer imbalance, and no
layer is cascading while others sit inert.
**Yet at the SAME per-layer flip rate (TR 5e-4), first-post-warmup damage scales sharply with coverage:**
  8 layers  0.7358 → 0.5804  (improves immediately)
  16 layers 0.7353 → 0.7535  (small dip, recovers to 0.5054 final)
  32 layers 0.7355 → 1.9080  (2.6x WORSE than baseline; recovering 1.36 → 1.20 → 0.83 but still above baseline
                              at step 100)
Uniform flips + sharply worse aggregate damage ⇒ **INTER-LAYER ERROR COMPOUNDING**: each layer's assignment
change perturbs its output and downstream layers see shifted inputs, so simultaneous updates compound
multiplicatively through depth. With stride 4 the 24 untouched layers act as a stabilising scaffold.
**⇒ STRUCTURAL, and WORSE AT 64 LAYERS (27B). Do NOT move all layers' assignments simultaneously.**
FIX (added): `--tw-layer-offset` — with `--tw-layer-stride N`, select layers where idx%N == offset, so N
SEQUENTIAL GROUP passes (offset 0..N-1, each warm-starting from the previous) cover every layer while only ever
perturbing 1/N at a time. This is the same reason block-AP already goes layer-by-layer.
ALSO CORRECTED: the earlier 8>16>32 ordering (§8t) was NOT a floor result — 8L had PLATEAUED (0.4648→0.4637→
0.4625) while 16L was still descending; and matched-ABSOLUTE-flips starves each layer of updates. More
trainable assignments must have a LOWER floor (32L strictly contains 8L's degrees of freedom); the correct
control is matched FRACTION + matched steps. 16L final at matched-absolute = 0.5054.

**8v. PER-LAYER TALR — implemented, but does NOT fix the depth blowup (2026-08-06).**
Built one optimizer group PER LATENT LAYER + a per-layer flip-rate servo (`--per-layer-tr`, default on;
`--no-per-layer-tr` restores the shared group). Motivated by a measured burst: on an 8-layer run ONE layer moved
15.5% of its assignments in the first post-warmup steps while the median layer sat at 3.3% (5x), invisible to a
global controller that only sees the aggregate.
**RESULT ON 32 LAYERS (TR 5e-4, the config that failed): global TALR 0.7355 → 1.9080; PER-LAYER TALR 0.7355 →
2.0105.** No improvement ⇒ **per-layer imbalance is NOT the cause of the depth blowup.** The controller does
work (per-layer gains differentiate, 0.05–0.36), but the failure is AGGREGATE perturbation across depth: 32
layers each moving ~3% compounds; 8 layers each moving ~3% does not.
CORRECTION to §8u: the burst layer IS present in 32L runs too (max 14.90 vs med 3.05 here) — the earlier
"uniform, ratio 2x" reading was one run's numbers and does not generalise. Per-layer TALR is still worth keeping
as robustness (it fixes a real, otherwise-invisible pathology), it is just not the depth lever.
REMAINING PATHS for full coverage: (a) SEQUENTIAL GROUP PASSES — pass 0 (8 layers, 120 steps) already reached
KL 0.7358 → 0.5126 and is preserved at output_4bpipe/seqassign/pass0; (b) 32 layers at a MUCH LOWER TR
(1.25e-4 = matched aggregate perturbation vs 8L@5e-4), the direct analogue of 16L@2.5e-4 which dipped then
recovered to 0.5054.

**8w. ★ FULL 32-LAYER COVERAGE WORKS — the depth "blowup" was an un-servoed BASE-LR burst (2026-08-06).**
The TR target barely mattered: 32L@TR5e-4 gave assign-moved 3.379% / KL 1.908, and 32L@TR1.25e-4 gave 3.354% /
1.864 — nearly identical. Reason visible in the TALR trace: measured rate 7.45e-4 vs a 9.17e-5 target (8x over)
with the gain already clamped to 0.22 → **the damage happens in the ~5 steps between warmup ending and TALR's
first measurement, where the BASE latent-lr is applied raw.** With 32 layers x ~24% of latents near a boundary,
that one window flips ~3.3% of ALL assignments at once. TR is irrelevant because the burst precedes the
controller; the real control for those steps is the base latent-lr, which had been tuned on 8 layers (5e-6).
**FAIR TEST — 32L with base latent-lr 5e-7 (10x lower):**
  step 20 (baseline) KL 0.7355   assign-moved 0.000%
  step 40            KL 0.7079   assign-moved 0.002%   [talr] rate 0 < target ⇒ gain OPENED to 2.20
  step 60            KL 0.6293   assign-moved 0.227%   [talr] rate 1.34e-4 vs target 7.5e-5, gain 0.79-2.86
Monotone improvement from the first eval, NO blowup, gradual flips, TALR in genuine two-sided control (it opens
up when under target, clamps when over).
**⇒ REVISES §8u: depth compounding is NOT the barrier. Simultaneous full-coverage assignment-QAT is viable;
the base latent-lr must simply be scaled DOWN as coverage grows** (8L:5e-6 → 32L:5e-7). For the 64-layer 27B,
size the base lr to the coverage (or ramp it from 0) rather than assuming layer-group passes are required.
(The per-layer ratio 9155x at step 40 is a divide-by-near-zero artifact — median 0.00% — not real imbalance;
by step 60 it is a healthy 3x.)

**8x. BUG — save path hung the worker under --latent-offload (2026-08-06, fixed).**
Symptom: after the last training step the run stalled indefinitely; RSS 44.9GB, CPU 385%, GPU idle, and
torchrun emitted continuous `RendezvousTimeoutError` heartbeat failures. Cause was the CPU-save path added for
the earlier save-time OOM (§ CE-stage work): it unconditionally did `core.to("cpu")` + dequant-in-RAM whenever
`_mem_eff` was on. Two faults: (1) under `--latent-offload` the latents are ALREADY off-GPU (only 7.2GB VRAM in
use, 16GB free) so the CPU move is unnecessary — and it cost a ~45GB CPU dequant that blocked the worker long
enough for torchrun's rendezvous heartbeat to time out; (2) `_dev = next(core.parameters()).device` resolved to
**cpu** when the first parameter was an offloaded latent, so `core.to(_dev)` never returned the model to the GPU.
FIX: only take the CPU-save path when `_mem_eff and NOT latent_offload`, and resolve `_dev` from the first CUDA
parameter with a fallback to `device`.

**8y. BUG — torch.optim.Adam `foreach=True` blew host RAM and wedged the offloaded run (2026-08-06, fixed).**
Symptom: the 32L offloaded run stopped progressing at step 100, log silent for 50 min, process in **D state**
(uninterruptible sleep), RSS **43.8GB**, CPU 167%, GPU 0%. The cgroup guard was MemoryHigh=42G, so RSS crossed
it and the kernel throttled the cgroup into synchronous reclaim — the same signature as the block-AP throttle.
ROOT CAUSE: identified host-RAM budget was only ~19.6GB (latents 3.02 + Adam state 6.04 + grads 3.02 + teacher
cache 6.10 + Wlm 1.27 + calib 0.13), a 24GB gap. `torch.optim.Adam` defaults to **foreach=True**, which
allocates same-size temporaries across the WHOLE param group during the step — for a 755M-param CPU group that
is many extra GB.
FIX: `torch.optim.Adam(opt_groups, foreach=False)` for the offload path (per-tensor stepping; bounded memory).
**RSS 43.8GB → 23.9GB, state D → S, CPU 167% → 690%, and ~8s/step vs ~36s/step** — so most of the "CPU offload
is slow" impression was actually this reclaim thrashing, not PCIe traffic. Guard for offloaded runs sized to
MemoryHigh 47G / MemoryMax 50G (steady 24G sits far below; deliberately under the 52G that froze the host
2026-06-18).
NOTE: the `lr=0.00e+00` shown on step lines is NOT a bug — that field prints the SCALE lr, which is 0 by design
under `--lr 0` (V-phase). The latent lr appears in the `[talr]` lines.

**8z. LATENT-LR SCALING RULE (note for the 64-layer 27B).**
`--latent-lr` is applied RAW between warmup ending and TALR's first measurement (~tr_every steps); too large for
the coverage ⇒ that window flips a large fraction of ALL assignments and the model blows up (TALR clamps too
late). Measured on 4B down_proj: 8L/189M latents @5e-6 = OK (first-eval 3.2%, KL→0.4625); 32L/755M @5e-6 =
BURST 3.3% ⇒ KL 1.91; 32L/755M @**5e-7** = OK (first-eval 0.002%, KL→0.4513 still falling). So 4x the latents
needed ~10x lower lr (≈N^-1.66, faster than 1/N).
27B: 64 layers x down_proj[5120,17408] = **5704M latents** = 7.6x the 4B-32L case ⇒ extrapolates to 6.6e-8
(∝1/N) … 1.7e-8 (∝N^-1.66), i.e. **~1e-8..7e-8 — but CALIBRATE, don't extrapolate**: run with a small
--eval-every and require first post-warmup `assign-moved` ≲0.05%; drop 10x and repeat if higher. Starting too
LOW is self-correcting (TALR opened gain to 2.20 on the good 32L run); starting too HIGH is not.
BETTER FIX (unimplemented): ramp the latent lr from 0 over ~tr_every x4 steps after warmup so no raw lr is ever
applied — removes the per-coverage hand-tuning entirely.

**8aa. ★ TEST 1 SETTLED — FULL 32-LAYER COVERAGE IS BEST (2026-08-06).** 360 steps, base latent-lr 5e-7,
TR 1.25e-4, scales FROZEN, fp-spread init, latent-offload.
  step:   20(base)   40      80     120     160     200     240     280     320     360
  KL:     0.7355  0.6684  0.5329  0.4819  0.4708  0.4675  0.4587  0.4513  0.4495  0.4493  (converged)
  FINAL COMPARISON (from the same 0.7355 skeleton baseline):
    8L  120 steps  trits moved 4.75%  KL 0.5126  (-30%)
    8L  240 steps  trits moved 3.66%  KL 0.4625  (-37%)
    16L 240 steps  trits moved 3.43%  KL 0.5054  (-31%)
    **32L 360 steps trits moved 0.80%  KL 0.4493  (-39%)  ← best floor, ~5x FEWER flips**
Gradual TALR-controlled flips are far better targeted than a burst. Per-layer spread stayed healthy (2x).
INTEGRITY VERIFIED on the saved 10.59GB model: down_proj trits changed 0.754% and its scales are BIT-IDENTICAL
(Δ=0.00e+00 ⇒ `--lr 0` genuinely froze them, a pure V-phase); gate_proj/up_proj TRITS 100% unchanged (0.000%)
with only a ~0.2% per-block scale shift from `--scale-qat-bits 8` re-quantising onto the 8-bit log grid at save;
output is foldable g64 ternary. Save completed cleanly (the §8x fix held).
**⇒ The "depth barrier" of §8u was TWO ORDINARY BUGS — an un-rescaled base latent-lr (§8z) and
Adam(foreach=True) exhausting host RAM (§8y) — NOT anything about depth. More trainable assignments do have a
lower floor, as expected. For the 27B: full simultaneous 64-layer coverage should work; no sequential
layer-group passes needed.**

**8ab. ★ TEST 2 — TR SWEEP: 1.25e-4 IS NEAR-OPTIMAL, CLEAN INTERIOR OPTIMUM (2026-08-06).**
32L config held fixed (base latent-lr 5e-7, scales frozen, fp-spread init, offload, 360 steps, annealed
schedule --tr-final-frac 0.2); ONLY --target-tr varied. Baseline 0.7355.
  TR 5e-5    trits moved 0.396%   final KL 0.4572   converged (0.4573→0.4572)
  TR 1.25e-4 trits moved 0.796%   final KL **0.4493**  converged (0.4495→0.4493)   ← BEST
  TR 5e-4    trits moved 2.633%   final KL 0.4849   NOT converged (0.4948→0.4849, still descending)
**Both neighbours worse ⇒ genuine interior optimum, not an edge.** Too few flips plateaus HIGHER (5e-5 had
converged, so it is a real ceiling from insufficient movement, not slower pacing); too many flips picks WORSE
ones (5e-4 moves 3.3x more trits and is still 0.036 behind at equal budget).
CAVEAT: 5e-4 had NOT converged, so its ceiling is unknown — the defensible claim is that it is less efficient
per step at equal budget, not that its floor is higher.
**KEY CONFIRMATION: TR 5e-4 was completely STABLE here (0.7355→0.4849 monotone), whereas the SAME TR at base
latent-lr 5e-6 destroyed the model (→1.91).** So the transition rate was never the destabilising variable — the
earlier catastrophe was entirely the pre-TALR base-lr burst (§8z). Under proper control a high TR merely
degrades quality; it does not blow up.
⇒ RECIPE SETTING for the 27B: target-tr ~1.25e-4 (annealed to 0.2x), with the base latent-lr CALIBRATED to the
coverage (§8z), NOT swept.

**8ac. GRADIENT RELEASE — the right way to fit larger assignment scopes (2026-08-07).**
Goal: full MLP scope (gate+up+down) at 32 layers = 2.26B latents. With offloaded fp32 latents the host budget is
16B/latent (params 4 + grads 4 + Adam m 4 + v 4) = 36.2GB + ~12.9GB fixed = **49.1GB ⇒ throttles** at
MemoryHigh=47G.
REJECTED: swapping Adam for SGD+momentum (12B/latent). It saves the memory but CHANGES THE OPTIMISER SEMANTICS,
and the probe stalled with no steps in 3.5min for unrelated reasons. Not worth debugging a shortcut.
**ADOPTED — `--latent-grad-release`:** register a post-accumulate-grad hook on each latent so it takes its Adam
step the instant its gradient exists, then sets `.grad = None`. At most one layer's gradients are ever live, so
the 4B/latent grad term leaves the peak: **16B → 12B/latent ⇒ mlp@32L ≈ 40GB (fits)**. Crucially this is EXACT
Adam (the update is per-parameter): a unit test vs standard Adam over 5 steps gives **max|Δparam| = 0.000e+00**
and confirms the grad is freed. Integration details: the lr schedule + TALR must be applied BEFORE backward
(latents step during it), and the latent groups' lr is zeroed inside `opt.step()` so they cannot double-step.
VALIDATED on the known-good down@32L config: KL @40/@80 = 0.6624/0.5237 vs the 0.6684/0.5329 reference (tracks,
slightly better), steady RSS **21.1GB vs 24-26GB**, saved cleanly.
OPEN: the SAVE transient spiked to 46.1GB on down@32L (it completed, but that is at the 47G line). mlp@32L
trains at ~40GB steady, so the save spike — not training — is the remaining risk for the full-MLP scope.

**8ad. SAVE-TRANSIENT FIX — peak RSS 46.1GB → 21.1GB (2026-08-07).**
Root cause: `save_student` claims to stream "shard-by-shard", but this student has **ONE shard**, so its
per-shard dict accumulates the ENTIRE dequantised fp16 model (~10.6GB at 4B) and `save_file()` copies it again
during serialisation — a ~20GB transient on top of whatever training still holds. `safetensors.save_file` has no
streaming API (it takes a full dict), so the fix must reduce what is resident BEFORE the call.
FIX: `save_export(tag, final=True)` on the final save now also clears the OPTIMIZER STATE (Adam's 2 fp32 buffers
per latent) — training is over at that point, so the state is dead weight. Grads were already dropped.
MEASURED on down@32L: peak RSS **46.1GB → 21.1GB**, "[save] released 6.5GB of optimizer state before writing",
model saved cleanly.
PROJECTION for mlp@32L (2.26B latents) with grad-release + this fix: training 40.0GB steady, final save 43.1GB
(would have been 61.2GB) — both under the MemoryHigh=47G cap.

**8ae. mlp@32L (full MLP scope, 2.26B latents) — NOT REACHED on this box (2026-08-07).**
Built three memory mechanisms, all correct and unit-tested, and still could not run it:
  1. `--latent-grad-release` — per-latent Adam step in a post-accumulate-grad hook, frees each grad
     immediately. EXACT Adam (unit test max|Δ|=0.000e+00). Validated on down@32L: RSS 24-26 → 21.1GB, KL
     trajectory preserved (0.6624/0.5237 vs 0.6684/0.5329 reference).
  2. §8ad save fix — release optimizer state before the final save. down@32L peak 46.1 → 21.1GB.
  3. `--latent-state-nvme` — Adam exp_avg/exp_avg_sq in np.memmap buffers (unit test vs in-RAM: 0.000e+00
     over 20 steps). Frees 18.1GB of RAM at mlp@32L.
MEASURED COST at mlp@32L: **18.1 bytes/latent** (not the predicted 12) ⇒ ~48GB resident, over the 47G cap. With
NVMe state it trained but at 47.6GB (memmap dirty pages count toward the cgroup until written back) — 10 steps
per 3 min, i.e. throttled. Lowering MemoryHigh to 38G to force early writeback made it worse: 9 min with ZERO
steps, D state, load 15.6, no state files created — stuck thrashing in the FIRST backward.
**STATUS: full-MLP scope at full 32-layer coverage is not reachable on a 60GB host with this design.** The
mechanisms are sound and reusable (grad-release + save fix are pure wins already in use); the blocker is that
2.26B fp32 latents + their transients simply exceed the box. Options for the scope question: (a) gate+down @32L
= 1.51B latents (~34GB, fits, keeps full coverage); (b) mlp @16L (~27GB, but confounds scope with coverage);
(c) revisit with q8/bf16 latents once the fp32 requirement is re-examined (bf16 was ruled out because the Adam
update ~2.5e-4 is below bf16 resolution 3.9e-3 — but a fp32 master-copy + bf16 compute variant was never tried).

**8af. bf16 LATENT COMPUTE — REJECTED, it degrades the result (2026-08-07).**
Tried standard mixed precision: fp32 master latent on the CPU, BF16 copy streamed to the GPU for the forward
(and hence a bf16 grad). Rationale was that the model already runs bf16 and `forward()` casts dequant()'s
output to x.dtype anyway, so the fp32 GPU copy looked free to drop (~6B/latent = 13.6GB at mlp@32L).
**MEASURED on down@32L (same seed/config as the fp32 grad-release reference): step40 KL 0.9495 vs 0.6624,
step80 0.6741 vs 0.5237 — clearly WORSE, with FEWER flips (0.154% vs 0.209%) at the same ~21-24GB RSS.**
WHY (the reasoning I got wrong): the STE gate is `|L| < 1.5·s` and the update is driven by each latent's
DISTANCE FROM ITS DECISION BOUNDARY — a small difference of similar-magnitude numbers (s ~1e-2, latents sitting
NEAR the boundary by construction after fp-spread init). bf16's ~3 decimal digits cannot resolve that
difference, so the gate admits/blocks the wrong latents and gradients land on the wrong weights. **The fp32
requirement is not only about the Adam update magnitude (2.5e-4 vs bf16 3.9e-3) — the STE FORWARD itself needs
fp32 to resolve boundary proximity.** Reverted.
⇒ Full-MLP scope at 32 layers stays out of reach on this box. Falling back to gate+down @32L (1.51B latents,
~34GB) which keeps FULL layer coverage and isolates the scope variable against the down@32L reference.

**8ah. ★ TEST 3 — SCOPE vs COVERAGE: coverage is ~2.5x the better lever (2026-08-07).**
Matched pair, both 8 layers (stride 4), TR 1.25e-4 annealed, 360 steps, scales FROZEN, fp-spread init,
grad-release + offload. Baseline 0.7355.
  down only (0.189B latents, lr 8e-7)      final KL 0.4718   (-36%)
  mlp gate+up+down (0.566B, lr 1.5e-7)     final KL 0.4631   (-37%)
⇒ **3x the trainable assignments buys 0.0087 KL.**
CROSS-REFERENCE with the coverage result (same TR/steps, down scope):
  8 layers  0.189B -> 0.4718
  32 layers 0.755B -> 0.4493
⇒ **4x the COVERAGE buys 0.022 KL — ~2.5x more per unit of latent budget than scope**, and coverage is the
CHEAPER one to run (one projection across all layers needs less memory than three across a quarter of them).
CAVEAT: the pair differs in lr as well as scope (1.5e-7 vs 8e-7) because more latents need a lower lr to avoid
the burst; "scope doesn't pay" and "the lr penalty for scope outweighs it" are not fully separable here. The
queued ablation's gate/up/down arms are IDENTICALLY SIZED (0.189B each) so they share one lr and settle that.
**ACTIONABLE FOR THE 27B: prioritise full 64-layer coverage of down_proj over adding projections at partial
depth.** It also reframes §8ae/§8ag — the mlp@32L memory fight was chasing a lever worth only ~0.009.

**8ai. ★ SCOPE ABLATION @8L — ATTENTION IS THE BEST SINGLE SCOPE (2026-08-07).**
All arms: 8 layers (stride 4), TR 1.25e-4 annealed, 240 steps, scales FROZEN, fp-spread init, grad-release +
offload, from the same block-AP skeleton (baseline KL 0.7355). MLP arms are IDENTICALLY sized (0.189B latents)
so gate/up/down share lr 8e-7 — no calibration confound between them.
  arm    latents     KL     gain   gain/B-latent
  attn    0.630B  0.4668  0.2687      0.427     <- BEST
  down    0.189B  0.4935  0.2420      1.280
  up      0.189B  0.4966  0.2389      1.264
  gate    0.189B  0.5022  0.2333      1.234
**FINDINGS:** (1) attention beats the best MLP projection by 0.027 — 3x the ENTIRE spread among the three MLP
projections, and more than full 3-projection MLP scope gained in §8ah. (2) The three MLP projections span only
0.009 ⇒ largely SUBSTITUTABLE; which one you pick barely matters (down marginally best, consistent with it
consuming the SwiGLU intermediate and writing the residual stream). (3) **attn is 3x LESS EFFICIENT PER LATENT** (0.427 gain/B vs
~1.27 for every MLP projection): it has 3.3x more trainable weights, and that size is the whole reason it wins
in absolute terms. (I first wrote the opposite here — 'competitive per-latent' — which the gain/B column I had
just computed directly contradicts. Corrected.)
**⇒ WHICH SCOPE TO PICK DEPENDS ON THE BINDING CONSTRAINT:** best ABSOLUTE result from a single scope = attn
(0.4668); best VALUE per unit of memory/compute = any MLP projection (~1.27 vs 0.43). On the 27B memory is the
binding limit, so the per-latent figure is the relevant one and MLP projections stay preferable there; attention
is only worth it with spare headroom.
**SURPRISE / CORRECTION:** this whole test line focused on down_proj because block-AP's QAT is `--qat-attn-only`
and I assumed attention was already handled and the MLP was the untapped part. For the ASSIGNMENT stage the
opposite holds — attention has the most recoverable assignment error.
Sequential chain (down->up->gate->attn, each warm-starting from the previous) running to test complementarity.

**8aj. ★ SEQUENTIAL CHAIN BEATS SIMULTANEOUS — and costs LESS memory (2026-08-07).**
Chain @8L (240 steps/stage, each warm-starting from the previous saved model, scales frozen throughout):
  down            0.4935
  + up            0.4555   (-0.038)   <- big
  + gate          0.4580   (+0.003)   <- NOTHING (MLP projections are substitutable, cf §8ai spread of 0.009)
  + attn          0.4536   (-0.004)   <- small but real
COMPARISONS:
  all-MLP SIMULTANEOUSLY (§8ah, 360 steps) 0.4631  — the chain reaches 0.4536 in 240 steps/stage
  single scope at 32L coverage (§8aa)      0.4493  — chain@8L nearly matches it with 1/4 the coverage
**⇒ THE KEY OPERATIONAL FINDING: sequential scope holds only ONE projection's latents at a time, so peak memory
stays at the single-scope level (~26GB at 32L) NO MATTER how many stages are chained.** Simultaneous mlp@32L
needed 48GB and was OOM-killed (§8ag). The memory wall that consumed much of this session is avoidable by
ORDERING, not by more offload engineering — the more useful lesson for the 64-layer 27B.
**⇒ RECIPE IMPLICATION: chain down -> up (skip gate, it is redundant) and optionally -> attn.**
NEXT (queued): `run_seq32.sh` = the same chain at FULL 32-layer coverage (down -> up @32L, then attn @16L
because attn@32L = 2.52B latents ~ 77GB by the measured 29.1 B/latent model and will not fit). Tests whether
the two levers COMPOSE: coverage (-0.022) + sequential scope (-0.038) should land below the current best 0.4493.

**8ak. BUG — grad-release leaked a DUPLICATE Adam state; RSS 23.9→42.8GB and wedged (2026-08-07, fixed).**
seq32's first stage (down@32L, a config that had run stable at 21-26GB) climbed to 42.8GB anon by step 100 and
wedged in D state with the GPU at 0%. cgroup memory.stat showed **anon 39.4GB / file 2.0GB** ⇒ a real leak, not
page-cache accounting.
ROOT CAUSE: with `--latent-grad-release` the latents step inside backward via `_adam_step_one`, and I stopped
`opt.step()` from double-UPDATING them by zeroing the latent groups' lr. But **a zero-lr Adam step still
ALLOCATES exp_avg/exp_avg_sq** (verified directly: one step at lr=0.0 creates both buffers). So every latent
carried TWO sets of Adam state — an extra 6.0GB at 32L plus allocator slack.
Why it was missed: the grad-release validation runs were 30-90 steps; the leak only becomes fatal past ~100.
FIX: temporarily REMOVE the latent groups from `opt.param_groups` around `opt.step()` instead of zeroing their
lr, so the optimizer never visits them. **Verified: RSS flat at 15-24GB through step 90** (the buggy version was
at 42.8GB and wedging by step 100).
LESSON: validate memory behaviour over a run length comparable to the real one — a 30-step smoke cannot see a
per-step allocation leak.

**8al. ★ ROOT CAUSE of the repeated step-100 wedges: the PERIODIC CHECKPOINT SAVE, not a leak (2026-08-07).**
Three seq32 attempts all stalled at EXACTLY step 100 in D state. I chased it as a memory leak and made two
real-but-secondary fixes (see below). The user's observation — "it's a ~20GB SPIKE, system goes 35GB->50GB" —
identified it: **`--ckpt-every` defaults to 100**, so at step 100 `save_export()` runs `save_student()`, which
builds the entire fp16 model dict (~10.6GB at 4B) plus `save_file()`'s serialisation copy = a **~20GB
transient** on top of ~30GB of training state. That crosses the cgroup limit and throttles the process into
uninterruptible sleep. Explains everything: the exact step number, the spike shape, the D-state (throttle, not
OOM-kill, hence nothing in the journal), and why my "leak fixes" moved the number without curing it (they
lowered base RSS, so the same spike landed at 33.7GB/R-state instead of 40.9GB/D-state).
The §8ad fix released optimizer state only on the FINAL save, not on periodic checkpoints.
**FIXES:** (1) `--ckpt-every 0` in run_seq32.sh — we use `--select final`, so mid-training checkpoints are
useless here. (2) **`SAVE_MAX_SHARD_GB` (new, opt-in, default off)** — re-shards the OUTPUT into bounded files
so `d` and save_file's copy are each capped. The student has ONE source shard, so the default path always
accumulated the whole model. At 2GB: final-save peak ~34GB instead of ~51GB. VERIFIED: writes 5 shards +
rewritten index, and `build_student` loads it back (each seq32 stage warm-starts from the previous, so this had
to work). Deploy/export path unchanged (flag off by default).
**SECONDARY FIXES made while chasing this (both genuine, both kept):**
  - `_snap()` cloned all of `scales` (which INCLUDES the 3GB of latents) on every held-out improvement ⇒
    repeated multi-GB alloc/free. Now preallocated buffers, copied into: O(1) allocations for the run.
  - `_adam_step_one` computed `(exp_avg_sq.sqrt()/c).add_(eps)`, allocating TWO full-size fp32 temporaries per
    latent per step (~3GB/step of churn at 32L). Now one reusable scratch buffer, all ops in-place. Unit-tested
    identical to torch Adam (max|Δ| 4.8e-07 over 30 steps).
**LESSON: an exact, reproducible failure STEP is a code path, not a gradual leak.** I should have grepped for
step-100 triggers before hypothesising about allocator fragmentation.

---

## 9. Assignment-stage program (2026-08-05→11): cold-start fix → data scaling → scope decomposition → E2E composition

**HEADLINE: the assignment stage DOES scale with data (+1.67 pt eval2k agreement per doubling, no
saturation to 16M tokens) — that was the open problem. But every STRUCTURAL change we tried lands in the
noise, because whatever the first intervention fixes is nearly all that any of these mechanisms can fix at
this data budget.** Deployable best = **assignments + E2E, 81.32% / KL 0.3042** (`output_4bpipe/e2e_on_assign`).

### 9a. Data scaling of `down`@32L (from the raw skeleton, scales frozen, eval2k NP=1946 SEQ=1024)

| arm | steps | unique seqs | tokens | agreement% | meanKL |
|---|---|---|---|---|---|
| A | 360 | 120 (×3ep) | 0.31M | 71.03 | 0.6075 |
| B | 360 | 360 | 0.92M | 71.49 | 0.5868 |
| C | 1080 | 1080 | 2.77M | 73.72 | 0.5100 |
| D | 3240 | 3240 | 8.29M | 76.21 | 0.4289 |
| E | 3240 | 360 (×9ep) | 0.92M | 73.66 | 0.5083 |
| **F** | **6244** | **6244** | **15.97M** | **78.35** | **0.3637** |

Both axes pay and ADD: fixed compute + 9× data = **+2.55**; fixed data + 9× compute = **+2.17**; both = **+4.72**.
Fresh tokens ≈ **3× repeats** (E lands on C: 9 epochs over 360 seqs == 1 epoch over 1080). ⇒ **+0.685 pt per
doubling of STEPS** on fixed data. Unique calib caps at **15.0M tok** (6244×2560).

### 9b. THE DIMINISHING-RETURNS LAW (five independent confirmations)

Every mechanism gains hugely on a weak model and ~nothing on a strong one:

| intervention | on a weak model | on a strong model |
|---|---|---|
| E2E scale distillation | skeleton 61-64% → **80.86** (+18) | assign-trained 78.35 → **81.32** (+2.97) |
| assignment training | skeleton 61-64% → **78.35** (+15) | E2E'd 80.86 → **81.32** (+0.46) |
| `attn` after `down` | 0.4575 → 0.4488 (crossed) | from arm F 0.2341: **never crossed** in 3000 steps |
| later layer groups | group0 **+0.3836** | groups 1/2/3 +0.0144 / +0.0040 / **0.0000** |
| joint all-MLP vs `down` alone | mlp8 0.4631 vs down8 0.4718 | **+0.5 pt** at 32L (75.09 vs ~74.6 interp) |

**Scale-training and assignment-training are SUBSTITUTES, not complements**, despite touching disjoint
parameters (`W ≈ scale_g ⊙ trit`). Research prompt filed: `research_prompts/e2e_diminishing_returns_prompt.md`.

### 9c. Scope decomposition — what works and what does not

- **A 2nd MLP projection sequentially ALWAYS damages** (4×, incl. 3000 steps from the 16M `down`: entry
  0.2341 → 0.3261, never beat entry). SwiGLU `down(silu(gate)·up)` — gate/up multiply, so tuning one
  co-adapts the others' CURRENT assignments. **Recipe = `down` + `attn` only.**
- **Layer-sequential DOES stack** (group-joint, stride 4, all-MLP): every group dipped ~4%, recovered, and
  finished ahead — vs the scope axis's 39% dip that never recovered. The decomposition is sound; its
  *value* is only +0.5 pt, so the joint-scope program is **CLOSED**.
- `attn` was never budget-matched (half the steps, half the data, half the coverage) — still open.

### 9d. Two silent measurement bugs, both fixed in `src/e2e_qp_distill.py`

1. **No entry baseline** ⇒ `best_ho_kl` started at ∞, so the first post-training eval "won" by default and a
   stage could ship a model 33% WORSE than its input while logging a clean `best=`. Fixed: a step-0 held-out
   eval logs `<- ENTRY BASELINE` and seeds `best_ho_kl`/`best_ho_snap`.
2. **Abort threshold is in STEPS**, not evals (`ho_worse × eval_every ≥ 100 × abort_patience` ⇒ 300 at the
   default 3). At a coarse `--eval-every` that is ONE eval. It killed E2E at step 3000/12488 **while it was
   improving monotonically**. `init_ho_kl` is now deliberately anchored to the first POST-training eval, NOT
   the entry, so dip-then-recover stages (attn, E2E) are not aborted. Long stages need `--abort-patience 30+`.

Also fixed: the cold-start latent-lr burst (linear lr ramp + TALR gated until the ramp completes) — paired
proof 0.4935 → 0.4891 from 11% fewer flips in 17% fewer effective steps.

### 9e. Memory model — CORRECTED

Old **27.3 B/latent** came from only the two 8L points (40.6M apart, a weak lever arm) and was **44% high**.
Refit over the full range (attn@8L 336.9M→13.11GB, gateup@8L 377.5M→14.22GB, down@32L 755M→21.14GB):

> **18.9 B/latent + 6.92 GB base** (predicts all three within 0.2 GB)

= 4 (fp32 latent) + 4 (exp_avg) + 4 (exp_avg_sq) + 4 (snapshot) + ~2.9 overhead.

**27B latent inventory, measured from the checkpoint: 26.049B** quantizable (excl. embed; MLP 17.11B = 66%,
attention 5.54B, lm_head 1.271B, mtp ~0.42B). RAM for ALL latents at 64 layers: **499 GB** today, **239 GB**
with 8-bit Adam + snapshot eviction, **111 GB** for bare fp32 latents alone. Does not fit 42 GB by any route
(would need 1.35 B/latent). `down`@64L alone = 5.704B = 115 GB (58 GB patched).

### 9f. Built but NOT yet applied

- `src/adam8bit_cpu.py` — block-wise 8-bit Adam moments for CPU-resident latents (bitsandbytes' 8-bit
  optimizers are CUDA-only; our latents are host-side under `--latent-offload`). **Measured: −6.0 B/latent,
  99.85% identical flip decisions vs fp32 at a ~10% flip rate, +10.3% step time.**
- `tools/apply_mem_wins.py` — applies both wins behind opt-in flags (`--adam8bit`, `--snap-nvme`);
  `--check` verifies anchors + parse without writing. Snapshot eviction measured to keep 4 B/latent off
  ANONYMOUS rss (memmap ⇒ reclaimable page cache).
- `run_full_pipeline.sh` **Phase 4.5** (assignment training) is wired: `down` 1 epoch + `attn` stride 2,
  Phase 5 now starts from `$ASSIGNED`. `ASSIGN=0` restores the old behaviour exactly.

---

## 10. Condensed from removed raw artifacts (logs/*.log, logs/*.json — deleted 2026-08-11)

Everything below was the *only* copy of these numbers. The raw files are gone; the `.txt` tables remain.

### 10a. Free-gen gate, 2048-token budget (n=48, think mode, temp 0.6, tau 4.0)

| model | loop | trunc | commit | mean comp-ratio | think len | n_closed |
|---|---|---|---|---|---|---|
| **FP teacher** (`output_4b/rot`) | 25.0% | 25.0% | 75.0% | 2.399 | — | — |
| **TQ1_64 sim, FINAL gate** | **22.9%** | **18.8%** | **79.2%** | 2.849 | 674 | 39/48 |
| TQ1_64 sim (earlier) | 27.1% | 20.8% | 75.0% | 2.838 | 652 | 38/48 |
| final_g64q8 | 29.2% | 20.8% | 75.0% | 3.080 | 648 | 38/48 |
| chatfix v1 | 29.2% | 37.5% | 62.5% | 3.338 | — | — |
| combined-2560 | 39.6% | 43.8% | 56.3% | 3.153 | — | — |
| stageB | 56.3% | 41.7% | 50.0% | 3.911 | — | — |

**The deployed TQ1_64 sim BEAT the FP teacher on loop rate (22.9 vs 25.0) and commit (79.2 vs 75.0).** Max
comp-ratio is the outlier metric: FP 3.86 vs ternary 8.5-18.5, i.e. the tail degenerates even when the mean
does not. stageB (56.3% loop) is the clearest example of a variant that fails free-gen while looking fine
teacher-forced.

### 10b. Oracle: first-order flip ranking is DEAD (`down`, 755M trits, ce_weight 0.5)

Baseline KL 0.2949 / CE 1.4403 / agreement **77.72%** (FP gap 22.28 pt). Rank candidate trit flips by
`ḡ·s·(t′−t)`, commit top-K unconstrained, measure true held-out:

| committed flips | KL | agreement | Δ agreement |
|---|---|---|---|
| 736 | 0.2963 | 77.57 | −0.15 |
| 7,520 | 0.3492 | 75.54 | −2.18 |
| 75,498 | 1.8215 | 52.21 | −25.51 |
| 755,008 | 14.93 | 6.75 | −70.97 |
| 7.55M / 37.8M | 36.7 / 50.9 | 0.00 | −77.72 |

**Monotonically worse at every budget**, and only flips with *predicted* Δloss<0 were ever committed. A flip
moves a weight by ~100% of its own magnitude, so the linear term carries no information.

### 10c. Oracle: GPTQ re-solve at the post-E2E operating point is DEAD

19.66% of assignments change. Agreement 77.72 → **62.48%** (own absmax scales) / **67.61%** (E2E-trained
scales kept). Keeping the trained scales recovers ~5 pt, so it is NOT a scale confound — GPTQ's preferred
assignments are genuinely worse for the END loss because it optimises `‖X(W_fp−Q)‖²`, not the task.

### 10d. Activation-Gram diagnostic (rotbase, d=2560, 2.62M tokens/layer)

| layer | stable rank | r_eff (tr/λmax) | spectral conv. (rel F-norm vs 1024 samples) | held-out energy in calib top-512 |
|---|---|---|---|---|
| 8 | 5.4 | 2.3 | 128→0.878, 256→0.753, 512→0.502, 1024→0.0 | 0.776 |
| 16 | 7.0 | 2.7 | same shape | 0.773 |
| 24 | 21.0 | 4.7 | same shape | 0.693 |

The Gram is extremely low-rank (stable rank 5-21 out of d=2560) and converges by ~1024 calibration samples.
**This is the mechanistic reason fixed-teacher block-local stages saturate so early** — data enters only
through an O(d²) statistic that is essentially converged after ~1M tokens. Deeper layers are richer
(stable rank 21 at L24 vs 5.4 at L8) and retain less held-out energy in the calib subspace (0.693 vs 0.776).

### 10e. Long-context KL gap (83 seqs, `eval_long8k.json`) — NO degradation with length

| ctx | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|
| mean KL | 0.3274 | 0.3151 | 0.3156 | 0.3263 | 0.3377 |

gap growth **+0.0103** against SE **0.0136** ⇒ **not significant**. Training at seq 2560 does not cost
long-context fidelity out to 8192.

### 10f. Misc single numbers from deleted logs

- `e2e_pass2` (second E2E pass): agreement **69.00%**, loop rate 31.2% (15/48).
- `eval2k_c4`: KL 3.4764 nats, agreement **70.53%**.
- `grcheck`: best held-out KL 0.5237.
- `mlp32` / `granularity_test`: recorded FAILED (the 32-layer all-MLP OOM and the granularity sweep aborts);
  superseded by §9's corrected 18.9 B/latent memory model.

---

## 11. FULL-PIPELINE COLD-START VALIDATION (2026-08-11/12, 4M smoke test) — PIPELINE WORKS

`run_full_pipeline.sh` run end-to-end from scratch on the 4B (ORIG_MODEL=output_4b/untied_4b, fresh
ROT_BASE, CALIB_TOKENS=4000000, NGPU=1, EVAL_EACH=1). **19:23:35 total, ZERO failures**, valid final artifact.
Models deleted afterwards; records kept in `logs/pipeline_smoke_*`.

| phase | wall | result |
|---|---|---|
| 1 rotation (cold) | **0:00:43** | **rot-check PASS**: KL 0.0006, agreement 98.66%, FP-confident **75.96 → 76.13%** |
| 2a chat pool | 3:32:46 | 1866 rollouts → **705 kept** (require-close ≈38% yield), 1.80M tok, 843 `<think>` |
| 2b calib | 8s | — |
| 3 block-AP skeleton | ~1:55 | eval2k **63.09%** / KL 0.9211 |
| 4 teacher cache | 8m | — |
| **4.5a `down`** (1555 steps) | ~3:54 | eval2k **76.49%** / 0.4299 (**+13.40**) |
| **4.5b `attn`** (1555, stride 2) | ~3:40 | eval2k **77.96%** / 0.3844 (**+1.47**) |
| 5 E2E (2ep, 3110 steps) | ~3:30 | eval2k **81.00%** / 0.3194 (**+3.04**) |
| 6 gates | 2:12:19 | Gate A **PASS** · Gate B **FAIL** |

**The rotation post-condition works on a cold rotation** — 75.96→76.13% FP-confident is the exact metric that
read 78.87→0.00% during the §8k lm_head norm-fold bug.

**Phase 4.5 (assignments in-pipeline) works**, and **`attn` EARNS its slot at realistic budgets**: +1.47 pt
here from a 4M `down`, versus a complete no-op from arm F's 16M `down`. Same diminishing-returns law — the
two stages are partly INTERCHANGEABLE, not additive. Do not drop `attn` from the pipeline.

**TOKEN EFFICIENCY — the strongest datapoint we have.** Correct per-stage accounting (the "16M path" is a
misnomer: its `attn` got 3000 steps AND was a no-op):

| | `down` | `attn` | E2E | total token-steps | unique | final |
|---|---|---|---|---|---|---|
| 16M path | 6244 / 16.0M | 3000 / 8.3M **(no-op)** | 12488 / 16.0M ×2ep | **55.6M** | 16.0M | 81.32% |
| **4M smoke** | 1555 / 4.0M | 1555 / 4.0M | 3110 / 4.0M ×2ep | **15.9M** | 4.0M | **81.00%** |

**3.5× fewer token-steps and 4× less unique data for −0.32 pt.** NOT a controlled comparison (different
calib AND skeleton) — the clean test is this same pipeline at 16M.

### 11a. GATE B WAS NEVER ENFORCED — found and fixed

The free-gen gate printed its targets and printed its results but **never compared them**; the
`|| echo "...continuing"` caught only a crash. This 4M model — **Gate A 81.00% PASS** — fails every
free-gen criterion and would have gone to GGUF export and printed ALL DONE:

| metric | got | target | FP teacher |
|---|---|---|---|
| loop_rate | **37.5%** (18/48) | ≤30% | 25% |
| commit_rate | **64.6%** | ≥68% | 75% |
| mean_comp_ratio | **3.52** (max 18.2) | ≤3.1 | 2.40 |

Now **fail-closed** in Phase 6 (parses `loop_gate.json`, exits 1 on any breach; `GATE_ADVISORY=1` restores
the old print-only behaviour). This is the script's own stated philosophy — "GATES ARE FREE-GEN,
teacher-forced KL is PROVEN BLIND" — which it was not actually implementing.

Likely cause of the failure is the 4M budget starving the chat pool (commit rate tracks corpus
`</think>`-density; §8). Validated 16M recipe reference: loop 29.2% / commit 75.0% / comp 3.08.

---

## 12. Gate B across the stage chain (2026-08-12) — E2E is MANDATORY, and teacher-forced is blind

Ran the free-gen gate (n=48, 2048 tok, think, temp 0.6) on every stage of the 16M chain, to test whether
the 4M pipeline's Gate-B failure was (A) too little data or (B) the assignment stage damaging generation.

| model | eval2k | loop | commit | trunc | comp mean/max | gate |
|---|---|---|---|---|---|---|
| block-AP skeleton | 63.09% | **97.9%** | **2.1%** | 87.5% | 20.02 / 82.7 | catastrophic |
| + `down` 16M (arm F) | 78.35% | 47.9% | 72.9% | 25.0% | 3.34 / 10.8 | FAIL |
| **+ E2E (best model)** | **81.32%** | **27.1%** | **75.0%** | 22.9% | **2.90 / 10.6** | **PASS** |
| scale-only E2E (no assignments) | 80.86% | 29.2% | 75.0% | — | 3.08 | PASS |
| FP teacher | — | 25.0% | 75.0% | 25.0% | 2.40 / 3.86 | — |

**(B) is DEAD — assignment training does NOT hurt free-gen.** The assignment-trained model at 16M *beats*
the previously validated no-assignment model on loop (27.1 vs 29.2) and comp (2.90 vs 3.08), matching
commit exactly, within 2.1 pt of the FP teacher's own loop rate. **(A) was right: 4M was too little data.**

**Per-stage contributions — both stages matter, for DIFFERENT reasons:**
```
commit:  2.1 → 72.9 → 75.0    assignments +70.8, E2E +2.1    (assignments do 97%)
loop:   97.9 → 47.9 → 27.1    assignments −50.0, E2E −20.8   (assignments do 71%)
comp:  20.02 → 3.34 → 2.90    assignments −16.7, E2E −0.44   (assignments do 97%)
```
Assignments do the BULK of the repair; E2E does the final approach on loop rate, and that is what crosses
the 30% line. **This revises §9b**: the two stages are substitutes on eval2k AGREEMENT only — on
free-generation they are not interchangeable, and E2E cannot be skipped no matter how good agreement looks.

Likely mechanism for the split: assignments sharpen top-1 while leaving the distribution tail rougher
(hence assignments+E2E has better agreement but *worse* meanKL than scale-only E2E, 0.3042 vs 0.2977), and
a rough tail is what degenerate repetition feeds on; E2E re-fits the per-g64 scales and smooths it.

**The skeleton row is the case for Gate B existing**: 63.09% teacher-forced agreement, loops on 98% of
prompts, emits `</think>` on 2%. Teacher-forced metrics are blind to this by construction.

Raw table: `logs/gateB_compare.txt`.

## 13. Throughput campaign for arm B on the 2x3090 box (2026-08-20→29) — 294.0 -> 149.3 s/step CONFIRMED (-49%); four correctness bugs; the real lesson is measurement

Goal: cut s/step for full-latent assignment training (arm B) so a 160M-token run is not absurdly long.
All timings: 27B student, seq 2560, `--train-weights all --tw-layer-stride 2`, `--lr 0 --latent-lr 0`
(throughput only), steps 2→8 from the timestamped step lines. "s/microbatch" = s/step ÷ `--pipe-parallel-mb`.

### 13a. Multi-process pipeline parallelism — the 1F1B handoff was the whole story

In-process pipelining is impossible here (one python thread cannot feed both GPUs: every layer's
`dequant()` blocks on a host→device latent transfer). Two ranks as separate interpreters do work, but
the first schedule ran at **123.8 s/microbatch — exactly the single-process baseline (119–132)** even
though both GPUs sampled at 100%.

**Cause: rank 0 released microbatch k+1 only AFTER `backward(g_k)`.** Rank 1 therefore idled for the
entire duration of stage-0's backward, every microbatch. Cycle was `max(F0, stage1) + B0` instead of
`max(F0 + B0, stage1)`. Fix = `isend` the next activation BEFORE the backward, with rank 1 pre-posting
the matching `irecv` (a blocking eager send deadlocks: NCCL watchdog timeout).

| schedule | s/microbatch |
|---|---|
| blocking handoff (release after backward) | 123.8 |
| **non-blocking isend + pre-posted irecv** | **112.5** (−9.1%) |

This is the ONLY lever in the campaign that survived end-to-end measurement (13d, 13g).

**GPU utilisation is a MISLEADING signal here and cost three wrong diagnoses.** A rank blocked in
`recv` reads 0%; a rank doing host-bound work reads low while owning the wall clock. Only BLOCKED time
disambiguates. `PP_PROF=1` reports it per rank.

### 13b. Stage profile at split 40 (PP_PROF=1, per step, 2 microbatches)

```
rank0:  fwd  16.4    BLOCKED   0.0    bwd 235.0     <- never waits: OWNS the critical path
rank1:  fwd 108.6    BLOCKED  53.6    bwd 102.4
```

**Rank 0's backward is 14x its forward** (235.0 vs 16.4 s). A checkpointed backward should be ~3x.
That anomaly, not the schedule, is ~89% of the step — and it is still UNEXPLAINED. An op-level trace
attributes 89.4% of rank 0's self-CPU to `cudaMemcpyAsync` while `aten::mm` is 2.56% of self-CUDA, so
the step is overwhelmingly not compute; but the one fix derived from that trace measured WORSE (13d),
so the attribution is not yet understood well enough to act on.

### 13c. Offloaded latents cost 20.7x resident ones IN ISOLATION — which did NOT predict the real model

| latent placement (one 47.8M-latent linear) | forward | backward | total |
|---|---|---|---|
| offloaded (pinned CPU) | 51.4 ms | 232.0 ms | **283.4 ms** |
| GPU-resident | 8.5 ms | 5.2 ms | **13.7 ms** |

A checkpointed 2-layer synthetic stack reproduced it (1.011 vs 0.077 s/layer, 13x). **Neither predicted
the real model.** `--latent-gpu-budget 8` promoted 56/160 of rank 0's latents and measured 243.8 s/step
against a 225.0 baseline — worse, though that run was still TRENDING DOWN when killed (266, 267, 263,
220, 203) while every other run was flat from step 1. **Residency is UNSETTLED, not refuted**: it needs
a longer run to steady state. Do not quote either verdict.

Why the isolation benchmarks mislead: they have almost no surrounding compute, so they measure a
transfer at full cost that the real model (288 linears/rank, pinned async H2D) already overlaps. There
is also a plausible mechanism for residency being actively BAD — the CPU-side Adam step and gradient
accumulate run CONCURRENTLY with GPU work, so promoting a latent moves that work onto the already
saturated critical-path GPU and idles a CPU that was doing it for free.

### 13d. Refuted hypotheses — every one of them inferred from a proxy, then killed by end-to-end measurement

This section is the main deliverable of the campaign. **Five successive proposals, each derived from a
plausible indirect signal, each neutral or worse when measured.**

* **H2D contention between the GPUs.** Aggregate bandwidth SCALES: 3.23+4.57=7.80 GB/s solo vs
  3.55+4.39=7.94 GB/s concurrent. Pinning the forward copy also bought ~0 (3.31 vs 3.23).
* **Grad-buffer reallocation.** Reuse (`set_to_none=False`) is 26% SLOWER (316.5 vs 251.3 ms): it turns
  a straight assignment into a read-modify-write.
* **Rebalancing the split.** The profile showed rank 1 idle 44-54 s/step and rank 0 NEVER blocking,
  which says to move layers to rank 1. Doing it (40→36) measured **15.6% WORSE** (260.0 vs 225.0).
  **CORRECTED 2026-08-29 (13m): under controlled placement the real penalty is 2.2%** (241.6 vs 236.3,
  spreads 0.46%/0.85%), not 15.6%. The DIRECTION survives -- split 40 is slightly better -- but the
  mechanism story ("rank 1 owns the loss and crosses over to critical within 4 layers") was built on a
  7x-inflated number and was explaining noise. Split 40 stands, barely.
* **GPU utilisation as a bottleneck signal.** Cost three wrong diagnoses. A rank blocked in `recv`
  reads 0%; a rank doing host-bound work reads low while owning the wall clock. Only BLOCKED time
  disambiguates (`PP_PROF=1`).
* **Pinned gradient staging.** An op-level trace showed `Memcpy DtoH (Device -> Pageable)` at 87.75 s
  / 320 calls / 274 ms each, i.e. 0.70 GB/s vs 3.84 GB/s for the pinned H2D — apparently 5.5x too
  slow, and `cudaMemcpyAsync` was 89.4% of rank 0 self-CPU. Fixing it measured **6.2% WORSE**
  (239.3 vs 225.0). **The trace was misread: a blocking memcpy's duration on the CUDA timeline
  INCLUDES time queued in the stream, so 274 ms was occupancy, not bandwidth.** The 0.70 GB/s figure
  was itself the tell — that is below even a normal pageable D2H (~2-3 GB/s). Profiler attribution
  says where time is *charged*, not what is *causal*.

**Method note, learned the hard way.** One run per variable. A budget-8 result at split 36 was compared
against a budget-0 result at split 40 and "un-confounded" using a profiler-overhead ratio measured at a
third config; that produced a confident -21% that was pure artifact. Profiler overhead is +17.6% at
split 40 and does NOT transfer across configs. Never compare a profiled run to a clean one.

### 13e. Four correctness bugs on the `--pipe-parallel` path (invisible to throughput tests at `--lr 0`)

These, not the timings, are what the campaign actually banked.

1. **Saves shipped a HALF-TRAINED model.** Latents exist only for owned layers and rank 0 alone writes
   the file, so stage 1's trained weights were silently replaced by entry values. A completed arm B run
   would have looked successful and thrown away half its training. Fixed: `pp_sync_for_save()` folds
   rank 1's latents to trits and ships packed+scale before any save.
2. **No loss signal at all** — `log()` is rank-0-only and rank 0 computes no loss, so every step printed
   `kl=0.0000`. Fixed by broadcasting KL from rank 1 (verified live: `kl=9.4206`).
3. **Held-out eval skipped entirely** → no best-checkpoint selection, no abort. Fixed:
   `pipe_heldout_kl_flips()`, stage-aware, result broadcast so both ranks branch identically (a
   one-sided branch desyncs the next collective into a watchdog timeout).
4. **`--latent-pin-grad` was a silent no-op** without `--latent-prefetch` (`_GRAD_PIN_ON` was assigned
   only inside the prefetch branch — as was the warning meant to catch it). Now hoisted out and wired
   into the non-prefetch `dequant()` path via `_UseTransferred`, so the flag does what it documents.
   **It is measured 6.2% SLOWER — the flag is honest now, but do not enable it.**

### 13f. `--latent-gpu-budget` — two accounting bugs that made it unusable

It could not work with the default optimizer at all: `adam-blockv` allocated `v_blk` with a bare
`torch.zeros()` (no device), so a GPU-resident latent hits a device mismatch on step 1. Fixed to
`device=param.device`.

Budget accounting then OOM'd, because **Adam state is allocated LAZILY at the first step via
`zeros_like(param)`, so promoting a latent silently promotes its state too.** A "12 GB" budget
committed 11.9 GB of latent+state and died in the checkpoint recompute. Fixes: `state_mult` (2.0 for
adam-blockv, 3.0 for full Adam) charges the budget for the state, and a `pending_state` term makes the
free-VRAM check look ahead at state not yet allocated. Measured headroom, rank 0 at split 40: 16.33 GB
free after build, ~4.2 GB activation peak in the recompute ⇒ ~11 GB is the real ceiling, hence
`reserve_bytes=6e9`. Full residency is unreachable regardless: 7.66B latents need ~61 GB at 8 B/latent.

### 13g. Net result and recommended config

All unprofiled, split 40 unless stated, step-line timestamps, mean of consecutive deltas.
WITHIN-run variance is <1% (+-1-2 s), so these are repeatable; the older "4-19% noise floor" does not
apply to the pipeline path.

| config | s/step | s/microbatch |
|---|---|---|
| blocking handoff (first working pipeline) | 247.6 | 123.8 |
| **non-blocking 1F1B handoff** | **225.0** | **112.5** |
| + pinned grad staging | 239.3 | 119.7 |
| + `--latent-gpu-budget 8` | 243.8 (unsettled, trending down) | — |
| split 36 instead of 40 | 260.0 | 130.0 |

**Net: -9.1%, from the 1F1B handoff alone.** Single-process baseline was 119-132 s/microbatch, so the
pipeline's whole contribution is that one scheduling fix; every other lever measured neutral or worse.

Recommended: `--pipe-parallel --pipe-parallel-mb 2`, `MP_SPLIT=40`, no `--latent-gpu-budget`, no
`--latent-pin-grad`.

**Still unexplained**: `cudaMemcpyAsync` at 89.4% of rank 0's self-CPU time, with `aten::mm` at only
2.56% of self-CUDA. The step is overwhelmingly not compute. That observation stands even though the
pageable-D2H reading of it was wrong; the next attempt should time individual copies with explicit
stream synchronisation rather than trusting profiler attribution.

### 13h. WHERE THE TIME ACTUALLY GOES — per-latent stage breakdown (2026-08-25)

Measured with explicit `cuda.synchronize()` per stage, because profiler attribution misled us twice
(a blocking memcpy's timeline duration includes time QUEUED, not just transferred).

**True PCIe on this box**, 191 MB fp32 buffers: H2D pinned **4.89**, D2H pinned **5.09**, D2H pageable
**3.67** GB/s. Pageable D2H is only **28%** slower than pinned — not the 5.5x the trace implied. This
retires pinning as a lever and explains why `--latent-pin-grad` measured worse.

**One linear, 47.8M latents (= rank 0's per-linear average), seq 2560:**

| stage | ms |
|---|---|
| H2D latent fp32 (pinned) | 63.5 |
| STE dequant (6 full-size temporaries) | 4.7 |
| GEMM `F.linear` | 3.1 |
| backward, latent GPU-RESIDENT | **13.9** |
| backward, latent OFFLOADED | **433.0** |
| — of which D2H 72.4 + CPU grad accumulate 26.4 | 98.8 |
| — **unexplained per-tensor overhead** | **320.3** |

**Arithmetic ~8 ms. PCIe ~136 ms. ~320 ms/latent is NEITHER.** GPU residency runs the same backward in
13.9 ms, a 31x gap that transfer volume cannot explain.

Whole-model corroboration (rank 0, 40 layers, 160 latents, 2 microbatches, step 225.0 s):

* Bytes the step MUST move: H2D 122.8 GB (fwd + checkpoint recompute, x2 mb) + D2H 61.4 GB = **184.2 GB**
  ⇒ a **37.2 s floor, 17% of the step**. Transfers are NOT the dominant term.
* `aten::mm` = **2.56%** of self-CUDA. The step is NOT compute.
* `cudaMemcpyAsync` = **89.4% of self-CPU** (283.7 s of 317.4 s) while real transfer is ~17% ⇒ the
  calling thread blocks far longer than the copies take. ~960 forced sync points per step
  (640 H2D + 320 D2H) serialise CPU and GPU instead of overlapping them.
* Matches the older result that step time is **linear in latent COUNT** (64/249/497 tensors →
  102.6/165.7/327.1 s) while halving BYTES did nothing.

**Conclusion: the dominant cost is per-TENSOR overhead in the offloaded-parameter autograd path, not
bandwidth and not FLOPs.** Leading untested suspects: allocation + first-touch page faults on a fresh
~191 MB pageable CPU grad buffer every backward; `AccumulateGrad` on a CPU leaf; implicit stream
synchronisation per transfer. Attack surface is the ~960 per-tensor ops/step, not the 184 GB.

Written up for external review as `research_prompts/armb_hotspot_prompt.md`.


### 13i. FUSED OFFLOADED-LATENT GRADIENT — the second win (-17.0%, 2026-08-25)

**Mechanism.** For a CPU leaf, autograd's `AccumulateGrad` allocates a FRESH full-size host grad tensor
every backward. At 191 MB that is an mmap the kernel must fault in and zero. Measured on the real path
(H2D + backward of one 47.8M-latent tensor):

| path | ms | minor faults |
|---|---|---|
| plain `.to()` -> AccumulateGrad allocates | 400.7 | 46,721 |
| return a persistent buffer -> AccumulateGrad still clones (`--latent-pin-grad`) | 168.9 | 46,721 |
| **consume the grad in backward, return None** | **80.5** | **0** |

320.2 ms/latent — which independently matches the 320.3 ms that 13h's stage breakdown could not
account for. Fix: the transfer's `backward` stages the grad into a persistent host buffer, hands it to
the EXISTING grad-release hook, and returns `None` so AccumulateGrad never runs. Verified
**bit-identical** gradients with `param.grad is None`.

**RETRACTED — the -17.0% did not replicate and the mechanism does not engage in the real trainer.**

First measurement (back-to-back, split 40, 9 steady deltas each):

| arm | s/step |
|---|---|
| fused | 200.4 |
| `--no-fused-latent-grad` (old) | 241.3 |

That looked like -17.0%. Three later measurements of the SAME fused config say otherwise:

| run | position | s/step |
|---|---|---|
| fused (first, morning) | 1 | **200.4** |
| fusedref (evening) | 2 | 242.9 |
| posA (night) | 1 | **247.2** |
| old path, for reference | 2 | 241.3 |

Position is NOT the explanation (posA was position 1 and slow). Two of three fused runs land at ~245,
indistinguishable from the un-fused 241.3, so **200.4 was an outlier**.

**DIRECT DISPROOF, and the cheapest test of the whole campaign.** The fused path's entire mechanism is
removing ~46,721 minor page faults per latent per backward. Counting faults on the LIVE trainer
(`/proc/<pid>/stat` field 10) while it ran the fused config:

```
rank0  24,434,369 minor faults in 120 s
rank1  25,042,656 minor faults in 120 s
```

If the path engaged this would be ~0. It is instead ABOVE the ~7.3M/120s that the un-fused fallback
would produce. **The fused path is not doing what the unit test says it does.** The unit test asserted
bit-identical gradients and `param.grad is None` and passed twice -- it verified CORRECTNESS, never
that the mechanism ENGAGED. A test that had counted page faults would have caught this immediately.

**Status: the fused path is unproven, not a win.** The code is bit-identical and safe to keep, but it
must not be counted as a speedup until the fault count in a real run drops. Root cause not yet found;
the unit test engages the path (param.grad is None) while the real trainer evidently does not, so the
difference is in gradient checkpointing, pipeline parallelism, or the 2-D latent shape.

**The prediction over-shot by 2.5x** (102.5 s predicted, 40.9 s actual). The microbenchmark freed and
re-mmap'd ONE 191 MB block per iteration so it faulted every time; the real trainer cycles 160 buffers
and glibc reuses freed blocks, so the old path already avoided many faults. **Isolated microbenchmarks
systematically over-predict on this code** — same lesson as the 20.7x residency figure (13c).

**A memory regression found and fixed before it shipped:** the first implementation kept a persistent
buffer PER PARAM = 30.7 GB on rank 0, reinstating exactly what `--latent-grad-release` exists to remove.
The grad is consumed synchronously (stage -> step -> release), so one shared arena sized to the largest
latent suffices (~356 MB), and reusing one buffer also keeps its pages warm.

**Cross-run drift, again:** the oldpath arm measured 241.3 where the same config measured 225.0 a day
earlier (+7%). Only the back-to-back comparison is trustworthy.

**What did NOT work from the same analysis** (report proposed, measured): pinning the grad destination
(1.6 ms/latent — pageable D2H is 3.67 vs 5.09 GB/s) and an async copy stream (0.3 ms). The lever was
the ALLOCATION, not the copy. Also: the latent gradient is **96.1% nonzero** (the `|L|<1.5s` gate is
wide), so index/value compaction of the D2H would lose to a dense copy.

Cumulative CONFIRMED: pipeline start 123.8 -> 1F1B **112.5 s/microbatch**. The fused-grad step is
retracted (see above).

**MEASUREMENT LESSON (the most expensive of the campaign).** A single back-to-back A/B is NOT enough on
this box: same-config runs have differed by 7%, 16% and 21%. Any claimed win under ~20% needs the
config measured at least twice, in both positions, AND a direct check that its MECHANISM engages
(here: page-fault count). Correctness tests do not establish engagement.

### 13j. NUMA MEMORY PLACEMENT — the reason nothing in this campaign was reproducible (2026-08-26)

**The box has 4 NUMA nodes (~161 GB each) and BOTH GPUs sit on node 1.** Each rank needs ~52 GB
resident (pinned latents + Adam state), so the two ranks cannot both live on the GPU-local node.
Linux first-touch fills the local node then spills to whichever node has the most room -- which is
node 3, the FARTHEST from both GPUs (distance 30 vs 10 local). Nothing in the launcher constrained
this, so **which node a run's 52 GB lands on was a lottery re-rolled every run.**

ABBA, fused path throughout, `numa` = `numactl --cpunodebind=N --preferred=N` (rank0 N=1, rank1 N=0):

| arm | s/step | CPUs | where rank 0's 52 GB landed | faults/step |
|---|---|---|---|---|
| 1_numa | 212.0 | 16 | node 1 (dist 10) — 99.6% local | 8.48M |
| 2_nonuma | 271.4 | 64 | **node 3 (dist 30) — 0.3% local** | 8.30M |
| 3_nonuma | 175.3 | 64 | node 2 (dist 20) | 8.45M |
| 4_numa | 175.0 | 16 | node 1 (dist 10) | 8.43M |

**Node 3 costs 55% (271.4 vs 175.3) with fault counts IDENTICAL across all four arms** — so the only
difference is how far rank 0's memory sits from its GPU. This is a first-order effect on a workload
moving 184 GB/step across PCIe.

**This explains the entire measurement crisis:**
* 205.7 vs 310.3 s/step for identical code — one run drew a near node, the other node 3.
* Runs internally stable to +-1% while differing 51% between runs — placement is fixed at allocation.
* Build time never varied (sequential reads) while step time did (PCIe-bound).
* GPU health always clean — the GPU was never the problem.
* The apparent "14.7%/arm drift" that vanished at slot 4 — never a trend, just draws from a
  multi-modal distribution.

**⇒ EVERY throughput number in 13a-13i was measured without placement control and is unreliable**,
including the 1F1B -9.1% and the fused-grad result. They must be re-measured under binding.

**What binding buys:** it prevents the node-3 disaster (spread 1.21x bound vs 1.55x unbound; means
193.5 vs 223.4). It does NOT make the box fully reproducible.

**STILL UNEXPLAINED: ~21% residual.** Slots 1 and 4 have identical placement (node 1, 99.6% local),
identical CPU binding (16 cores), identical code -- and differ 212.0 vs 175.0. A CPU-restriction
hypothesis (16 vs 64 cores) was proposed and REFUTED by slot 4, which is CPU-bound and fast.

**Recommended:** bind memory away from node 3 (`--membind=1,2` or `--preferred=1`). CPU binding shows
no clear effect either way (slot 4 is fast with 16 cores) -- do not assume it helps.

**METHOD NOTE, repeated for the third time in this campaign:** the `numa` treatment bundled memory
binding AND CPU binding into one variable, so the arms could not separate them; slot 4 happened to
disambiguate by accident. One variable per arm, always.


### 13k. FUSED LATENT GRAD — CONFIRMED at -19.6%, and the un-retraction (2026-08-28)

13i retracted this result. **The retraction was wrong**, and so was the original claim's evidence base;
both were made on a box whose measurement noise (7-51% run to run) exceeded the effect. Once placement
was controlled the question became answerable in four arms.

**Setup that made it measurable** (all four arms identical except the one variable):
* `numactl --interleave=0,1,2` — deterministic round-robin placement, node 3 excluded. Unbound runs
  drew a different node each time; node 3 (distance 30) costs 55% (13j).
* `--no-final-save` — the 52 GB checkpoint each arm wrote was pure waste at `--lr 0` AND was
  OOM-killing the container against its 576 GB cgroup limit (13l).
* Counterbalanced order base,nofused | nofused,base so slot position cancels.

| config | run 1 | run 2 | mean | within-config spread |
|---|---|---|---|---|
| **fused** | 237.3 | 235.3 | **236.3** | **0.85%** |
| `--no-fused-latent-grad` | 290.9 | 297.0 | 294.0 | 2.1% |

**-19.6% (118.2 vs 147.0 s/microbatch).** The gap is 10-20x the noise; t ~ 18 on n=2 per group.

**Mechanism confirmed in the same runs** — page faults on rank 0, per step:

| config | run 1 | run 2 | agreement |
|---|---|---|---|
| fused | 2,254,592 | 2,264,782 | 0.45% |
| nofused | 17,206,673 | 17,206,720 | **0.0003%** |

**7.6x fewer faults.** The fault counts are essentially deterministic, which also proves both arms did
identical work — the timing gap cannot be some divergence between runs. Placement (`n0 34%`) and
cgroup peak (244 GB) were identical across all four arms, so neither explains it either.

**Why the earlier numbers were untrustworthy in BOTH directions.** The original -17.0% was a single
back-to-back pair on an uncontrolled box; the retraction rested on (a) two later arms that happened to
draw bad placement and (b) a page-fault count sampled during the BUILD phase, when the model loader
faults tens of millions of pages by design. Neither the claim nor the retraction had the resolution to
decide. **Effect size is meaningless without a measured noise floor** — that is the lesson, and it cost
roughly a week.

Cumulative CONFIRMED: **118.2 s/microbatch** with the fused path, vs 147.0 without. The 1F1B result is
STILL unmeasured under controlled conditions (its A/B flag restored the old schedule on rank 0 only,
hanging rank 1 in RECV until the NCCL watchdog fired; fixed to restore both ranks, arms queued last).


### 13m. THE CONTROLLED SWEEP — every lever re-measured, and what the campaign actually taught (2026-08-29)

12 arms, 2 reps per config, counterbalanced, all under `numactl --interleave=0,1,2` + `--no-final-save`,
8 steps/arm, split 40 unless stated. Placement (`n0 34%`) and cgroup peak (244 GB) were IDENTICAL in
every arm, so neither can explain any difference.

| config | run 1 | run 2 | mean s/step | spread | vs base |
|---|---|---|---|---|---|
| spike (P1 ceiling, NOT a real impl) | 195.4 | 195.6 | **195.5** | 0.10% | **-17.3%** |
| **base** (fused, non-blocking 1F1B) | 237.3 | 235.3 | **236.3** | 0.85% | — |
| split 36 | 241.0 | 242.1 | 241.6 | 0.46% | +2.2% |
| `--pipe-blocking-handoff` (pre-1F1B) | 267.0 | 268.3 | 267.7 | 0.49% | +13.3% |
| `--no-fused-latent-grad` (pre-fused) | 290.9 | 297.0 | 294.0 | 2.10% | +24.4% |
| `--latent-gpu-budget 8` | CUDA OOM | skipped | — | — | not viable |

**CONFIRMED WINS, both of which I claimed early, then doubted, retracted or under-stated:**
* **Fused offloaded-latent gradient: -19.6%** (originally claimed -17.0%, then RETRACTED in 13i).
  Mechanism verified in the same arms: 2.26M vs 17.21M page faults/step -- **7.6x fewer** -- with the
  nofused fault counts reproducing to **0.0003%**.
* **1F1B non-blocking handoff: -11.7%** (originally claimed -9.1%, then downgraded to "probable").

**Both were real all along. The measurement was the problem, not the optimisations.**

**Metric quality note:** page-fault counts reproduce to 0.0003-0.5% and track known structural changes
exactly (split 36 faults are 10% below split 40, matching the 36/40 layer ratio). On a noisy box a
mechanism metric that precise is worth more than the stopwatch -- the fault counts, not the timings,
were what first showed the fused path genuinely engages.

**THE LESSON, which cost about a week.** Effect size is meaningless without a measured noise floor.
This box ran at 7-51% run-to-run variance while I chased 9-20% effects through it, producing a claim,
a retraction, and an un-retraction of the SAME result. Three separate "disproofs" were themselves
artifacts: a page-fault count sampled during the BUILD phase (the model loader faults tens of millions
of pages by design), an RSS-based watchdog that structurally could not see a cgroup limit, and a
`--membind` control that made things worse than no control at all. Fixing the measurement -- NUMA
determinism plus removing 52 GB of per-arm checkpoint churn -- took the noise floor under 1%, and
every question then resolved in four arms.

**RESOLVED in 13n below — co-located placement is worth -37.0%, far more than the ~175 target.**
The ~175 s/step seen in uncontrolled runs (13j, arms 3_nonuma/4_numa) is 26% below this sweep's 236.3
base. Those runs had memory CONCENTRATED on one node with threads co-located; interleaving
spreads pages across 3 nodes so every access may be remote. A placement sweep (co-located vs
interleaved, 2 reps each) tests whether co-location now reproduces at ~175 -- the 175.0/212.0 variance
that made me abandon it was measured BEFORE `--no-final-save`, so checkpoint-cache churn is the prime
suspect for that variance rather than the policy itself.


### 13n. NUMA CO-LOCATION — the single largest win of the campaign, -37.0% (2026-08-29)

Counterbalanced A,B,B,A; only the placement wrapper differs. Fused path, `--no-final-save`, split 40,
8 steps/arm throughout.

| policy | runs | mean s/step | spread | rank-0 memory |
|---|---|---|---|---|
| **co-located** `--cpunodebind=N --membind=N` | 149.9, 148.6 | **149.3** | 0.87% | node 1, **99%** |
| interleaved `--interleave=0,1,2` | 237.3, 235.3, 237.3, 238.3 | 237.1 | 1.27% | n0 34% (spread over 3) |

**-37.0%.** Page faults identical across all six arms (2.25-2.26M, +-0.5%), so the ONLY variable is
where rank 0's 52 GB lives relative to the GPU that reads it. Both GPUs sit on node 1; co-location
pins rank 0's threads AND memory there (rank 1 -> node 0), while interleaving scatters pages across
three nodes so most accesses are remote on a workload moving 184 GB/step across PCIe.

**I had this configuration and threw it away.** 13j measured co-location at 175.0 and 212.0 and I read
that 21% spread as "unreproducible", switching to interleaving for determinism -- surrendering ~37% of
throughput. That variance was measured BEFORE `--no-final-save`, when every arm wrote a 52 GB
checkpoint; page-cache churn is what perturbs NUMA locality. With the churn gone, co-location
reproduces to 0.87%. **Same error as the fused-grad retraction: judging a LEVER through a broken
INSTRUMENT.**

### 13o. RECOMMENDED PRODUCTION CONFIG and the measured total

```
numactl --cpunodebind=$NODE --membind=$NODE   # rank0 -> node 1 (GPU-local), rank1 -> node 0
--pipe-parallel --pipe-parallel-mb 2  MP_SPLIT=40
(fused latent grad is the default; do NOT pass --no-fused-latent-grad)
--no-final-save for measurement runs only
NOT recommended: --latent-gpu-budget (OOMs), --latent-pin-grad (measured slower), split 36 (+2.2%)
```

Measured endpoints, both under controlled placement:

| config | s/step | s/microbatch |
|---|---|---|
| `--no-fused-latent-grad` + interleaved | 294.0 | 147.0 |
| **fused + co-located** | **149.3** | **74.6** |

**-49.2% between measured endpoints**, from two changes that were each independently confirmed
(fused -19.6%, co-location -37.0%). The 1F1B handoff (-11.7%) is inside both numbers, having shipped
before this sweep.

**All of the above is at `--lr 0`** (throughput only, nothing trains). Before committing to a long arm
B run, one validation at a real learning rate is required: the fused path changes WHERE the optimizer
step happens (inside the transfer's backward rather than an autograd hook), and while its gradients are
bit-identical by unit test, that has never been exercised with a nonzero `--latent-lr` at scale.


### 13p. ARM B AT FULL SCOPE RUNS — three memory bugs, and the first real step time (2026-08-30)

`--train-weights all --tw-layer-stride 1` (311 latents / 16.49B on rank 0, ~26B total), `--latent-lr
5e-7` (REAL optimizer updates, not `--lr 0`), co-located NUMA, split 40, mb 2, seq 2560.

| | |
|---|---|
| **step time** | **501.9 s/step = 250.9 s/microbatch** (7 steady deltas; 496,495,496,496,495,541,495,496 -> median 496) |
| **peak memory** | **274 GB of 620 GB (44%)**, anon 218 GB, flat from step 1 to step 10 |
| result | rc=0, 10/10 steps, KL 9.43 -> 9.65 across the run |

Before tonight this configuration could not complete a SINGLE step: it OOM-killed the container
(exit 137) every time, always between the backward and the step line.

**THREE MEMORY BUGS, ~463 GB, all invisible at stride 2 and fatal at stride 1:**

| bug | cost at stride 1 | mechanism |
|---|---|---|
| `randperm(n)[:32768]` retained the whole permutation | **146 GB** | the slice is a VIEW over the full n-element int64 tensor, and `.to(device)` is a no-op when the latent is already on CPU, so nothing forced a copy. `.clone()` fixes it. |
| duplicate Adam state | **211 GB** | `torch.optim.Adam.step()` ALLOCATES exp_avg/exp_avg_sq for every param it visits. The non-pipeline path already removed the latent groups first; both pipeline paths called a bare `opt.step()`, giving every latent a SECOND full set on top of grad-release's. |
| held-out snapshot cloned every latent | **106 GB** | `scales = scale_only + latents`, and `_snap()` does `t.detach().clone()` over `scales` -- a THIRD full fp32 copy. Now `--no-latent-snapshot`. |

**How they were found, after four blind OOM kills.** Polling could never catch these: with swap
disabled the kernel's reclaim starves userspace samplers exactly when memory spikes, so a 0.5 s
catcher logged nothing for the final minutes before a kill. Two instruments cracked it:
* `host_mem_audit()` -- cgroup memory at named phases, which localised the randperm retention to one
  list comprehension (anon 125 -> 271 GB across a single line).
* `host_tensor_inventory()` -- every live CPU tensor grouped by dtype+shape, DEDUPED BY STORAGE so
  views are charged once and genuine duplicates show as xN. Run at `--tw-layer-stride 64` (one layer,
  ~17 GB, cannot OOM) it showed every latent shape at **x2 before step 1 and x3 after**, where 1 and 2
  were correct. Scaling down until the bug fits in a safe run is what made it visible.

**Two of the three were documented behaviour the code did not have**: the randperm comment said "move
only the 32k survivors" (the clone is what makes that true), and the memory model in 9e already named
"that third full copy of every latent (latents + Adam moment + snapshot)".

**CAVEAT THAT INVALIDATES THE CAMPAIGN'S ABSOLUTE NUMBERS.** Every throughput arm in 13a-13n ran at
`--lr 0`, where `_lat_step_one` is SKIPPED (`if param.grad is not None and g["lr"] != 0.0`). The
optimizer never ran and Adam state was never allocated. Those runs compare CONFIGURATIONS validly
against each other, but they understate real training: stride 2 measured 149.3 s/step at lr 0, while
stride 1 with real updates is 501.9 s/step against a 2.15x latent count. Plan arm B from 13p, not 13n.

**Projection at this rate**: 5120 tokens/step (mb 2 x seq 2560) -> 16M tokens = 3125 steps = **18.2
days**. Still open as levers: the fp32 lm_head copy on rank 0 (~5 GB), Wlm held twice (~4.7 GB), the
teacher cache materialising 6244 sequences for a 12-sample run, and P1 (13m, ceiling -17.3%).

## 13q. The 503 s step was a single-threaded optimizer and a wasted temporary (2026-08-31) — 501.9 -> 234.7 s/step CONFIRMED (-53.2%)

Acting on the external deep-research report for §13p's breakdown (CPU Adam 47.0%, H2D 42.5%, D2H
8.3%, GPU compute 2.2%). Its ranked Stage-1 recommendation was right, but its diagnosis was wrong in
a way worth recording, because the wrong reason would have sent the next round of work somewhere
useless.

**FIRST: the report's hardware model is not this machine.** It reasoned throughout about AMD EPYC
Rome/Milan in NPS4 (single-core ~30 GB/s STREAM, ~65 GB/s per node, AVX2 fallback for
DeepSpeedCPUAdam). This box is a 4-socket **Intel Xeon E5-4650 (Sandy Bridge-EP, 2012), `avx` only —
no AVX2, no FMA**, 8 cores/socket, 4 NUMA nodes of 161 GB. The ~8.1 GB/s single-thread figure the
report quotes *as a contrast* is in fact this machine's own ceiling. Any future prompt must state
the CPU.

**SECOND: there was no "3.5x Adam anomaly" to explain.** The report compared the measured 381.6
ms/latent against 108 ms implied by an older 2.04 ns/element benchmark and inferred a 3.5x shortfall
caused by page faults, dispatch overhead and DMA contention. Rebuilding the exact kernel standalone
(53.0M-element fp32 latent, node-bound) measured **384.1 ms at 1 thread = 238.1 s across 620 calls,
against 236.6 s in production — a 0.6% reproduction.** Nothing was missing. The kernel was simply
single-threaded, because torchrun sets `OMP_NUM_THREADS=1` for every rank when it is unset and
nothing in this repo overrode it.

The report's premise came from a byte undercount. It assumed ~3 DRAM passes (394 GB/step). The
kernel actually moved **44 bytes/element**: `mul_` and `add_` are separate passes over `exp_avg`
(20n, not 12n), and `gb.pow(2)` allocates a FULL-SIZE 202 MiB temporary that is written and then
re-read to produce a result 256x smaller (12n). That is **1.45 TB/step, a 3.7x undercount**, and it
is what produced the report's "~1.5 GB/s aggregate = ~2% of one node, therefore a concurrency wall,
not a bandwidth wall". Corrected: the old kernel ran at **6.07 GB/s = 76% of this CPU's single-core
ceiling** — near-saturated *for one core*. Threading was still the right lever, but the report
explicitly ruled out bytes, and bytes were worth 1.94x on their own.

**Measured on the real kernel (ms per 53.0M-element latent, node-bound, 9 reps, median):**

| variant | 1 thr | 8 thr | 16 thr | bytes/elem |
|---|---|---|---|---|
| as-shipped | 384.1 | 116.7 | 120.9 | 44n |
| `+ vector_norm` (kills the `pow(2)` temp) | 197.9 | 71.9 | 74.7 | 32n |
| `+ lerp_` (both) | 344.9 | **67.2** | **62.6** | 28n |

Three changes, all in `_adam_blockv_step_one` plus one new flag:
* **`exp_avg.lerp_(gr, 1 - b1)`** IS the Adam moment update (`b1*m + (1-b1)*g`) in one fused
  read-modify-write, where `mul_` then `add_` costs two passes. 20n -> 12n.
* **`torch.linalg.vector_norm(gb, dim=1).pow_(2).div_(256)`** replaces `gb.pow(2).mean(dim=1)`,
  reducing each 256-wide row in a single streaming pass with no full-size temporary. Against an fp64
  reference it is *no less* accurate than the original (err 2.9e-7 vs 2.0e-7, both fp32 rounding).
* **`--cpu-threads N`** -> `torch.set_num_threads(N)`, which overrides torchrun's env default
  (verified in-log: `intra-op=8 (OMP_NUM_THREADS=1)`).

**Placement note the report got backwards.** It worried both ranks contend for node 1's cores.
`numa_colocate.sh` already splits rank 0 -> node 1, rank 1 -> node 0, so they do not. Measured
directly with two concurrent processes: **co-located 115 ms each, split 70 ms each** (solo is 67 ms).
8 threads/rank is contention-free as configured; 16 buys nothing once both ranks run.

**Correctness.** Six successive steps of old vs new on identical gradients: parameter divergence is
**flat at 2.3e-07 and does not accumulate**, and **zero ternary assignments differ** — assignments
being the only thing that moves KL. Production KL differs from baseline by +0.028, +0.027, -0.010 at
steps 1-3: random sign, i.e. run-to-run GPU nondeterminism (bf16 reduction order), not a systematic
effect of the rewrite.

**Result, stride 1 / `--train-weights all`, same config otherwise:**

| | baseline (13p) | 13q | change |
|---|---|---|---|
| CPU Adam | 236.6 s | **40.1 s** | **5.90x** |
| latent H2D | 214.1 s | **148.1 s** | 1.45x |
| gradient D2H | 41.6 s | 40.0 s | — |
| **s/step** | **501.9** | **234.7** | **-53.2%** |

**The H2D fell 66 s although nothing on that path was touched** — the one place the report's
contention argument was right, for a reason it did not give. Host DRAM traffic dropped 1.45 -> 0.92
TB/step when the `pow(2)` temporary went away, and the pageable H2D's driver staging copy reads host
DRAM, so it now contends with far less optimizer traffic. The report predicted contention from the
Adam being *slow*; it was contention from the Adam being *wasteful*.

**New composition: H2D 63.3%, D2H 17.1%, Adam 17.1%, everything else 2.5%.** The step is still ~97%
latent plumbing, but the target has moved to the transfer.

**Final: 234.7 s/step, rc=0, 6/6 steps, peak 273/620 GB** (steady deltas 234/234/235/235/234 -- a
tight spread, unlike the 7-51% placement noise of 13a-13n). Baseline 501.9 -> **-53.2%**.

### 13q-i. `--latent-prefetch` is BOTH broken and unusable with the fused path — ARM KILLED

Tried next because `_PROF_ACC["h2d"]` is incremented ONLY in the non-prefetch branch, which means
every H2D number in 13p and above was measured with prefetch OFF: those 1240 copies/step were fully
blocking and had no overlap even attempted.

**Correctness bug found by the arm, not by the timing.** Step 1 reported `adam n=440` where 620 is
correct -- 180 latents (29%) silently received NO optimizer step -- and `d2h` fell to 27.7 s, exactly
the 440/620 ratio of 40 s. Cause: on a prefetch MISS the `_PF_ON` branch fell back to a bare
`self.latent.to(...)` with no `_UseTransferred` wrapper. Under `--latent-offload
--latent-grad-release` the post-accumulate hook is deliberately not registered (`if not
_OFFLOAD_STEP_ON`), so `_UseTransferred.backward` is the ONLY thing that steps a latent. Fixed to
mirror the non-prefetch branch. This is the sixth `--pipe-parallel`-adjacent path found to be
silently inert rather than wrong-answered.

**Not worth re-measuring even fixed.** The code's own startup line already says it: *"without
--latent-pin the copies are pageable and therefore synchronous, so prefetch cannot overlap"*. Pageable
`cudaMemcpyAsync` blocks the host thread on the driver's staging copy, so the side stream buys
nothing, and `--latent-pin` is the rejected 106 GB mlock. Measured anyway: rank 0 bwd 186.9 s vs
136.7 s for 13q on the same step, i.e. SLOWER while doing 29% less optimizer work.

**Also rejected on arithmetic, not run:** the report's Stage-1 gradient accumulation across the 2
microbatches. It needs a persistent per-latent accumulator (the D2H currently stages through ONE
shared `_LAT_GRAD_ARENA`), so **+106 GB anon, 212 -> 318 GB of the 620 GB cap** on a swapless box with
four prior OOM kills, to buy 40 s of a 234.7 s step (17.1%). It was sized against a 236.6 s Adam
where it was worth 27.7%; the threading fix removed the reason to want it.

**Next lever, measured but NOT yet implemented.** The 1240 H2D transfers are 620 latents x 2 --
forward plus gradient-checkpoint RECOMPUTE. Eliminating the recompute transfer is worth ~74 s (31%
of the step). Caching the fp32 latent is the known-impossible option (65 GB on rank 0), but the
recompute does not need it: in `deq = q.detach()*s + (Lb - Lb.detach())*mask` the forward VALUE is
just `q*s`, and the `(Lb - Lb.detach())` term contributes zero to the value while carrying gradient
`*mask`. So only `q` (2-bit), `mask` (1-bit) and the small per-block `s` are required --
**3 bits/latent = 6.1 GB on rank 0's 40/64 layers, against 8.3 GB free**. Unlike the 13m SPIKE probe
(which skipped the H2D with a deliberately wrong all-ones mask and was timing-only), this is exact.

## 13r. Where the H2D actually goes (2026-08-31) — split NEGATIVE, code-transfer a WASH, and the
## profiler's h2d attribution REFUTED (real ceiling: 151.0 s/step)

After 13q the step is 234.7 s: H2D 148.1 s (63.3%), D2H 40.0 s, Adam 40.1 s. Everything below targets
the H2D. Rank 0 holds **311 latents = 66.0 GB fp32**, each transferred 4x/step (2 microbatches x 2 for
the gradient-checkpoint recompute) = **264 GB/step at 1.78 GB/s**.

### Pipeline split re-tune: NEGATIVE, and the balance model is wrong

r0 was active 211.8 s vs r1 149.0 s with r1 idling 83.7 s, so a balance model predicted split 34 at
~204 s (-13.2%). Measured **split 36 = 243.7 s/step, +3.8%** -- worse, reproducing §13m's +2.2% under
a completely different cost structure. The model is wrong, and the profiler says why:

| | split 40 | split 36 |
|---|---|---|
| r0 latents | 311 (66.0 GB) | 280 (59.9 GB) |
| r0 h2d | 148.1 s (n=1240) | 164.1 s (n=1116) |
| per transfer | 119.4 ms | **147.0 ms** |
| effective BW | 1.78 GB/s | **1.45 GB/s** |

Rank 0 got **9.2% fewer bytes but 11.4% MORE time**. Moving layers off the critical path made the
critical path slower, so r0's cost is NOT independent of r1's load. Do not re-tune the split by
balancing active time; the coupling term dominates the balance term.

### Four transfer facts, measured on this box (GPU 0, node-1 bound, 202 MiB fp32 unless stated)

| test | result | meaning |
|---|---|---|
| size scaling 3 -> 202 MiB | flat **0.31 ms/MiB** (3.34-3.63 GB/s) | H2D is **bandwidth-bound, linear in bytes** |
| `MADV_HUGEPAGE` | 61.7 vs 60.8 ms | THP is a **non-lever** (THP is `madvise`-only here, 32 MB in use) |
| two GPUs concurrently | r0 60.9 -> 66.3 ms (+8.9%) | cross-rank PCIe contention is **NOT** the bottleneck |
| pinned vs pageable | 3.47 -> **5.62 GB/s** | pinning is worth **1.62x**, contradicting §13's "pinned 4.89 vs pageable 4.88, no difference" |

### REFUTED: "the cost tracks the NUMBER of per-latent operations, not bandwidth"

`src/e2e_qp_distill.py:807` records that halving the bytes via `--latent-bf16-compute` did not help
(+3.8%), and concludes H2D cost is per-operation rather than per-byte. **The experiment never halved
the bytes.**

| | time | bytes moved |
|---|---|---|
| fp32 host -> fp32 gpu | 61.78 ms | 202 MiB |
| fp32 host -> bf16 gpu, `.to(dtype=)` | **62.00 ms** | **202 MiB — the cast runs ON THE GPU** |
| bf16 host -> bf16 gpu | **30.69 ms** | 101 MiB |

`.to(device, dtype=)` from a pageable fp32 source moves fp32 and casts on the device, so
`--latent-bf16-compute` paid a cast for zero byte reduction -- exactly the +3.8% observed. Casting on
the HOST first halves the time, as the size-scaling row predicts. The conclusion drawn from that arm
is unfounded and the size scaling above is the direct refutation: **bytes are what matter.**

Consequence: a smaller host-side payload converts to time roughly proportionally, which is what makes
the code-transfer design worth building.

### The h2d timer OVER-ATTRIBUTES by 63 s, and "GPU compute is 2.2%" is WRONG

`SPIKE_NO_H2D=1` removes only the latent H2D (both forward and recompute) and keeps D2H + the Adam
step via `_stage_and_step`. §13m measured -17.3% for it, but at **stride 2 and `--lr 0`** where the
optimizer never ran, so that number never applied here. Re-measured at the 13q config:

| | 13q | SPIKE | |
|---|---|---|---|
| h2d | 148.1 s | 0 | bypassed |
| **d2h** | 40.0 s | **100.8 s** | **+60.8 s on a path that was not touched** |
| adam | 40.1 s | 40.8 s | unchanged |
| r0 fwd | 22.0 s | **2.2 s** | |
| **s/step** | **234.7** | **151.0** | **-35.7%** |

D2H nearly tripled while nothing on that path changed. **The GPU wait did not disappear, it
RELOCATED** from the h2d timer into the d2h timer. A pageable `.to()` is enqueued on the current
stream and blocks until prior kernels drain, so `_PROF_ACC["h2d"]` was charging GPU compute to the
transfer. Of the 148.1 s attributed to H2D, **~84.7 s is real transfer and ~63.4 s was GPU wait**.

**This retracts the headline of the §13p decomposition** (and of
`research_prompts/armb_step_breakdown_prompt.md`, which was built on it): "GPU compute is 2.2% of the
step" is false. Real GPU work is ~60-85 s/step, hidden inside the transfer timers. Any future
decomposition must be validated by REMOVING a phase and seeing whether the step actually shrinks by
the attributed amount -- a wall-clock timer around a blocking call measures the queue, not the work.

### `--latent-code-transfer`: implemented, bit-exact, and a WASH (-1.0%) on this hardware

The GPU never needs the fp32 master, only `q = round(L/s).clamp(-1,1)` and `mask = |L| < 1.5s`,
because `deq = q.detach()*s + (L - L.detach())*mask` is `q*s` in VALUE -- the second term is
identically zero and exists solely to carry `grad*mask` to the CPU leaf. So one uint8 code suffices
(0,1,2 = q with the gate open; 3,5 = saturated), at 1/4 the bytes. Verified against the fp32 path:
**max|dvalue| = 0, max|dgrad_L| = 0, max|dgrad_s| = 0, zero q/mask mismatches** across three scale
regimes -- bit-exact, unlike a bf16 mirror which would shift roundings at bin boundaries and silently
move assignments.

The encode had to be fused to be viable: eager it materialised ~8 full-size 202 MiB temporaries and
cost **314 ms per latent, 5x the entire Adam step**. `torch.compile` brings it to **72.5 ms**.

MEASURED, full run, 6/6 steps rc=0, peak 291/620 GB (`--latent-code-transfer`, 310 latents,
15.2 GB of codes replacing 60.9 GB of fp32 per pass):

| | 13q | code | delta |
|---|---|---|---|
| h2d | 148.1 s | **93.0 s** | **-55.1** |
| adam (encode is charged here) | 40.1 s | **92.1 s** | **+52.0** |
| d2h | 40.0 s | 39.8 s | — |
| **s/step** | **234.7** | **232.3** | **-1.0%** |

The transfer saving landed close to prediction (-55.1 measured vs -63.5 projected) and the encode cost
close too (+52.0 vs +44.9), so the two nearly cancel: **-1.0%, a wash. Left OFF by default.**

**Why it is only marginal, and the general rule:** host DRAM here runs at 22 GB/s and PCIe at
3.4 GB/s -- only **6.5x apart**. Spending a CPU pass over the fp32 latent to avoid *sending* that
latent is therefore barely profitable, and it stays barely profitable for any payload size, because
the encode cost is set by reading L, not by what is written. The compiled encode achieves only
3.65 GB/s effective, so it is compute-bound (round + type conversion) rather than bandwidth-bound;
a hand-written kernel could close part of the gap. A comparison-based reformulation would be faster
but breaks bit-exactness at round-half-to-even ties (round(0.5)=0, round(1.5)=2), which is not worth
trading away.

**Fusing the encode into the Adam tail FAILED.** The encode reads `param` immediately after Adam
writes it, so in principle the updated value is already in registers. Measured: adam 69.2 ms + encode
80.8 ms = 150.0 ms separately, but **1924 ms** when the addcdiv_ and the encode were put in one
`torch.compile` region -- 13x WORSE. Inductor cannot handle the in-place `addcdiv_` on an
`expand`-ed view and falls back to something pathological. Making this path pay would need a
hand-written C++/OpenMP kernel; at a -1.0% starting point that is not justified.

**What is actually left.** The step is 232-235 s against a 151 s floor with zero latent H2D. Of the
~84 s gap, ~21 s is residual transfer and the rest is GPU compute that the timers had been hiding.
The next real lever is therefore GPU-side work, not host transfer -- a different problem from
everything in 13a-13r.

## 13s. Sequences per microbatch: the only lever left is TOKENS/s, not s/step (2026-08-31)

13r established that every component of the 234.7 s step sits at a wall: GPU compute cannot shed its
recompute (checkpointing is mandatory, below), H2D payload reduction is a wash (host DRAM is only
6.5x PCIe), and the Adam and gradient D2H are both near their bandwidth limits. What was still on the
table is the OTHER axis.

**The latent plumbing is paid PER MICROBATCH, not per token.** A latent is transferred, stepped and
its gradient shipped once per microbatch forward, however many sequences ride along in the batch
dimension. At 13q that fixed cost is ~158 s of the 233 s step (H2D 85 + Adam 40 + D2H 33); only the
~75 s of GPU compute scales with tokens. Grouping G sequences into each microbatch therefore gives

    s/step  = 158 + 75*G          tokens/step = 5120*G

which RAISES s/step while raising tokens/s toward an asymptote of 5120/75 = 68 tok/s.

`--mb-seqs G` (default 1) implements this. Note the metric change: **s/step gets worse on purpose.**
The quantity that sets the wall-clock cost of arm B is tokens/s, which is what the day count in 13p
was always derived from.

**Semantics change, deliberately.** With grad-release each latent takes one Adam step per microbatch,
so at G>1 that step sees a G-sequence gradient instead of a 1-sequence one -- equivalent to a G-fold
larger batch. Less gradient noise, generally an improvement, but the latent lr is tuned against G=1
and would eventually want revisiting.

**Batch-1 assumptions found and fixed** (all silent, none would have raised an error except the last):
* `pipe_mem_eff_loss` computed the CE term from `h[0]` / `ids[0]` -- with G>1 it would have trained
  cross-entropy on the FIRST sequence only and quietly ignored the rest.
* the inter-rank activation landing pad was `torch.empty(1, seq, H)`, hard-coding the batch dim.
* rank 0's stage-0 forward and rank 1's `_ids_l` both indexed a single `batches[i]`.
* `cache["idx"][bi].unsqueeze(0)` faked a batch dim for one sample. `cache["idx"]` is a LIST of
  [seq, K] tensors, not a tensor, so a group needs `torch.stack`, not list indexing -- the one that
  did raise (`TypeError: list indices must be integers or slices, not list`).

`chunked_hidden_state_loss` (flattens B*T) and `pipe_make_ctx` (expands to `inputs_embeds.shape[0]`)
were already batch-general and needed nothing.

### WHY the recompute cannot be removed instead (measured, 13s)

The obvious alternative -- stop recomputing, since checkpointing is what doubles both the H2D and the
forward GEMMs -- is not affordable, and by a wide margin. Instrumented with
`torch.autograd.graph.saved_tensors_hooks`, the STE dequant saves **five FULL-SIZE fp32 tensors,
20.03 bytes per latent element**:

    rank 0 holds 16.5e9 latent elements -> 330 GB per microbatch, 661 GB with 2 in flight
    free VRAM: 8.3 GB

So `--grad-ckpt-stride 2` (un-checkpointing 20 of 40 layers) is off by ~40x, not marginally short.
Gradient checkpointing here is LOAD-BEARING, and the 2x H2D it causes is structural. Even a custom
autograd Function saving a packed 3-bit code instead of the five fp32 tensors (6.2 GB/microbatch,
12.4 GB for two) would still not fit, so this direction is closed, not merely expensive.

### Measured: --mb-seqs 2 is +83% tokens/s, nearly double the +51% predicted

| | G=1 (13q) | G=2 | |
|---|---|---|---|
| latent H2D count | 1240 | **1240** | unchanged, as the model requires |
| latent Adam count | 620 | **620** | unchanged |
| h2d | 148.1 s | 163.8 s | |
| adam | 40.1 s | 40.4 s | |
| d2h | 40.0 s | 46.3 s | |
| **s/step** | **234.7** | **256.7** | **+9.4%** |
| tokens/step | 5120 | 10240 | |
| **tokens/s** | **21.8** | **39.9** | **+83%** |
| **days for 16M tokens** | **8.5** | **4.6** | |

Steady deltas 257/255/257/258 -- tight. The plumbing counts are IDENTICAL at 1240 H2D and 620 Adam
while tokens doubled, which is the whole thesis confirmed directly rather than inferred.

**It beat the +51% projection because the projection assumed GPU compute would double. It rose only
~22 s, not 75.** At batch 1 the GEMMs are WEIGHT-LOAD BOUND -- a 27B ternary weight streamed from VRAM
for a single 2560-token sequence -- so the second sequence reuses the already-loaded weight and rides
along at roughly a third of the cost of the first. This is the same arithmetic-intensity argument as
the host-side one in 13r, one level down the memory hierarchy, and it means the per-token GPU cost
FALLS as G grows.

**VRAM is the binding constraint, not compute.** GPU 0 went 16.3 -> 19.4 GB of 24.6 (~3.1 GB per extra
sequence), so G=3 (~22.5 GB) is the practical ceiling on rank 0 and G=4 (~25.6 GB) would OOM. Rank 1
has far more headroom (12.7 GB used), which is a REASON TO REVISIT THE SPLIT: 13r found split 36
slower per step, but per-step time is no longer the objective -- moving layers to rank 1 to buy VRAM
headroom for a larger G could win on tokens/s even while losing on s/step.

### The G sweep, and where it stops

| G | s/step | tokens/step | tok/s | vs G=1 | days for 16M | GPU0 VRAM |
|---|---|---|---|---|---|---|
| 1 | 234.7 | 5120 | 21.8 | — | 8.5 | 16.3 / 24.6 GB |
| 2 | 256.7 | 10240 | 39.9 | **+83%** | 4.6 | 19.4 GB |
| 3 | **275.0** | 15360 | **55.9** | **+156%** | **3.3** | **22.4 GB (2.2 free)** |
| 4 | — | — | — | — | — | ~25.9 GB — **would OOM** |

Steady deltas at G=3: 275/273/273/279. The marginal cost of each extra sequence FALLS (+22.0 s for
the 2nd, +18.3 s for the 3rd) because the GEMMs are weight-load bound, so tokens/s is still climbing
when VRAM runs out. **G=3 is the ceiling on the current split**, and the limit is memory, not compute.

**Session total: 501.9 s/step at 5120 tokens (10.2 tok/s, 18.1 days) -> 275.0 s/step at 15360 tokens
(55.9 tok/s, 3.3 days) = 5.5x throughput.**

Note the two halves are independent and multiply: 13q made the step 2.14x cheaper by fixing the
optimizer, and 13s got 2.57x more tokens through each step by amortising what the step pays per
microbatch.

### Reaching G=4 is a VRAM problem, and the recorded lever for it is STALE

G=4 needs ~3.5 GB more per rank. Rebalancing cannot supply it: at G=3 rank 0 holds 22.4 GB and rank 1
19.6 GB, so split 36 would land both near 24 GB. The §13p note listing "the fp32 lm_head copy on rank
0 (~5 GB), Wlm held twice (~4.7 GB)" as open levers is **stale for VRAM** -- `_Wlm` is already
`.to(torch.bfloat16).cpu()` and pinned, i.e. it is HOST memory and was already offloaded. Whatever is
holding rank 0's 22.4 GB has not been identified; `vram_audit()` exists but is gated behind
`VRAM_AUDIT=1`, which no arm in this campaign set. **Run one arm with `VRAM_AUDIT=1` before
speculating about G=4** -- that is the single highest-value diagnostic left, because tokens/s is now
limited by VRAM and nothing else.

## 13t. Where rank 0's VRAM actually goes, and what that permits (2026-08-31)

13s left tokens/s limited by VRAM: G=3 ran with ~1 GB free on rank 0 while the marginal cost per
extra sequence was still FALLING. First run in the campaign with `VRAM_AUDIT=1` (the flag existed all
along; no arm had ever set it).

**Rank 0, `--mb-seqs 3`, mid-training:**

    cuda:0  packed 3.80  other params 2.55  scale 0.24        -> 7.40 GB persistent
    cuda:0  allocated 7.40  reserved 23.39  peak 22.32  free 1.03 / 25.21 GB

**Fragmentation is NOT the problem, which kills the obvious fix.** Peak ALLOCATED is 22.32 GB and
reserved is 23.39 -- barely 1 GB of allocator slack. Rank 0's memory is genuinely live, so
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` has almost nothing to reclaim there. (Rank 1 does
carry ~3.5 GB of slack -- allocated 3.47, reserved 19.95 -- but rank 1 is not the constraint.)

So of rank 0's 22.3 GB: **7.4 GB is persistent weights and ~14.9 GB is activations + dequant
transients**, and the latter is exactly what scales with G. Measured VRAM by G: 16.3 / 19.4 / 22.4 GB
for G = 1 / 2 / 3, i.e. **~3.0 GB per extra sequence**, which puts G=4 at ~25.4 GB against 25.21 GB
usable. It misses by ~2 GB after the 1 GB already free.

**RETRACTED: "a 5.1 GB fp32 latent is GPU-resident on rank 0."** Inferred in 13s from the
code-transfer log (311 latents = 66.0 GB, but only 310 = 60.9 GB got codes). The audit shows NO
`latent` category on either device. The exclusion test is `latent.device != scale.device`, and the
excluded module has BOTH on CPU -- it belongs to the other rank's stage -- so it was never in VRAM.
Nothing to reclaim there.

**Also stale, confirmed:** the §13p note listing "the fp32 lm_head copy on rank 0 (~5 GB), Wlm held
twice (~4.7 GB)" as open VRAM levers. `_Wlm` is already `.to(torch.bfloat16).cpu()` and pinned --
host memory, already offloaded, and 2.55 GB of "other params" on rank 0 is the bf16 embedding table.

**Minor bug in the instrument:** `vram_audit` counts `packed` twice, once by module attribute and
once as a registered buffer, which is why every report shows `buffers` equal to `packed` and a
NEGATIVE "unnamed". The per-category numbers are right; the `named` total is inflated by exactly the
packed size.

**The real lever for G>=4 is the dequant transient.** `_LATENT_DEQUANT_MIN_ELEMS` defaults to
268M elements, so only tensors ABOVE that chunk at all -- but a typical 27B latent is 53M, so the
common case takes the unchunked path and materialises ~6 full-size temporaries (~1.3 GB for a 202 MiB
latent). Lowering the threshold makes every latent chunk, bounding that transient. It costs a
`torch.cat` per dequant and does not touch the persistent 7.4 GB.

### G=4 REACHED: 68.5 tok/s, 2.70 days -- by freeing ~3 GB, not by finding slack

The first `--mb-seqs 4` attempt OOMed, and the traceback named the exact site:

    File "e2e_qp_distill.py", line 280, in dequant
      return torch.cat(outs, dim=0).reshape(...)
    OutOfMemoryError: Tried to allocate 340.00 MiB ... 310.06 MiB is free.
    Of the allocated memory 22.32 GiB is allocated by PyTorch, and 114.20 MiB is
    reserved by PyTorch but unallocated.

**114 MiB reserved-but-unallocated proves `expandable_segments:True` worked and that there was no
fragmentation to reclaim** -- the run needed ~3 GB of real reduction. Two changes supplied it:

1. **The chunked dequant's `torch.cat` doubled its own peak.** `outs` holds a full output's worth of
   chunks while `cat` allocates a SECOND full-size result. Writing each chunk into a preallocated
   tensor makes the peak 1x the output plus one chunk. Verified against `cat`: max|dvalue|,
   max|dgrad_L| and max|dgrad_s| all exactly 0, with non-trivial gradients.
2. **bf16 decode on the code path.** `dequant()` built the weight in fp32 (q, mask and the output all
   4 B/element) and `forward()` then cast to bf16, so ~12 B/element of transient per weight, with
   ~7 weights alive per layer during a checkpoint recompute. Decoding in bf16 halves that to 6.

**bf16 is safe HERE and only here.** Plain `--latent-bf16-compute` rounds `L/s` in bf16, whose 8-bit
mantissa puts ~0.5% of latents close enough to a bin boundary to flip spuriously -- against a real
motion rate of only ~3.55%, that is noise comparable to the signal. With `--latent-code-transfer` the
assignment q and the STE gate are computed on the CPU in fp32 and shipped exactly, so bf16 touches
only the weight VALUE q*s, which the GEMM consumes in bf16 anyway. **The code path, which was a WASH
on speed at G=1 (13r, -1.0%), is what makes the memory saving admissible.**

Result: rank 0's peak allocated is **22.32 GiB at G=4 -- identical to G=3** (22.80 GiB by step 4).

| G | s/step | tokens/step | tok/s | vs G=1 | days for 16M | GPU0 peak |
|---|---|---|---|---|---|---|
| 1 | 234.7 | 5120 | 21.8 | — | 8.5 | ~16.3 GB |
| 2 | 256.7 | 10240 | 39.9 | +83% | 4.6 | ~19.4 GB |
| 3 | 275.0 | 15360 | 55.9 | +156% | 3.3 | 22.32 GiB |
| **4** | **299.0** | **20480** | **68.5** | **+214%** | **2.70** | **22.80 GiB, 0.70 free** |

Steady deltas at G=4: 299/299/299. **s/step rose 27% while tokens/s rose 214%.**

**CAVEAT -- G=4's arm is not config-identical to G<=3.** It adds `--latent-code-transfer` and
`--latent-bf16-compute`, so the latent gradient and the scale gradient are bf16 (the fp32 master and
the assignment decisions are untouched). Adam normalises by sqrt(v), so a ~0.4% relative gradient
error should be benign, but this is a PRECISION CHANGE that no quality metric has been run against.
Validate assign-moved ratio and held-out KL before committing to it for a production run; G=3 at
55.9 tok/s needs neither flag and is the conservative fallback.

**G=5 is not reachable**: 0.70 GB free at G=4, and each sequence costs ~0.5 GB even after the bf16
saving. The next real headroom would have to come from the ~7.4 GB of persistent weights or from
rank 1's loss transients, neither of which this campaign has attacked.

## 13u. Quality check on the G=4 config — and why the obvious experiment has NO POWER (2026-09-01)

13t shipped `--mb-seqs 4` at 68.5 tok/s but flagged that its arm is not config-identical to G<=3: it
adds `--latent-code-transfer --latent-bf16-compute`, making the latent GRADIENT bf16. This is that
check.

### The training-run comparison was ABANDONED because it cannot discriminate

Design was three arms at equal DATA (96 sequences each): A = G3 fp32, B = G3 + code + bf16 (isolating
precision at fixed batch), C = G4 + code + bf16. Arm B's first eval:

    [held-out] ENTRY  KL=8.2820 flips=96.26%
    [held-out] step 4 KL=8.2820 flips=96.26% assign-moved=0.000%

**Bit-identical, and zero assignments moved.** Not a short-run artefact -- it is what this config
does. `--latent-init center` sets `L = tern*scale`, so every latent sits EXACTLY on a bin centre;
`init_latent`'s own docstring records "0.000% of weights within 0.05 of a decision boundary" and calls
it DEGENERATE, with a measured bimodality of **0% flips at lr<=2e-4, 18%->37% runaway at 5e-3**. Every
throughput arm in 13a-13t ran `center` init at `--latent-lr 5e-7`, i.e. deep in the inert branch.

Since the forward VALUE is `q*s`, if no `q` changes the output is bit-identical whatever the gradient
precision. All three arms would have returned the same numbers after ~5 h, and reporting that as
"G=4 passes" would have been a **false pass from a test with no power**. Stopped at that point.

The regime where assignments actually move is fp-spread init + `--lr 0` + latent-lr 1e-4 + TALR
(§8r). `--latent-init fp-spread` requires `--fp-model <rotated FP dir>`, which does not exist on this
box (only the base 27B and the naive student), so the faithful experiment is not runnable here.

**Note for anyone reading 13a-13t as training results: they are not.** At `center` init and lr 5e-7
nothing moves, so those arms measure throughput on a model that never changes. That is exactly what
they were for, but it also means no quality claim can be read out of them.

### Direct measurement instead: does a bf16 gradient move a DIFFERENT SET of latents?

Quality can only change through `q`, so the question is not the size of the gradient error but
whether it pushes different latents across a boundary. Simulated in `L/s` units (center init puts
every latent on an integer; the boundary is 0.5 away) through the REAL adam-blockv kernel (lerp_ +
block-256 vector_norm second moment). Both arms consume an IDENTICAL gradient sequence; the only
difference is bf16 rounding, exactly as `--latent-bf16-compute` delivers it. 384k latents, 200 steps.

| regime | moved (fp32) | bf16 differs | as % of moved | fp32-eps control |
|---|---|---|---|---|
| signal-dominated | 40.174% | 0.0016% | 0.00% | 0.0000% |
| balanced | 32.117% | 0.0060% | 0.02% | 0.0000% |
| noise-dominated | 1.103% | 0.0021% | **0.19%** | 0.0000% |

Aggregate motion is unchanged to three decimals (40.174 vs 40.175, 32.117 vs 32.116, 1.103 vs 1.103).
bf16 relocates **at most 0.19% of the assignments that move**, and 0.00% in the signal-dominated
regime that real training should occupy.

The control matters: perturbing the fp32 gradient at fp32 rounding level (1.2e-7 relative, what a
different GPU reduction order does) changes **0.0000%** of flips. So flip decisions are NOT
chaotically sensitive, which means bf16's effect is a real, bounded signal rather than noise
amplification -- and that the measurement is sensitive enough to have detected a problem.

**A CONTROL BUG WORTH RECORDING.** The first version drew the perturbation from the SAME generator as
the gradients, which advanced the stream and made the "control" a different RUN rather than the same
run perturbed. It reported the control changing 2.29-69.43% of flips -- 150-575x MORE than bf16 --
which would have inverted the conclusion. A control that consumes randomness is not a control.

### Verdict

**bf16 latent gradients are safe for this use**, with the residual risk quantified rather than
asserted: identical aggregate motion, <=0.19% of flip decisions relocated, 0% where the gradient
signal dominates. Combined with the fact that the code path already keeps the assignment decision
itself exact (q and the STE gate are computed on the CPU in fp32 and shipped as a uint8 code), the
G=4 config's only numerical difference from G<=3 is bounded at this level.

**CAVEAT, stated plainly:** this is a mechanism study with synthetic gradients, not an end-to-end
training comparison. It isolates precision -> flip decisions correctly, but does not capture real
gradient structure or its correlation across steps. The end-to-end check still wants fp-spread init
and a rotated FP model; until then, G=3 at 55.9 tok/s remains the option with no precision change at
all, and G=4 at 68.5 tok/s carries this bounded and measured risk.

## 13v. First END-TO-END pipeline run with the throughput work (4B, all latents, G=4) — 2026-09-01/02

Everything in 13a-13u measured throughput on a config that never moved an assignment. This is the
first run that actually trains through the full recipe and reaches the deploy gates.

**Why the 4B and not the 27B.** A 27B quality run needs three upstream phases that do not exist on
this box -- a 27B rotation, a 27B teacher cache, and a block-AP skeleton -- and, decisively, **the
teacher cache every throughput arm used is the 4B one** (`output_4bpipe` is 32 layers / hidden 2560;
`_scratch_naive27b` is 64 / 5120). Distilling a 27B student toward a 4B teacher is fine for timing
and useless for quality. The 4B also has every prerequisite on disk AND is the testbed the recipe's
constants were calibrated on, so its gate thresholds actually apply.

**Setup.** Fresh `output_4b_g4/` symlinking the 4B rotation, the 16M-token calib (6244 seqs =
15.98M tok), the teacher cache and the block-AP skeleton, so phases 1-4 skip in 50 s and nothing in
`output_4bpipe` is touched. Assignment run as ONE full-latent pass (`--train-weights all
--tw-layer-stride 1`) instead of the validated down->attn pair, at `--mb-seqs 4 --cpu-threads 8`
with `--pipe-parallel --pipe-parallel-mb 2`. **This config actually trains**: `--latent-init
fp-spread --fp-model $ROT` + `--target-tr 1.25e-4` (TALR), not the inert `center` init of the
throughput arms.

### Stage results (frozen 1946-seq eval2k referee)

| stage | agreement | mean KL |
|---|---|---|
| block-AP skeleton | 62.92% | 0.9222 |
| + assignment, all latents, G=4 | **73.63%** | **0.5098** |
| + E2E (2 ep) | (pending) | (pending) |
| **Gate A threshold** | **>=77%** | — |

The full-latent assignment pass alone bought **+10.71 points of agreement and -45% KL** -- the payoff
for the scope the whole throughput campaign existed to make affordable. **And it did so on HALF the
intended token budget** -- see the selection bug below -- so 16M tokens of assignment should do
better than this.

### THE DELIVERED ASSIGNMENT MODEL IS THE STEP-400 STATE, NOT STEP 780

    [22:10:51]  [held-out] step 400 KL=0.2899 flips=21.16% assign-moved=0.590%  best=0.2899
       restored best held-out checkpoint (KL 0.2899, flips_used 0)
       final save (select=final)

The end-of-training restore is UNCONDITIONAL -- it ignores `--select final` by design ("deliverable =
best HELD-OUT-KL checkpoint"). That is defensible alone, but it is fatal in combination with the eval
cadence: `--eval-every 400` with `steps=780` evaluates ONLY step 400, so the final state is never
scored, can never be "best", and the run silently reverts. **Steps 401-780 were discarded: 48.7% of
the assignment compute, ~13.6 h, and half the 16M-token budget.** The stage still reports its full
27:53:31 wall time, so the cost was paid and thrown away.

`--select final` being silently overridden is a second wart: the flag reads as "keep the final
state" and does not. Behaviour left as-is (it is the validated recipe's intent), but the name lies.

FIX (both eval sites): `if held_idx and (opt_step % eval_every == 0 or opt_step >= args.steps)` --
always evaluate the final step so it is a selection candidate. The same bug was about to cost E2E its
last 244 of 6244 steps (3.9%), since 300 does not divide 6244 either.

LESSON: any run whose deliverable is "the best evaluated checkpoint" MUST evaluate its last step, and
`eval_every` should divide `steps`. A cadence that does not divide silently truncates training.

Assignment held-out trajectory (4-seq set, not comparable to the referee): entry KL 0.5595 /
flips 29.47% -> step 400 KL 0.2899 / flips 21.16% / **assign-moved 0.590%**. Non-zero motion is the
point: it confirms the run is training, against the flat 0.000% that made the 13u quality comparison
vacuous.

### TWO BUGS THE THROUGHPUT ARMS COULD NEVER HAVE CAUGHT

Both were in code this session had touched, and both were silent.

1. **`--mb-seqs` broke the epoch budget.** `spe = n // (world * accum)` assumes one sequence per rank
   per step. Under `--pipe-parallel` the ranks are STAGES, not replicas -- both walk the SAME
   sequences -- so `world` does not divide the data; a step consumes `mb * mb_seqs`. The formula
   agreed only by coincidence (mb defaulted to 2, world was 2). At G=4 it over-counted 4x: a 1-epoch
   16M-token budget became 3122 steps of 8 sequences = **64M tokens, 124 h instead of 31**. It would
   have over-trained silently and produced a wrong "16M tokens took X" number. The log now prints
   `seqs/step` so the budget is checkable at a glance.
2. **`host_tensor_inventory` ran every step, ungated** -- `gc.get_objects()` plus a 14-line dump
   after every optimizer step, left over from the 13p memory hunt. Measured cost: **143.0 -> 107.3
   s/step, i.e. 25% of the step.** It inflated every throughput number taken after 13p equally, so
   the relative comparisons in 13q-13t still hold, but the absolute s/step figures there are ~25%
   pessimistic.

### Timings (lib_timing.sh, per phase)

| phase | wall |
|---|---|
| 1-4 (rotation, chat pool, calib, block-AP, teacher) | 00:00:50 (all skipped/reused) |
| 4.5 assignment — all latents, stride 1, G=4, 780 steps | **27:53:31** |
| 5 E2E — 2 epochs, 6244 steps | (pending) |
| 6 gates A + B | (pending) |

Assignment ran 780 steps at ~107 s/step early, drifting to ~122 s; 16M tokens at ~191 tok/s.

### RESULT: Gate A PASS, Gate B marginal FAIL

| stage | eval2k agreement | mean KL |
|---|---|---|
| block-AP skeleton | 62.92% | 0.9222 |
| + assignment (all latents, G=4, ~8.2M tok effective) | 73.63% | 0.5098 |
| + E2E (2 ep, commit-beta 1.5) | **78.50%** | **0.3800** |
| **Gate A threshold** | **>=77% → PASS (+1.50)** | |

E2E held-out trajectory was monotonic throughout: KL 0.2776 -> 0.2088 (-24.8%), flips 23.56% ->
20.74%, no instability at G=4.

**Gate B (the deploy gate) — FAIL on 2 of 3, both marginal:**

| criterion | result | target | FP teacher |
|---|---|---|---|
| commit_rate | **0.7917 PASS** | >=0.68 | 0.75 |
| loop_rate | **0.3125 FAIL** | <=0.30 | 0.25 |
| mean_comp_ratio | **3.2481 FAIL** | <=3.1 | 2.40 |

(n=48, temp 0.6, maxnew 2048; trunc 0.1875, mean_think_len 748, max_comp_ratio 14.08.)

**The commit rate BEATS the FP teacher** (79.2% vs 75%) -- the failure mode §8 was built to prevent
(the model never emitting `</think>`) is solidly fixed. The two failures are loop/repetition.

**The loop_rate failure is not statistically distinguishable from a pass.** n=48, so 0.3125 is
15/48; 14/48 = 0.2917 would pass. The threshold sits **0.19 standard errors** away (SE = 6.7 pts).
One sequence decides it. `mean_comp_ratio` is likewise a mean over a heavy tail (max 14.08).
Do NOT read this as "G=4 degrades quality" -- the sample cannot support that claim either way.

**Two confounds, both favouring a re-run before drawing conclusions:**
1. The assignment stage effectively trained on **~8.2M tokens, not 16M** (the eval/restore truncation
   above). Half the intended budget.
2. The scope was ONE full-latent pass (`--train-weights all --tw-layer-stride 1`), not the validated
   `down` -> `attn` pair the gate thresholds were calibrated against.

So this run does not show the recipe failing; it shows a non-validated scope, on half the assignment
budget, landing one sequence outside a noisy gate.

### Timings — the answer to "how long does this take"

| phase | wall |
|---|---|
| 1-4 rotation / chat pool / calib / block-AP / teacher | 00:00:50 (reused) |
| 4.5 assignment — all latents, stride 1, G=4, 780 steps | **27:53:31** |
| 5 E2E — 2 epochs, 6244 steps | **06:10:59** |
| 6 gates (eval2k + free-gen) | ~00:30 |
| **total (with phases 1-4 reused)** | **~34.6 h** |

From scratch on the 4B, phases 1-4 would add roughly a day (block-AP + teacher cache dominate).
Assignment ran ~107 s/step early, drifting to ~122 s (191 tok/s); E2E ~2.76 s/step.

## 13w. GATE B IS NOT REPRODUCIBLE — and that retracts 13v's verdict (2026-09-03)

Testing OPSA (arXiv 2608.31046) against Gate B surfaced a measurement failure that invalidates the
gate's own prior result.

### The finding

`loop_gate.py` samples at `TEMP=0.6` with `do_sample=True` and had **NO SEED ANYWHERE**, so every
invocation drew a different RNG stream. Three runs on ONE UNCHANGED model (4B `e2eqp`), identical
`think/temp/maxnew/tau`, identical model path:

| run | loop_rate | commit_rate | comp_ratio |
|---|---|---|---|
| pipeline, N=48 | **0.3125** | 0.7917 | 3.248 |
| re-run, N=48 | **0.7083** | 0.3542 | 4.629 |
| re-run, N=96 | **0.6875** | 0.4062 | 4.573 |

**loop_rate spans 0.396** (sd 0.223) on a model that never changed. That is ~6 binomial SE, so it is
not sampling noise of the kind the n=48 SE implies: looping is BISTABLE per prompt and this model
sits near that boundary, so a single RNG stream flips many prompts together. The N=96 prompt set was
verified to CONTAIN the N=48 set (deterministic round-robin over EASY=40 / REASON=32 / PLAIN=16), so
the prompts are not the explanation.

### What this retracts

**13v concluded "Gate B fails on loop_rate by one sequence (15/48 vs 14/48), 0.19 SE from the
threshold, not statistically distinguishable from a pass."** That reading came from the 0.3125 run,
which two subsequent measurements identify as the OUTLIER. Two of three runs put loop_rate at
**0.69-0.71, ~6 SE ABOVE the 0.30 gate**. The 4B G=4 model is not marginally failing Gate B; on the
weight of evidence it is failing it badly. The commit_rate claim ("beats the FP teacher's 0.75") came
from the same outlier run and does not survive either: the other two runs give 0.35-0.41.

### Fix

`SEED` env (default 0), seeding torch + CUDA at import, recorded in the output JSON. A seeded run is
reproducible; **a single run still must not be used to compare two models** -- report a mean over
seeds and its spread. Any gate that samples and does not seed is not a gate.

### Consequence for OPSA

The paired N=96 comparison showed OPSA moving both target metrics the right way -- loop_rate
0.6875 -> 0.6042 (-8.3 pts), comp_ratio 4.573 -> 4.063 (-0.51), commit 0.4062 -> 0.3750 (-3.1 pts) --
which is exactly its intended mechanism. But an 8.3-pt effect cannot be read against a 39.6-pt
run-to-run swing. **The OPSA result is currently UNINTERPRETABLE**, not negative. A 5-seed paired
re-measurement is the minimum needed to say anything.

## 13x. OPSA CONFIRMED on a seeded gate (5/5 seeds, all three metrics) — and SoftWater is REJECTED

First paper method from the 2026-09 batch tested end to end. Prerequisite was 13w's seeding fix:
without it the gate swung 0.396 on an unchanged model and could not resolve anything.

### OPSA (arXiv 2608.31046) — teacher-free tail suppression

5 seeds x 2 models, N=48, `--seed 0..4`, everything else identical. PAIRED by seed:

| seed | base loop | opsa loop | base commit | opsa commit | base comp | opsa comp |
|---|---|---|---|---|---|---|
| 0 | 0.7292 | 0.5208 | 0.3542 | 0.4583 | 4.964 | 3.425 |
| 1 | 0.7500 | 0.5417 | 0.2917 | 0.4583 | 4.777 | 3.611 |
| 2 | 0.6875 | 0.6250 | 0.3750 | 0.4167 | 5.307 | 4.255 |
| 3 | 0.6667 | 0.6250 | 0.3542 | 0.3750 | 3.959 | 3.761 |
| 4 | 0.7083 | 0.5000 | 0.3333 | 0.4167 | 5.276 | 4.092 |

| metric | base | OPSA | paired delta | seeds improved | paired t (df 4) |
|---|---|---|---|---|---|
| loop_rate | 0.7083 ± 0.0329 | **0.5625 ± 0.0589** | **-0.1458** | **5/5** | 3.80 |
| commit_rate | 0.3417 ± 0.0316 | **0.4250 ± 0.0349** | **+0.0833** | **5/5** | 3.27 |
| mean_comp_ratio | 4.8567 ± 0.5483 | **3.8287 ± 0.3410** | **-1.0280** | **5/5** | 4.61 |

**Every metric improves on every seed**, all three significant at p<0.05. This is the first
CONFIRMED quality win in the campaign, and it cost 200 steps / 12h24m with NO teacher forward.

**It does NOT pass Gate B.** loop 0.5625 vs the <=0.30 target -- it closes 36% of the gap. That is
consistent with its mechanism: it suppresses degenerate tails, while 8a attributes the underlying
failure to distribution collapse at the `assistant\n<think>\n` position caused by the TERNARY
lm_head. OPSA treats the symptom well; it is not the cure.

**The seeding fix is what made it measurable.** Seeded baseline sd is 0.0329 -- 12x tighter than the
0.396 unseeded range. The pipeline's original 0.3125 sits **12.0 sd** from the seeded baseline mean,
which retires it as an outlier for good.

**Retraction of an interim claim.** The single-run N=96 comparison in 13w reported commit_rate
getting WORSE under OPSA (-3.1 pts). With 5 seeds it is clearly BETTER (+8.3 pts, 5/5). A single
unseeded run got the SIGN wrong, not merely the magnitude.

### SoftWater (arXiv 2608.12026) — REJECTED, incompatible with TQ1_64

Rated "Strongly Adopt" in the incoming analysis and targeted at exactly our failure (the ternary
lm_head). Reading the paper rejects it on the SAME grounds the same analysis used to reject ECASQ:

* SoftWater's entire benefit is **unequal rate across classes** -- "fine grids to frequent,
  low-variance classes and coarse grids to rare ones".
* Alg. 1 step 10 is `B_i <- EC(Z_SIC[:,i])`, entropy coding per column, and §2.2 states outright:
  **"any unequal-rate scheme needs: entropy coding, which turns the integer codes into a bitstream"**.
* TQ1_64 is "1.7812 bpw, uniform (self-contained, no row term)" on a fixed 512-weight superblock.
  Variable-length codes break the strided GEMV layout -- **the exact reason ECASQ (§7) was rejected**.

The paper's "the decode format does not change" means unchanged relative to WaterSIC, its baseline,
which already entropy-codes. It is NOT a statement of fixed-stride compatibility.

**There is no free residual to salvage at fixed rate.** `β_k` is normalised to unit geometric mean
(Alg. 1 step 2) and "rows are quantized independently against the same L" (§4), so a per-row scalar
CANCELS in the rounding decisions. Without variable rate SoftWater reduces to plain GPTQ.

Cost comparison for the same goal (fixing the ternary head), computed from real tensor shapes --
embed+head is 2.543B of 27.78B (**9.2%**) on the 27B and 1.271B of 5.30B (**24.0%**) on the 4B:

| option | 27B bpw | 27B size | delta |
|---|---|---|---|
| all ternary TQ1_64 | 1.7812 | 6.19 GB | — |
| q4_K head+embed | 2.0300 | 7.05 GB | +0.86 GB (+14.0%) |
| int4 + g64 fp16 scales | 2.0072 | 6.97 GB | +0.79 GB (+12.7%) |
| SoftWater | — | — | **format-incompatible** |

So the head can be fixed by SPENDING 14% bpw, but not for free via SoftWater. Note the 4B is the
worst case for this failure mode (24% of params in embed+head vs 9.2% at 27B) and it is what we are
testing on.

## 13y. SchurOpt grid refit GRAFTED ONTO GPTQ — NEGATIVE, and the graft is the reason (2026-09-04)

Second paper method from the batch. Result: **56.11% -> 5.59% eval2k agreement**. The method is not
refuted; my ADAPTATION of it is.

### What was implemented

SchurOpt (arXiv 2608.15567) names two gaps in GPTQ-family PTQ: (a) group decisions ignore what the
continuous suffix can absorb, and (b) "discrete refinements typically keep the affine quantization
grid fixed". Our `_gptq_ternary` had exactly (b): the per-row scale came from `_block_scale`'s
UNWEIGHTED MSE search, chosen once BEFORE any code was picked, never revisited. For symmetric ternary
the zero-point drops out of Prop. 2, leaving a clean closed form

    s* = diag(Z S W^T) / diag(Z S Z^T)

alternated with code updates. Implemented as `_schur_refit`, wired as `--gptq-refit-iters` (default 0).

Pre-run checks all passed: converges by ~2 iterations, every scale positive and finite, and on
synthetic groups the weighted reconstruction error falls in proportion to Hessian anisotropy
(0.11% isotropic, **16.87%** at a realistic condition number ~4e3).

### Clean A/B at the skeleton (same harness, only the flag differs)

| arm | eval2k agreement | mean KL | block-MSE (L31) | sparsity |
|---|---|---|---|---|
| refit=0 (control) | **56.11%** | 1.1988 | 3.793e-02 | 0.456 |
| refit=4 | **5.59%** | 6.7739 | **3.667e-02** | 0.462 |

**The refit did exactly what it was designed to do and the model still collapsed.** Local block-MSE
IMPROVED (3.793e-2 -> 3.667e-2), sparsity is unchanged, no NaN, no scale explosion, and every 64-wide
block is still exactly {-s, 0, +s} (2560 blocks sampled, 0 violations) so the TQ1_64 format is intact.

### Why the graft is invalid

SchurOpt's refit is defined INSIDE SchurOpt's optimizer -- Schur-conditioned coordinate descent with
the suffix's continuous response eliminated analytically. Their ablation's "refit" arm refits within
THAT optimizer. I grafted Eq. 16 onto GPTQ's sequential error-feedback pass, a hybrid the paper never
proposes and whose numbers therefore do not apply.

Mechanically: GPTQ pushes each column's rounding residual forward through Hinv, so the scale is not a
free local choice -- it sets the residual structure the compensation then propagates. Optimising the
grid for the block's ISOLATED weighted error changes those residuals, and the error accumulates over
40 blocks x 32 layers. Better per-block reconstruction with catastrophically worse end-to-end output
is the exact pattern the paper warns about ("tighter reconstruction does not consistently improve
end-model metrics"), here in the extreme.

**Testing SchurOpt properly requires replacing GPTQ's optimizer, not decorating it** -- Schur
conditioning and code descent together, since the ablation shows the refit's value is conditional on
the curvature it is refitting against. That is a much larger change than this was.

### Two measurement notes

* **The recorded 62.92% skeleton was NOT a valid control.** It came from `output_4bpipe`, an earlier
  run with different settings. The same-harness control is **56.11%**. Comparing the refit arm against
  62.92% would have overstated the damage; building the control was necessary and cheap.
* **Stale `_recovery_staging` silently produces an UNRECOVERED model.** The resume check only tests
  that `inputs_after_l` + `layer_l` exist for `l < NUM_HIDDEN_LAYERS`. A crashed run leaves those
  behind; the next run then sets `start_layer = 32`, executes an EMPTY recovery loop, and saves the
  input model. Symptoms: `recovery_report.json` shows `layers: {}`, block-AP "finishes" in ~2 min, and
  eval2k reads 3.75%. The A/B driver now asserts `layers_recovered == 32`. Root cause of that crash was
  mine: `ORIG_MODEL` must be EXPORTED because `config.py` binds `NUM_HIDDEN_LAYERS` at import and
  defaults to the 27B's 64 -- a 4B run then walks off the end at layer 33.

`--gptq-refit-iters` is left in the tree, defaulted OFF, with this result recorded against it.

---

## 13z. SchurOpt as a SIBLING optimizer — ABANDONED after a verified-correct implementation diverged
## on the real model (2026-09-05)

13y ended with "testing SchurOpt properly requires replacing GPTQ's optimizer, not decorating it."
That was done: `_schuropt_ternary()` implements Alg. 1 as a sibling of `_gptq_ternary()`, dispatched by
`_ternary_fit()` behind `--quant-optimizer gptq|schuropt` (default **gptq**, unchanged). It is
abandoned NEGATIVE. No skeleton was ever scored, because the smoke never produced a finite model.

### The failure

Per-layer mean block-MSE, 4B, `--samples 32 --qat-epochs 1`, against a GPTQ control run at the
IDENTICAL reduced config (this control was built specifically to rule out a low-sample confound):

| layer | GPTQ (same config) | SchurOpt |
|---|---|---|
| L0 | 8.760e-05 | **1.851e+01** |
| L1 | 6.681e-05 | 1.281e-03 |
| L2 | 1.246e-04 | **4.242e+06** |
| L3 | 2.062e-04 | **nan** |
| L17 | 1.634e-03 | (dead) |

GPTQ holds 1e-5..1e-3 across all 18 layers and drifts up only gently. SchurOpt is **five orders worse
at L0** — before any depth accumulation exists — and NaNs by L3. The scale-guard fire rate escalates
with depth (0.489, 0.000, 0.001, ..., 2.222, 4.775, 3.854%), i.e. scales run away systematically
rather than failing on isolated rows.

### The algebra is NOT the bug

Verified against independent references before and after the failure: the Schur complement `S` to
8e-6, `K = -P[:g, g:].T @ S` (Alg. 1 line 6) against `solve(G_rr, G[i2:,k])` to 6e-7, the `inv(G_rr)`
block-recursion identity, and PSD preservation across all 144 chunks. On synthetic problems the
sibling BEATS GPTQ on the weighted objective. Two fix attempts were made and both failed: the
effective-target init `W_eff = solve(S, T.T).T` with an `a_init`-relative `[0.1, 10]x` clamp cleaned up
the synthetic case (guard 0/184320) and pushed the blow-up from L0/L1 to L2/L3, but did not remove it.

### The structural reading

SchurOpt eliminates the suffix *analytically, assuming it responds continuously*. Our suffix is
ternarised too. At 3 symmetric levels with no zero-point there is far less absorption capacity than at
the paper's 2-bit **asymmetric, zero-pointed, g=128** grid, so `W_eff = S^-1 T` drifts to targets the
ternary grid cannot represent and each chunk's unabsorbed error feeds the next. That is consistent
with both the paper's large reported gains and our blow-up, and with 13y's separate finding that the
scale refit helps the isolated block objective while destroying the end model. Measured upside here
was +1.26% on the local objective vs GPTQ, against the paper's +11.88 pp — the premise the gain rests
on is the part that does not hold at ternary.

### Process notes (mine)

* I called it "Fixed" after reading only L0/L1 of the smoke log. It was not; the user caught it.
* I did not flag that L0's 1.851e+01 was already ~3 orders above the control's ~3.8e-02 — the failure
  was visible in the FIRST layer of the first smoke and I read it as a warm-up transient.
* The reduced-config GPTQ control should have been built with the first SchurOpt smoke, not after two
  fix attempts; without it, "is 1.85e+01 bad?" was unanswerable.

`--quant-optimizer schuropt` stays in the tree, defaulted OFF, with this result recorded against it.
**SchurOpt (paper #6) is closed.** Along with SoftWater (#3, 13x), that retires both remaining
"Strongly Adopt" Phase-3 entries except ICBQ (#10).

---

## 13aa. ICBQ seam refinement — a SMALL POSITIVE (+0.44 pp), and a name-aliasing bug that first
## faked a 99.8% local win while destroying the model (2026-09-05)

Third paper from the batch, and the first Phase-3 entry that did not fail. ICBQ (arXiv:2608.09595) is
a pure SCHEDULE: partition depth into chunks of K, and at each chunk close re-optimise the TWO-BLOCK
windows spanning the boundary, so each seam pair is refined twice (end of chunk c, start of c+1) and
the student stream is re-rolled. The inner quantizer, the grid and TQ1_64 are untouched -- which is
exactly why it does not rest on grid capacity the way 13y/13z's SchurOpt did.

Implemented as `--icbq-chunk K` (default 0 = off). `K >= n_layers` reproduces the paper's "K = L"
Sequential-CBQ baseline. Deliberate deviation: their window ends at pair (b, b+1) with b+1 unquantized
and carried as a "provisional copy"; our sweep stages each finished layer and reverts it to meta, so
the window is shifted one block left instead. Seam count and pairs-per-window (K+1) are unchanged.

### Result (same harness, only the flag differs)

| arm | agreement | mean KL | KL(conf>0.5) | %flips |
|---|---|---|---|---|
| control (icbq off) | 56.11% | 1.1988 | 1.0197 | 43.89% |
| **ICBQ K=4** | **56.55%** | **1.1767** | **0.9959** | **43.45%** |

38 pairs, 7 seam revisits, exactly as scheduled. All four metrics move the same way. Per-layer
block-MSE improves ~6% on average (L16 -15.5%, L20 -5.9%, L24 +1.6%, L27 -6.6%, L31 -2.2%), and the
seam's SECOND visit still finds -20.9% after the first visit's -9%, which is the paper's predicted
extra contraction observed directly.

**NOT yet established as a reliable win.** N=1 per arm. Over 1.99M positions the difference between
these two MODELS is certain (binomial SE 0.035 pp), but block-AP has real run-to-run RNG -- L0..L3
block-MSEs moved 0.3-1% between runs on layers ICBQ never touched. The noise floor for a
skeleton A/B in this harness has never been measured; +0.44 pp may sit inside it. Cost to measure:
one repeat control run (~2.4 h), which would calibrate EVERY future skeleton A/B here.

### The bug that first reported 1.19%, and why three metrics endorsed it

The first run scored **1.19% agreement / KL 10.2583**. Cause was mine and mundane:

```
if full in weight_map:      # "model.layers.12..." -> MISSES, always
    ...load FP weights...
else:
    ...zeros...             # every param took this branch
```

This checkpoint indexes layers as `model.language_model.layers.N`, and the alias is applied INSIDE
`load_tensor_from_shards`. A raw dict pre-check bypasses it, so **the FP teacher for all 38 pairs was
an all-zero block** and every pair was trained toward a passthrough.

What makes this worth recording is that THREE separate signals endorsed the broken model:

* pair-MSE looked plausible (3.4e-03) -- a dead teacher is invisible to it;
* per-layer block-MSE looked SPECTACULAR (L12 -99.8%, L16 -99.9%) -- because it is measured as
  `FP_block(x) vs Q_block(x)` on the SAME stream, so when the stream degrades both sides degrade
  together and the metric improves;
* the run exited `rc=0` with `layers_recovered=32`.

Only the end-to-end referee caught it. The real tell was in the activations: the residual stream
stopped growing (h28 rms 0.209 vs FP 0.753, absmax 0.50 vs 4.25) because every block had been trained
to contribute nothing.

**Local reconstruction metrics cannot detect a corrupted stream** -- they are measured against a
target computed FROM that stream. That is the same trap as 13y (better block-MSE, model destroyed)
arriving by a completely different route, and it is now two for two.

### Guards added (all verified passing before the real run)

* a missing WEIGHT raises; only a tequila-folded bias may be zeroed;
* per-pair `teacher ms vs input ms` logged, with a hard abort when equal (a passthrough teacher is
  invisible to pair-MSE alone);
* `_icbq_load_staged` raises if any live param is absent from the staged file;
* `ICBQ_SELFTEST=1` (first window only, so ~free): re-roll must reproduce the sweep's stream
  BIT-EXACTLY -- measured rel=0.000e+00 -- and deploy->stage->reload must round-trip exactly (0
  mismatched params). The bit-exact re-roll is what isolated the fault to the teacher by elimination.

With a live teacher the honest per-pair gain is **-5% to -9%**, not the fake run's -96% to -99.6%.
That contrast is the cleanest evidence the fix took.

---

## 13ab. NAP preconditioning — NEGATIVE (-2.20 pp), and the reason generalises: preconditioning is
## incompatible with a TEACHER-MATCHING pipeline (2026-09-05)

NAP (arXiv:2608.03919) identifies normalization affine parameters as a low-dimensional high-leverage
subspace and, for PTQ, freezes the backbone and tunes ONLY those affines under the target
fake-quantization graph on the FP model, "proactively boosting quantization friendliness before
downstream reconstruction". Implemented as `--nap-epochs` (writes a preconditioned FP model, skips
recovery); format-free, since norm weights ship fp in GGUF.

### Result (identical block-AP recipe on the preconditioned model)

| arm | agreement | mean KL | KL(conf>0.5) | %flips |
|---|---|---|---|---|
| control | 56.11% | 1.1988 | 1.0197 | 43.89% |
| ICBQ K=4 | 56.55% | 1.1767 | 0.9959 | 43.45% |
| **NAP precondition** | **53.91%** | **1.2763** | **1.1089** | **46.09%** |

-2.20 pp and +0.078 KL — 5x ICBQ's effect, so comfortably outside any plausible noise floor.

It also made the model measurably HARDER to quantize, growing with depth (each arm's block-MSE
against its OWN FP model, so this is a fair read of quantization friendliness — the exact quantity
NAP claims to improve): L12 +3.2%, L20 +25.4%, L28 +32.5%, L31 +24.3%.

### Why — and this is the part that generalises

**Our pipeline matches a TEACHER; NAP's evaluates a TASK.** Block-AP's target, assignment training,
E2E distillation and the eval2k referee all measure agreement with the ORIGINAL FP model. NAP's whole
mechanism is to MOVE the FP model. In a task-accuracy setting that is free — you may move the FP model
anywhere that scores better. Here the FP model *is* the objective, so every bit of movement is a debt
the quantization gain has to repay. It did not.

**The composition is strictly lossy.** NAP tunes the norms so that `Q(W; g_new) ~= FP(g_old)`. Block-AP
then reconstructs the preconditioned model against `FP(g_new)` — it re-anchors to the shifted model.
So NAP's compensation is DISCARDED by the next stage while its drift is KEPT: worst of both.

**The ordering question is now answered, and the answer is the opposite of the paper's.** `--col-scale`
(S7) applies the SAME correction — a per-input-channel gain folded into the same RMSNorm — but AFTER
reconstruction, and it is a small positive (paired dKL -0.00038, kept in the 27B recipe). NAP applies
it BEFORE and costs -2.20 pp. For a teacher-matching pipeline, correct after; do not precondition.

### Diagnostics worth keeping

* **The movement concentrates 10-20x on exactly the norms QuaRot zeroed.** Mean |d(1+w)|:
  `post_attention_layernorm` 0.0357 and `input_layernorm` 0.0280 (both folded by QuaRot, so they
  start at gain exactly 1.0) versus `k_norm` 0.0038, `q_norm` 0.0033, `linear_attn.norm` 0.0015
  (all keep pretrained values). Uniform Adam drift would move every type by the same ABSOLUTE amount,
  so this is gradient-driven — a real interaction between our Phase 1 and this method.
* **Magnitude disagreed with prior evidence by 5x and that was the true warning.** NAP's own optimum
  wants median 3% gain changes; col-scale measured ~0.6% on the same axis. I first blamed the
  fake-quant graph (RTN instead of the GPTQ graph we ship — a real bug, fixed: L0 fakequant block-MSE
  2.393e-03 -> 8.567e-05) but correcting it did NOT shrink the band, which is what redirected the
  diagnosis to the objective rather than the quantizer.
* **A mean-based trust region does not bound a min/max-reported band.** `--nap-trust` penalises mean
  squared relative movement; the reported extremes are tails. It restrained the bulk and looked like
  it worked on 3 sampled layers, while 25 of 32 were outside +/-2%. Kept, defaulted OFF.

### Process note

I twice concluded from a prefix of the sweep — 3 layers of a 32-layer band, and L0/L1 of the SchurOpt
smoke in 13z. Both times the full sweep contradicted it. Read the whole pass before concluding; these
probes cost 40 minutes, not 4 hours, so there is no excuse for sampling.

---

## 13ac. THE NOISE FLOOR — and it RETRACTS 13aa's ICBQ positive (2026-09-06)

Every paper A/B in this harness has been N=1 per arm, and the only surviving positive (ICBQ, +0.44 pp)
sat at a magnitude nobody had shown was resolvable. block-AP has NO global seed — the QAT loop's
`torch.randperm` draws from the unseeded global RNG — so re-running the byte-identical config on the
identical calibration data samples exactly the variance in question. Two repeats + the banked control:

| run | agreement | mean KL |
|---|---|---|
| ctl run 1 (banked) | 56.11% | 1.1988 |
| repeat r2 | **56.63%** | **1.1718** |
| repeat r3 | 56.08% | 1.1985 |
| **mean ± SD** | **56.273% ± 0.309 pp** | **1.1897 ± 0.0155** |

Empirical spread 0.55 pp. **3-SD resolution limit: 0.93 pp agreement / 0.0465 KL.**

### What this does to the two results

| arm | agreement | vs control mean | KL | vs control mean | verdict |
|---|---|---|---|---|---|
| ICBQ K=4 | 56.55% | **+0.89 SD** | 1.1767 | **-0.84 SD** | **WASH** |
| NAP precondition | 53.91% | **-7.64 SD** | 1.2763 | **+5.59 SD** | **REAL (negative)** |

**13aa's ICBQ positive is RETRACTED.** The +0.44 pp is inside the noise floor, and the null repeat r2
moved FURTHER (+0.52 pp, KL -0.0270) than ICBQ did (+0.44 pp, KL -0.0221) — a run that changed nothing
but the random seed beat the treated arm. ICBQ is a wash, not a win. The implementation is still
correct (bit-exact re-roll, 38 pairs / 7 seam revisits, all guards passing); it simply buys nothing
measurable at this scale. Do not spend GPU on K=2 / K=L variants.

NAP's negative is unaffected: -7.6 SD, and independently corroborated by the friendliness metric
moving the wrong way with depth (L20 +25.4%, L28 +32.5%, L31 +24.3%).

### A caution about the SD itself

n=3 is a crude SD, and the three runs are not evenly spread: ctl1 and r3 land almost on top of each
other (56.11 vs 56.08; KL 1.1988 vs 1.1985 — 0.03 pp and 0.0003 apart) while r2 sits away from both on
BOTH metrics. So run-to-run variance behaves like ONE latent factor — the QAT permutation trajectory —
rather than independent per-metric noise, and it may be occasional-excursion rather than Gaussian. The
honest operating rule is the empirical spread, not a t-test on n=3.

### Operating rule going forward

**A single-run skeleton A/B in this harness cannot resolve anything below ~0.9 pp agreement / ~0.05 KL.**
Screen with N=1 for LARGE effects only; anything smaller needs seeds before it is claimed. This is
retroactive: it is why 13y's -50.5 pp, 13z's NaN and 13ab's -2.20 pp were always safe to call, and why
13aa's +0.44 pp never was. Cost of the calibration: 4.7 h, once, for every future A/B here.

---

## 13ad. CAT-Q is a WASH, QUASAR is a large NEGATIVE — and the batch's structural finding
## (2026-09-07)

Both screened against the measured noise floor (13ac: control n=3 = 56.273 +/- 0.309 %,
KL 1.1897 +/- 0.0155; 3-SD limit 0.93 pp).

| arm | agreement | vs control | mean KL | vs control | verdict |
|---|---|---|---|---|---|
| CAT-Q, alpha learned (v1) | 53.72% | -8.26 SD | 1.3833 | +12.49 SD | MY BUG, not the method |
| **CAT-Q, alpha frozen (v3)** | **56.51%** | **+0.77 SD** | **1.1954** | **+0.37 SD** | **WASH** |
| QUASAR self-inconsistent (v1) | 2.24% | — | 9.1156 | — | MY BUG |
| **QUASAR self-consistent (v2)** | **45.97%** | **-33.3 SD** | **1.8395** | **+41.9 SD** | **REAL NEGATIVE** |

### CAT-Q (ScaleQ-1.58 Eq. 2) — neutral once its scale is stabilised

Verified format-exact first: `deploy()` is BIT-IDENTICAL to `_deploy_ternary` (mu=0, D=0.5 makes it a
strict generalisation of our grid), and the forward anneals to it (rel err 9.6e-03 at t=0.01 -> 0.0
at t=1). Two confounded runs preceded the real answer, both mine:

* **v1: the zero region collapsed** — mean sparsity 0.4568 -> 0.4142 with blocks at EXACTLY 0.0000,
  i.e. ternary degenerating to BINARY. I blamed the learnable redistribution mean mu and clamped it.
* **v2: the clamp changed nothing** (0.4142 -> 0.4162, still min 0.0000). The region is
  `|W-mu| < 0.5*alpha`, so it also closes when ALPHA shrinks — and alpha enters CAT-Q TWICE, inside
  the tanh via (W-mu)/alpha and again in the output T*alpha. The tanh path contributes
  `-ts*W/alpha^2*sech^2`, which blows up as alpha shrinks. Measured |grad|: CAT-Q alpha **1.2e+00**
  vs the STE control's scale **1.6e-07** — a **7.7e6** gap that LSQ's 1/sqrt(bs) damping cannot close.
* **v3, alpha frozen at the GPTQ level**, making the soft-vs-hard forward the ONLY variable: **WASH**.

So CAT-Q's mechanism is neutral at ternary, and its learnable parameterisation is unstable here. The
other half of that paper, AYOT, was already in this pipeline (13ae).

### QUASAR — improves its OWN objective and destroys the model

`--qat-quasar N` refits each block scale every N steps by saliency-weighted least squares over a
clipping search, with Adam's second moment as the diagonal-Fisher saliency. Two ternary adaptations
were needed before it could even run correctly:

1. **The clipping grid.** The paper searches f*amax with f in (0,1] because at 2-4 bits the optimal
   clip sits near the max. At ternary the optimum is ~0.5*amax — BELOW their entire grid — so the
   search never reached the useful region and LOST to the plain MSE scale on QUASAR's own weighted
   objective (1.433e-01 vs 1.250e-01). Caught by unit test, before any GPU. Candidates re-centred on
   the incumbent scale with f=1.0 included, so the refit can never return something worse.
2. **Self-consistency.** QUASAR fits (s,z) for FIXED codes q because its format stores codes and
   dequantizer separately. TQ1_64 has ONE scale doing both jobs, so writing the free WLS optimum back
   re-derives codes DIFFERENT from the q it was fit for. Measured: the scale escaped its own [0.7,1.3]
   search to **0.26-2.56x**, ran systematically small, and the zero region collapsed (sparsity 0.4568
   -> 0.2326) — **2.24% agreement**. Fixed by alternating codes <-> scale to a fixed point.

With both fixed it still costs **-10.3 pp**. And the reason it matters:

**QUASAR IMPROVES THE LOCAL BLOCK OBJECTIVE AT DEPTH WHILE LOSING 10.3 pp END TO END** — L8 -10.9%,
L20 -14.9%, L31 -10.7%. That is 13y's exact pattern (better reconstruction, worse model) reached by a
completely different route, and it is now the **fourth** instance: 13y's grid refit, ICBQ's zeroed
teacher, NAP's inverse friendliness, and now this.

Mechanism: our block QAT already learns the scale by gradient on the TRUE block-output MSE. QUASAR
replaces that with a fit to a weight-space PROXY (saliency-weighted reconstruction error). Swapping a
directly-optimised parameter for a proxy-fitted one is a downgrade, however good the proxy looks on
its own terms. QUASAR's argument is about a QAT LOSS FLOOR over long training; our block QAT is 4
epochs at tiny LR from a GPTQ init, where the weights barely move, so the floor it targets is not the
binding constraint. Its proper home is the ~28 h assignment stage — untested, and not obviously worth
the day given two negatives at block scope.

### THE BATCH'S STRUCTURAL FINDING

Four methods now fail because **the paper assumes a richer parameterisation than ternary provides**:

| method | assumes | ternary reality |
|---|---|---|
| SchurQuant (13y/13z) | a continuous suffix that absorbs chunk error | suffix is ternarised too |
| NAP (13ab) | tunable normalization affines | QuaRot FOLDS them away (stored w == 0) |
| CAT-Q (13ad) | a stably learnable scale inside the soft map | alpha is ill-conditioned, 7.7e6 grad gap |
| QUASAR (13ad) | codes and dequantizer stored separately | ONE scale does both jobs |

This is predictive, and it is the cheapest screen we have: before implementing, ask what the method
needs to vary that TQ1_64 does not give it. Three of these four were diagnosable on paper or by unit
test; only NAP needed a GPU run to see.

### Second-order lesson: local metrics have never once caught a failure here

pair-MSE, block-MSE and `rc=0` all endorsed the model whose residual stream had collapsed (13aa);
block-MSE improved while SchurOpt's refit destroyed the model (13y); NAP's friendliness metric was the
only local signal that pointed the right way, and QUASAR's pointed the WRONG way at three of four
depths. What HAS caught every failure: the end-to-end referee, and cheap **structural invariants of
the format** — the sparsity histogram (found both CAT-Q's and QUASAR's zero-region collapse), the
activation-norm probe (found the collapsed stream), and a gradient-magnitude comparison (found
CAT-Q's ill-conditioned alpha). None of those are the training loss.

---

## 13ae. SQuaT — NULL BY CONSTRUCTION for a weight-only pipeline (2026-09-07)

Last of the 13. Resolved on paper, and the paper argument is STRONGER than a run would be.

SQuaT (arXiv:2608.10709) removes the "unattainable residual" in QAT+KD by projecting teacher features
onto the STUDENT'S quantization lattice: Pi_phiS,l(z) is defined (their Eq. 6) by "the student's
forward quantization path at layer l", and Eq. 7 matches the student's "actual (already quantized)
student output" against Pi_phiS(f_T). Eq. 1-2 build that lattice by quantizing x -- "either weights
or activations" -- to a b-bit uniform grid.

**That lattice is the student's FEATURE/ACTIVATION grid. We are weight-only.** There is no activation
quantizer anywhere in this repo (grep: no act_quant / quantize_act / activation-quant path); student
features are bf16. So Pi_phiS is the identity and Eq. 7 collapses EXACTLY to plain feature MSE --
which is verbatim what our `feature_loss()` already computes:

    return F.mse_loss(student_h.float(), teacher_h.float())

SQuaT would reduce to the baseline it exists to beat. The residual it eliminates is CREATED by the
student's activation quantizer; weight-only quantization never incurs it. Running it would mean
either mislabelling plain feature-KD as SQuaT, or inventing an activation quantizer we would never
ship -- measuring a model that does not exist.

This is the batch's structural screen (13ad) applied one more time, and the fifth hit: **ask what the
method needs to vary that TQ1_64 does not give it.** SQuaT needs a feature lattice; we have none.

### Two facts that complete the picture

* **`--feat-weight` is off for a MEMORY bug, not a quality result.** The hidden teacher cache is ~34 GB
  and every DDP rank `torch.load`s the whole thing (2x34 GB > 60 GB RAM -> swap-death + DDP socket
  timeout). Recorded at the time as "dropped (unvalidated anyway; 4B baselines used feat=0)". So
  feature distillation has never actually been evaluated in this pipeline.
* **We already do feature distillation where it matters most.** block-AP's entire objective IS
  hidden-state MSE against the FP block, applied at all 32 layers. SQuaT's family is well represented
  here already; `--feat-weight` only adds it to the E2E logit-KD stage on top.

### The one honest adjacent experiment (NOT run; cost decision)

Does feature-KD help at the E2E stage at all? That is SQuaT's own baseline, genuinely untested here.
If it does not help, no SQuaT variant could have anything to improve on. Cost is a different class
from this batch's screens: a memory-safe (mmap / streamed / single-rank) cache loader, then an E2E
PAIR at ~6 h each (~12 h) versus 2.4 h for a skeleton screen. Left for an explicit budget decision.

---

## 13af. E2E feature distillation — a WASH, which closes SQuaT's family (2026-09-07)

13ae showed SQuaT is null by construction here (it needs a student FEATURE lattice; we are
weight-only). That left the prior question, open in this repo since §1 and never answered: **does
feature-KD at the E2E stage help at all?** If not, no SQuaT variant could have anything to improve on.

Design: ONE teacher cache built with hidden states, used by BOTH arms, so `--feat-weight` is the only
difference. Both start from the banked control skeleton. seq stays 2560; `--max-samples 1000` keeps
the hidden cache at 14 GB instead of the 82 GB the full 6244-seq calib would need at that length.

| arm | agreement | mean KL | KL(conf>0.5) | %flips | held-out KL |
|---|---|---|---|---|---|
| E2E feat=0 | 70.89% | 0.6262 | 0.4429 | 29.11% | 0.3497 |
| E2E feat=1.0 | 70.50% | **0.6259** | 0.4485 | 29.50% | 0.3569 |

**-0.39 pp agreement, and mean KL identical to 3 decimal places (0.6262 vs 0.6259).** Inside the
noise floor. Feature-KD at the E2E stage buys nothing.

The striking part is HOW little it changed given how much it changed the trajectory. The feature term
carried 41-59% of the KL's weight throughout (logged `w*feat/KL` = 0.411 -> 0.588) and made training
KL **1.5-1.9x worse** at matched steps (step 210: 0.5709 vs 0.9897; step 310: 0.5353 vs 0.9243). It
substantially redirected the optimisation and the final model landed in the same place.

Two reasons this is unsurprising in hindsight, both already true of our pipeline:
* **E2E-QP trains only SCALES** (`assign-moved=0.000%` in both arms), so any extra loss term has
  limited leverage — the code's own log says so.
* **block-AP's objective ALREADY IS hidden-state MSE**, applied at every one of the 32 layers. The
  features are matched structurally before E2E ever runs; adding a final-hidden MSE on top is
  redundant with work already done.

### Instrumentation note

`hidden_state_loss`'s docstring said "watch the printed feat vs KL magnitudes" — nothing had ever
printed them, so the weight would have been a pure guess. Added the log first. It immediately
corrected my own estimate: I predicted `w*feat/KL ~ 0.008` from an assumed ~10% hidden discrepancy,
but the measured feature MSE is **0.3337**, giving 0.411 — a 50x error. Guessing from that estimate
would have set the weight ~40x too high and produced a "feature-KD destroys the model" result that
was purely my hyperparameter. That is the same failure that cost runs on CAT-Q's alpha and NAP's gain
range; this time the instrumentation came first and cost nothing.

### Scope

One weight (1.0), one run per arm, and the E2E stage's own noise floor is unmeasured (13ac measured
the SKELETON's, SD 0.309 pp). A smaller weight might be neutral-to-positive rather than neutral, but
with KL identical to 0.0003 and the trajectory evidence above, there is no signal to chase.

**Side result worth recording:** both E2E arms took the skeleton from 56.27% to ~70.7% agreement
(KL 1.19 -> 0.626). E2E remains by far the largest lever in this pipeline — bigger than every paper
method in this batch combined, all of which were washes or negatives.

---

## 13ag. PLAER — Gate B is MIXED (0.40), and the report's marker list is wrong for our model
## (2026-09-08)

The deep-research report's most useful contribution was not a method but a FALSIFIER. Gate B's
numbers (commit 0.4583, loop 0.5208, comp 3.4253 on the OPSA model, N=48 seed 0) are produced by two
pathologies needing OPPOSITE fixes: COMMITMENT failure (answer derived, model cannot stop -> fixable
at decoding time, 0 bpw) vs PATH-FINDING failure (no answer ever derived -> needs better weights).
PLAER separates them: of looped rollouts, what fraction already held an answer BEFORE loop onset?

**PLAER = 0.400 (10/25). MIXED.** Decoding-time control can address roughly the commitment share,
so size any expected Gate B gain by ~0.4, NOT by a paper's headline. The report asserted commitment
failure outright; that is only 40% right.

### The first answer was wrong, and how it was caught

Detector v1 (`\boxed{}` + "the answer is") gave **PLAER 0.120**, which would have said PATH-FINDING
dominates and killed the whole decoding branch. It was an artifact: **`\boxed` appears in ZERO of 48
rollouts**, and v1 fired on only **30.4%** of the SUCCESSFUL (non-looped) rollouts. An answer
detector that cannot find answers in traces that worked cannot be trusted on traces that failed.
Detector v2 matches the forms this model actually emits ("Common knowledge: Paris", "*Response:* …"),
fires on **69.6%** of successful rollouts, and gives 0.400.

**Rule: calibrate any detector on the positive class before applying it to the negative class.**

### What the traces actually show

Loop onset is EARLY — median 192 words, 7/25 inside 100 words, some at 8-13 — so a large share of
loops begin before any derivation could finish. That contradicts the report's model ("derives the
answer within 300-600 tokens, then loops verifying"). Two representative failures:

* *LCM of 1-6* (comp 4.17, never closed, onset@13w): the model loops **re-reading the prompt** —
  "Wait, let me re-read carefully" / "reading the prompt again" — and hallucinates a different
  question ("discretely divisible"). It never derives 60. No answer exists to commit.
* *Capital of France* (comp 2.35, onset@192w): knows "Common knowledge: Paris" immediately, then
  loops **polishing** — "Paris, also known as The Hague or London? No, Paris is correct" / "Wait, I
  should keep it simple". Genuine commitment failure. **804 tokens for "What is the capital of
  France?"**

Both are stuck in a meta-cognitive loop inside a rigid `Thinking Process:` / `*Draft:*` /
`*Refinement:*` / `*Final Polish:*` scaffold. The template itself looks like part of the attractor.

### The report's marker list is WRONG for this model (mean occurrences/rollout)

| marker | looped | non-looped | |
|---|---|---|---|
| "actually" | 6.7 | 2.0 | **3.4x enriched** |
| "Wait" | 5.8 | 2.4 | **2.4x enriched** |
| "re-read / read again" | 0.5 | 0.0 | looped-only |
| "Alternatively" | 0.1 | 0.0 | ~absent |
| **"However"** | **0.1** | **0.4** | **ANTI-correlated** |

The report's Candidate 3 penalises ~50 curated markers including "Alternatively" and "However".
Applied blindly here it would penalise a token that appears **4x more often in HEALTHY generations**
and one that barely occurs. Any marker-penalty arm must use THIS list (Wait, actually, re-read),
measured on our own traces.

### Consequences for the report's roadmap

* **Candidate 2 (DRY)** is the best first test: already native in `llama.cpp`, 0 bpw, and it
  suppresses verbatim continuation regardless of whether an answer exists — the only candidate whose
  value does not scale with PLAER.
* **Candidate 1 (loop rescue)** needs an answer to extract, so discount its claimed -30..-45 pp by
  ~0.4.
* **Candidate 3 (marker penalty)** only with the corrected list above.
* **Methodological guard:** Gate B's bars come from the FP teacher (commit .75 / loop .25 /
  comp 2.40). A sampler change must be applied to the TEACHER too, or the comparison is unmatched.
  And given the measured 0.396 loop-rate swing on an UNCHANGED model, arms must be seeded and run
  one at a time — the report's advice to stack three interventions at once would confound attribution.

---

## 13ah. DRY passes 2 of 3 Gate B bars — and proves the commit failure is INDEPENDENT of looping
## (2026-09-08)

First decoding-time intervention, chosen because PLAER=0.400 (13ag) meant it was the only candidate
whose value does not scale with PLAER. Implemented as a `LogitsProcessor` in the existing HF gate
(Z-algorithm longest-suffix match, penalty `mult*base**(L-allowed)`, llama.cpp's sequence breakers),
default OFF so every prior Gate B number stays reproducible — verified: `dry=off seed=0` reproduces
the earlier run exactly.

**Paired by seed** (same prompts, same base RNG), 3 seeds x N=48. Pairing matters: an UNCHANGED model
swung loop_rate 0.396 across reruns (13w), and the paired loop deltas here agree to +/-0.043.

| metric | DRY off | DRY on | delta | bar | seeds passing |
|---|---|---|---|---|---|
| **loop_rate** | 0.5625 | **0.1319** | **-0.4306 +/- 0.0434** | <=0.30 | **3/3 PASS** |
| **mean_comp_ratio** | 3.7637 | **2.6194** | **-1.1443 +/- 0.4668** | <=3.1 | **3/3 PASS** |
| commit_rate | 0.4444 | 0.4931 | +0.0486 +/- 0.1185 | >=0.68 | **0/3** |

**Gate B overall: still 0/3 seeds.** Two bars solved decisively, one untouched — the commit delta's SD
is 2.4x its mean, i.e. indistinguishable from zero.

### The finding that matters more than the pass/fail

| arm | loop | trunc | trunc but NOT looping | mean_think_len |
|---|---|---|---|---|
| DRY off | 0.5625 | 0.4722 | -0.09 (truncation ~ fully explained by looping) | 855 |
| DRY on | 0.1319 | 0.3819 | **+0.2500** | **928** |

**DRY converted "looping until budget" into "rambling until budget."** A quarter of rollouts now run
to the 2048-token cap without repeating AND without emitting a stop token, and mean think length went
UP (855 -> 928). Removing 76% of the looping moved commit by +0.049.

**So the commit failure is NOT caused by looping.** It is an independent pathology: the model does not
emit `</think>`. Every model of this failure we have been carrying — including the research report's
("derives the answer, loops verifying until the budget is gone, so break the loop and it commits") —
is wrong on this point. Break the loop and it does not commit; it produces non-repetitive verbosity
instead.

### Consequences

* **Candidate 1 (loop rescue) is now nearly pointless**: with DRY applied only 13% of rollouts loop,
  and PLAER says ~40% of those hold an answer, so its ceiling is ~5 pp of commit — against a 19 pp gap.
* **The target is the stop-token decision itself.** And we already built the tool: §8e records a
  post-hoc `</think>`-row lm_head calibration (`run_thinkcal.sh` + `fold_think_scale.py`) that scales
  `scale[row*40:(row+1)*40]`, leaving assignments untouched — on-grid, TQ2_0-exact, **0 bpw**. It was
  built and NOT applied because at the time "FINAL is already at FP-parity commit, so the smallest c
  reaching parity is c=1.0". That premise no longer holds: with DRY the model is at commit 0.4931
  against a 0.68 bar, and the blocker is precisely the `</think>` decision that tool targets.
* DRY should be adopted regardless — it is 0 bpw, native in llama.cpp, and passes two bars.

### Still pending

The FP teacher with the SAME sampler (3 paired seeds, queued). Gate B's bars (commit .75 / loop .25 /
comp 2.40) were measured on the teacher WITHOUT DRY, and ternary+DRY now loops at 0.1319 — BETTER than
the teacher's no-DRY 0.25. Scoring a DRY'd student against a non-DRY'd teacher is not like-for-like;
if DRY ships in the inference config it applies to both and the bar moves with it.

### 13ah-i. Two harness gotchas worth not rediscovering

* **`loop_gate.py` ternarises whatever you give it.** It defaults to `build_student()`, so pointing it
  at an FP model silently RTN-ternarises it with no recovery. A first "FP teacher" arm scored
  **commit 0.0000 / trunc 0.9792 / comp 1.3507** against the recorded reference of .75/.25/2.40 —
  the log line `replaced 249 linears with packed TernaryScaleLinear` is the tell. **`MODEL_KIND=fp`
  is REQUIRED** for any FP-side measurement; that is how the original reference was produced.
* **Never gate a chained run on `pgrep -f`.** Zombie (`<defunct>`) processes match it forever in this
  container — a teacher run chained on `while pgrep -f "dry_ab.sh"` waited on two dead shells and
  idled the GPU **~6.5 h**. Gate on the completion ARTIFACT the driver writes (`.dry_done`). Same
  family as the recurring `pkill -f` exit-144 problem, which matches this session's own shell.

---

## 13ai. The matched-teacher arm QUALIFIES 13ah: DRY closes the loop gap, but "narrows" the commit
## gap mainly by DEGRADING the reference (2026-09-09)

Gate B's bars (commit .75 / loop .25 / comp 2.40) were measured on the FP teacher WITHOUT DRY. Since
ternary+DRY reached loop 0.1319 — better than that no-DRY teacher — the comparison was no longer
like-for-like, so the teacher was re-run with the SAME sampler, 3 paired seeds, N=48.
(`MODEL_KIND=fp` is required; see 13ah-i.)

| metric | teacher no-DRY | teacher +DRY | ternary no-DRY | ternary +DRY |
|---|---|---|---|---|
| loop_rate | 0.1806 | 0.0486 | 0.5625 | 0.1319 |
| commit_rate | 0.8056 | **0.6944** | 0.4444 | 0.4931 |
| mean_comp_ratio | 2.3823 | 2.3520 | 3.7637 | 2.6194 |

### DRY has a real, reproducible COMMIT COST

**Teacher commit falls -0.1111 +/- 0.0241 under DRY** (per-seed -0.125, -0.083, -0.125; 3/3 negative,
SD one fifth of the effect), and trunc_rate rises by exactly +0.1111 — DRY converts commits into
truncations. On a model with almost no looping to fix, only the cost is visible.

Mechanism, and it is not subtle: **committing means restating.** "Therefore the answer is 60" repeats
tokens the trace already contains, and DRY penalises precisely the continuations that extend earlier
context. It taxes conclusion-writing along with degenerate looping.

### What this does to 13ah's framing

Both framings are defensible and they disagree, so both are recorded:

* **Against the project's established ABSOLUTE bars** (the deploy gate): ternary+DRY passes loop and
  comp, fails commit — 13ah's result stands as stated.
* **Against a LIKE-FOR-LIKE teacher at the same sampler**: the teacher improves too, so the bars move
  to <=0.0486 / >=0.6944 / <=2.3520 and ternary+DRY **fails all three** (gaps 0.083 / 0.201 / 0.267).

Gap closure is the honest measure of what DRY bought:

| gap (student - teacher) | no-DRY | with DRY | closed |
|---|---|---|---|
| loop_rate | 0.3819 | 0.0833 | **78%** |
| mean_comp_ratio | 1.3814 | 0.2674 | **81%** |
| commit_rate | -0.3611 | -0.2014 | 44% — but see below |

**The commit "improvement" is mostly an artifact.** The student gained +0.049 while the teacher LOST
0.111, so most of that 44% is the reference falling, not the model rising. Reporting it as a gain
would be wrong.

### Consequences

1. **DRY is a genuine win on looping and compression** — 78-81% of those gaps closed, at 0 bpw. Adopt.
2. **Stock DRY settings are mistuned for our binding constraint.** We used llama.cpp defaults
   (mult 0.8, base 1.75, allowed 2). Commit is the metric we cannot afford to lose, and DRY costs
   ~0.11 of it. A gentler configuration (lower multiplier, larger allowed_length) should keep most of
   the loop benefit at less commit cost — worth a sweep.
3. **The commit deficit is now unambiguous and isolated**: -0.201 against a like-for-like reference,
   untouched by removing 76% of the looping (13ah). The `</think>`-row sweep now running targets
   exactly this, and it is the right next lever.

**Process note:** this arm was nearly skipped as a formality, and it inverted a headline conclusion.
Measuring the reference under the same intervention as the treatment is not bookkeeping — DRY moved
the reference by more than the treatment moved the model on the metric that matters.

---

## 13aj. `</think>`-row gain: c=1.20 closes most of the remaining gap — and c=1.40 FAKE-PASSES the
## whole of Gate B (2026-09-09)

13ah isolated the blocker to the stop-token decision; `fold_think_scale.py` / `THINK_ROW_SCALE`
(built in S8e, shelved because commit was at FP parity then) targets exactly that. Swept at the new
operating point (DRY on), seed 0, N=48. Format: multiplies the per-block scales of lm_head row 248069
— assignments untouched, on-grid, TQ2_0-exact, **0 bpw**.

| c | commit | loop | comp | **think_len** | n_closed |
|---|---|---|---|---|---|
| 1.00 | 0.5417 | 0.1042 | 2.5568 | 999 | 27 |
| 1.05 | 0.5417 | 0.1042 | 2.5403 | 948 | 28 |
| 1.10 | 0.5833 | 0.0833 | 2.5158 | 917 | 30 |
| **1.20** | **0.6250** | 0.0625 | 2.4205 | **737** | 32 |
| 1.40 | **0.8958** | 0.0417 | 1.9396 | **224** | 45 |
| *teacher +DRY* | *0.6944* | *0.0486* | *2.3520* | *586* | *33.3* |
| *teacher no-DRY* | *0.8056* | *0.1806* | *2.3823* | *648* | *38.7* |

### c=1.40 PASSES EVERY GATE B BAR AND IS WORTHLESS

commit 0.8958 (bar >=0.68 — beats even the no-DRY teacher), loop 0.0417 (bar <=0.30), comp 1.9396
(bar <=3.1), truncation zero. **All three bars pass.** And think_len is **224** against the teacher's
586-648: the model has stopped reasoning and answers immediately. Compression falls BELOW the
teacher's for the same reason — there is no trace left to repeat.

**Gate B cannot see this.** Every metric it scores says PASS while the model has been lobotomised.
The only exposure is think_len measured against the TEACHER's, which is why it was logged and the
acceptance rule fixed in advance ("smallest c reaching the bar without collapsing think_len"), and
why `run_thinkcal.sh`'s own docstring warns of premature closing. **Add think_len to Gate B**: a
model whose think_len is far below the teacher's must fail regardless of the other three.

### c=1.20 is the honest result, and it is a large one

The student over-thinks by ~54% at c=1.0 (999 vs the teacher's 648). Rising c does not cause
premature closing at first — it CORRECTS that bias, pulling think_len toward the teacher. At c=1.20,
737 is still above the teacher's 586-648, so it has not collapsed.

Gap to the like-for-like teacher (both with DRY):

| gap | DRY only | DRY + c=1.20 | closed |
|---|---|---|---|
| commit_rate | 0.2014 | **0.0694** | 66% |
| loop_rate | 0.0833 | **0.0139** | 83% |
| mean_comp_ratio | 0.2674 | **0.0685** | 74% |

Two 0-bpw decoding/format-safe changes take the ternary model from failing Gate B on every
like-for-like metric to within 0.07 of the FP teacher on all three.

### One nuance not to lose

At c=1.20 the student closes 32/48 vs the teacher's 33.3/48 — nearly matched — yet commit is 0.6250
vs 0.6944. It CLOSES about as often but some closes carry no valid answer. That is a different defect
from failing to stop, and it is what the residual 0.07 consists of.

### Pending

* c=1.25 / 1.30 to bracket where think_len crosses the teacher's floor.
* c=1.20 on 3 paired seeds before it is claimed (seed-0 scan only so far).
* DRY strength sweep (13ai): stock settings cost the teacher 0.111 commit, and commit is the binding
  constraint, so a gentler DRY may give some of that back.

---

## 13ak. DRY tuning: my "gentler DRY" hypothesis is FALSIFIED — stock settings are already best
## (2026-09-09)

13ai measured a reproducible commit COST from DRY on the FP teacher (-0.111 +/- 0.024) and reasoned
that, since committing means restating and DRY penalises continuations that extend earlier context, a
gentler configuration should keep the loop benefit at less commit cost. Swept on the student, seed 0:

| mult | allowed | commit | loop | comp | trunc |
|---|---|---|---|---|---|
| **0.8** | **2** (stock) | **0.5417** | **0.1042** | **2.5568** | **0.3542** |
| 0.4 | 2 | 0.3750 | 0.3958 | 3.0739 | 0.4792 |
| 0.8 | 4 | 0.4167 | 0.3542 | 2.9555 | 0.4792 |
| 0.8 | 8 | 0.3750 | 0.6250 | 3.2459 | 0.5625 |
| 0.3 | 4 | 0.4583 | 0.5000 | 3.1176 | 0.5208 |

**Every gentler setting is worse on BOTH metrics.** Stock llama.cpp (mult 0.8, allowed 2) wins
outright. The hypothesis is dead.

**Why it was wrong.** The teacher and the student have opposite dominant terms. On the TEACHER there
is almost no looping, so only DRY's restatement tax is visible and it reads as a pure cost. On the
STUDENT, looping *causes* truncation, and truncation *prevents* commitment — so suppressing loops
raises commit by more than the restatement tax lowers it. Weakening DRY gives back the loop
suppression and commit falls with it. Generalising the teacher's cost to the student was the error:
**a cost measured on the reference does not transfer to a model whose failure mode is different.**

Stock DRY stays. The commit deficit is closed by the `</think>`-row gain (13aj), not by detuning DRY.

### Harness note (third of this kind)

`tc_confirm.sh` died instantly on `local c=$1 s=$2 O=...${c}...` — under `set -u`, bash expands every
argument of a single `local` BEFORE assigning any of them, so `${c}` is unbound. It was chained with
output to `/dev/null`, so it failed silently and idled the GPU ~25 min. **Chained drivers must keep
their logs**; the earlier zombie-pgrep stall (13ah-i) was invisible for the same reason.

---

## 13al. c=1.20 confirmed on 3 seeds (+0.097 commit) — and TWO seed-0 claims retracted (2026-09-09)

13aj's sweep was seed 0 only. Repeating c=1.20 on 3 paired seeds:

| c=1.20 vs c=1.0, paired | delta | per-seed |
|---|---|---|
| **commit_rate** | **+0.0972 +/- 0.0120** | +0.083, +0.104, +0.104 |
| loop_rate | +0.0069 +/- 0.0524 | -0.042, +0.062, 0.000 |
| mean_comp_ratio | -0.0877 +/- 0.0430 | -0.136, -0.072, -0.054 |
| mean_think_len | -182 +/- 72 | — |

### Retractions

* **The `</think>`-row gain does NOT reduce looping.** 13aj reported loop 0.1042 -> 0.0625 at c=1.20;
  paired over 3 seeds the effect is **+0.0069 +/- 0.0524**, i.e. nothing. That was seed-0 noise. The
  cleaner (and correct) division of labour: **DRY owns looping, the row gain owns commitment.**
* **"c=1.25 is essentially at teacher parity (commit gap 0.028)" is NOT established.** Seed 0 only.
  Absolute commit_rate has **SD 0.103** across seeds (c=1.0: 0.5417 / 0.3750 / 0.5625), so a
  single-seed 0.6667 is consistent with anything from ~0.57 to ~0.77. c=1.25/1.30/1.40 remain
  seed-0 provisional.

### What IS established

`c=1.20` buys **+0.097 +/- 0.012 commit**, 3/3 seeds, SD one eighth of the effect. It moves mean
commit **0.4931 -> 0.5903**. Real, reproducible, and still **short of the 0.68 bar**.

### The measurement lesson, again

Gate B's ABSOLUTE rates are far noisier (commit SD 0.103) than its PAIRED deltas (SD 0.012 for the
same quantity). Every claim of the form "this configuration passes the bar" needs multiple seeds;
claims of the form "this change helps by X" survive on paired seeds. 13w established this for
loop_rate and it was re-learned here for commit_rate — the same trap, one metric over.

---

## 13am. THE GATE IS MEASURING THE WRONG THING — GSM8K accuracy is ~0 while Gate B improves
## (2026-09-10)

Gate B scores commit / loop / compression and **never checks whether the committed answer is
correct**. 13aj showed c=1.40 passing every bar with think_len collapsed to 224; think_len is only a
proxy, so GSM8K was scored under the IDENTICAL decoding config (same harness, same DRY processor,
same THINK_ROW_SCALE, same template/temp/seed), N=48, MAXNEW=2048.

| arm | accuracy | closed | **acc \| closed** | think_len |
|---|---|---|---|---|
| **teacher (FP)** | 0.3750 | 15/48 | **0.9333** | 970 |
| student c=1.00 | 0.0000 | 4/48 | 0.0000 | 584 |
| student c=1.25 | 0.0208 | 17/48 | 0.0588 | 707 |
| student c=1.30 | 0.0417 | 28/48 | 0.0714 | 597 |
| student c=1.40 | 0.0208 | **46/48** | **0.0217** | 331 |

### 1. The teacher's low overall score is a BUDGET artifact; its reasoning is intact

**acc|closed 0.9333 vs acc|truncated 0.1212.** When the FP model finishes it is 93% correct; when
truncated, 12%. So overall accuracy at MAXNEW=2048 mostly measures *whether it finished*, and
**closing is the dominant term in correctness** — which is why commit_rate looked like the right
thing to optimise.

### 2. The student's reasoning is GONE, and that is not a budget artifact

At c=1.30 the student closes 28/48 — a well-powered sample — and is right **7.1%** of the time. The
same model at full precision is right **93.3%**. This is a CAPABILITY loss at 1.78 bpw, not a
behavioural one, and **no decoding-time intervention can recover it.**

### 3. Raising c is SAFE but empty

acc|closed across c: 0.0000 -> 0.0588 -> 0.0714 -> 0.0217. Flat within noise up to c=1.30, so the
row gain does NOT trade correctness for closure — that answers the c=1.25 vs c=1.30 question this run
was built for. But it is flat because there is almost nothing to trade. **c=1.40 is the exception and
the proof of the trap: it closes 46/48 and scores acc|closed 0.0217 — it converts nearly every
rollout into a confidently-wrong completion, and PASSES EVERY GATE B BAR while doing it.**

### 4. What this retracts

Gate B's `commit_rate` counts a close plus a non-empty, non-looping answer. It never asked whether the
answer was right. **Every commit gain in this program has been measured without that check** — OPSA's
+0.083, DRY's, the row gain's +0.097. Those are increases in the rate of *confidently-wrong
completions*. They are not wrong as measurements; they are wrong as evidence of quality.

Confirmed alongside the repo's standing warning that teacher-forced metrics are blind here: eval2k
agreement ~70%, Gate A 78.50% PASS, and GSM8K free-generation accuracy ~2-4%.

### 5. Consequences

* **Add a correctness term to the gate.** `commit_rate` should require a CORRECT answer on a scorable
  subset, or be reported alongside `acc|closed`. As it stands the gate is gameable, and c=1.40 games it.
* **DRY and c=1.20 remain adopted** — 0 bpw, and they genuinely fix looping/compression/commit
  behaviour. They just do not, and cannot, restore capability.
* **The real question is upstream**: a 4B at 1.78 bpw retains 70% teacher-forced agreement and ~7% of
  the teacher's multi-step arithmetic. That gap is where the remaining work is — not in the sampler.
* **Re-run the accuracy check at a larger MAXNEW** before quoting absolute numbers: at 2048 even the
  teacher truncates 69% of the time.

---

# ===== SESSION STATUS (2026-09-05 → 09-10): where the project actually stands =====

Two campaigns ran back to back. Read this before planning the next one.

## A. The 13-paper batch — 13/13 resolved, ZERO resolvable improvements

| outcome | methods |
|---|---|
| confirmed win (earlier batch) | OPSA — but see the retraction in §13am |
| already implemented here | AYOT (independently derived as `build_chat_calib`; calib is 50.0% `<think>`-bearing) |
| wash (inside noise) | ICBQ (+0.89 SD), CAT-Q (+0.77 SD, once its alpha is frozen) |
| real negative | NAP (-7.6 SD), QUASAR (-33 SD), SchurQuant (diverged) |
| rejected on analysis | SoftWater, ECASQ, FlashQuant, ExTernD, SQuaT (null by construction) |
| parked (need custom GEMV) | LCD, AWSRC |

**The transferable output is a predictive screen, not a method.** Five failures shared one cause:
*the paper assumes a richer parameterisation than ternary provides* — SchurQuant's continuous suffix,
NAP's unfolded affines, CAT-Q's stably-learnable scale, QUASAR's separate code/dequantizer, SQuaT's
feature lattice. **Four of five were diagnosable on paper or by unit test.** Ask that question first.

**Noise floor (§13ac):** skeleton A/B SD **0.309 pp**; a single run resolves ~0.93 pp and nothing
finer. This retracted ICBQ's apparent +0.44 pp win — a null repeat moved further.

## B. The Gate B campaign — real behavioural gains, then a hard stop

Adopted, both **0 bpw** and format-safe:
* **DRY sampler** — loop -0.4306 +/- 0.0434, comp -1.1443, 3/3 seeds. Closes 78%/81% of the loop and
  compression gaps to the FP teacher. Stock llama.cpp settings are optimal (§13ak falsified my own
  "gentler DRY" hypothesis).
* **`</think>`-row gain c=1.20** — commit +0.0972 +/- 0.0120, 3/3 seeds. On-grid, TQ2_0-exact.

**Then §13am stopped the line.** Scored GSM8K under the identical decoding config:

| | acc \| closed |
|---|---|
| FP teacher | **0.9333** |
| ternary student (c=1.30, 28/48 closed) | **0.0714** |

The ternary 4B retains ~70% teacher-forced agreement and **~7% of the teacher's multi-step
arithmetic**. That is a CAPABILITY loss no sampler can fix. And `commit_rate` never checked
correctness, so every commit gain in this program — OPSA's +0.083, DRY's, the row gain's +0.097 —
measured the rate of *confidently-wrong completions*. c=1.40 passes **every** Gate B bar at 2.2%
accuracy.

## C. What to do next, in order

1. **Fix the gate before optimising against it again.** `commit_rate` must require a CORRECT answer on
   a scorable subset, or always be reported beside `acc|closed`. Add `think_len` vs the teacher's as a
   guard (§13aj). Until then Gate B is gameable and has been gamed.
2. **Re-measure capability at a larger MAXNEW.** At 2048 even the teacher truncates 69% of the time,
   so the absolute accuracies above are floors, not capability figures.
3. **The remaining work is upstream, not in decoding.** The gap to close is 70% teacher-forced
   agreement vs ~7% free-generation arithmetic. E2E is still the largest lever ever measured here
   (skeleton 56.27% -> ~70.7%, §13af) — larger than every paper method in the batch combined.
4. Do NOT spend further GPU on sampler variants, ICBQ chunk sizes, or CAT-Q/QUASAR follow-ups.

## D. Method notes that cost real time this session

* **Local metrics have never once caught a failure here.** block-MSE endorsed a collapsed residual
  stream, improved while SchurOpt destroyed the model, and pointed the WRONG way for QUASAR. What
  caught every failure: the end-to-end referee, and cheap format invariants — sparsity histograms,
  activation-norm probes, gradient-magnitude comparisons.
* **Calibrate a detector on the positive class first.** PLAER read 0.120 (path-finding) from a
  detector that fired on only 30% of SUCCESSFUL rollouts; recalibrated it read 0.400 (mixed).
* **Absolute gate rates are ~9x noisier than paired deltas** (commit SD 0.103 vs 0.012). "Helps by X"
  survives on pairing; "passes the bar" needs seeds.
* **Harness:** `loop_gate` ternarises whatever it is given — `MODEL_KIND=fp` is REQUIRED for FP arms.
  Gate chained runs on the completion ARTIFACT, never `pgrep -f` (zombies match forever; cost 6.5 h).
  Never redirect a chained driver to `/dev/null` (a silent `set -u` abort cost 25 min).
