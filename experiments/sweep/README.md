# Experiment drivers

Every driver that produced a number in `logs/RESULTS_SUMMARY.md`. **These were untracked until
2026-09-14** — caught by the `output*/` rule in `.gitignore` while living in `output_sweep/` — which
meant no experiment in this project was reproducible by anyone, including us after a disk failure.
That was T0.1 of `paper/PLAN.md` and this directory is the fix.

## Layout

* **drivers live here** (`experiments/sweep/*.sh`), tracked
* **outputs still go to `output_sweep/`**, untracked, and the scripts write there unchanged

Do not move outputs into a tracked directory. Model checkpoints and rollout dumps are large and
regenerable; the drivers are small and are the thing that cannot be reconstructed.

## Running

Scripts `cd` to the repository root themselves, so they run from anywhere:

    ./experiments/sweep/rebase.sh 0 base130 tern output_sweep/opsa/modified_model 1.30

Common environment: `SEED`, `N_SCORE`, `TEACHER_THINK_LEN`, `DENSE_INFER`, `HEAD_MODE`, `EMBED_MODE`,
`CUDA_VISIBLE_DEVICES`. `tools/capture_env.sh` records the ones that matter.

## Before any campaign whose numbers will be published

    ./tools/capture_env.sh          # -> paper/env/env-YYYY-MM-DD.md

Cite that filename in the results entry. A results log without a matching environment snapshot is a
claim, not a record. The script refuses to pretend: it flags a dirty working tree, because a snapshot
taken against uncommitted code does not describe what ran.

## Current drivers of record

| script | what it produces |
|---|---|
| `rebase.sh` | Gate B re-baseline arms under the fixed gate (§13as) |
| `traces.sh`, `arm.sh` | GSM8K trace capture, one arm per GPU |
| `exacc.sh`, `xa_one.sh` | ExAccErr / divergence-profile analysis (§13an) |
| `head_ga.sh`, `head_gsm.sh` | head/embed swap arms, Gate A and GSM8K (§13ao) |
| `densecheck2.sh`, `dc_dense_only.sh` | `DENSE_INFER` token-identity validation (§13aq-ii) |
| `lg_regress.sh` | loop_gate refactor regression (§13aq-i) |
| `noisefloor.sh` | skeleton A/B noise floor (§13ac) |
| `mathcorrect.sh`, `plaer.sh` | GSM8K correctness; pre-loop answer extraction |
| `icbq_ab.sh`, `nap_ab.sh`, `catq_ab.sh`, `quasar_ab.sh`, `schur_ab.sh`, `feat_ab.sh` | the 13-paper batch A/Bs |
| `dry_ab.sh`, `drytune.sh`, `thinkcal_dry.sh`, `tc_confirm.sh` | DRY and `</think>`-row calibration (DRY is curbed — §13ap-ii) |

Older drivers (`sweep.sh`, `g4.sh`, `place_sweep.sh`, `numa_*.sh`, `spike*.sh`, `stride*.sh`,
`vaudit.sh`, …) are kept because they produced recorded results. Nothing here is deleted for being
superseded: an unreproducible past result is the problem this directory exists to solve.
