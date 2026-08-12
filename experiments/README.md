# experiments/

One-off scripts from the 4B campaigns. **Historical record — not maintained.** They still contain
absolute `/home/kyler/...` paths and reference `output_4bpipe/<dir>` model directories that were deleted
after their results were condensed into `logs/RESULTS_SUMMARY.md` (§7–§11).

Kept because the *settings* are the evidence: each script's header documents why its knobs were chosen and
what the run was testing. If you want to re-run one, point `ORIG`/`ORIG_MODEL` at `$MODEL_4B` (see
`../env.sh`) and expect to regenerate its inputs.

The live entry points are in the repo root:

| script | purpose |
|---|---|
| `run_full_pipeline.sh` | the whole conversion: rotation → calib → block-AP → **assignments (Phase 4.5)** → E2E → gates |
| `run_eval2k.sh` | score any model on the frozen 1946-seq eval2k referee |
| `export_4b_gguf.sh` | GGUF export for the llama.cpp eval harness |
| `env.sh` | machine-specific paths (override with `env.local.sh`) |
