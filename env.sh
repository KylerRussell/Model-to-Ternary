#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# MACHINE-SPECIFIC PATHS. Source this from any run_*.sh; every value is overridable
# from the environment, so a new host only needs the ones that actually differ.
#
#   source ./env.sh
#
# Copy to env.local.sh to override without touching the tracked file (env.local.sh
# is gitignored and sourced automatically at the bottom of this script).
# ──────────────────────────────────────────────────────────────────────────────

# Repo root, derived from THIS file's location — never hardcode it.
REPO_ROOT=${REPO_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
export REPO_ROOT

# ─────────── models ───────────
# 4B testbed (untied embeddings — see src/untie_embeddings.py). Fast screen for the 27B.
MODEL_4B=${MODEL_4B:-$REPO_ROOT/output_4b/untied_4b}
# 27B target. Default resolves the newest snapshot in the HF cache rather than pinning a hash,
# so a fresh `huggingface-cli download` on a new box just works.
HF_CACHE=${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}
if [ -z "${MODEL_27B:-}" ]; then
  MODEL_27B=$(ls -d "$HF_CACHE"/models--Qwen--Qwen3.6-27B/snapshots/*/ 2>/dev/null | head -1)
  MODEL_27B=${MODEL_27B%/}
fi
export MODEL_4B MODEL_27B HF_CACHE

# ─────────── data ───────────
# Calibration parquet folders used by the diverse-calibration builder (Phase 2b). Now IN-REPO:
# populate with `python tools/fetch_data.py` (17 Nemotron sources, one part file each). The old
# out-of-repo default ($HOME/Documents/CTM-Transformer/data_cache) still works if you point at it.
CTM_DATA=${CTM_DATA:-$REPO_ROOT/data}
# Frozen 1946-seq eval2k referee. THE cross-run comparable metric — keep it identical across hosts.
EVAL2K_JSON=${EVAL2K_JSON:-$REPO_ROOT/output_4b/eval2k.json}
export CTM_DATA EVAL2K_JSON

# ─────────── llama.cpp (GGUF export + the eval harness) ───────────
LLAMA_DIR=${LLAMA_DIR:-$HOME/llama.cpp}
LLAMA_QUANTIZE=${LLAMA_QUANTIZE:-$LLAMA_DIR/build/bin/llama-quantize}
LLAMA_SERVER=${LLAMA_SERVER:-$LLAMA_DIR/build/bin/llama-server}
export LLAMA_DIR LLAMA_QUANTIZE LLAMA_SERVER

# ─────────── python ───────────
VENV_PY=${VENV_PY:-$REPO_ROOT/.venv/bin/python}
export VENV_PY

# ─────────── host capability (drives the memory-bound knobs) ───────────
# The assignment stage costs ~18.9 bytes/latent + ~6.9 GB base (measured; see RESULTS_SUMMARY §9e).
# On a 60 GB box that caps `down` at ~21 of 64 layers for the 27B; a large-RAM host can hold far more,
# which is the whole point of moving. Scripts read HOST_RAM_GB to size layer groups.
HOST_RAM_GB=${HOST_RAM_GB:-$(free -g | awk '/^Mem:/{print $2}')}
export HOST_RAM_GB

[ -f "$REPO_ROOT/env.local.sh" ] && source "$REPO_ROOT/env.local.sh"

if [ "${ENV_VERBOSE:-0}" = "1" ]; then
  echo "[env] REPO_ROOT=$REPO_ROOT"
  echo "[env] MODEL_4B=$MODEL_4B  $([ -d "$MODEL_4B" ] && echo OK || echo MISSING)"
  echo "[env] MODEL_27B=${MODEL_27B:-<unset>}  $([ -n "${MODEL_27B:-}" ] && [ -d "$MODEL_27B" ] && echo OK || echo MISSING)"
  echo "[env] CTM_DATA=$CTM_DATA  $([ -d "$CTM_DATA" ] && echo OK || echo MISSING)"
  echo "[env] EVAL2K_JSON=$EVAL2K_JSON  $([ -f "$EVAL2K_JSON" ] && echo OK || echo MISSING)"
  echo "[env] LLAMA_QUANTIZE=$LLAMA_QUANTIZE  $([ -x "$LLAMA_QUANTIZE" ] && echo OK || echo MISSING)"
  echo "[env] HOST_RAM_GB=$HOST_RAM_GB"
fi
