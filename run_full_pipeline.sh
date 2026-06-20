#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Full ternary pipeline for Qwen3.6-27B (hybrid Gated-DeltaNet/attention), text-only TQ2_0.
#
#   rotation(≈1.0) → Block-AP → E2E-QP (+feature distillation) → [TQ2_0 export]
#   then GATE on the metrics that actually predict DEPLOYED quality:
#     • out-of-domain perplexity (WikiText-2)        — in-domain ppl is best-case/optimistic
#     • long free-generation degeneration            — the failure ppl is structurally blind to
#
# Progress baked in (see memory/diagnostics): calibration is now a DIVERSE 17-source mix (general
# web + QA + math + code + reasoning), not in-domain-only — the in-domain-only calib gave 2.24×
# OOD ppl. Feature distillation is ON (FEAT_WEIGHT>0) to fight free-gen degeneration. QAT is REMOVED
# (it regressed ppl: assignments were fine but group-wise scale retraining was overdosed). config
# trains at BLOCK_SIZE=256 so the TQ2_0 export is lossless (no g256 collapse).
#
# Specify ONLY the model + data locations (everything else defaults sensibly):
#   ORIG_MODEL=<hf snapshot dir>  CTM_DATA=<local data_cache dir>  bash run_full_pipeline.sh
# Staging:  CALIB_TOKENS=500000 WORK=output_diverse_0p5M   (then 4000000 / 16000000, new WORK each)
# Export:   RUN_EXPORT=1 .   Background; logs to run_full_pipeline.log.  Stages skip if already done.
# ──────────────────────────────────────────────────────────────────────────────
(
set -e
export PYTHONUNBUFFERED=1

# ─────────── the only two things you must specify (env-overridable) ───────────
ORIG_MODEL=${ORIG_MODEL:-/home/kyler/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9}
CTM_DATA=${CTM_DATA:-$HOME/Documents/CTM-Transformer/data_cache}   # local CTM parquet folders (12 'ctm' sources)

# ─────────── host-safety RAM cap (cgroup): kill a runaway STEP before it freezes the box ──────────
# The cap MUST leave real headroom under physical RAM. If it doesn't, GLOBAL memory pressure freezes
# the host before the cgroup OOM-killer fires — the 2026-06-18 crash: cap=52G on a 60G box led to a
# "Under memory pressure" hard freeze with NO clean cgroup kill. So default the cap to physical−18G
# (≈42G on this 60G box), and add MemoryHigh to throttle/reclaim BEFORE the hard MemoryMax kill.
TOTAL_GB=$(free -g | awk '/^Mem:/{print $2}')
DEF_MAX=$(( TOTAL_GB>=48 ? TOTAL_GB-18 : (TOTAL_GB>=24 ? TOTAL_GB-10 : TOTAL_GB*2/3) ))
MEM_MAX=${MEM_MAX:-${DEF_MAX}G}
MM_NUM=${MEM_MAX%G}; MEM_HIGH=${MEM_HIGH:-$(( MM_NUM>8 ? MM_NUM-4 : MM_NUM ))G}   # soft reclaim before hard kill
if systemd-run --user --scope --quiet -p MemoryMax=64M -p MemorySwapMax=0 /bin/true 2>/dev/null; then
  MEMGUARD="systemd-run --user --scope --quiet -p MemoryHigh=$MEM_HIGH -p MemoryMax=$MEM_MAX -p MemorySwapMax=0"
  echo "[memguard] per-step RAM: high=$MEM_HIGH max=$MEM_MAX swap=off (physical ${TOTAL_GB}G) — a CPU leak kills the step, not the host"
elif [ "${FORCE_NO_MEMGUARD:-0}" = "1" ]; then
  MEMGUARD=""; echo "[memguard] DISABLED (FORCE_NO_MEMGUARD=1) — host is NOT protected from a runaway-RAM hard freeze"
else
  echo "[memguard] FATAL: systemd --user scope unavailable — refusing to run UNGUARDED (a CPU memory leak"
  echo "           could hard-freeze the host). Fix the user scope, or re-run with FORCE_NO_MEMGUARD=1 to override."
  exit 1
fi
PY="$MEMGUARD ./.venv/bin/python"
# Bound GPU allocator growth/fragmentation; compute runs on cuda:0 (NOT the display GPU), so a CUDA
# OOM throws and kills the step — it does not freeze the host (the 13:05 crash was RAM, not GPU).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ─────────── knobs ───────────
NGPU=${NGPU:-2}                       # GPUs for data-parallel E2E-QP
CALIB_TOKENS=${CALIB_TOKENS:-500000}  # diverse-calib size. STAGING: 0.5M test → 4M → 16M
SEQ=1024
E2E_STEPS=${E2E_STEPS:-2000}
FEAT_WEIGHT=${FEAT_WEIGHT:-0.5}       # hidden-state feature distillation (generation-fidelity fix). >0 ⇒ teacher caches hidden states.
WORK=${WORK:-output_diverse}          # per-run output tree (use a DISTINCT one per staging size)
EVAL_SAMPLES=96; GEN_PROMPTS=8; GEN_LEN=48   # in-domain teacher-forced sanity eval (stage tracking)

# Phase 7 (deploy/export)
RUN_EXPORT=${RUN_EXPORT:-0}
LLAMA_QUANTIZE=${LLAMA_QUANTIZE:-/home/kyler/llama.cpp/build/bin/llama-quantize}
EMBED_TYPE=q4_K; OUTPUT_TYPE=q6_K     # embed→4bit / lm_head→6bit (≈ −3.3 GB, biggest non-body size win)

# ─────────── derived paths ───────────
ROT=./output/modified_model                  # rotation (data-independent; shared/reused across runs)
CALIB=$WORK/calibration_data.json            # block_ap reads <WORK>/calibration_data.json by convention
CALIB_EVAL=$WORK/calib_eval.json
TEACHER=$WORK/teacher_topk.pt
RECOVERED=$WORK/modified_model               # Block-AP output
E2EQP=$WORK/e2eqp/modified_model             # FINAL deployable model
SHARED_REF=$WORK/_eval_ref.pt

evalstep () {  # $1 label  $2 model dir  $3 optional flag (--plain-hf) — in-domain teacher-forced sanity
  echo "=== Eval [$1]: $2 vs FP teacher ==="
  $PY eval_ternary.py --fp-path "$ORIG_MODEL" --ternary-path "$2" --orig-config-path "$ORIG_MODEL" \
      --calib "$CALIB_EVAL" --seq "$SEQ" --max-samples "$EVAL_SAMPLES" \
      --gen-prompts "$GEN_PROMPTS" --gen-len "$GEN_LEN" \
      --ref-file "$SHARED_REF" --reuse-ref $3 || echo "(eval [$1] failed; continuing)"
}

# ─────────── preflight ───────────
echo "########## ternary pipeline (diverse-calib + feat-distill) — $(date) ##########"
echo "  ORIG_MODEL=$ORIG_MODEL"; echo "  CTM_DATA=$CTM_DATA"; echo "  CALIB_TOKENS=$CALIB_TOKENS  WORK=$WORK  FEAT_WEIGHT=$FEAT_WEIGHT"
[ -x ./.venv/bin/python ] || { echo "FATAL: ./.venv/bin/python not found (run from repo root)"; exit 1; }
[ -d "$ORIG_MODEL" ] || { echo "FATAL: model snapshot not found: $ORIG_MODEL"; exit 1; }
[ -d "$CTM_DATA" ] || { echo "FATAL: CTM data dir not found: $CTM_DATA"; exit 1; }
mkdir -p output "$WORK"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# ─────────── Phase 1: rotation (QuaRot, generated FROM $ORIG_MODEL each cold start) ───────────
# Cold-start contract: only $ORIG_MODEL and $CTM_DATA are assumed to pre-exist; everything else
# (rotation, calib, recovery, distillation) is generated here. The skip is a resume convenience only.
if [ ! -d "$ROT" ]; then
  echo "=== Phase 1: QuaRot rotation-only from $ORIG_MODEL (FP16 rotated model) ==="
  $PY convert.py --model-path "$ORIG_MODEL" --output-dir ./output --rotation-only --skip-gguf
else
  echo "=== [skip] Phase 1: $ROT already exists (resume) ==="
fi

# ─────────── Phase 2: DIVERSE calibration (17-source mix) ───────────
if [ ! -f "$CALIB" ]; then
  echo "=== Phase 2: diverse calibration ($CALIB_TOKENS tokens: web/QA/math/code/reasoning) ==="
  $PY build_diverse_calib.py --orig-model "$ORIG_MODEL" --ctm-data "$CTM_DATA" \
      --tokens "$CALIB_TOKENS" --out "$CALIB" --eval-out "$CALIB_EVAL"
else
  echo "=== [skip] Phase 2: $CALIB already exists ==="
fi
NSAMP=$(./.venv/bin/python -c "import json;print(len(json.load(open('$CALIB'))))")
echo "  calib sequences: $NSAMP"

evalstep "rotation (sanity ~1.0)" "$ROT" "--plain-hf"   # first eval → computes the shared FP reference

# ─────────── Phase 3: Block-AP per-linear recovery (GPTQ error feedback; optional QEP) ───────────
# QEP=1 adds inter-layer (cross-depth) error compensation: a 2nd clean-FP activation stream lets each
# weight pre-correct for accumulated upstream quantization error before the GPTQ fit (QEP_ALPHA ridge).
QEP_FLAG=""; [ "${QEP:-0}" = "1" ] && QEP_FLAG="--qep --qep-alpha ${QEP_ALPHA:-0.5}"
SSR_FLAG=""; [ "${SSR:-0}" = "1" ] && SSR_FLAG="--ssr"
CDQ_FLAG=""; [ "${CDQ:-0}" = "1" ] && CDQ_FLAG="--cdquant --cd-sweeps ${CD_SWEEPS:-4}"
if [ ! -f "$RECOVERED/model.safetensors.index.json" ]; then
  echo "=== Phase 3: Block-AP per-linear recovery (GPTQ${QEP_FLAG:+ + QEP α=${QEP_ALPHA:-0.5}}${SSR_FLAG:+ + SSR}${CDQ_FLAG:+ + CDQuant}) ==="
  $PY block_ap_recovery.py --model-path "$ROT" --orig-config-path "$ORIG_MODEL" \
      --output-dir "$WORK" --samples "$NSAMP" --iters 200 --lr 1e-3 $QEP_FLAG $SSR_FLAG $CDQ_FLAG
else
  echo "=== [skip] Phase 3: $RECOVERED already exists ==="
fi
evalstep "block-ap" "$RECOVERED"

# ─────────── Phase 4: FP teacher cache (+hidden states when FEAT_WEIGHT>0) ───────────
CACHE_HIDDEN_FLAG=""; [ "$FEAT_WEIGHT" != "0" ] && CACHE_HIDDEN_FLAG="--cache-hidden"
if [ ! -f "$TEACHER" ]; then
  echo "=== Phase 4: FP teacher top-k logits${CACHE_HIDDEN_FLAG:+ + hidden states} ==="
  $PY e2e_qp_distill.py --precompute-teacher --teacher-path "$ORIG_MODEL" \
      --calib "$CALIB" --teacher-cache "$TEACHER" --seq "$SEQ" --topk 64 \
      --gpu-mem 20GiB --cpu-mem 120GiB --max-samples "$NSAMP" $CACHE_HIDDEN_FLAG
else
  echo "=== [skip] Phase 4: $TEACHER already exists ==="
fi

# ─────────── Phase 5: E2E-QP scale distillation + FEATURE distillation ───────────
if [ ! -f "$WORK/e2eqp/.done" ]; then
  echo "=== Phase 5: E2E-QP scale distillation (+feature distillation, feat=$FEAT_WEIGHT, DDP x$NGPU) ==="
  mkdir -p "$WORK/e2eqp"
  $PY -m torch.distributed.run --nproc_per_node="$NGPU" --standalone \
      e2e_qp_distill.py --train --student-path "$RECOVERED" --orig-config-path "$ORIG_MODEL" \
      --calib "$CALIB" --teacher-cache "$TEACHER" --out "$E2EQP" \
      --seq "$SEQ" --steps "$E2E_STEPS" --lr 2e-5 --max-samples "$NSAMP" \
      --loss-fn cakld --feat-weight "$FEAT_WEIGHT"
  touch "$WORK/e2eqp/.done"
else
  echo "=== [skip] Phase 5: $WORK/e2eqp/.done present ==="
fi
evalstep "e2e-qp (+feat-distill)" "$E2EQP"

FINAL="$E2EQP"

# ─────────── Phase 6: deployment-realistic GATES (the metrics that matter) ───────────
echo "=== Gate A: out-of-domain perplexity (WikiText-2; in-domain baseline was 1.2302×, OOD 2.24×) ==="
$MEMGUARD env ORIG="$ORIG_MODEL" FP_DIR="$ROT" E2E_MODEL="$E2EQP" \
    ./.venv/bin/python ood_ppl.py || echo "(OOD gate failed; continuing)"
echo "=== Gate B: long free-generation degeneration (greedy; baseline ternary rep4 → 89% vs teacher 39%) ==="
$MEMGUARD env ORIG="$ORIG_MODEL" FP_DIR="$ROT" E2E_MODEL="$E2EQP" \
    EVAL_DATA="$CALIB_EVAL" NP="${GATE_NP:-8}" GEN="${GATE_GEN:-480}" \
    ./.venv/bin/python long_gen_eval.py || echo "(degeneration gate failed; continuing)"

# ─────────── Phase 7: deploy to TQ2_0 GGUF (text-only) ───────────
if [ "$RUN_EXPORT" = "1" ]; then
  echo "=== Phase 7: GGUF export (text-only) + TQ2_0 with compressed embed/head ==="
  F16=$WORK/model-f16.gguf; OUTG=$WORK/model-TQ2_0.gguf
  $PY convert_hf_to_gguf_patched.py "$FINAL" --outfile "$F16" --outtype f16 \
      || { echo "FATAL: convert failed"; exit 1; }
  [ -x "$LLAMA_QUANTIZE" ] || { echo "FATAL: llama-quantize not executable: $LLAMA_QUANTIZE"; exit 1; }
  "$LLAMA_QUANTIZE" --token-embedding-type "$EMBED_TYPE" --output-tensor-type "$OUTPUT_TYPE" \
      "$F16" "$OUTG" TQ2_0 || { echo "FATAL: llama-quantize failed"; exit 1; }
  echo "=== exported -> $OUTG ==="; ls -la "$F16" "$OUTG" 2>/dev/null
else
  echo "=== [skip] Phase 7 export (set RUN_EXPORT=1 to build the TQ2_0 GGUF) ==="
fi

echo "########## ALL DONE — final ternary model: $FINAL ##########"
echo "Compare the Gate A (OOD ppl) + Gate B (degeneration) blocks above to the in-domain baseline"
echo "(1.2302× in-domain ppl / 2.24× OOD ppl / 79% free-gen degeneration) to judge the diverse+feat run."
) > run_full_pipeline.log 2>&1 &

echo "Pipeline started in the background (PID $!)."
echo "Watch it:   tail -f run_full_pipeline.log"
