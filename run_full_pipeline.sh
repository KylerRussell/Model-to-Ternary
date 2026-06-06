#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Full ternary pipeline for Qwen3.6-27B, with a fidelity eval after EVERY stage so
# you can watch the ppl ratio improve step by step:
#   rotation(≈1.0) → Block-AP → E2E-QP → QAT, each vs the FP teacher.
# All per-stage evals share ONE FP-teacher pass (computed by the first eval, reused
# by the rest via --reuse-ref), so the extra evals are cheap.
#
# Runs in the background, logs to run_full_pipeline.log. Stages skip if already done
# (training stages use a .done marker). To force a re-run, delete the output dir
# (and output_*/.done). If you change the EVAL data, delete $SHARED_REF too.
#
# Run:    bash run_full_pipeline.sh        Watch:  tail -f run_full_pipeline.log
# ──────────────────────────────────────────────────────────────────────────────
(
set -e
export PYTHONUNBUFFERED=1

# ───────────────────────────── config ─────────────────────────────
PY=./.venv/bin/python
ORIG=/home/kyler/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9
NEMOTRON=~/Documents/CTM-Transformer/data_cache
NGPU=2                  # GPUs for data-parallel E2E-QP (set 1 to disable DDP)

# knobs (tune freely)
CALIB_SAMPLES=128       # calibration sequences. RAISE for more data (e.g. 256/512) — but then
                        # delete output_recovery/teacher_topk.pt + output_recovery/modified_model
                        # so Phase 3/4 regenerate on the larger set. QAT benefits most from this.
EVAL_SAMPLES=96         # held-out sequences for fidelity eval
SEQ=1024                # sequence length for calib / teacher / E2E-QP / eval
QAT_SEQ=512             # shorter seq for QAT (per-block latents are memory-heavy)
QAT_GROUP=2             # adjacent blocks trained jointly
QAT_PASSES=3            # full sweeps over the model
QAT_STEPS=120           # steps per group per pass (>= CALIB_SAMPLES so each group sees ALL data)
E2E_STEPS=500           # E2E-QP scale-distillation steps

# derived paths
CALIB=./output_recovery/calibration_data.json
EVAL=./output_recovery/eval_data.json
TEACHER=./output_recovery/teacher_topk.pt
SHARED_REF=./output_recovery/_eval_ref.pt     # FP-teacher reference, shared across all evals
ROT=./output/modified_model
RECOVERED=./output_recovery/modified_model
E2EQP=./output_e2eqp/modified_model
QAT=./output_qat/modified_model

# eval helper: $1 label, $2 model dir, $3 optional extra flag (e.g. --plain-hf)
evalstep () {
  echo "=== Eval [$1]: $2 vs FP teacher ==="
  $PY eval_ternary.py --fp-path "$ORIG" --ternary-path "$2" --orig-config-path "$ORIG" \
      --calib "$EVAL" --seq "$SEQ" --max-samples "$EVAL_SAMPLES" \
      --ref-file "$SHARED_REF" --reuse-ref $3 || echo "(eval [$1] failed; continuing)"
}

# ──────────────────────── preflight ────────────────────────
echo "########## ternary QAT pipeline — $(date) ##########"
[ -x "$PY" ] || { echo "FATAL: $PY not found (run from repo root)"; exit 1; }
[ -d "$ORIG" ] || { echo "FATAL: base model snapshot not found: $ORIG"; exit 1; }
[ -d "$NEMOTRON" ] || { echo "FATAL: Nemotron data dir not found: $NEMOTRON"; exit 1; }
mkdir -p output output_recovery output_e2eqp output_qat
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

# ──────────────────── Phase 1: rotation-only ────────────────────
if [ ! -d "$ROT" ]; then
  echo "=== Phase 1: QuaRot rotation-only (FP16 rotated model) ==="
  $PY convert.py --download --output-dir ./output --rotation-only --skip-gguf
else
  echo "=== [skip] Phase 1: $ROT already exists ==="
fi

# ──────────────────── Phase 2: calibration data ────────────────────
if [ ! -f "$CALIB" ]; then
  echo "=== Phase 2: calibration data (train, seed 42) ==="
  $PY calibration.py --local-dir "$NEMOTRON" --samples "$CALIB_SAMPLES" \
      --seq-length "$SEQ" --output "$CALIB"
else
  echo "=== [skip] Phase 2: $CALIB already exists ==="
fi
if [ ! -f "$EVAL" ]; then
  echo "=== Phase 2b: held-out eval data (seed 123) ==="
  $PY calibration.py --local-dir "$NEMOTRON" --samples "$EVAL_SAMPLES" \
      --seq-length "$SEQ" --seed 123 --output "$EVAL"
else
  echo "=== [skip] Phase 2b: $EVAL already exists ==="
fi

# eval the rotation-only model (should be ≈1.00 — confirms rotation is lossless).
# This is the first eval, so it computes the shared FP reference the others reuse.
# (Best-effort: --plain-hf loads it as a full FP model; if the dir is multimodal it
#  may not load, but the FP reference still gets saved for the later evals.)
evalstep "rotation (sanity ~1.0)" "$ROT" "--plain-hf"

# ──────────────────── Phase 3: Block-AP recovery ────────────────────
if [ ! -f "$RECOVERED/model.safetensors.index.json" ]; then
  echo "=== Phase 3: Block-AP per-linear recovery ==="
  $PY block_ap_recovery.py --model-path "$ROT" --orig-config-path "$ORIG" \
      --output-dir ./output_recovery --samples "$CALIB_SAMPLES" --iters 200 --lr 1e-3
else
  echo "=== [skip] Phase 3: $RECOVERED already exists ==="
fi
evalstep "block-ap" "$RECOVERED"

# ──────────────────── Phase 4: FP teacher logit cache ────────────────────
if [ ! -f "$TEACHER" ]; then
  echo "=== Phase 4: precompute FP teacher top-k logits ==="
  $PY e2e_qp_distill.py --precompute-teacher --teacher-path "$ORIG" \
      --calib "$CALIB" --teacher-cache "$TEACHER" --seq "$SEQ" --topk 64 \
      --gpu-mem 20GiB --cpu-mem 120GiB --max-samples "$CALIB_SAMPLES"
else
  echo "=== [skip] Phase 4: $TEACHER already exists ==="
fi

# ──────────────────── Phase 5: E2E-QP scale distillation ────────────────────
if [ ! -f output_e2eqp/.done ]; then
  echo "=== Phase 5: E2E-QP global scale distillation (data-parallel x$NGPU) ==="
  $PY -m torch.distributed.run --nproc_per_node="$NGPU" --standalone \
      e2e_qp_distill.py --train --student-path "$RECOVERED" --orig-config-path "$ORIG" \
      --calib "$CALIB" --teacher-cache "$TEACHER" --out "$E2EQP" \
      --seq "$SEQ" --steps "$E2E_STEPS" --lr 2e-5 --max-samples "$CALIB_SAMPLES"
  touch output_e2eqp/.done
else
  echo "=== [skip] Phase 5: output_e2eqp/.done present ==="
fi
evalstep "e2e-qp" "$E2EQP"

# ──────────────────── Phase 6: group-wise multi-pass QAT ────────────────────
# To resume an interrupted QAT at a later pass: add --start-pass N and remove output_qat/.done.
if [ ! -f output_qat/.done ]; then
  echo "=== Phase 6: group-wise multi-pass QAT (both GPUs) ==="
  $PY block_qat.py --student-path "$E2EQP" --orig-config-path "$ORIG" \
      --calib "$CALIB" --teacher-cache "$TEACHER" --out "$QAT" \
      --seq "$QAT_SEQ" --group-size "$QAT_GROUP" --passes "$QAT_PASSES" \
      --steps-per-block "$QAT_STEPS" --adam8bit
  touch output_qat/.done
else
  echo "=== [skip] Phase 6: output_qat/.done present ==="
fi
evalstep "qat (final)" "$QAT"

# ──────────────────── Optional Phase 7: deploy to TQ2_0 ────────────────────
# echo "=== Phase 7: GGUF + TQ2_0 ==="
# $PY convert_hf_to_gguf_patched.py "$QAT" \
#     --outfile output/Qwen3.6-27B-ternary-final-f16.gguf --outtype f16
# /path/to/your-fork/llama-quantize \
#     output/Qwen3.6-27B-ternary-final-f16.gguf \
#     output/Qwen3.6-27B-ternary-final-TQ2_0.gguf TQ2_0

echo "########## ALL DONE — final ternary model: $QAT ##########"
echo "Scan the Eval blocks above: rotation ≈1.0 → block-ap → e2e-qp → qat ppl ratios."
) > run_full_pipeline.log 2>&1 &

echo "Pipeline started in the background (PID $!)."
echo "Watch it:   tail -f run_full_pipeline.log"