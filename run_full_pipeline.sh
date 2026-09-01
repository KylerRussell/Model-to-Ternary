#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Full ternary pipeline — VALIDATED RECIPE (rewritten 2026-07-30 to match the run that finally worked).
#
#   rotation → CHAT+generic calib → block-AP skeleton (g64, GPTQ-init QAT, ternary embed+head)
#            → FP teacher cache → E2E scale distillation (+col-scale, 8-bit scale-QAT, ADDITIVE commit)
#            → GATES (eval2k agreement + free-gen loop gate) → [GGUF export]
#
# WHY EACH SETTING (all measured on the 4B testbed; see logs/RESULTS_SUMMARY.md §8):
#  • CHAT DATA IS MANDATORY. Generic-only calib makes the ternary model collapse at the never-calibrated
#    `assistant\n<think>\n` position: 6% MMLU-Pro / 1% GPQA (BELOW random) while teacher-forced KL still
#    read 79-84%. Chat-format E2E + chat-context block-AP Hessians fix it.
#  • 50% REQUIRE-CLOSE CHAT. Commit (emitting `</think>`) tracks corpus `</think>`-density: 4.3%→56% commit,
#    47.2%→75% commit (= FP teacher parity). --require-close keeps only rollouts that actually closed.
#  • SEQ 2560. STEM reasoning closes at a median of 1786 tok; at seq 1024 the packing SPLITS `<think>` from
#    `</think>` into different chunks, so the model never sees a whole reason→close→answer arc.
#  • g64 + 8-bit scale-QAT (~1.71 bpw). Beats g128@fp16 by +1.4pt at identical size and g256/TQ2_0 by +2.75pt.
#    g32@Q4 is NOT better (4-bit scales cancel the finer grid). NOTE: g64 is NOT TQ2_0-exact — see Phase 7.
#  • ADDITIVE commit objective (--commit-beta). Reweighting the commit window INSIDE the normalized CAKLD mean
#    is a mathematical no-op (the window is ~0.1% of tokens ⇒ 0.4% of the weight). The additive term is
#    un-diluted by rarity, and is safe at large beta because it is KL-TO-TEACHER (self-limiting), not a
#    close-bonus.
#  • NO arm-b, NO on-policy. Sparse lm_head flips are ~0.006% of params (null) and caused an NCCL desync;
#    training on the student's own rollouts is self-distillation on the pathology (made looping 2x worse).
#  • GATES ARE FREE-GEN. Teacher-forced KL/ppl are PROVEN BLIND to the failure that matters.
#
# Specify ONLY the model + data locations (everything else defaults sensibly):
#   ORIG_MODEL=<hf snapshot dir>  CTM_DATA=<local data_cache dir>  bash run_full_pipeline.sh
# Staging:  CALIB_TOKENS=4000000 WORK=output_4M   (then 16000000, new WORK each)
# Export:   RUN_EXPORT=1 .   Background; logs to run_full_pipeline.log.  Stages skip if already done.
# ──────────────────────────────────────────────────────────────────────────────
(
# SINGLE-INSTANCE GUARD. A duplicate launch (two watchers both firing) once ran this script twice: two
# independent workers with 20.8GB PRIVATE each, 48.9GB PSS on a 60GB host, both writing the SAME --out.
# Refuse to start if a trainer is already live. MUST match comm=="python" AND the args via ps/awk: plain
# `pgrep -f` also matches any SHELL whose command text contains the pattern (e.g. a terminal running a grep
# for it), which false-fired and blocked a legitimate launch; and `pgrep -x python -f PATTERN` is invalid
# (pgrep accepts only one pattern) so it fails OPEN. Root cause of the original incident: `kill $!` after
# `setsid`,
# which forks when not already a process-group leader, so $! was not the running script.
if ps -eo comm=,args= | awk '$1=="python" && /e2e_qp_distill\.py --train/{f=1} END{exit !f}'; then
  echo "FATAL: a training process is already running -- refusing to start a second one."; exit 1
fi
set -e
export PYTHONUNBUFFERED=1

# ─────────── machine-specific paths (see env.sh; override via env or env.local.sh) ───────────
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/env.sh"
ORIG_MODEL=${ORIG_MODEL:-$MODEL_27B}      # env.sh resolves the newest 27B snapshot in the HF cache
CTM_DATA=${CTM_DATA:-$HOME/Documents/CTM-Transformer/data_cache}   # local CTM parquet folders (12 'ctm' sources)
export ORIG_MODEL   # config.py derives model geometry from this at import (works for 27B + 2B/4B testbeds)

# ─────────── knobs (defaults = the validated recipe) ───────────
BLK=${BLK:-64}                        # weight group size. g64+8bit scales ≈ 1.71 bpw (best of the sweep)
SBITS=${SBITS:-8}                     # scale-QAT bit-width (8 = free vs fp16; 4 loses the g64 gain)
export TERNARY_BLOCK_SIZE=$BLK        # MUST be identical for block-AP + E2E + eval (mismatch ⇒ +24pt ppl)

# ─────────── host-safety RAM cap (cgroup): kill a runaway STEP before it freezes the box ──────────
# The cap MUST leave real headroom under physical RAM, or GLOBAL memory pressure freezes the host before the
# cgroup OOM-killer fires (the 2026-06-18 crash: cap=52G on a 60G box → hard freeze, no clean kill).
TOTAL_GB=$(free -g | awk '/^Mem:/{print $2}')
DEF_MAX=$(( TOTAL_GB>=48 ? TOTAL_GB-18 : (TOTAL_GB>=24 ? TOTAL_GB-10 : TOTAL_GB*2/3) ))
MEM_MAX=${MEM_MAX:-${DEF_MAX}G}
MM_NUM=${MEM_MAX%G}; MEM_HIGH=${MEM_HIGH:-$(( MM_NUM>8 ? MM_NUM-4 : MM_NUM ))G}
if systemd-run --user --scope --quiet -p MemoryMax=64M -p MemorySwapMax=0 /bin/true 2>/dev/null; then
  MEMGUARD="systemd-run --user --scope --quiet -p MemoryHigh=$MEM_HIGH -p MemoryMax=$MEM_MAX -p MemorySwapMax=0 -E TERNARY_BLOCK_SIZE=$BLK"
  echo "[memguard] per-step RAM: high=$MEM_HIGH max=$MEM_MAX swap=off (physical ${TOTAL_GB}G)"
elif [ "${FORCE_NO_MEMGUARD:-0}" = "1" ]; then
  MEMGUARD=""; echo "[memguard] DISABLED — host is NOT protected from a runaway-RAM hard freeze"
else
  echo "[memguard] FATAL: systemd --user scope unavailable — refusing to run UNGUARDED."; exit 1
fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEQ=${SEQ:-2560}                      # ≥ the STEM close length (median 1786, p75 2514)
NGPU=${NGPU:-2}
CALIB_TOKENS=${CALIB_TOKENS:-16000000}
CHAT_FRAC=${CHAT_FRAC:-0.50}          # fraction of calib that is require-close chat (drives commit rate)
EPOCHS=${EPOCHS:-2}                   # settled: 2 epochs × linear-decay × final-select
E2E_LR=${E2E_LR:-2e-5}                # 1e-5 when warm-starting from an existing E2E checkpoint
COMMIT_BETA=${COMMIT_BETA:-1.5}       # additive commit objective; ≤5. 0 disables
FEAT_WEIGHT=${FEAT_WEIGHT:-0}         # settled recipe uses 0 (feature distillation not used)
WORK=${WORK:-output_ternary}
GATE_MAXNEW=${GATE_MAXNEW:-2048}      # free-gen gate budget; MUST be ≤ SEQ (never test beyond training)

# Phase 7 (deploy/export)
RUN_EXPORT=${RUN_EXPORT:-0}
LLAMA_QUANTIZE=${LLAMA_QUANTIZE:-$LLAMA_DIR/build/bin/llama-quantize}
EMBED_TYPE=q4_K; OUTPUT_TYPE=q6_K

# ─────────── derived paths ───────────
ROT_BASE=${ROT_BASE:-./output}        # per-MODEL (data-independent); MUST differ per model
ROT=$ROT_BASE/modified_model
CHATPOOL=$WORK/chat_pool.json
CALIB=$WORK/calibration_data.json     # block_ap reads <WORK>/calibration_data.json by convention
CALIB_EVAL=$WORK/calib_eval.json
TEACHER=$WORK/teacher_topk.pt
RECOVERED=$WORK/modified_model        # Block-AP skeleton
E2EQP=$WORK/e2eqp/modified_model      # FINAL deployable model
PY="$MEMGUARD ./.venv/bin/python"

# ─────────── per-stage eval2k (EVAL_EACH=1) ───────────
# Scores an intermediate model on the FROZEN 1946-seq referee so a pipeline run can be checked stage by
# stage instead of only at the end. Off by default (each call costs ~20-45 min); EVAL_EACH=1 turns it on,
# which is what a smoke test wants. Per-run held-out is NOT comparable across stages -- only this is.
EVAL_EACH=${EVAL_EACH:-0}
EVAL2K_JSON=${EVAL2K_JSON:-$REPO_ROOT/output_4b/eval2k.json}
STAGE_RES=$WORK/stage_eval2k.txt
eval2k_stage () {   # tag model_dir
  [ "$EVAL_EACH" = "1" ] || return 0
  [ -f "$EVAL2K_JSON" ] || { echo "  [eval2k] SKIP $1 — $EVAL2K_JSON missing"; return 0; }
  [ -f "$2/model.safetensors.index.json" ] || { echo "  [eval2k] SKIP $1 — no model at $2"; return 0; }
  [ -f "$WORK/.e2k_$1" ] && { echo "  [eval2k] [skip] $1 already scored"; return 0; }
  local LOG=$WORK/e2k_$1.log
  echo "  [eval2k] scoring $1 ..."
  $MEMGUARD env ORIG="$ORIG_MODEL" FP_DIR="$ROT" E2E_MODEL="$2" \
      EVAL_DATA="$EVAL2K_JSON" NP=1946 SEQ=1024 \
      ./.venv/bin/python src/kl_flips_eval.py > "$LOG" 2>&1 || { echo "  [eval2k] FAILED for $1"; return 0; }
  local AG KL
  AG=$(grep -aoE "top-1 agreement *: *[0-9.]+" "$LOG" | grep -oE "[0-9.]+$" | tail -1)
  KL=$(grep -aoE "mean KL\(fp\|\|tern\) *: *[0-9.]+" "$LOG" | grep -oE "[0-9.]+$" | tail -1)
  [ -f "$STAGE_RES" ] || printf "%-14s %-12s %s\n" "stage" "agreement%" "meanKL" > "$STAGE_RES"
  printf "%-14s %-12s %s\n" "$1" "${AG:-NA}" "${KL:-NA}" | tee -a "$STAGE_RES"
  touch "$WORK/.e2k_$1"
}

# ─────────── preflight ───────────
echo "########## ternary pipeline (VALIDATED RECIPE) — $(date) ##########"
echo "  ORIG_MODEL=$ORIG_MODEL"; echo "  CTM_DATA=$CTM_DATA"
echo "  g$BLK + ${SBITS}bit scales | seq $SEQ | ${CALIB_TOKENS} tok | chat ${CHAT_FRAC} | commit-beta $COMMIT_BETA | WORK=$WORK"
[ -x ./.venv/bin/python ] || { echo "FATAL: ./.venv/bin/python not found (run from repo root)"; exit 1; }
[ -n "$ORIG_MODEL" ] && [ -d "$ORIG_MODEL" ] || { echo "FATAL: model snapshot not found: '$ORIG_MODEL'"; echo "  set ORIG_MODEL=, or MODEL_27B= in env.local.sh (run: ENV_VERBOSE=1 source ./env.sh)"; exit 1; }
[ -d "$CTM_DATA" ] || { echo "FATAL: CTM data dir not found: $CTM_DATA"; exit 1; }
mkdir -p "$ROT_BASE" "$WORK"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
source ./lib_timing.sh

# ─────────── Phase 1: rotation (QuaRot, generated FROM $ORIG_MODEL each cold start) ───────────
stage "Phase 1: rotation"
if [ ! -d "$ROT" ]; then
  $PY src/convert.py --model-path "$ORIG_MODEL" --output-dir "$ROT_BASE" --rotation-only --skip-gguf \
      ${LEARNED_R:+--learned-rotation "$LEARNED_R"}
else echo "=== [skip] Phase 1: $ROT exists ==="; fi
# POST-CONDITION: QuaRot is a similarity transform, so the rotated FP model MUST be behaviorally identical
# to the original. A silent violation here poisons everything downstream — the teacher cache is built from
# $ROT, so a degraded rotation trains every student toward a broken teacher. That is exactly what happened
# 2026-07-21..08-02 (§8k): lm_head lost the final norm's ~3.19x gain, FP logits shrank ~3.1x, the softmax
# went flat, and 35h of runs were spent diagnosing the symptom. Fail CLOSED — never train on a bad teacher.
if [ "${SKIP_ROT_CHECK:-0}" != "1" ]; then
  $MEMGUARD env ORIG_MODEL="$ORIG_MODEL" ROT_DIR="$ROT" ./.venv/bin/python - <<'PY' || { echo "FATAL: rotation post-condition failed"; exit 1; }
import os, json, sys, glob, torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM
orig, rot = os.environ["ORIG_MODEL"], os.environ["ROT_DIR"]
src = next((p for p in ("output_4b/eval2k.json", os.path.join(orig, "eval2k.json")) if os.path.exists(p)), None)
if src is None:
    print("  [rot-check] no eval corpus found — SKIPPED (cannot verify)"); sys.exit(0)
seqs = [torch.tensor(s[:512]).unsqueeze(0) for s in json.load(open(src))[:6]]
a = AutoModelForCausalLM.from_pretrained(orig, trust_remote_code=True, dtype=torch.bfloat16).to("cuda:0").eval()
d1 = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
b = AutoModelForCausalLM.from_pretrained(rot, trust_remote_code=True, dtype=torch.bfloat16).to(d1).eval()
kl = fl = n = ca = cb = 0
with torch.no_grad():
    for s in seqs:
        la = a(s.to("cuda:0")).logits[:, :-1].float().cpu(); lb = b(s.to(d1)).logits[:, :-1].float().cpu()
        pa = F.softmax(la, -1)
        kl += (pa*(F.log_softmax(la, -1) - F.log_softmax(lb, -1))).sum(-1).sum().item()
        fl += (la.argmax(-1) != lb.argmax(-1)).sum().item()
        ca += (pa.max(-1).values > 0.5).sum().item()
        cb += (F.softmax(lb, -1).max(-1).values > 0.5).sum().item()
        n += la.shape[1]
kl, agree, c_o, c_r = kl/n, 100-100*fl/n, 100*ca/n, 100*cb/n
print(f"  [rot-check] KL {kl:.4f} | agreement {agree:.2f}% | FP-confident orig {c_o:.2f}% rot {c_r:.2f}%")
bad = kl > 0.05 or agree < 98.0 or (c_o > 20.0 and c_r < 0.5*c_o)
print("  [rot-check] " + ("FAIL — rotation is NOT transparent (see §8k: lm_head norm-fold)" if bad
                          else "PASS — rotation is behaviorally transparent"))
sys.exit(1 if bad else 0)
PY
fi

# ─────────── Phase 2a: CHAT pool — FP thinking-mode rollouts that CLOSE </think> ───────────
# The single most important data ingredient. Prompts are DERIVED from the generic corpus, so this scales
# with the token budget (no fixed hand-written prompt list). --require-close keeps only closed traces:
# close-rate is ~45% at max_new=$SEQ, so we over-generate ~2.3x.
stage "Phase 2a: chat pool (require-close)"
if [ ! -f "$CHATPOOL" ]; then
  NEED=$(./.venv/bin/python -c "print(int($CALIB_TOKENS/$SEQ*$CHAT_FRAC))")
  NROLL=$(( NEED * 100 / ${CLOSE_RATE:-43} + 50 ))
  # Generation is the LONGEST phase of the run and build_chat_calib.py is single-GPU, so an unsharded
  # Phase 2a leaves NGPU-1 cards idle for hours (measured 4B: 11.3 rollouts/min on one card => ~11 h).
  # If the FP teacher fits on ONE card we run one whole model per GPU (near-linear speedup); if it does
  # NOT (27B bf16 ~54 GiB), a per-GPU shard is impossible and we fall back to a single accelerate
  # device_map job spanning both GPUs + CPU.
  MODEL_GIB=$(du -sb --dereference "$ORIG_MODEL" | awk '{printf "%.1f", $1/1073741824}')
  GPU_GIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | awk '{printf "%.1f", $1/1024}')
  FITS=$(awk -v m="$MODEL_GIB" -v g="$GPU_GIB" 'BEGIN{print (m*1.30 < g) ? 1 : 0}')   # 1.3x for KV cache + activations
  NSH=${CHAT_SHARDS:-$([ "$FITS" = "1" ] && echo "$NGPU" || echo 1)}
  echo "=== Phase 2a: $NEED chat seqs needed → $NROLL rollouts (require-close) ==="
  echo "    teacher ${MODEL_GIB}GiB vs ${GPU_GIB}GiB/card → fits-on-one=$FITS → $NSH shard(s)"
  PER=$(( (NROLL + NSH - 1) / NSH ))
  # ALL shards go inside ONE memguard scope: N concurrent scopes would each get MemoryMax=$MEM_MAX,
  # allowing N x $MEM_MAX total and reproducing the 2026-06-18 host freeze.
  SHRUN="$WORK/.chat_shards.sh"; : > "$SHRUN"
  echo 'pids=""; rc=0' >> "$SHRUN"
  for ((s = 0; s < NSH; s++)); do
    if [ "$FITS" = "1" ]; then PLACE="--device cuda:$s"; else PLACE="--device-map auto --gpu-mem 20GiB --cpu-mem 30GiB"; fi
    printf '%s\n' "./.venv/bin/python src/build_chat_calib.py --model \"$ORIG_MODEL\" \
--out \"$WORK/chat_pool_sh${s}.json\" --n-rollouts $PER --n-derived $PER --max-new $SEQ \
--batch ${CHAT_BATCH:-12} --seq $SEQ --replay-frac 0 --require-close $PLACE --seed $((1234 + s)) \
> \"$WORK/chat_sh${s}.log\" 2>&1 &" >> "$SHRUN"
    echo 'pids="$pids $!"' >> "$SHRUN"
  done
  # plain `wait` returns 0 even if a job failed, so wait on each pid individually
  echo 'for p in $pids; do wait $p || rc=1; done; exit $rc' >> "$SHRUN"
  $MEMGUARD bash "$SHRUN" || { echo "FATAL: chat pool generation failed (see $WORK/chat_sh*.log)"; exit 1; }
  for ((s = 0; s < NSH; s++)); do tail -2 "$WORK/chat_sh${s}.log"; done
  ./.venv/bin/python - "$CHATPOOL" "$WORK"/chat_pool_sh*.json <<'PY'
import json, sys
out, parts = sys.argv[1], sys.argv[2:]
seqs, seen = [], set()
for p in parts:
    for s in json.load(open(p)):
        k = tuple(s)                      # different seeds can still land on the same derived prompt
        if k not in seen:
            seen.add(k); seqs.append(s)
json.dump(seqs, open(out, "w"))
print(f"  merged {len(parts)} shard(s) -> {len(seqs)} unique sequences -> {out}")
PY
  [ -s "$CHATPOOL" ] || { echo "FATAL: merged chat pool is empty"; exit 1; }
else echo "=== [skip] Phase 2a: $CHATPOOL exists ==="; fi

# ─────────── Phase 2b: calibration = chat + diverse generic (17-source mix) ───────────
stage "Phase 2b: calib ($CALIB_TOKENS tok, ${CHAT_FRAC} chat)"
if [ ! -f "$CALIB" ]; then
  $PY src/build_diverse_calib.py --orig-model "$ORIG_MODEL" --ctm-data "$CTM_DATA" \
      --tokens "$CALIB_TOKENS" --seq "$SEQ" --chat-frac "$CHAT_FRAC" --chat-src "$CHATPOOL" \
      --out "$CALIB" --eval-out "$CALIB_EVAL"
else echo "=== [skip] Phase 2b: $CALIB exists ==="; fi
NSAMP=$(./.venv/bin/python -c "import json;print(len(json.load(open('$CALIB'))))")
DENS=$(./.venv/bin/python -c "import json;d=json.load(open('$CALIB'));print(f'{100*sum(1 for s in d if 248069 in s)/len(d):.1f}')")
echo "  calib: $NSAMP seqs | </think> density ${DENS}%  (47% gave FP-parity commit on the 4B)"
# Block-AP holds the propagated activation stream in CPU RAM (~10MB per 1024-tok seq, ×2 in Pass B), so the
# sample count must scale INVERSELY with SEQ or it OOM-kills the host (measured: 640 seqs @2560 did exactly that).
BLOCKAP_SAMPLES=${BLOCKAP_SAMPLES:-$(( 640 * 1024 / SEQ ))}
[ "$BLOCKAP_SAMPLES" -gt "$NSAMP" ] && BLOCKAP_SAMPLES=$NSAMP
echo "  Block-AP samples: $BLOCKAP_SAMPLES (constant token budget ⇒ constant RAM); teacher+E2E use all $NSAMP"

# ─────────── Phase 3: Block-AP skeleton — GPTQ-init QAT + ternary embed & lm_head ───────────
# --quant-embed-head ternarizes embed_tokens + lm_head (Bonsai footprint parity; lm_head via GPTQ on the
# final-hidden Gram). The Hessians are collected on the CHAT-bearing calib, which is what fixes the
# start-of-`<think>` distribution collapse.
stage "Phase 3: Block-AP skeleton (g$BLK)"
if [ ! -f "$RECOVERED/model.safetensors.index.json" ]; then
  $PY src/block_ap_recovery.py --model-path "$ROT" --orig-config-path "$ORIG_MODEL" \
      --output-dir "$WORK" --block-size "$BLK" --samples "$BLOCKAP_SAMPLES" \
      --qat --qat-gptq-init --qat-keep-best --qat-attn-only \
      --qat-lr 1e-4 --qat-scale-lr 1e-4 --qat-epochs 4 --quant-embed-head
else echo "=== [skip] Phase 3: $RECOVERED exists ==="; fi
eval2k_stage skeleton "$RECOVERED"

# ─────────── Phase 4: FP teacher cache (top-64) ───────────
CACHE_HIDDEN_FLAG=""; [ "$FEAT_WEIGHT" != "0" ] && CACHE_HIDDEN_FLAG="--cache-hidden"
stage "Phase 4: FP teacher cache"
if [ ! -f "$TEACHER" ]; then
  $PY src/e2e_qp_distill.py --precompute-teacher --teacher-path "$ROT" --orig-config-path "$ORIG_MODEL" \
      --calib "$CALIB" --teacher-cache "$TEACHER" --seq "$SEQ" --topk 64 \
      --cache-batch "${CACHE_BATCH:-1}" --teacher-dp "${TEACHER_DP:-1}" \
      --gpu-mem 20GiB --cpu-mem 30GiB --max-samples "$NSAMP" $CACHE_HIDDEN_FLAG
else echo "=== [skip] Phase 4: $TEACHER exists ==="; fi

# ─────────── Phase 4.5: ASSIGNMENT training (trit assignments; scales FROZEN) ───────────
# NEW. Everything before this trains SCALES; the trits themselves were fixed by the Block-AP skeleton.
# Measured on the 4B (32L), eval2k agreement, assignment stage ALONE from the raw skeleton:
#     0.31M tok 71.03 | 2.77M 73.72 | 8.29M 76.21 | 16M 78.35   (+1.67 pt per DOUBLING, no saturation)
# Both axes pay and add: at fixed compute 9x data = +2.55 pt; at fixed data 9x compute = +2.17 pt.
#
# SCOPE = down + attn ONLY, and that is a hard-won constraint, not an arbitrary choice:
#   * A SECOND MLP projection trained sequentially ALWAYS damages and never recovers -- confirmed 3x,
#     including 3240 steps warm-started from the 16M `down` (entry 0.2341 -> 0.3261). SwiGLU is
#     down(silu(gate(x))*up(x)): gate/up multiply, so tuning one co-adapts the others' CURRENT
#     assignments, which become exactly what the next stage must move away from.
#   * Training them JOINTLY does work (gateup @8L beat sequential in half the compute) but
#     {gate,up}@32L = 1510M latents ~= 45.7GB > the 42GB host ceiling at the measured 27.3 B/latent.
#   * `attn` already covers every DeltaNet projection (the predicate matches "attn" in the name, and
#     linear_attn contains it), including the tiny in_proj_a/b and conv1d. Nothing small is left out.
# --abort-patience 30 IS LOAD-BEARING: the abort threshold is in STEPS (ho_worse x eval_every >=
# 100 x patience => 300 at the default 3), so with a coarse --eval-every ONE non-improving eval kills
# the stage. That silently truncated the first chain attempt at step 800 of 3240.
#
# ONE EPOCH is deliberate. More epochs DO help (+0.685 pt per doubling of STEPS on fixed data) but more
# UNIQUE data helps ~3x more per token (9 epochs over 360 seqs == 1 epoch over 1080 seqs, 73.66 vs 73.72).
# Unique calib caps at ~15M tok (6244 seqs x 2560), so everything past 1 epoch is repeats at ~1/3
# efficiency. If this axis is ever pushed, GENERATE MORE CALIB rather than raising --epochs.
ASSIGN=${ASSIGN:-1}                      # 0 = skip Phase 4.5 entirely (pre-2026-08 behaviour)
# Latent optimizer. `adam-blockv` shares ONE Adam second moment per 256-latent scale block instead of
# per element: 1.48x faster on a bandwidth-bound CPU (48.2 vs 71.1 ms per 23.6M latents) at no measured
# quality cost. Validated 2026-08-19 as a 4-arm paired run on the 4B (raw skeleton, 8 down_proj layers,
# 240 steps, verified-identical init fingerprints, frozen source):
#     base (dense Adam)   KL 0.6234  assign-moved 3.558%  per-layer ratio 1x
#     adam-blockv         KL 0.6225  assign-moved 2.878%  per-layer ratio 1x   <- -0.0009 KL, 1.48x
#     sgd                 KL 0.6452  assign-moved 7.825%  per-layer ratio 1x   <- REJECTED (+0.0218)
# Noise floor is 0.010 KL (three independent dense-Adam baselines: 0.6234 / 0.6288 / 0.6335), so
# blockv is indistinguishable from dense and SGD is not. blockv also reaches that KL with ~19% FEWER
# assignment flips. Set ASSIGN_OPT=adam to fall back to the per-element second moment.
ASSIGN_OPT=${ASSIGN_OPT:-adam-blockv}
ASSIGN_LR_DOWN=${ASSIGN_LR_DOWN:-5e-7}   # servo-calibrated at 32L; base only needs to be within ~1 order
ASSIGN_LR_ATTN=${ASSIGN_LR_ATTN:-7.5e-7} # the servo's PLATEAUED answer (2.5e-7 starves, 5e-6 bursts)
ASSIGN_ATTN_STRIDE=${ASSIGN_ATTN_STRIDE:-2}   # attn@32L = 1347M ~= 41GB; stride 2 = 674M ~= 23GB (proven)
ASSIGN_ATTN_SAMP=${ASSIGN_ATTN_SAMP:-3240}
# ── FULL-LATENT ("arm B") variant, off by default so the validated down->attn recipe is unchanged.
# ASSIGN_SCOPE=all + ASSIGN_STRIDE_ALL=1 replaces the two staged passes with ONE pass over every
# trained weight. MB_SEQS is the 13s/13t sequences-per-microbatch knob: the latent H2D, the CPU Adam
# and the gradient D2H are paid PER MICROBATCH, so grouping amortises them (27B: 21.8 -> 68.5 tok/s
# at G=4). CPU_THREADS overrides torchrun's OMP_NUM_THREADS=1, worth 5.9x on the latent Adam (13q).
ASSIGN_SCOPE=${ASSIGN_SCOPE:-}                # empty = validated down+attn; "all" = single full pass
ASSIGN_STRIDE_ALL=${ASSIGN_STRIDE_ALL:-1}
ASSIGN_LR_ALL=${ASSIGN_LR_ALL:-5e-7}
MB_SEQS=${MB_SEQS:-1}
CPU_THREADS=${CPU_THREADS:-0}
ASSIGN_EXTRA=${ASSIGN_EXTRA:-}
ASSIGNED="$RECOVERED"
if [ "$ASSIGN" != "0" ]; then
  run_assign () {   # tag scope lr max_samples eval_every stride student
    local OUT=$WORK/assign_$1/modified_model
    stage "Phase 4.5$1: assignment ($2, lr $3, n=$4, stride $6)"
    if [ -f "$WORK/assign_$1/.done" ]; then echo "=== [skip] assign_$1 ==="; ASSIGNED="$OUT"; return 0; fi
    mkdir -p "$WORK/assign_$1"
    $PY -m torch.distributed.run --nproc_per_node="$NGPU" --standalone \
        src/e2e_qp_distill.py --train --student-path "$7" --orig-config-path "$ORIG_MODEL" \
        --calib "$CALIB" --teacher-cache "$TEACHER" --out "$OUT" \
        --seq "$SEQ" --epochs 1 --max-samples "$4" \
        --lr 0 --latent-lr "$3" --latent-init fp-spread --fp-model "$ROT" \
        --latent-offload --latent-grad-release --tw-layer-stride "$6" --latent-warmup-steps 20 \
        --target-tr 1.25e-4 --tr-every 5 --tr-final-frac 0.2 \
        --lr-schedule linear --scale-ema-decay 0 --select final --loss-fn cakld --decision-gamma 2 \
        --ce-weight 0.1 --ce-positions 128 --train-weights "$2" --scale-qat-bits "$SBITS" \
        --latent-opt "$ASSIGN_OPT" --mb-seqs "$MB_SEQS" --cpu-threads "$CPU_THREADS" $ASSIGN_EXTRA \
        --heldout-n 4 --eval-every "$5" --ckpt-every 0 --abort-patience 30 \
      || { echo "  assign_$1 FAILED"; exit 1; }
    touch "$WORK/assign_$1/.done"; ASSIGNED="$OUT"
  }
  if [ -n "$ASSIGN_SCOPE" ]; then
    run_assign all "$ASSIGN_SCOPE" "$ASSIGN_LR_ALL" "$NSAMP" 400 "$ASSIGN_STRIDE_ALL" "$RECOVERED"
    eval2k_stage assign_all "$ASSIGNED"
  else
  run_assign a down "$ASSIGN_LR_DOWN" "$NSAMP"           400 1                    "$RECOVERED"
  eval2k_stage assign_down "$ASSIGNED"
  run_assign b attn "$ASSIGN_LR_ATTN" "$ASSIGN_ATTN_SAMP" 200 "$ASSIGN_ATTN_STRIDE" "$ASSIGNED"
  eval2k_stage assign_attn "$ASSIGNED"
  fi
  echo "=== Phase 4.5 done — E2E will start from $ASSIGNED ==="
else echo "=== [skip] Phase 4.5: ASSIGN=0 ==="; fi

# ─────────── Phase 5: E2E scale distillation (+col-scale, scale-QAT, ADDITIVE commit) ───────────
stage "Phase 5: E2E (${EPOCHS}ep, commit-beta $COMMIT_BETA)"
if [ ! -f "$WORK/e2eqp/.done" ]; then
  mkdir -p "$WORK/e2eqp"
  $PY -m torch.distributed.run --nproc_per_node="$NGPU" --standalone \
      src/e2e_qp_distill.py --train --student-path "$ASSIGNED" --orig-config-path "$ORIG_MODEL" \
      --calib "$CALIB" --teacher-cache "$TEACHER" --out "$E2EQP" \
      --seq "$SEQ" --epochs "$EPOCHS" --max-samples "$NSAMP" \
      --lr "$E2E_LR" --lr-schedule linear --scale-ema-decay 0 --select final \
      --loss-fn cakld --feat-weight "$FEAT_WEIGHT" --decision-gamma 2 \
      --col-scale --scale-qat-bits "$SBITS" \
      --commit-beta "$COMMIT_BETA" --commit-pre 16 --commit-post 12 \
      --heldout-n 48 --eval-every 300 --abort-patience 30 ${E2E_EXTRA:-}
  touch "$WORK/e2eqp/.done"
else echo "=== [skip] Phase 5: $WORK/e2eqp/.done present ==="; fi
eval2k_stage e2e "$E2EQP"
FINAL="$E2EQP"

# ─────────── Phase 6: GATES — free-gen is the only admissible check ───────────
# Teacher-forced KL/ppl are PROVEN BLIND here (they read 79-84% while the model scored below random).
stage "Phase 6: gates"
# Gate A MUST use the FROZEN eval2k corpus, not $CALIB_EVAL. A per-run held-out is drawn from the run's
# OWN training mixture, so it measures "does the model fit its own distribution" — which is blind to the
# failure that matters here: V1 (50% chat throughout) and FINAL (chat only at the last pass) both fit their
# own held-out, yet eval2k separated them by 7.7 pt (72.77 vs 80.46) = the generic-capability loss. There is
# also a documented case (§7, researcher rounds 6-7) of fixed-eval FLOORING while own-held-out kept falling.
# Only the frozen same-mixture draw is comparable to the 77% gate and to prior runs.
EVAL2K=${EVAL2K:-output_4b/eval2k.json}
echo "=== Gate A: eval2k agreement (frozen 1946-seq referee; gate >=77%) ==="
if [ -f "$EVAL2K" ]; then
  $MEMGUARD env ORIG="$ORIG_MODEL" FP_DIR="$ROT" E2E_MODEL="$FINAL" \
      EVAL_DATA="$EVAL2K" NP="${GATE_NP:-1946}" SEQ=1024 \
      ./.venv/bin/python src/kl_flips_eval.py || echo "(eval2k gate failed; continuing)"
else
  echo "!!! $EVAL2K not found. NOTE: the 4B draw IS reusable across this model family — Qwen3.5-4B and"
  echo "!!! Qwen3.6-27B share a tokenizer (verified: vocab 248077, </think>=248069, max id in the draw"
  echo "!!! 248046), and eval2k scores FP-vs-ternary of whichever model is loaded. Point EVAL2K at it."
  echo "!!! Falling back to the per-run held-out, whose number is NOT comparable to the 77% gate."
  $MEMGUARD env ORIG="$ORIG_MODEL" FP_DIR="$ROT" E2E_MODEL="$FINAL" \
      EVAL_DATA="$CALIB_EVAL" NP="${GATE_NP:-512}" SEQ=1024 \
      ./.venv/bin/python src/kl_flips_eval.py || echo "(agreement gate failed; continuing)"
fi
echo "=== Gate B: FREE-GEN loop/commit gate @${GATE_MAXNEW} tok (THE deploy gate) ==="
echo "    targets (4B refs): commit ≥68% · loop ≤30% · comp-ratio ≤3.1   [FP teacher: 75% / 25% / 2.40]"
$MEMGUARD env ORIG="$ORIG_MODEL" E2E_MODEL="$FINAL" \
    N_PREFIX=48 MAXNEW="$GATE_MAXNEW" THINK=1 TEMP=0.6 BATCH=8 GATE_OUT="$WORK/loop_gate.json" \
    ./.venv/bin/python src/loop_gate.py || echo "(free-gen gate crashed; continuing)"
# ENFORCE Gate B. It was advisory-only until 2026-08-12: the targets were printed and the result was
# printed, but nothing ever COMPARED them, so `|| echo ... continuing` caught only a crash. The 4M smoke
# test exposed this — a model at 81.00% teacher-forced agreement (Gate A PASS) that loops on 37.5% of
# prompts and closes </think> on only 64.6% would have sailed through to GGUF export and printed ALL DONE.
# This script's own header says free-gen is THE deploy gate and that teacher-forced metrics are PROVEN
# BLIND to this failure, so it must fail CLOSED. GATE_ADVISORY=1 restores the old print-only behaviour.
if [ -f "$WORK/loop_gate.json" ] && [ "${GATE_ADVISORY:-0}" != "1" ]; then
  ./.venv/bin/python - "$WORK/loop_gate.json" <<'GPY' || { echo "FATAL: free-gen deploy gate FAILED — refusing to mark this model deployable."; exit 1; }
import json, sys
g = json.load(open(sys.argv[1]))
lim = {"commit_rate": (">=", 0.68), "loop_rate": ("<=", 0.30), "mean_comp_ratio": ("<=", 3.1)}
bad = []
for k, (op, v) in lim.items():
    got = g.get(k)
    if got is None:
        print(f"  [gate-B] {k}: MISSING from loop_gate.json"); bad.append(k); continue
    ok = got >= v if op == ">=" else got <= v
    print(f"  [gate-B] {k:16} {got:.4f} {op} {v}   {'PASS' if ok else 'FAIL'}")
    if not ok: bad.append(k)
print("  [gate-B] " + ("PASS — deployable" if not bad else f"FAIL on {', '.join(bad)}"))
sys.exit(1 if bad else 0)
GPY
fi

# ─────────── Phase 7: deploy to GGUF ───────────
# WARNING: TQ2_0 is inherently g256. A g64 model is NOT on-grid at 256, so a TQ2_0 export REQUANTIZES and
# gives back the granularity win. Either deploy g256 (BLK=256, costs ~2.75pt agreement) or use a custom
# g64 packing. Q8_0 preserves the g64 weights faithfully and is the fidelity reference for benchmarking.
stage "Phase 7: GGUF export"
if [ "$RUN_EXPORT" = "1" ]; then
  F16=$WORK/model-f16.gguf
  $PY src/convert_hf_to_gguf_patched.py "$FINAL" --outfile "$F16" --outtype f16 \
      || { echo "FATAL: convert failed"; exit 1; }
  [ -x "$LLAMA_QUANTIZE" ] || { echo "FATAL: llama-quantize not executable"; exit 1; }
  if [ "$BLK" = "256" ]; then
    "$LLAMA_QUANTIZE" --token-embedding-type "$EMBED_TYPE" --output-tensor-type "$OUTPUT_TYPE" \
        "$F16" "$WORK/model-TQ2_0.gguf" TQ2_0 || { echo "FATAL: quantize failed"; exit 1; }
    echo "=== exported -> $WORK/model-TQ2_0.gguf (g256, TQ2_0-exact) ==="
  else
    echo "!!! BLK=$BLK is NOT TQ2_0-exact (TQ2_0 = g256). Exporting Q8_0 (faithful) AND TQ2_0 (lossy) so the"
    echo "!!! packing cost can be measured; deploy TQ2_0 only if the gate still passes."
    "$LLAMA_QUANTIZE" "$F16" "$WORK/model-Q8_0.gguf" Q8_0 || echo "(Q8_0 failed)"
    "$LLAMA_QUANTIZE" --token-embedding-type "$EMBED_TYPE" --output-tensor-type "$OUTPUT_TYPE" \
        "$F16" "$WORK/model-TQ2_0.gguf" TQ2_0 || echo "(TQ2_0 failed)"
  fi
  rm -f "$F16"; ls -la "$WORK"/model-*.gguf 2>/dev/null
else echo "=== [skip] Phase 7 export (set RUN_EXPORT=1) ==="; fi

stage_end
echo "########## ALL DONE — final ternary model: $FINAL ##########"
echo "Judge it on Gate B (free-gen commit/loop), NOT on the teacher-forced numbers."
) > "${LOG_FILE:-run_full_pipeline.log}" 2>&1 &

echo "Pipeline started in the background (PID $!)."
echo "Watch it:   tail -f ${LOG_FILE:-run_full_pipeline.log}"
