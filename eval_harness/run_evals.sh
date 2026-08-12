#!/usr/bin/env bash
# run_evals.sh — EVAL-OUTER ordering: for each eval, run ALL models, then the next eval.
# Gives a complete cross-model comparison for each eval as early as possible (the fast MCQ
# 3-way lands within ~1-2h, before the slow math). Models reload per (eval,model) cell — cheap
# vs the eval cost, and the only way to run 3 models that can't co-reside in 48 GB.
#
# Models (served on the SAME patched llama.cpp engine):
#   ternary  TQ2_0 27B (8 GB)   -> 2 instances, one per GPU (data-parallel)
#   9b       Q8_0 9B (~9.6 GB)  -> 2 instances, one per GPU (data-parallel)
#   fp       Q8_0 27B (~27 GB)  -> 1 instance, tensor-split both GPUs (can't fit one)
set -u
cd "$(dirname "$0")/.."
source ./lib_timing.sh
BIN=${LLAMA_SERVER:-${LLAMA_DIR:-$HOME/llama.cpp}/build/bin/llama-server}
PY=.venv-eval/bin/python
RES=eval_harness/results; mkdir -p "$RES"
declare -A GGUF=( [ternary]=eval_gguf/ternary_4M-TQ2_0.gguf [fp]=eval_gguf/fp_27b-Q8_0.gguf [9b]=eval_gguf/qwen9b-Q8_0.gguf )

MCQ_SAMPLES=${MCQ_SAMPLES:-4}
MATH_SAMPLES=${MATH_SAMPLES:-2}        # TEST=2; FINAL run set MATH_SAMPLES=4
MMLU_LIMIT=${MMLU_LIMIT:-300}
MODELS=${MODELS:-"ternary 9b fp"}      # order within each eval
EVALS=${EVALS:-"mmlu_pro gpqa aime26 hmmt26"}
MAXTOK_MCQ=${MAXTOK_MCQ:-128}; MAXTOK_MATH=${MAXTOK_MATH:-11264}

wait_health () { for i in $(seq 1 120); do curl -s "$1/health" 2>/dev/null | grep -q ok && return 0; sleep 3; done; return 1; }
kill_servers () { for p in $(pgrep -f 'build/bin/llama-server'); do kill -9 "$p" 2>/dev/null; done; sleep 2; }

# serve_model <tag> <ctx> <parallel> -> sets globals EPS (endpoints) and CONC (client concurrency)
serve_model () {
  local m=$1 ctx=$2 par=$3 g=${GGUF[$1]}
  kill_servers
  if [ "$m" = "fp" ]; then
    nohup $BIN -m "$g" --host 127.0.0.1 --port 8080 -c "$ctx" -ngl 99 -ts 1,1 --jinja --reasoning-format none --parallel "$par" > eval_harness/serve_fp.log 2>&1 &
    wait_health http://127.0.0.1:8080 || { echo "FATAL: fp serve"; kill_servers; exit 1; }
    EPS="http://127.0.0.1:8080/v1"; CONC=$par
  else
    CUDA_VISIBLE_DEVICES=0 nohup $BIN -m "$g" --host 127.0.0.1 --port 8081 -c "$ctx" -ngl 99 --jinja --reasoning-format none --parallel "$par" > eval_harness/serve_${m}0.log 2>&1 &
    CUDA_VISIBLE_DEVICES=1 nohup $BIN -m "$g" --host 127.0.0.1 --port 8082 -c "$ctx" -ngl 99 --jinja --reasoning-format none --parallel "$par" > eval_harness/serve_${m}1.log 2>&1 &
    wait_health http://127.0.0.1:8081 && wait_health http://127.0.0.1:8082 || { echo "FATAL: $m serve"; kill_servers; exit 1; }
    EPS="http://127.0.0.1:8081/v1,http://127.0.0.1:8082/v1"; CONC=$(( par * 2 ))
  fi
}

run_eval () {  # <eval> <model>  (uses EPS, CONC)
  local ev=$1 m=$2 out="$RES/${1}_${2}.json"
  case "$ev" in
    mmlu_pro) $PY eval_harness/mcq_eval.py  --base-urls "$EPS" --type mmlu_pro --limit "$MMLU_LIMIT" --samples "$MCQ_SAMPLES" --concurrency "$CONC" --max-tokens "$MAXTOK_MCQ"  --out "$out" ;;
    gpqa)     $PY eval_harness/mcq_eval.py  --base-urls "$EPS" --type gpqa --samples "$MCQ_SAMPLES" --concurrency "$CONC" --max-tokens "$MAXTOK_MCQ"  --out "$out" ;;
    aime26)   $PY eval_harness/math_eval.py --base-urls "$EPS" --dataset MathArena/aime_2026     --samples "$MATH_SAMPLES" --concurrency "$CONC" --max-tokens "$MAXTOK_MATH" --out "$out" ;;
    hmmt26)   $PY eval_harness/math_eval.py --base-urls "$EPS" --dataset MathArena/hmmt_feb_2026 --samples "$MATH_SAMPLES" --concurrency "$CONC" --max-tokens "$MAXTOK_MATH" --out "$out" ;;
  esac || echo "($ev / $m failed)"
}

kill_servers
for ev in $EVALS; do
  case "$ev" in
    mmlu_pro|gpqa) CTX=4096;  PAR=8 ;;   # MCQ letter-only: tiny gens, many slots
    *)             CTX=49152; PAR=4 ;;   # math: long CoT
  esac
  for m in $MODELS; do
    [ -f "${GGUF[$m]}" ] || { echo "(skip $ev/$m — missing ${GGUF[$m]})"; continue; }
    stage "[$ev / $m] serve"
    serve_model "$m" "$CTX" "$PAR"
    stage "[$ev / $m] run (conc=$CONC)"
    run_eval "$ev" "$m"
  done
done
kill_servers
stage_end
echo; echo "=== 3-WAY COMPARISON ==="; $PY eval_harness/compare.py "$RES"
