#!/usr/bin/env bash
# run_evals_4b.sh — the 4B SAME-MEMORY demo: does 4B-ternary (TQ2_0, ~1GB) beat a similar/larger-memory
# fp8 (Q8_0) model on reasoning? NOT a 27B gate — a "ternary beats fp8 at the same size" demonstration.
# Reuses the 27B harness's mcq_eval.py/math_eval.py/compare.py unchanged; only the model set + serving differ.
# All 4 models are small (≤~4GB) so each serves 2 data-parallel instances (one per GPU) — no tensor-split.
#
#   tern4b : 4B ternary TQ2_0 (A6 winner)  ~1.0 GB   <- ours
#   q0_8b  : Qwen3.5-0.8B  Q8_0            ~0.85 GB  <- FAIR same-memory competitor
#   q2b    : Qwen3.5-2B    Q8_0            ~2.1 GB   <- 2x memory (stronger claim if we still win)
#   q4b    : Qwen3.5-4B FP Q8_0            ~4.2 GB   <- FP-4B ceiling (how much ternary gives up)
set -u
cd "$(dirname "$0")/.."
source ./lib_timing.sh
BIN=${LLAMA_SERVER:-${LLAMA_DIR:-$HOME/llama.cpp}/build/bin/llama-server}
PY=.venv-eval/bin/python
RES=eval_harness/results_4b; mkdir -p "$RES"
declare -A GGUF=( [tern4b]=eval_gguf/tern4b_a6-TQ2_0.gguf [q0_8b]=eval_gguf/qwen0.8b-Q8_0.gguf \
                  [q2b]=eval_gguf/qwen2b-Q8_0.gguf [q4b]=eval_gguf/qwen4b-Q8_0.gguf \
                  [final4b]=eval_gguf/final4b-TQ2_0.gguf \
                  [final4b_tq164]=eval_gguf/final4b-TQ1_64.gguf [final4b_q8]=eval_gguf/final4b-Q8_0.gguf )

MCQ_SAMPLES=${MCQ_SAMPLES:-4}
MATH_SAMPLES=${MATH_SAMPLES:-2}
MMLU_LIMIT=${MMLU_LIMIT:-300}
MODELS=${MODELS:-"tern4b q0_8b q2b q4b"}
EVALS=${EVALS:-"mmlu_pro gpqa"}                 # MCQ reasoning; add "aime26 hmmt26" for math (slow)
MAXTOK_MCQ=${MAXTOK_MCQ:-128}; MAXTOK_MATH=${MAXTOK_MATH:-11264}

wait_health () { for i in $(seq 1 120); do curl -s "$1/health" 2>/dev/null | grep -q ok && return 0; sleep 3; done; return 1; }
kill_servers () { for p in $(pgrep -f 'build/bin/llama-server'); do kill -9 "$p" 2>/dev/null; done; sleep 2; }

# serve_model <tag> <ctx> <parallel> -> two data-parallel instances (one per GPU); sets EPS, CONC
serve_model () {
  local m=$1 ctx=$2 par=$3 g=${GGUF[$1]}
  kill_servers
  CUDA_VISIBLE_DEVICES=0 nohup $BIN -m "$g" --host 127.0.0.1 --port 8081 -c "$ctx" -ngl 99 --jinja --reasoning-format none --parallel "$par" > eval_harness/serve_4b_${m}0.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 nohup $BIN -m "$g" --host 127.0.0.1 --port 8082 -c "$ctx" -ngl 99 --jinja --reasoning-format none --parallel "$par" > eval_harness/serve_4b_${m}1.log 2>&1 &
  wait_health http://127.0.0.1:8081 && wait_health http://127.0.0.1:8082 || { echo "FATAL: $m serve"; kill_servers; exit 1; }
  EPS="http://127.0.0.1:8081/v1,http://127.0.0.1:8082/v1"; CONC=$(( par * 2 ))
}

run_eval () {  # <eval> <model>
  local ev=$1 m=$2 out="$RES/${1}_${2}.json"
  case "$ev" in
    mmlu_pro) $PY eval_harness/mcq_eval.py  --base-urls "$EPS" --type mmlu_pro --limit "$MMLU_LIMIT" --samples "$MCQ_SAMPLES" --concurrency "$CONC" --max-tokens "$MAXTOK_MCQ" --out "$out" ;;
    gpqa)     $PY eval_harness/mcq_eval.py  --base-urls "$EPS" --type gpqa --samples "$MCQ_SAMPLES" --concurrency "$CONC" --max-tokens "$MAXTOK_MCQ" --out "$out" ;;
    aime26)   $PY eval_harness/math_eval.py --base-urls "$EPS" --dataset MathArena/aime_2026     --samples "$MATH_SAMPLES" --concurrency "$CONC" --max-tokens "$MAXTOK_MATH" --out "$out" ;;
    hmmt26)   $PY eval_harness/math_eval.py --base-urls "$EPS" --dataset MathArena/hmmt_feb_2026 --samples "$MATH_SAMPLES" --concurrency "$CONC" --max-tokens "$MAXTOK_MATH" --out "$out" ;;
  esac || echo "($ev / $m failed)"
}

echo "########## 4B SAME-MEMORY EVAL $(date) | models: $MODELS | evals: $EVALS ##########"
echo "=== GGUF footprints (the same-size axis) ==="
for m in $MODELS; do [ -f "${GGUF[$m]}" ] && printf "  %-8s %s\n" "$m" "$(ls -la ${GGUF[$m]} | awk '{print $5}') ${GGUF[$m]}" || echo "  $m MISSING ${GGUF[$m]}"; done

kill_servers
for ev in $EVALS; do
  case "$ev" in
    mmlu_pro|gpqa) CTX=4096;  PAR=8 ;;
    *)             CTX=49152; PAR=4 ;;
  esac
  for m in $MODELS; do
    [ -f "${GGUF[$m]}" ] || { echo "(skip $ev/$m — missing ${GGUF[$m]})"; continue; }
    stage "[$ev / $m] serve"; serve_model "$m" "$CTX" "$PAR"
    stage "[$ev / $m] run (conc=$CONC)"; run_eval "$ev" "$m"
  done
done
kill_servers
stage_end
echo; echo "=== 4B SAME-MEMORY COMPARISON ==="; $PY eval_harness/compare.py "$RES"
