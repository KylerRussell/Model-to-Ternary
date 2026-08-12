#!/usr/bin/env bash
# serve_gguf.sh <gguf> [port] [ctx] — serve a GGUF via llama-server (OpenAI-compatible API),
# both GPUs, embedded chat template. Used to serve both the ternary (TQ2_0) and the FP (Q8_0)
# references on the SAME engine so evals are apples-to-apples.
set -u
GGUF=${1:?usage: serve_gguf.sh <gguf> [port] [ctx]}
PORT=${2:-8080}
CTX=${3:-16384}
BIN=${LLAMA_SERVER:-${LLAMA_DIR:-$HOME/llama.cpp}/build/bin/llama-server}
# --reasoning-format none: keep the whole <think>…</think> + answer in message.content (don't split
# into reasoning_content) so eval harnesses that read content can extract the final answer.
exec "$BIN" -m "$GGUF" --host 127.0.0.1 --port "$PORT" -c "$CTX" \
     -ngl 99 -ts 1,1 --jinja --reasoning-format none --parallel 1
