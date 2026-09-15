#!/bin/bash
# 27B family sweep, chained behind the shard download. Gate on the artifact, not the process.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
while ! grep -q "^DONE" output_sweep/q27_dl.log 2>/dev/null; do sleep 60; done
SNAP=$(ls -d /home/kasm-user/.cache/huggingface/hub/models--Qwen--Qwen3.5-27B/snapshots/*/ | head -1)
echo "27B snapshot: $SNAP" > output_sweep/fam27_meta.txt
run(){ PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$1 SRC="$SNAP" N_T=$2 KINDS="$3" EMB="$4" \
       OUT=output_sweep/fam27_$5.json ./.venv/bin/python src/run_family_sweep.py \
       > output_sweep/fam27_$5.log 2>&1; }
run 0 3 "down_proj,gate_proj,up_proj" "" mlp &
run 1 0 "" "model.language_model.embed_tokens.weight" emb &
wait
touch output_sweep/.fam27_done
