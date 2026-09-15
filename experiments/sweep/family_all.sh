#!/bin/bash
# Family sweep across tensor ROLES -- MLP, attention, embedding. F11c showed the tensor sample
# decides the answer, so the frontier must be reported per role, not pooled.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
run(){ PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$1 N_T=$2 KINDS="$3" EMB="$4" \
       OUT=output_sweep/fam_$5.json ./.venv/bin/python src/run_family_sweep.py \
       > output_sweep/fam_$5.log 2>&1; }
run 0 3 "down_proj,gate_proj,up_proj" "" mlp &
run 1 2 "in_proj_qkv,out_proj" "model.language_model.embed_tokens.weight" attn_emb &
wait
touch output_sweep/.famall_done
