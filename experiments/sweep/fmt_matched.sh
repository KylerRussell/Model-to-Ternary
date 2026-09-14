#!/bin/bash
# Matched-tensor-set format comparison, rotated vs unrotated, same tensors in both arms.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
T="model.language_model.layers.13.mlp.up_proj.weight,model.language_model.layers.19.self_attn.o_proj.weight,model.language_model.layers.25.mlp.down_proj.weight"
run(){
  PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$1 env TENSORS="$T" FMT_SRC="$2" OUT="$3" \
    ./.venv/bin/python src/fmt_recon.py > "$4" 2>&1
}
run 0 output_4bpipe/rotbase/modified_model output_sweep/fm_rot.json output_sweep/fm_rot.log &
run 1 output_4b/untied_4b                  output_sweep/fm_unrot.json output_sweep/fm_unrot.log &
wait
touch output_sweep/.fm_done
