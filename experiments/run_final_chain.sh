#!/bin/bash
# run_final_chain.sh — researcher round-6 "THE ONE RUN".
#   Data:      50% require-close chat (100%-closing) + 50% generic, seq 2560, ~16M tok
#   Objective: L = CAKLD_all + beta*mean(KL over commit window)   [ADDITIVE, un-diluted: the Stage-A fix]
#              beta 1.5, forward-KL bulk, NO student rollouts (Stage B was self-distillation on the pathology)
#   Init:      the combined-2560 checkpoint; lr 1e-5; scale-only (col-scale + scale-QAT 8bit, g64)
#   Gate@2048: commit >=68%, loop <=30%, comp-ratio <=3.1, eval2k >=77%
# Then run_thinkcal.sh (post-hoc, zero-training) folds a </think>-row lm_head scale.
set -u; cd "$(dirname "$0")"; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
PY=./.venv/bin/python; CTM=$HOME/Documents/CTM-Transformer/data_cache; ORIG=output_4b/untied_4b
ROT=output_4b/rot/modified_model; BASE=output_4b/combined2560_g64q8/e2e/modified_model
SEQ=2560; BLK=64; SBITS=8; BETA=${BETA:-1.5}; CHAT_FRAC=0.50; TOTAL=16000000
NEED=$(( TOTAL/SEQ*50/100 ))                       # chat seqs needed (~3125)
POOL=output_4b/chat_pool_final.json; CORPUS=output_4b/final_calib.json
SW=output_4b/final_g${BLK}q${SBITS}; TCH=$SW/teacher.pt; EOUT=$SW/e2e/modified_model
mkdir -p "$SW" logs; export ORIG_MODEL="$ORIG"; export TERNARY_BLOCK_SIZE=$BLK
TG=$(free -g | awk '/^Mem:/{print $2}'); DM=$(( TG>=48 ? TG-18 : TG*2/3 )); MEM_MAX=${MEM_MAX:-${DM}G}
MM=${MEM_MAX%G}; MG="systemd-run --user --scope --quiet -p MemoryHigh=$(( MM-4 ))G -p MemoryMax=$MEM_MAX -p MemorySwapMax=0 -E TERNARY_BLOCK_SIZE=$BLK"
echo "########## FINAL RUN $(date) | 50% require-close chat + 50% generic | beta=$BETA | seq $SEQ ##########"

# ---- (1) top up the require-close chat pool to $NEED seqs (2 GPU shards; existing 917 reused) ----
if [ ! -f "$POOL" ]; then
  HAVE=$($PY -c "import json;print(len(json.load(open('output_4b/chat_pool_rc2560.json'))))")
  MORE=$(( NEED - HAVE )); [ "$MORE" -lt 0 ] && MORE=0
  # require-close keeps ~43% of rollouts, ~1 packed seq each -> generate MORE/0.43 split over 2 GPUs
  PER=$(( MORE * 100 / 43 / 2 + 50 ))
  echo "--- [1] have $HAVE, need $NEED -> generate 2x$PER rollouts (require-close) $(date) ---"
  for SH in 0 1; do
    $MG env CUDA_VISIBLE_DEVICES=$SH $PY src/build_chat_calib.py --model $ORIG \
      --out output_4b/_rcmore_sh${SH}.json --n-rollouts $PER --n-derived $PER --max-new $SEQ \
      --batch 12 --seq $SEQ --replay-frac 0 --require-close --seed $((17+SH)) --device cuda:0 \
      > logs/final_chat_sh${SH}.log 2>&1 &
  done
  wait
  $PY - <<PY
import json
base=json.load(open("output_4b/chat_pool_rc2560.json"))
add=[]
for sh in (0,1):
    try: add += json.load(open(f"output_4b/_rcmore_sh{sh}.json"))
    except Exception as e: print("shard missing:", e)
out=base+add
json.dump(out, open("$POOL","w"))
cl=sum(1 for s in out if 248069 in s)
print(f"chat pool: {len(base)} existing + {len(add)} new = {len(out)} seqs, </think> {100*cl/len(out):.0f}%")
PY
  [ -f "$POOL" ] || { echo "FATAL 1"; exit 1; }
fi

# ---- (2) corpus: 50% chat + 50% generic @seq 2560 ----
if [ ! -f "$CORPUS" ]; then
  echo "--- [2] mix 50/50 @seq $SEQ $(date) ---"
  $PY src/build_diverse_calib.py --orig-model $ORIG --ctm-data "$CTM" --tokens $TOTAL --seq $SEQ \
    --chat-frac $CHAT_FRAC --chat-src "$POOL" --out "$CORPUS" --eval-out output_4b/final_eval.json \
    > logs/final_mix.log 2>&1 || { echo "FATAL 2 (logs/final_mix.log)"; exit 1; }
  grep -aE 'CHAT|wrote.*train' logs/final_mix.log | tail -2
  $PY -c "import json;d=json.load(open('$CORPUS'));print(f'  corpus </think> density: {sum(1 for s in d if 248069 in s)/len(d)*100:.1f}% (target ~50%; comb2560 was 4.3%)')"
fi
NS=$($PY -c "import json;print(len(json.load(open('$CORPUS'))))")

# ---- (3) teacher cache ----
if [ ! -f "$TCH" ]; then
  echo "--- [3] teacher top-64 $(date) ---"
  $MG env CUDA_VISIBLE_DEVICES=0,1 ORIG_MODEL=$ORIG $PY src/e2e_qp_distill.py --precompute-teacher \
    --teacher-path $ROT --orig-config-path $ORIG --calib "$CORPUS" --teacher-cache "$TCH" \
    --topk 64 --seq $SEQ --teacher-dp 1 > logs/final_teacher.log 2>&1
  [ -f "$TCH" ] || { echo "FATAL 3 (logs/final_teacher.log)"; exit 1; }
fi

# ---- (4) E2E: additive commit objective, from the combined-2560 checkpoint ----
if [ ! -f "$SW/e2e/.done" ]; then
  echo "--- [4] E2E 2ep + col-scale + scale-qat $SBITS + commit-beta $BETA $(date) ---"; mkdir -p "$SW/e2e"
  $MG env ORIG_MODEL=$ORIG $PY -m torch.distributed.run --nproc_per_node=2 --standalone src/e2e_qp_distill.py --train \
    --student-path "$BASE" --orig-config-path $ORIG --calib "$CORPUS" --teacher-cache "$TCH" \
    --out "$EOUT" --seq $SEQ --epochs 2 --max-samples $NS \
    --lr 1e-5 --lr-schedule linear --scale-ema-decay 0 --select final \
    --loss-fn cakld --feat-weight 0 --decision-gamma 2 --col-scale --scale-qat-bits $SBITS \
    --commit-beta $BETA --commit-pre 16 --commit-post 12 --abort-patience 30 \
    --heldout-n 48 --eval-every 300 > logs/final_e2e.log 2>&1 \
    && touch "$SW/e2e/.done" || { echo "FATAL 4 (logs/final_e2e.log)"; exit 1; }
fi

# ---- (5) gate @2048 + eval2k ----
echo "--- [5] eval2k ---"
OUTP=$($MG env CUDA_VISIBLE_DEVICES=0,1 ORIG=$ORIG FP_DIR=$ROT E2E_MODEL="$EOUT" \
  EVAL_DATA=output_4b/eval2k.json NP=1946 SEQ=1024 $PY src/kl_flips_eval.py 2>&1 | tee logs/final_e2k.log)
AG=$(echo "$OUTP" | grep -aoE 'agreement *: *[0-9.]+' | grep -aoE '[0-9.]+$' | head -1)
echo "--- [5] free-gen gate @2048 ---"
$MG env ORIG=$ORIG E2E_MODEL="$EOUT" N_PREFIX=48 MAXNEW=2048 THINK=1 TEMP=0.6 BATCH=8 \
  GATE_OUT=logs/final_gate2048.json $PY src/loop_gate.py 2>&1 \
  | grep -aE 'loop_rate|trunc_rate|commit_rate|comp_ratio' | tee logs/final_gate.log
echo ""
echo "=== FINAL RUN | eval2k=$AG% (gate >=77) ==="
echo "  gate targets: commit >=68% | loop <=30% | comp-ratio <=3.1   (FP 75/25/2.40; V1 62.5/29.2/3.34; comb2560 56.2/39.6/3.15)"
echo "########## FINAL DONE $(date) ##########"
