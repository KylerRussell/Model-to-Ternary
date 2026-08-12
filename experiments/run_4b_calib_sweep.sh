#!/bin/bash
# run_4b_calib_sweep.sh — the DATA-SIZE SCALING CURVE. How do TIME and ACCURACY scale with E2E calibration
# data size, at the FIXED optimal recipe and FIXED epochs? Vary ONLY the E2E calib {0.5M,4M,16M,64M} tokens,
# each at --epochs 2 (the sweet spot). Fixed skeleton = a1_lr1e-4_ep4 (A6 base, block-AP pre-E2E, the CONTROL).
# Fixed eval = calib_eval.json (ID) + ood_ppl (OOD) + kl_flips (behavioral). Then fit data↔time & data↔OOD.
#
# Purpose: the researcher calls E2E the "primary knob" for 27B (3-4× steps for cross-block accumulation) —
# this tells us the accuracy-per-token and time-per-token so we can budget the 27B E2E rationally.
set -u; cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1
ORIG=output_4b/untied_4b; ROT=output_4b/rot/modified_model; PY=./.venv/bin/python
CTM=$HOME/Documents/CTM-Transformer/data_cache
SKEL=output_4b/strengthen/a1_lr1e-4_ep4/modified_model      # fixed control skeleton (A6's block-AP output)
CALIB_EVAL=output_4b/calib_eval.json; EPOCHS=2
export ORIG_MODEL="$ORIG"
SW=output_4b/calib_sweep; mkdir -p "$SW" logs
RES=logs/calib_sweep_results.txt
[ -f "$RES" ] || echo "tag tokens seqs epochs steps e2e_sec ID OOD KL flips" > "$RES"

# ---- host-safety RAM cap ----
TOTAL_GB=$(free -g | awk '/^Mem:/{print $2}'); DEF_MAX=$(( TOTAL_GB>=48 ? TOTAL_GB-18 : TOTAL_GB*2/3 ))
MEM_MAX=${MEM_MAX:-${DEF_MAX}G}; MM=${MEM_MAX%G}; MEM_HIGH=${MEM_HIGH:-$(( MM-4 ))G}
if systemd-run --user --scope --quiet -p MemoryMax=64M -p MemorySwapMax=0 /bin/true 2>/dev/null; then
  MG="systemd-run --user --scope --quiet -p MemoryHigh=$MEM_HIGH -p MemoryMax=$MEM_MAX -p MemorySwapMax=0"
  echo "[memguard] max=$MEM_MAX swap=off"
else echo "[memguard] FATAL: no user scope"; exit 1; fi

[ -f "$SKEL/model.safetensors.index.json" ] || { echo "FATAL: skeleton $SKEL missing"; exit 1; }
[ -d "$CTM" ] || { echo "FATAL: CTM source $CTM missing"; exit 1; }
echo "########## 4B CALIB-SIZE SWEEP $(date) | skeleton=a1_lr1e-4_ep4 | epochs=$EPOCHS ##########"

run_size () {  # tag  tokens
  local tag=$1 TOK=$2 CAL TCH
  if [ "$tag" = "4M" ]; then CAL=output_4b/calibration_data.json; TCH=output_4b/teacher_topk.pt
  else CAL=$SW/calib_${tag}.json; TCH=$SW/teacher_${tag}.pt; fi
  grep -qaE "^$tag " "$RES" && { echo "=== [skip] $tag already in results ==="; return 0; }
  echo "===== SIZE $tag ($TOK tokens) ====="

  # 1) calib generation (train-only; eval stays fixed)
  if [ ! -f "$CAL" ]; then
    echo "  [$tag] build_diverse_calib $TOK tokens"
    $MG env CUDA_VISIBLE_DEVICES=0 $PY src/build_diverse_calib.py --orig-model $ORIG --ctm-data "$CTM" \
      --tokens $TOK --seq 1024 --eval-frac 0 --out "$CAL" --eval-out "$SW/calib_${tag}_eval_unused.json" \
      > logs/calib_gen_${tag}.log 2>&1 || { echo "  FATAL calib-gen $tag"; return 1; }
  fi
  local NS=$($PY -c "import json;print(len(json.load(open('$CAL'))))")

  # 2) FP teacher top-64 cache for THIS calib
  if [ ! -f "$TCH" ]; then
    echo "  [$tag] precompute teacher ($NS seqs)"
    $MG env CUDA_VISIBLE_DEVICES=0,1 $PY src/e2e_qp_distill.py --precompute-teacher \
      --teacher-path $ORIG --calib "$CAL" --teacher-cache "$TCH" --seq 1024 --topk 64 \
      --teacher-dp 0 --gpu-mem 20GiB --cpu-mem 30GiB --max-samples $NS > logs/teacher_${tag}.log 2>&1 \
      || { echo "  FATAL teacher $tag"; return 1; }
  fi

  # 3) E2E at fixed epochs (timed)
  local OUT=$SW/e2e_${tag}/modified_model
  if [ ! -f "$SW/e2e_${tag}/.done" ]; then
    echo "  [$tag] E2E --epochs $EPOCHS"; mkdir -p "$SW/e2e_${tag}"
    local t0=$(date +%s)
    $MG env ORIG_MODEL=$ORIG $PY -m torch.distributed.run --nproc_per_node=2 --standalone src/e2e_qp_distill.py --train \
      --student-path $SKEL --orig-config-path $ORIG --calib "$CAL" --teacher-cache "$TCH" \
      --out "$OUT" --seq 1024 --epochs $EPOCHS --max-samples $NS \
      --lr 2e-5 --lr-schedule constant --scale-ema-decay 0.999 --scale-ema-start-frac 0.5 --select ema \
      --loss-fn cakld --feat-weight 0 --decision-gamma 2 > logs/e2e_${tag}.log 2>&1 \
      && echo $(( $(date +%s) - t0 )) > "$SW/e2e_${tag}/.sec" && touch "$SW/e2e_${tag}/.done" \
      || { echo "  FATAL E2E $tag"; return 1; }
  fi
  local SEC=$(cat "$SW/e2e_${tag}/.sec" 2>/dev/null || echo NA)
  local STEPS=$(grep -aoE "steps=[0-9]+" logs/e2e_${tag}.log | head -1 | cut -d= -f2)

  # 4) eval (fixed eval sets) — ID / OOD / KL / flips
  rm -f "$SW/_ref_${tag}.pt"
  local ID=$($MG env CUDA_VISIBLE_DEVICES=0 $PY src/eval_ternary.py --fp-path $ORIG --ternary-path "$OUT" \
     --orig-config-path $ORIG --calib $CALIB_EVAL --seq 1024 --max-samples 96 --ref-file "$SW/_ref_${tag}.pt" --reuse-ref 2>&1 \
     | grep -aoE 'ratio \(tern[^0-9]*[0-9.]+' | grep -aoE '[0-9.]+$')
  local OOD=$($MG env CUDA_VISIBLE_DEVICES=0 ORIG=$ORIG FP_DIR=$ROT E2E_MODEL="$OUT" $PY src/ood_ppl.py 2>&1 \
     | grep -aoE 'OOD ppl ratio [0-9.]+' | grep -aoE '[0-9.]+$')
  local KLF=$($MG env CUDA_VISIBLE_DEVICES=0,1 ORIG=$ORIG FP_DIR=$ROT E2E_MODEL="$OUT" EVAL_DATA=$CALIB_EVAL NP=24 SEQ=1024 \
     $PY src/kl_flips_eval.py 2>&1)
  local KL=$(echo "$KLF" | grep -aoE 'mean KL\(fp\|\|tern\) *: *[0-9.]+' | grep -aoE '[0-9.]+$' | head -1)
  local FL=$(echo "$KLF" | grep -aoE '%flips[^0-9]*[0-9.]+' | grep -aoE '[0-9.]+' | head -1)
  echo "$tag $TOK $NS $EPOCHS ${STEPS:-NA} $SEC ${ID:-NA} ${OOD:-NA} ${KL:-NA} ${FL:-NA}" | tee -a "$RES"
}

run_size 0.5M 500000
run_size 4M   4000000
run_size 16M  16000000
run_size 64M  64000000

echo ""; echo "=== RESULTS ==="; column -t "$RES"
echo "=== CURVE FIT (time∝tokens? accuracy vs log tokens) ==="
$PY - "$RES" <<'PYEOF'
import sys, math
rows=[l.split() for l in open(sys.argv[1]) if l.strip() and not l.startswith("tag")]
def num(x):
    try: return float(x)
    except: return None
pts=[(num(r[1]),num(r[5]),num(r[7])) for r in rows if len(r)>=8]   # (tokens, e2e_sec, OOD)
pts=[p for p in pts if p[0] and p[1]]
if len(pts)>=2:
    import statistics
    print(" tokens      e2e_sec   sec/Mtok    OOD")
    for tok,sec,ood in pts:
        print(f"  {tok/1e6:6.1f}M   {sec:7.0f}   {sec/(tok/1e6):8.1f}   {ood if ood else 'NA'}")
    # time ~ linear in tokens (fixed epochs): slope = mean(sec/tok)
    sl=[sec/tok for tok,sec,_ in pts]
    print(f"\n time-per-token ~ {statistics.mean(sl)*1e6:.1f} sec/Mtok (fixed {2} epochs); "
          f"spread {min(sl)*1e6:.1f}-{max(sl)*1e6:.1f}")
    # OOD vs log2(tokens): each doubling of data improves OOD by ~slope
    op=[(math.log2(tok),ood) for tok,_,ood in pts if ood]
    if len(op)>=2:
        n=len(op); mx=sum(x for x,_ in op)/n; my=sum(y for _,y in op)/n
        b=sum((x-mx)*(y-my) for x,y in op)/sum((x-mx)**2 for x,_ in op)
        print(f" OOD vs log2(tokens): slope {b:+.4f} OOD-ratio per data-DOUBLING "
              f"(negative=better); extrapolate & judge diminishing returns")
PYEOF
echo "########## DONE $(date) ##########"
