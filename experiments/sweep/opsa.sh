#!/bin/bash
# OPSA (arXiv 2608.31046) refinement of the Gate-B-failing 4B E2E model.
#
# Gate B failed ONLY on loop_rate (0.3125 vs <=0.30, i.e. 15/48) and mean_comp_ratio (3.2481 vs 3.1);
# commit_rate 0.7917 already beats the FP teacher's 0.75. OPSA targets exactly that failure mode:
# suppress the student's own lowest-logp rollout tokens with an entropy-scaled negative advantage.
#
# --heldout-n 0 is DELIBERATE. The end-of-training restore is unconditional (see 537065e), and OPSA
# deliberately makes held-out KL worse -- it is not optimising KL -- so any snapshot would be
# "better" than the result and the restore would silently undo the entire stage. No held-out set =>
# no snapshot => no restore. Judge this stage on Gate B, which is the only metric it targets.
#
# rollout 128+512 @ T=0.6: §3 parked on-policy as UNDER-DOSED at rollout len 96 vs the ~480-token
# regime where looping appears, and T=0.6 matches loop_gate's own sampling temperature.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
W=/home/kasm-user/Documents/Model-to-Ternary/output_sweep
OUT=$W/opsa/modified_model
mkdir -p $W/opsa
echo "  [$(date +%F' '%H:%M:%S)] START opsa: ${OPSA_STEPS:-200} steps lr ${OPSA_LR:-1e-5} delta ${OPSA_DELTA:-1.0}" >> $W/opsa_results.txt
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 \
./.venv/bin/torchrun --nproc_per_node=1 --master_port=29851 \
  src/e2e_qp_distill.py --train \
  --student-path output_4b_g4/e2eqp/modified_model \
  --orig-config-path "$PWD/output_4b/untied_4b" \
  --calib output_4b_g4/calibration_data.json --teacher-cache output_4b_g4/teacher_topk.pt \
  --out "$OUT" \
  --seq 2560 --epochs 0 --steps "${OPSA_STEPS:-200}" --max-samples 512 \
  --lr "${OPSA_LR:-1e-5}" --lr-schedule linear --select final \
  --opsa-frac 1.0 --opsa-tail-frac 0.2 --opsa-delta "${OPSA_DELTA:-1.0}" \
  --rollout-prefix 128 --rollout-len 512 --rollout-temp 0.6 \
  --scale-qat-bits 8 --heldout-n 0 --eval-every 100000 --ckpt-every 0 \
  --abort-patience 1000000 --cpu-threads 8 \
  > $W/opsa.log 2>&1
rc=$?
echo "  [$(date +%F' '%H:%M:%S)] END opsa rc=$rc" >> $W/opsa_results.txt
