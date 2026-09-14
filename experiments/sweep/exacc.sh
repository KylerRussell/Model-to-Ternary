#!/bin/bash
# Stage 0 analysis. ONE model pair (FP teacher vs the c=1.30 ternary student) is held fixed across
# every arm; the only thing that varies is WHICH MODEL GENERATED THE TOKENS being teacher-forced.
# That is the whole experiment -- the difference between the two is the exposure-bias term:
#
#   ROWS=tr_teacher   tokens the FP teacher generated  -> KL at ON-distribution states  = epsilon
#   ROWS=tr_c1.30     tokens the student generated     -> KL at student-visited states  = regret R
#   ROWS=tr_c1.40     the known-collapsed arm (1/46)   -> must look WORSE than c1.30 or the metric
#                                                          is not measuring chain survival
#
# NOTE THE TRS ON THE TEACHER ARM. THINK_ROW_SCALE stays 1.30 for all three because the STUDENT is
# the thing being characterised and must be byte-identical across arms; only the token source moves.
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
source ./env.sh
STU=${STU:-output_sweep/opsa/modified_model}
FP=$PWD/output_4bpipe/rotbase/modified_model
RES=output_sweep/exacc_results.txt
run(){
  local tag=$1
  local src=$2
  local O=output_sweep/xa_${tag}.json
  if [ -f "$O" ]; then echo "  SKIP $tag" >> $RES; return 0; fi
  echo "  [$(date +%F' '%H:%M:%S)] START $tag (tokens from $src)" >> $RES
  env ORIG=$PWD/output_4b/untied_4b E2E_MODEL="$STU" FP_DIR="$FP" \
      ROWS="output_sweep/${src}_rows.json" THINK_ROW_SCALE=1.30 \
      OUT="$O" ./.venv/bin/python src/exaccerr.py > output_sweep/xa_${tag}.log 2>&1
  echo "  [$(date +%F' '%H:%M:%S)] END $tag rc=$?" >> $RES
  tail -14 output_sweep/xa_${tag}.log >> $RES
}
run teacher tr_teacher
run c1.30   tr_c1.30
run c1.40   tr_c1.40
echo DONE > output_sweep/.xa_done
