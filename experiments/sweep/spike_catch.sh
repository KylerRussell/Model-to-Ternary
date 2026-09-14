#!/bin/bash
# Fast spike catcher. Two jobs:
#   1. DIAGNOSE: when cgroup memory crosses WARN, snapshot py-spy stacks for both ranks + per-process
#      RSS + memory.stat, so we learn WHICH CODE is allocating -- the thing phase probes give only at
#      coarse granularity.
#   2. PROTECT: at KILLGB, SIGKILL the trainer so the CONTAINER survives with its logs.
# Polls at 0.5s: the pre-fix burst ran >=5.5 GB/s, so 2s sampling could miss ~11 GB per sample.
set -u
D=/home/kasm-user/Documents/Model-to-Ternary/output_sweep
WARN=${WARN:-340}          # normal peak measured at 306GB, so 340 = genuinely abnormal
KILLGB=${KILLGB:-470}      # below the 620 limit with room for one more 0.5s tick at 5.5GB/s
OUT=$D/spike_report.txt
PY=$(command -v py-spy || echo /home/kasm-user/Documents/Model-to-Ternary/.venv/bin/py-spy)
warned=0
while true; do
  CG="/sys/fs/cgroup$(awk -F: '{print $3}' /proc/self/cgroup 2>/dev/null | head -1)"
  [ -r "$CG/memory.current" ] || { sleep 0.5; continue; }
  CUR=$(( $(cat $CG/memory.current) / 1073741824 ))
  if [ "$CUR" -ge "$WARN" ] && [ "$warned" -eq 0 ]; then
    warned=1
    {
      echo "===== SPIKE SNAPSHOT $(date +%H:%M:%S) cur=${CUR}G ====="
      echo "--- memory.stat (top) ---"; head -12 $CG/memory.stat
      echo "--- per-process RSS ---"
      ps -eo pid=,rss=,comm=,args= | awk '$2>1048576{printf "  pid %s rss %.1fGB %s\n",$1,$2/1048576,$3}'
      for p in $(ps -eo pid=,args= | grep "[e]2e_qp_distill.py --train" | grep -v torchrun | awk '{print $1}'); do
        echo "--- py-spy dump pid $p ---"
        timeout 25 "$PY" dump --pid "$p" --nonblocking 2>&1 | head -40
      done
    } >> "$OUT" 2>&1
    echo "SPIKE at ${CUR}G -> snapshot written to spike_report.txt" >> "$OUT"
  fi
  [ "$CUR" -lt $(( WARN - 20 )) ] && warned=0     # re-arm once it settles back
  if [ "$CUR" -ge "$KILLGB" ]; then
    PIDS=$(ps -eo pid=,comm=,args= | awk '$2=="python" && /e2e_qp_distill\.py --train/{print $1}')
    echo "KILL at ${CUR}G >= ${KILLGB}G -> SIGKILL $PIDS ($(date +%H:%M:%S))" >> "$OUT"
    kill -9 $PIDS 2>/dev/null
    sleep 10
  fi
  sleep 0.5
done
