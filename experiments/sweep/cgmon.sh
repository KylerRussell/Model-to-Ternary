#!/bin/bash
# Sample cgroup memory against the real limit. RSS is the wrong quantity: the kill is done by the
# cgroup and it counts page cache, which never appears in RSS.
# The path is resolved at RUNTIME on every sample -- a container restart changes the pod ID, and an
# earlier version baked the path in at file-creation time, so after a restart it silently logged
# "No such file or directory" and a divide-by-zero instead of memory, losing the trace of two failures.
while true; do
  CG="/sys/fs/cgroup$(awk -F: '{print $3}' /proc/self/cgroup 2>/dev/null | head -1)"
  if [ -r "$CG/memory.max" ]; then
    LIM=$(( $(cat $CG/memory.max) / 1073741824 ))
    CUR=$(( $(cat $CG/memory.current 2>/dev/null || echo 0) / 1073741824 ))
    ANON=$(awk '/^anon /{printf "%d", $2/1073741824}' $CG/memory.stat 2>/dev/null)
    FILE=$(awk '/^file /{printf "%d", $2/1073741824}' $CG/memory.stat 2>/dev/null)
    OOM=$(awk '/oom_kill /{print $2}' $CG/memory.events 2>/dev/null)
    echo "$(date +%H:%M:%S) cur=${CUR}G/${LIM}G anon=${ANON}G cache=${FILE}G pct=$(( CUR*100/LIM ))% oom_kill=${OOM:-?}"
  else
    echo "$(date +%H:%M:%S) cgroup path unreadable: $CG"
  fi
  sleep 2
done
