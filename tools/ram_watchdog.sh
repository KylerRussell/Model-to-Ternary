#!/bin/bash
# RAM watchdog with an EXPECTED budget and a hard ceiling.
#
# WHY: an arm B run (25.6B latents) exhausted host RAM and forced a machine restart, taking the run and
# the container's /usr/lib layer with it. The kernel OOM killer is not a safety net — by the time it
# fires the box is already thrashing. This trips first, kills ONLY the trainer, and leaves the log and
# the machine intact. This container has no systemd, so run_eval2k.sh's `systemd-run --scope
# -p MemoryMax=` guard is unavailable; hence a userspace poller.
#
# It also ALERTS when a run stays under the ceiling but exceeds what we predicted — that means the
# memory model is wrong, which is worth knowing even when nothing crashes.
#
#   bash tools/ram_watchdog.sh --expect 300 --kill 400 [--poll 5] [--floor 40]
#     --expect GB  what we predict this run needs; exceeding it is an ALERT, not a kill
#     --kill   GB  hard ceiling on trainer RSS; exceeding it SIGKILLs the trainer
#     --floor  GB  also kill if system MemAvailable drops below this (protects the host)
set -u
EXPECT=0; KILL=0; POLL=5; FLOOR=40
while [ $# -gt 0 ]; do
  case "$1" in
    --expect) EXPECT=$2; shift 2;; --kill) KILL=$2; shift 2;;
    --poll) POLL=$2; shift 2;;     --floor) FLOOR=$2; shift 2;;
    *) echo "unknown arg $1"; exit 2;;
  esac
done
[ "$KILL" -gt 0 ] || { echo "--kill GB is required"; exit 2; }
# CGROUP-AWARE. The original poller compared trainer RSS to a ceiling and MemAvailable from free(1)
# to a floor. Inside a k8s pod both are the wrong quantity: the kill is done by the cgroup against
# memory.max (576 GB here), and that counts PAGE CACHE, which never shows up in RSS, while free(1)
# reports the HOST's memory, not the pod's. The container was OOM-killed (exit 137) three times with
# this watchdog armed and silent. CG_MAX/CG_CUR below read the real numbers.
CG=$(awk -F: '{print $3}' /proc/self/cgroup 2>/dev/null | head -1)
CG_DIR="/sys/fs/cgroup${CG}"
[ -r "$CG_DIR/memory.max" ] || CG_DIR=""
cg_max () { [ -n "$CG_DIR" ] && cat "$CG_DIR/memory.max" 2>/dev/null | grep -v max || echo ""; }
cg_cur () { [ -n "$CG_DIR" ] && cat "$CG_DIR/memory.current" 2>/dev/null || echo ""; }
LOG=${WATCHDOG_LOG:-/tmp/ram_watchdog.log}
ALERT=${WATCHDOG_ALERT:-/tmp/ram_watchdog_alert.txt}
# Write our own PID so this watchdog can be stopped by PID. Do NOT stop it by pattern-matching its
# command line: any shell whose own arguments contain "ram_watchdog.sh" — including the very command
# doing the killing — matches too, and kills itself instead (RESULTS_SUMMARY §5; hit four times).
#   kill "$(cat "$PIDFILE")"
PIDFILE=${WATCHDOG_PIDFILE:-/tmp/ram_watchdog.pid}
echo $$ > "$PIDFILE"
trap 'rm -f "$PIDFILE"' EXIT
: > "$ALERT"
PEAK=0; ALERTED=0; SEEN=0
CGMAXB=$(cg_max); CGMAX_GB=0
[ -n "$CGMAXB" ] && CGMAX_GB=$((CGMAXB / 1073741824))
# Kill at 78% of the cgroup limit. The container was OOM-killed at memory.max three times while an
# RSS-based watchdog watched and stayed silent; the headroom has to absorb the save transient and
# dirty page cache that reclaim has not caught up with.
CGKILL_GB=$(( CGMAX_GB * 78 / 100 ))
if [ "$CGMAX_GB" -gt 0 ]; then
  echo "[watchdog] armed: cgroup limit ${CGMAX_GB}GB, cgroup kill at ${CGKILL_GB}GB; RSS expect ${EXPECT}GB kill ${KILL}GB; poll ${POLL}s" | tee -a "$LOG"
else
  echo "[watchdog] armed: NO cgroup limit visible (falling back to RSS only) expect ${EXPECT}GB kill ${KILL}GB poll ${POLL}s" | tee -a "$LOG"
fi

while true; do
  # match comm=="python" AND the args. `pgrep -f` also matches any SHELL whose command text contains
  # the pattern (RESULTS_SUMMARY §5) — including this script — so it must not be used here.
  PIDS=$(ps -eo pid=,comm=,args= | awk '$2=="python" && /e2e_qp_distill\.py --train/{print $1}')
  if [ -z "$PIDS" ]; then
    if [ "$SEEN" -eq 1 ]; then
      echo "[watchdog] $(date +%H:%M:%S) trainer exited. PEAK RSS ${PEAK}GB (expected ${EXPECT}GB)" | tee -a "$LOG"
      [ "$EXPECT" -gt 0 ] && [ "$PEAK" -gt "$EXPECT" ] && \
        echo "ALERT: peak RSS ${PEAK}GB exceeded the expected ${EXPECT}GB (did not hit the ${KILL}GB ceiling)" | tee -a "$ALERT" "$LOG"
      SEEN=0; PEAK=0; ALERTED=0
    fi
    sleep "$POLL"; continue
  fi
  SEEN=1
  RSS_KB=$(ps -o rss= -p $(echo $PIDS | tr ' ' ',') 2>/dev/null | awk '{s+=$1} END{print s+0}')
  RSS_GB=$((RSS_KB / 1024 / 1024))
  AVAIL_GB=$(awk '/^MemAvailable:/{printf "%d", $2/1024/1024}' /proc/meminfo)
  [ "$RSS_GB" -gt "$PEAK" ] && PEAK=$RSS_GB
  # THE CHECK THAT ACTUALLY MATTERS: cgroup usage vs memory.max. Counts page cache, which RSS misses
  # entirely and which is what the 52 GB-per-arm checkpoint writes were filling.
  CGCURB=$(cg_cur); CGCUR_GB=0
  [ -n "$CGCURB" ] && CGCUR_GB=$((CGCURB / 1073741824))
  if [ "$CGMAX_GB" -gt 0 ] && [ "$CGCUR_GB" -ge "$CGKILL_GB" ]; then
    echo "KILL: $(date +%H:%M:%S) cgroup ${CGCUR_GB}GB >= ${CGKILL_GB}GB (limit ${CGMAX_GB}GB) -> SIGKILL $PIDS" | tee -a "$ALERT" "$LOG"
    kill -9 $PIDS 2>/dev/null
    sync; sleep 20
  fi

  if [ "$EXPECT" -gt 0 ] && [ "$RSS_GB" -gt "$EXPECT" ] && [ "$ALERTED" -eq 0 ]; then
    ALERTED=1
    echo "ALERT: $(date +%H:%M:%S) trainer RSS ${RSS_GB}GB is ABOVE the expected ${EXPECT}GB (ceiling ${KILL}GB, cgroup ${CGCUR_GB}/${CGMAX_GB}GB)" | tee -a "$ALERT" "$LOG"
  fi
  if [ "$RSS_GB" -ge "$KILL" ] || [ "$AVAIL_GB" -le "$FLOOR" ]; then
    WHY="RSS ${RSS_GB}GB >= ceiling ${KILL}GB"
    [ "$AVAIL_GB" -le "$FLOOR" ] && WHY="MemAvailable ${AVAIL_GB}GB <= floor ${FLOOR}GB"
    echo "KILL: $(date +%H:%M:%S) $WHY -> SIGKILL $PIDS (peak was ${PEAK}GB)" | tee -a "$ALERT" "$LOG"
    kill -9 $PIDS 2>/dev/null
    sleep 20
  fi
  sleep "$POLL"
done
