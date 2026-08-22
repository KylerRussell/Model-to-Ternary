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
echo "[watchdog] armed: expect ${EXPECT}GB, kill at ${KILL}GB, host floor ${FLOOR}GB, poll ${POLL}s" | tee -a "$LOG"

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

  if [ "$EXPECT" -gt 0 ] && [ "$RSS_GB" -gt "$EXPECT" ] && [ "$ALERTED" -eq 0 ]; then
    ALERTED=1
    echo "ALERT: $(date +%H:%M:%S) trainer RSS ${RSS_GB}GB is ABOVE the expected ${EXPECT}GB (ceiling ${KILL}GB, avail ${AVAIL_GB}GB)" | tee -a "$ALERT" "$LOG"
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
