# lib_timing.sh — per-stage wall-clock timing for the pipeline scripts (source this).
#
#   source lib_timing.sh
#   stage "Phase 1: rotation"      # before each stage; auto-closes the previous one w/ its duration
#   ... work ...
#   stage "Phase 2: calib"
#   ...
#   stage_end                      # after the last stage; prints its duration + TOTAL
#
# Every line is prefixed "[timing]" and records the START wall-clock of each stage + how long the
# previous one took, so a run log can be grepped to estimate how long a longer run will take:
#   grep '\[timing\]' run.log
_TIMING_T0=$(date +%s)
_TIMING_PREV_T=""
_TIMING_PREV_LABEL=""
_fmt_hms () { local s=$1; printf '%02d:%02d:%02d' $((s/3600)) $(((s%3600)/60)) $((s%60)); }

stage () {  # $1 = stage label
  local now; now=$(date +%s)
  if [ -n "$_TIMING_PREV_T" ]; then
    printf '[timing] done  %-34s took %s\n' "$_TIMING_PREV_LABEL" "$(_fmt_hms $((now - _TIMING_PREV_T)))"
  fi
  printf '[timing] START %-34s %s  (+%s into run)\n' \
         "$1" "$(date '+%Y-%m-%d %H:%M:%S')" "$(_fmt_hms $((now - _TIMING_T0)))"
  _TIMING_PREV_T=$now; _TIMING_PREV_LABEL="$1"
}

stage_end () {
  local now; now=$(date +%s)
  [ -n "$_TIMING_PREV_T" ] && \
    printf '[timing] done  %-34s took %s\n' "$_TIMING_PREV_LABEL" "$(_fmt_hms $((now - _TIMING_PREV_T)))"
  printf '[timing] ===== TOTAL %s  (run started %s) =====\n' \
         "$(_fmt_hms $((now - _TIMING_T0)))" "$(date -d "@$_TIMING_T0" '+%Y-%m-%d %H:%M:%S' 2>/dev/null)"
}
