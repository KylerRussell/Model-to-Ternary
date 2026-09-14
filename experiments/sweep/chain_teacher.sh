#!/bin/bash
set -u
cd /home/kasm-user/Documents/Model-to-Ternary
while pgrep -f "dry_ab.sh" >/dev/null 2>&1; do sleep 120; done
./experiments/sweep/dry_teacher.sh
