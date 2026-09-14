#!/bin/bash
cd /home/kasm-user/Documents/Model-to-Ternary
until [ "$(grep -c 'pp rank 0 mb=2 kl=' output_sweep/vaudit.log 2>/dev/null)" -ge 2 ]; do sleep 20; done
sleep 20
for p in $(ps -eo pid=,args= | grep '[e]2e_qp_distill.py --train' | awk '{print $1}'); do kill $p 2>/dev/null; done
