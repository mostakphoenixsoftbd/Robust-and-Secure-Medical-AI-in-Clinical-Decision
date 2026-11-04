#!/usr/bin/env bash
# Run run_task.py with automatic restarts (the shared 16 GB machine can run out of memory).
# Restarts resume: trained weights are cached and attack results are cached in <task>/adv_cache/.
#   bash run_retry.sh ehr                 # full run, log robust_outputs/logs/ehr_full.log
#   bash run_retry.sh ecg --stage train   # log robust_outputs/logs/ecg_train.log
set -u
cd "$(dirname "$0")"
task=$1; shift
tag=full; [[ " $* " == *" --stage train "* ]] && tag=train
log=/e/physionet/robust_outputs/logs/${task}_${tag}.log
for attempt in $(seq 1 30); do
    echo "=== attempt $attempt $(date +%H:%M:%S)" >> "$log"
    .venv/Scripts/python.exe -u run_task.py "$task" "$@" >> "$log" 2>&1 && { echo "=== finished OK" >> "$log"; exit 0; }
    echo "=== exited with error; retry in 120 s" >> "$log"
    sleep 120
done
echo "=== giving up after 30 attempts" >> "$log"
exit 1
