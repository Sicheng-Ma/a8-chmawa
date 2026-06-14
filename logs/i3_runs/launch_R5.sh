#!/bin/bash
cd ~/tpu-2026/scripts
set -a; source ~/.env; set +a
source ~/venvs/tunix/bin/activate
export RUN_NAME=R5 WANDB_NAME=R5 DATA_SOURCE=hf NUM_GENERATIONS=8 BETA=0.08
echo "=== R5 launch $(date -u) | G=8 grpo beta=0.08 (vs R0 isolate G; vs R3b isolate beta; completes 2x2) ==="
python -u train.py 2>&1 | tee -a ~/grpo_runs/R5/train.log
echo "=== R5 exited ${PIPESTATUS[0]} $(date -u) ==="
exec bash
