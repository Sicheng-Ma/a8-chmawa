#!/bin/bash
cd ~/tpu-2026/scripts
set -a; source ~/.env; set +a
source ~/venvs/tunix/bin/activate
export RUN_NAME=R3b WANDB_NAME=R3b DATA_SOURCE=hf NUM_GENERATIONS=8 BETA=0.0
echo "=== R3b launch $(date -u) | G=8 grpo beta=0 (isolate group size vs R4) ==="
python -u train.py 2>&1 | tee -a ~/grpo_runs/R3b/train.log
echo "=== R3b exited ${PIPESTATUS[0]} $(date -u) ==="
exec bash
