#!/bin/bash
cd ~/tpu-2026/scripts
set -a; source ~/.env; set +a
source ~/venvs/tunix/bin/activate
export RUN_NAME=R4 WANDB_NAME=R4 DATA_SOURCE=hf BETA=0.0
echo "=== R4 launch $(date -u) | BETA=0.0 (KL penalty OFF) ==="
python -u train.py 2>&1 | tee -a ~/grpo_runs/R4/train.log
echo "=== R4 exited ${PIPESTATUS[0]} $(date -u) ==="
exec bash
