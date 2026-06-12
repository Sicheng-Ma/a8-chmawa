#!/bin/bash
cd ~/tpu-2026/scripts
set -a; source ~/.env; set +a
source ~/venvs/tunix/bin/activate
export RUN_NAME=R1 WANDB_NAME=R1 DATA_SOURCE=hf BETA=0.3
echo "=== R1 launch at $(date -u) | BETA=0.3 (tighter KL leash) ==="
python -u train.py 2>&1 | tee -a ~/grpo_runs/R1/train.log
echo "=== R1 train.py exited code ${PIPESTATUS[0]} at $(date -u) ==="
exec bash
