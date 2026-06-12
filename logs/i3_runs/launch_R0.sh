#!/bin/bash
cd ~/tpu-2026/scripts
set -a; source ~/.env; set +a
source ~/venvs/tunix/bin/activate
export RUN_NAME=R0 WANDB_NAME=R0 DATA_SOURCE=hf
echo "=== R0 launch at $(date -u) (defaults: full baseline reproduction) ==="
python -u train.py 2>&1 | tee -a ~/grpo_runs/R0/train.log
echo "=== train.py exited code ${PIPESTATUS[0]} at $(date -u) ==="
exec bash
