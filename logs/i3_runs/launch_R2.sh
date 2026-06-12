#!/bin/bash
cd ~/tpu-2026/scripts
set -a; source ~/.env; set +a
source ~/venvs/tunix/bin/activate
export RUN_NAME=R2 WANDB_NAME=R2 DATA_SOURCE=hf LENGTH_PENALTY=1
echo "=== R2 launch $(date -u) | length penalty ON (soft 1500 / max 3000 chars) ==="
python -u train.py 2>&1 | tee -a ~/grpo_runs/R2/train.log
echo "=== R2 exited ${PIPESTATUS[0]} $(date -u) ==="
exec bash
