# Checkpoint archive — team chmawa, Part I.3 GRPO runs

Pulled from the chmawa TPU VM (`~/grpo_runs/<run>/ckpts/actor/<step>`) on **2026-06-14**,
before the VM was deleted. Each dir is a full **Orbax checkpoint** (LoRA params + optimizer
state, ~232 MB). Base model: **google/gemma-3-1b-it**; LoRA via qwix.

**Provenance**
- Code that produced these: `../vm_snapshot/` — incl. the exact per-run `grpo_runs/<run>/launch_<run>.sh` (env knobs) and our patched scripts.
- Upstream: `github.com/borisbolliet/tpu-2026@77c5a67` + our diff (`config/data/evaluate/rewards/train.py` modified, `prepare_test_tfds.py` new).
- Accuracy + full analysis: `../i3_results.md`. Figures: `../figures/` (F4 = the β×G 2×2).

## Contents — ALL 210 checkpoints (35 steps × 6 runs)
Every saved step is archived: **1, 100, 200, …, 3300, 3364** (35/run) for R0, R1, R2, R4, R3b, R5,
laid out as `ckpts_archive/<run>/<step>/`. Total **46 GB**, pulled 2026-06-14 (`ok=198 skip=12 fail=0`).

The **12 reported models** (best-val + final per run):
| path | run | G | β | extra | step | selection | W&B |
|---|---|---|---|---|---|---|---|
| `R0/700`   | R0  | 2 | 0.08 | — | 700  | best-val | `8c2785ut` |
| `R0/3364`  | R0  | 2 | 0.08 | — | 3364 | final (collapsed → 0%) | `8c2785ut` |
| `R1/600`   | R1  | 2 | 0.30 | — | 600  | best-val | `9p3kota8` |
| `R1/3364`  | R1  | 2 | 0.30 | — | 3364 | final (collapsed → 0%) | `9p3kota8` |
| `R2/400`   | R2  | 2 | 0.08 | + length penalty | 400  | best-val | `xt3d5b0e` |
| `R2/3364`  | R2  | 2 | 0.08 | + length penalty | 3364 | final | `xt3d5b0e` |
| `R4/1300`  | R4  | 2 | 0.00 | — | 1300 | best-val | `082vyug5` |
| `R4/3364`  | R4  | 2 | 0.00 | — | 3364 | final | `082vyug5` |
| `R3b/1000` | R3b | 8 | 0.00 | — | 1000 | best-val | `sgjawrsw` |
| `R3b/3364` | R3b | 8 | 0.00 | — | 3364 | final | `sgjawrsw` |
| `R5/1300`  | R5  | 8 | 0.08 | — | 1300 | best-val | `qu0uammy` |
| `R5/3364`  | R5  | 8 | 0.08 | — | 3364 | final | `qu0uammy` |

best-val step = argmax training-time `rewards/eval/mean` (held-out val split) → nearest 100-step checkpoint.

**β×G 2×2** (figure F4): R4 = (G2, β0) · R0 = (G2, β.08, *collapses*) · R3b = (G8, β0) · R5 = (G8, β.08).
R1 (β-sweep) and R2 (reward axis) are the other two variants, both G=2.

## How to evaluate later
On a JAX/TPU host with the env (`../vm_snapshot/tpu-2026/requirements.txt`, `create_tpu_env.sh`)
and scripts (`../vm_snapshot/*.py`). The archive flattens `<run>/<step>/`, so point `--ckpt-dir`
at the run dir and pass the step:

    RUN_NAME=R5 python evaluate.py --step 1300 --ckpt-dir <path>/ckpts_archive/R5 \
        --dump eval_R5_best.json            # restores LoRA params (restore_only_lora_params)

`--step -1` evaluates the BASE model (no restore). Eval needs only the LoRA params; the
optimizer_state in each dir is for resuming training, not inference.
