# A8 — GRPO finetuning of `gemma-3-1b-it` on GSM8K (team chmawa)

Part I of the Cambridge *Multi-Agent Systems & Agentic AI* practical: GRPO LoRA finetuning of
`google/gemma-3-1b-it` on GSM8K, on a single TPU v6e-1 with Tunix/JAX. Team **chmawa**
(sm3035, bc654, zw499) — runs jointly owned.

## Folder structure

| Path | What |
| --- | --- |
| `report/` | **the deliverable**: `report_part1.pdf` + its self-contained LaTeX source |
| `tpu-2026/` + `tpu-2026_our_changes.patch` | training code (detached upstream clone) + our changes as a single diff |
| `analysis/` | figure & CI tooling (`plot_report.py`, `paired_ci.py`, `export_wandb.py`) — run from the repo root |
| `training_logs/` | per-run W&B scalar exports (`tb_scalars_R0..R5.csv`) — sources for the training curves |
| `figures/`, `evals/` | report figures (F1–F4, S1–S3) · per-checkpoint eval dumps (n=64 and n=1319) |
| `scripts/` | TPU-VM orchestration (full-n eval sweep · checkpoint pull) |
| `vm_snapshot/`, `ckpts_archive/` | VM code/log archive · checkpoint manifest (the 46 GB of checkpoints are local-only) |

## Links

- **W&B runs** — project [`a8-grpo`](https://wandb.ai/sichengma0514-university-of-cambridge/a8-grpo)
- **Upstream training code** — [`borisbolliet/tpu-2026`](https://github.com/borisbolliet/tpu-2026) @ `77c5a67` (our changes: `tpu-2026_our_changes.patch`)
- **GRPO** — DeepSeekMath, [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- **DAPO** — [arXiv:2503.14476](https://arxiv.org/abs/2503.14476)
- **The N+ Implementation Details of RLHF with PPO** — Huang et al., [OpenReview](https://openreview.net/forum?id=kHO2ZTa8e3)
- **Model / dataset** — [`google/gemma-3-1b-it`](https://huggingface.co/google/gemma-3-1b-it) · [GSM8K (`openai/gsm8k`)](https://huggingface.co/datasets/openai/gsm8k)