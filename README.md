# A8 — GRPO finetuning of `gemma-3-1b-it` on GSM8K (team chmawa)

Part I of the Cambridge *Multi-Agent Systems & Agentic AI* practical: GRPO LoRA finetuning of
`google/gemma-3-1b-it` on GSM8K, on a single TPU v6e-1 with Tunix/JAX. Team **chmawa**
(sm3035, bc654, zw499) — runs jointly owned.

## Folder structure

| Path | What |
| --- | --- |
| `report/` | **the deliverable**: `main.tex` (Parts I.1–I.4 + II) → `build/main.pdf`, `references.bib`, and `figures/` |
| `scripts/` | training & eval code (`config`, `data`, `rewards`, `model`, `train`, `evaluate`, `chat`) and the plotting scripts (`make_curves.py`, `make_i3_figs.py`) |
| `logs/i3_runs/` | per-run W&B scalar exports (`tb_scalars_R0..R5.csv`), per-run `launch_R*.sh`, and `evals/` (per-checkpoint eval dumps, n=64 and n=1319) |
| `analysis/` | `paired_ci.py` (paired bootstrap 95% CIs behind the I.3 results table), `export_wandb.py` (W&B → CSV export) — run from the repo root |
| `tpu-2026_our_changes.patch` | our changes to the upstream baseline as a single diff |
| `ckpts_archive/MANIFEST.md` | checkpoint manifest (the 46 GB of LoRA checkpoints are local-only) |
| `docs/` | `coursework.pdf` (assignment spec) |
| `i3_runs_plan.md` · `i3_results.md` · `DEVLOG.md` | experiment plan · results summary · development log |
| `tunix.ipynb` | the upstream baseline notebook the `scripts/` were decomposed from |
| `bootstrap.sh` · `create_tpu_env.sh` · `tpu-setup.md` · `requirements.txt` | TPU-VM environment setup |

## Links

- **W&B runs** — project [`a8-grpo`](https://wandb.ai/sichengma0514-university-of-cambridge/a8-grpo)
- **Upstream training code** — [`borisbolliet/tpu-2026`](https://github.com/borisbolliet/tpu-2026/tree/324abbe4b4e229ea812223856393547db4fbb53e) @ `324abbe` (our changes: `tpu-2026_our_changes.patch`)
- **GRPO** — DeepSeekMath, [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- **DAPO** — [arXiv:2503.14476](https://arxiv.org/abs/2503.14476)
- **The N+ Implementation Details of RLHF with PPO** — Huang et al., [OpenReview](https://openreview.net/forum?id=kHO2ZTa8e3)
- **Model / dataset** — [`google/gemma-3-1b-it`](https://huggingface.co/google/gemma-3-1b-it) · [GSM8K (`openai/gsm8k`)](https://huggingface.co/datasets/openai/gsm8k)