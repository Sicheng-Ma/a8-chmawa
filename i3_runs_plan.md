# Part I.2–I.3 — Runs & metrics plan (team chmawa)

- Team **chmawa** (sm3035, bc654, zw499); all runs shared/jointly-owned, prose individual.
- Chip: one **v6e-1** ⇒ runs are **serial**; window closes **Mon 15 Jun 2026**.
- Full run ≈ 4 h 45 m; chip runs unattended ⇒ **~100 h left in window ≫ 5 full runs** (~25–40 h incl. R3). No compute shortcut needed.
- Refs: **GRPO** [arXiv:2402.03300] · **DAPO** [arXiv:2503.14476] · **N+ RLHF-PPO impl.** Huang et al. [OpenReview kHO2ZTa8e3].

## Decisions
- **Breadth**: 3 distinct modification axes (KL · reward · advantage), not depth-on-one.
- **Re-run baseline as R0** under final config (recover peak ckpt + instrumentation/config parity). Keep the existing shared full run for **I.1** only.
- **All runs = full `MAX_STEPS=3364`** (1 epoch, same horizon + LR schedule as baseline). Never lower `MAX_STEPS`: it rescales warmup + cosine decay (`WARMUP_STEPS=0.1·MAX_STEPS`, `decay_steps=MAX_STEPS`) ⇒ different optimisation = uncontrolled.
- **Early stopping = checkpoint *selection*, not shorter training**: report **final + best-val** checkpoint per model (same rule, e.g. max `rewards/eval/mean`). Handles the non-monotone curve, keeps the control.

## Runs (5; ≥3 satisfies brief)

| Run | Axis | Change (file · knob) | Hypothesis / tests | Ref · theory |
|---|---|---|---|---|
| **R0** | — | re-run default `config.py` + final ckpt/instrumentation | controlled reference; recover ~step-450 peak | GRPO |
| **R1** | KL | `BETA 0.08 → 0.3` | tighter leash ⇒ less drift/hacking, more held-out acc | trust region · I.4 Q2 |
| **R2** | reward | `rewards.py` + soft length penalty (DAPO Eq 13) | kill length blow-up + verbosity-hacking | DAPO §3.4 · I.3 |
| **R3** | advantage | `NUM_GENERATIONS 2 → 8` + leave-one-out (compute-normalised) | lower-variance Â, fewer dead groups, stabler | RLOO · I.4 Q1, I.2(b) |
| **R4** *(opt. 5th)* | KL | `BETA → 0` (KL off) | predicted **worse** drift — negative result | DAPO §2.3 · I.2/I.3(d) |

- **β-sweep** falls out free: R4/R0/R1 = β ∈ {0, 0.08, 0.3} → clean KL-drift story for I.3(d).
- R3 doubles as source for the Â-distribution / σ_r / degenerate-frac diagnostic (G=2 gives only ±1).
- Controlled: same data · seed 42 · eval split · **full 3364 steps**; only the one axis knob changes per run (leave `NUM_BATCHES/EPOCHS/FRACTION/ITERATIONS` at defaults). R3: same steps = **per-step-normalised** (G=8 = more rollout cost, same #updates); same-compute fallback = fewer steps, reported per-step.
- R4 alternatives if swapped: binary ±1 reward (DAPO §2.4) or LoRA-rank change.

---

## Plots & metrics (W&B)

- **Source:** ✅ already logged · ◐ derive from logged · ✚ needs instrumentation.
- W&B overlays runs on a shared panel natively → group **R0–R4**; mirror the 3 tiers below as W&B workspace **sections**.
- Report (I.1–I.3 ≤ 3 pp) fits ~4–6 figures ⇒ **Tier 1 = the report figures**; promote a Tier-2 plot only if space.

### Tier 1 — Main (the report figures; mandated)

| Fig | Shows | Source | Clause |
|---|---|---|---|
| **F1** reward + KL vs step (2 panels, shared x), **all runs overlaid** | reward trend + policy drift | ✅ `rewards/train/mean`, `actor/train/kl` | I.1(iii), I.3(b) |
| **F2** held-out accuracy: base / baseline / variants, **best-val + final, ± bootstrap CI** | does the variant actually help | eval + ✚ per-example CI dump | I.1(ii), I.3(a) |
| **F3** **KL vs task-accuracy** (over-optimisation frontier) | the core failure: drift ↔ acc loss | ◐ (KL ✅ × eval acc) | I.3(c) |

### Tier 2 — Supplementary (diagnostics; W&B + appendix, promote if space)

| Fig | Shows | Source | Why · ref |
|---|---|---|---|
| **S1** reward decomposition (4 terms) | format ↑ while correctness flat/↓ = hacking | ✅ `rewards/train/{match_format_exactly,_approximately,check_answer,check_numbers}` | I.2(c); DAPO §4.3 |
| **S2** response length + entropy vs step | length blow-up / entropy collapse | length ✅; entropy ✚ (proxy `perplexity` ✅) | I.3(c) alt; DAPO §3.1/3.3 |
| **S3** Â distribution / within-group σ_r / degenerate-frac | advantage variance, `K_eff`, dead groups | ✚ σ_r·Â at G>2; ◐ degenerate (`max==min`) | I.4 Q1, I.2(b); DAPO Fig 3b |
| **S4** length split by correct / incorrect | verbosity vs accuracy | ✚ | DAPO §3.3 |

### Tier 3 — Monitoring (live W&B health; per-run, usually not in report)

| Panel | Shows | Source |
|---|---|---|
| **H1** grad-norm + loss | instability / spikes (kill a diverging run early) | ✅ `actor/train/{grad_norm,loss}` |
| **H2** clip-frac / up-clipped prob | ≈0 at μ=1; informative if μ>1 / Clip-Higher | ✅ `actor/train/pg_clipfrac` |
| **H3** truncation frac (hit 768) | runaway length / degeneracy | ◐ `completions/*/max_length` vs cap |

> Not a plot: **I.3(d) discussion** (empirics → theory, incl. a contradicted prediction) — drafted from F1/F3/S1–S3.

---

## Operational (every run)
- **Keep dense ckpts** (enables best-val selection) — baseline lost its peak (`MAX_TO_KEEP=4`+`SAVE_INTERVAL=500` pruned all <2000). Set **`SAVE_INTERVAL_STEPS=100`, `MAX_TO_KEEP≥15`** (or keep-all).
- `CKPT_DIR` → **persistent** disk (not `/tmp`); export scalars to `tb_scalars_<run>.csv`.
- W&B entity = team `a8-grpo`; distinct run-id per run.

## Eval protocol
- GSM8K **test**, shuffle **seed 42**, **same N every model** (bump 64 → **≥256**); greedy; `--test-json` from `prepare_test_tfds.py` for an identical team split.
- Score every model at **final + best-val** checkpoint (same selection rule).
- **Uncertainty** = bootstrap CI over prompts (patch `evaluate.py` to dump per-example correctness) — no extra run needed.

## Instrumentation TODO
- `rewards.py`: soft length-penalty term (R2).
- train metrics (✚): entropy, within-group σ_r + Â stats, length-by-correctness.
- `config.py`: R3 leave-one-out flag (verify Tunix exposes it; else patch advantage calc).
- `evaluate.py`: per-example correctness dump → offline bootstrap CI.
- plotting: W&B export / overlay script over all `tb_scalars_*.csv`.

## Owners (serial chip)
- A: R0 + R1 + overlay/export script (F1).
- B: R2 + length term + reward-decomposition plot (S1).
- C: R3 + R4 + σ_r/entropy instrumentation (S2/S3) + `evaluate.py` bootstrap CI (F2).
