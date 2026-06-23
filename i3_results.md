# Part I.3 — Improving on the baseline (results & diagnosis)

Team **chmawa**. Controlled study: **R0** (baseline re-run) + 5 variants spanning KL (β),
reward, and advantage (group size G) — the **β×G corners form a clean 2×2** — all
**full 3364 steps**, same data (HF GSM8K, shuffle seed 42), same eval, dense checkpoints
(every 100, kept) → **best-val + final** reported. W&B project `a8-grpo`.
**Headline eval = full GSM8K test (n=1319)**, greedy, exact-match; paired bootstrap CIs over the shared prompts.

## Runs
| Run | Axis | Change vs baseline | W&B run | Wall-clock |
|---|---|---|---|---|
| **R0** | — (baseline) | none: β=0.08, G=2, std-normalised advantage | `8c2785ut` | ~5h05m |
| **R1** | KL | β 0.08 → **0.3** (tighter leash) | `9p3kota8` | ~3h35m |
| **R2** | reward | + soft length penalty (char proxy, soft 1500/max 3000) | `xt3d5b0e` | ~5h50m |
| **R4** | KL | β → **0.0** (KL penalty OFF) | `082vyug5` | ~4h50m |
| **R3b** | advantage | **G 2 → 8** (grpo, β=0, full steps) — isolate group size vs R4 | `sgjawrsw` | ~7h30m |
| **R5** | KL×G | **G=8 _and_ β=0.08** — 2×2 corner: vs R0 isolates G (β=.08), vs R3b isolates β (G=8) | `qu0uammy` | ~7h20m |

R3a (G=8 + **RLOO**, β=0 — isolate the *estimator* vs R3b) — code prepped, **not run** (parked).

## Held-out GSM8K accuracy (I.3a) — full test, n=1319
Greedy, exact-match, ±95% bootstrap CI; last column = **paired** bootstrap Δ vs base (same prompts).

| Model | step | acc | 95% CI | format | Δ vs base (paired) |
|---|---|---|---|---|---|
| Base gemma-3-1b-it | — | 47.38% | 44.7–50.0 | 4.0% | — |
| R0 (G2, β.08) | best 700 | 46.70% | 44.0–49.4 | 90.7% | −0.7  ns |
| R0 | final 3364 | **6.22%** | 4.9–7.5 | 11.8% | **−41.2** [−44.0,−38.3] *(collapse)* |
| R4 (G2, β0) | best 1300 | 52.77% | 50.1–55.5 | 93.6% | **+5.4** [+2.7,+8.2] |
| R4 | final 3364 | 48.60% | 45.9–51.3 | 96.0% | +1.2  ns |
| **R3b (G8, β0)** | best 1000 | 55.65% | 53.0–58.2 | 95.4% | **+8.3** [+5.5,+11.1] |
| **R3b** | final 3364 | **57.24%** | 54.6–59.9 | 93.4% | **+9.9** [+7.1,+12.7] |
| **R5 (G8, β.08)** | best 1300 | 53.45% | 50.7–56.2 | 95.6% | **+6.1** [+3.3,+8.9] |
| **R5** | final 3364 | 55.57% | 52.8–58.2 | 91.3% | **+8.2** [+5.5,+10.9] |
| R1 (G2, β.3) | best 600 | 37.15% | 34.5–39.8 | 91.4% | **−10.2** [−13.1,−7.4] |
| R1 | final 3364 | **0.00%** | 0–0 | 0.0% | **−47.4** [−50.0,−44.7] *(collapse)* |
| R2 (G2, +len-pen) | best 400 | 32.83% | 30.3–35.3 | 65.4% | **−14.6** [−17.5,−11.6] |
| R2 | final 3364 | 40.64% | 38.0–43.3 | 74.5% | **−6.8** [−9.5,−3.9] |

*R1 (tighter KL leash β=.3) and R2 (length penalty), both G=2, **significantly underperform base** at n=1319 — KL-sweep / reward-axis variants, both failures, not part of the 2×2. R1's final is a total collapse (0/1319, token salad); R2 was mis-calibrated (thresholds too loose, weight too small).*

## The G×β 2×2 — group size changes the KL story (figure **F4**)
n=1319, best-val / final accuracy:

|  | **β = 0** | **β = 0.08** |
|---|---|---|
| **G = 2** | R4 — 52.8 / 48.6 · stable | R0 — 46.7 / **6.2** · *collapse* |
| **G = 8** | R3b — 55.7 / **57.2** · stable | R5 — 53.4 / 55.6 · stable |

**Paired significance (n=1319, the controlled contrasts):**
- **G effect @ β=0** (R3b − R4): **+2.9 pp** best, **+8.6 pp** final — both **SIG**.
- **G effect @ β=.08** (R5 − R0): **+6.8 pp** best, **+49.4 pp** final — both **SIG** (R0 collapsed).
- **β effect @ G=2** (R4 − R0, final): **+42.4 pp** **SIG** — at G=2, the KL penalty is catastrophic.
- **β effect @ G=8** (R3b − R5, final): **+1.7 pp** **ns** — at G=8, the KL penalty is irrelevant.

- **G=8 is the winning intervention.** Both G=8 runs beat base (R3b +8–10 pp, R5 +6–8 pp; CIs exclude 0) *and* beat their matched G=2 run. **R3b (G=8, β=0) final 57.2% is the top model.**
- **The collapse is the G=2/β>0 corner only.** β=0.08 is lethal at G=2 (R0 → 6.2%) but benign at G=8 (R5 55.6%). Same β, opposite fate; the only changed variable is group size.
- **At G=8, β doesn't matter** (R3b ≈ R5, ns). **Mechanism (inferred):** larger G ⇒ lower-variance group advantage ⇒ smoother updates ⇒ the policy never lurches into the `exp(log-ratio)` KL-estimator blow-up that destroys R0/R1 (see β-paradox). G=8 buys back the stability that makes the KL penalty harmless.

## Headline findings
- **Group size is the fix.** Going G=2 → 8 turns the over-optimising baseline into a model that significantly beats base — the single most effective change tried.
- **Over-optimisation is a G=2 phenomenon.** G=2 peaks at best-val then *degrades* (R4 52.8 → 48.6; R0 46.7 → 6.2 collapse). **G=8 keeps improving to the end** (R3b 55.7 → 57.2; R5 53.4 → 55.6; final > best). Larger G removes the very failure mode that defined the I.1 baseline.
- **Best-val selection matters most at G=2** — it rescues R4 to +5.4 (SIG) while R4's *final* is only +1.2 (ns), and recovers R0/R1's usable mid-training checkpoint from their 0%/6% finals. For G=8 it barely matters (final ≥ best).
- **Base reconciled at 47.4%** (full test). The n=64 45.3% (HF) vs 51.6% (TFDS) gap was split-ordering noise (HF/TFDS overlap only 3/64 at seed 42); at n=1319 both sources = the entire test set.
- **n=64 was misleading on the key question.** At n=64 (±12 pt CIs) "G=8 gave no accuracy gain"; the firm-up at n=1319 (±2–3 pt) shows G=8's gain is real and significant. *The wide-CI caveat was load-bearing.*

## Training dynamics (`figures/`)
- **F1 reward+KL:** R0 = textbook over-optimisation arc; **R1 collapses ~step 1700** (length→~20 tok, KL spike); R4 = highest reward, low KL, stable.
- **S2 length + entropy proxy:** R0 length blows up to ~630 then declines; **R2 length NOT controlled** (~400–500); R1 length collapses ~1700; R4 stable ~300. *Entropy proxy = `actor/train/perplexity` ≈ exp(per-token entropy)* — monotonic proxy; true vocab entropy not logged.
- **F3 KL-vs-eval-reward frontier:** early/low-KL = high held-out reward; R1 trails out to KL≈32 with **negative** reward (the collapse).
- **S1 reward decomposition (2×2 per run):** format (shaping) terms dominate early while correctness terms stay flat; R0/R1 components degrade into collapse; R2's `length_penalty` term is visible and small.

## The β paradox — strongest KL penalty, yet KL *explodes*
β is the KL-penalty weight (`loss = −J_clip + β·KL(πθ‖π_ref)`): R1 β=.3 strongest, R0 .08, R4 0. Naively bigger β → nearer ref → lower KL, yet **R1's KL exploded (peak ~32.5) while R4 (β=0) stayed low**. Why:
- β is a **soft** penalty, not a hard cap — makes drift costly, doesn't forbid it.
- The explosion is a **numerical instability**, not gradual drift: Tunix's KL estimator has an `exp(log-ratio)` term that saturates/overflows when the policy lurches (`kl_clamp_value`, off by default, bounds it). A **larger β multiplies the blown-up KL gradient** → bigger destabilising kick → collapse. Strong β *amplified* the instability.
- Partly a **symptom**: once R1 collapsed to degenerate ~20-tok outputs (~step 1700), measured KL-to-ref spikes.
- **R4 (β=0)** has no KL term to blow up → no feedback loop → stable.
- **R5 (β=0.08 _but_ G=8) did NOT blow up** — same β as R0, opposite outcome ⇒ the blow-up is **G-gated**: the larger group's lower-variance advantage keeps updates smooth enough to stay out of the unstable regime. (Confirmed at n=1319: R5 55.6% vs R0 6.2%.)

## Completion examples (qualitative)
Last logged training completion per run — **same prompt** (answer = 15), 1 sample/group from `train.log`:
- **R4 (β=0) — coherent:** real on-track math — *"Let W… W = 2/5 × 50 = 20 cups… O = 3/10 × 50 = 15 cups…"*.
- **R0 (β=0.08 → collapse) — reward-*hacked*:** perfect tags around vacuous filler — `<reasoning>`*"…discerning an intuitive comprehension of the intricate nuances…"*`</reasoning><answer>1</answer>` — learned the **format** reward, no real reasoning.
- **R1 (β=0.3 → 0%) — token salad:** *"walaワンピース amiss told clean polic Zelda… quinoa… Reiki…"* — pure multilingual noise (the KL-blowup collapse).

## Group degeneracy (G=2 masks it, G=8 reveals it) — figure **S3**
- **G=2 (R0/R4):** ≈ **0%** degenerate within-group; *masking*: dense shaping reward rarely ties 2 samples — 0% does **not** prove diversity.
- **G=8 (R3b) — windowed:** `degenerate_frac` ≈ **0.27 → 0.37 (steady)**; `σ_r` ≈ **1.78 → 1.47 (steady, not → 0)**. ~⅓ of groups all-8-equal (zero advantage), ⅔ keep signal — *partial, steady* degeneracy, not collapse. Real ceiling is a **plateau**: format saturates (~94% of max by ~step 500) while correctness climbs slowly — but unlike G=2 it **keeps climbing to the end** (final > best, above).
- **Implication:** the σ_r→0 / `K_eff` issue (I.4 Q1) and what **DAPO's Dynamic Sampling** targets. Yet net effect of G=8 is *positive* (+8–10 pp): more rollouts → better, lower-variance gradient signal outweighs the dead groups.

## Theory connection (I.3d)
- **Contradicted prediction:** "baseline over-optimises → needs a *tighter* KL leash" — false at G=2 (tightening β→R1 collapsed; removing β→R4 only matches base at final). **The 2×2 refines it:** the over-optimisation is driven by **advantage variance (small G)**, not by missing regularisation. Fix the variance (G=8) and the model improves monotonically *and* tolerates any β we tried.
- KL penalty viability is **gated by G**, not universal — echoes **DAPO** dropping the KL term for the small-G regime; **Dr.GRPO** length-bias (R2); classic **over-optimisation** (best-val ≫ final at G=2).

## Caveats / next
- **Firm-up done:** all reported numbers are full GSM8K test (n=1319) with paired bootstrap CIs; base reconciled (HF/TFDS moot at full n). R5's n=1319 dumps were re-run after a transient I/O glitch during the concurrent checkpoint archive — values verified (best 53.45, final 55.57).
- R1/R2 evaluated at n=1319: both **significantly below base** (R1 best −10.2 / final 0%; R2 best −14.6 / final −6.8 pp), confirming the n=64 read. Secondary to the 2×2.
- Entropy is a perplexity proxy; true per-token entropy needs instrumentation. R3a (G=8 + RLOO, estimator isolation) still parked.
- Single seed (rollout RNG = `PRNGKey(0)` for all runs → seed-matched control). A 2nd seed would test robustness of the G=8 gain.

## Artifacts
- `figures/` — F1/F2/F3/S1/S2/S3 + **F4** (β×G 2×2) (PNG+PDF) + `F2_accuracy_table.tex`
- `evals/` — n=64 + **n=1319** `--dump` JSONs (base + R0/R4/R3b/R5 × best/final; R1/R2 pending), per-example + CI
- `analysis/paired_ci.py` (paired bootstrap over shared prompts) · `analysis/plot_report.py` (CSV+evals→figures, `--eval-suffix _n1319`) · `analysis/export_wandb.py`
- `ckpts_archive/` — all 210 checkpoints (46 GB) + `MANIFEST.md`; `vm_snapshot/` — code + per-run launch scripts + logs
- W&B: R0 `8c2785ut`, R1 `9p3kota8`, R2 `xt3d5b0e`, R4 `082vyug5`, R3b `sgjawrsw`, R5 `qu0uammy` (project `a8-grpo`)
