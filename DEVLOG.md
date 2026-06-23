# Dev log

Running log of changes to the Part I report and supporting docs. Newest first.

## 2026-06-22

### Repo cleanup + reorganisation (branch `cleanup/repo-structure`)
- **Verified (Q2):** all our training-code changes live in `tpu-2026/` — exactly 5 modified files (`config/data/evaluate/rewards/train.py`) + 1 new (`prepare_test_tfds.py`), confirmed by diffing against a fresh upstream `borisbolliet/tpu-2026@77c5a67` clone — and `tpu-2026_our_changes.patch` captures them all (its blob hashes match the current files). The root analysis tooling is separate, original work, not a change to the reference code.
- **New folders:** `training_logs/` (the `tb_scalars_*.csv` exports) · `analysis/` (`plot_report.py`, `paired_ci.py`, `export_wandb.py` — run from the repo root) · `scripts/` (`run_sweep_n1319.sh`, `pull_all_ckpts.sh`).
- **Deleted (recoverable from `main`):** stale I.1 artifacts `i1_results.md` + `i1_{diagnostics,eval_accuracy,reward_kl}.png`; superseded `i4_theory.tex` (the report is the single I.4 source), `tb_scalars_partial.csv` (→ `full`), `pull_ckpts.sh` (→ `pull_all`); five ephemeral pollers `watch_{sweep,r1r2,r5,r5final,r5redo}.sh`.
- **Left untouched:** `report/ figures/ evals/ ckpts_archive/ vm_snapshot/`, the `*.tgz` archives (gitignored), and `tpu-2026/`.
- **Reference fixes:** rewrote the README layout table (now lists `report/` as the deliverable, the new folders, and drops the deleted files); repointed `analysis/plot_report.py` + `analysis/export_wandb.py` docstrings/defaults at `training_logs/`; updated the `analysis/…` tool paths in `i3_results.md`.
- **Follow-up:** deleted `tb_scalars_full.csv` too — it was the W&B export of the *dropped* `bdbugenj` baseline (KL peak 41.0, 45 tags; the run I.1 was first written around, before R0 superseded it), **not** one of R0–R5 and unused by any current figure (F1 was regenerated as R0–R5). Repointed `plot_report.py`'s no-arg default `{baseline: …full.csv}` → `{R0: training_logs/tb_scalars_R0.csv}`. `training_logs/` now holds exactly R0–R5.

## 2026-06-21 (later 7)

### Re-expanded I.4.3(iii-b) (the (later 6) trim went too far)
- (iii-b) has no brevity constraint (unlike (v)), so restored substance per request. Added: (1) a **displayed** identity for why the outer weight alone is insufficient — $\E_m[w\,s_\theta(R-\mu_m)]=\E_p[s_\theta(R-\mu_p)]+(\mu_p-\mu_m)\E_p[s_\theta]$ — with the residual $(\mu_p-\mu_m)\E_p[s_\theta]$ vanishing only when $\E_p[s_\theta]=0$, i.e. exactly at $\theta=\theta_{\mathrm{old}}$ (score identity), not after an update step; (2) the self-normalised baseline now sits in a display **beside** the IS $\tfrac1K\sum w_jr_j$ (no longer inline prose). Replaced the terse one-line population-invariance parenthetical with this fuller math; kept the boxed reweighted advantage. Clean build, doc **9 → 10 pp**.
- Follow-up (per request, since the question says "derive"): added the **unbiasedness derivation** of $\widehat\mu_p^{\mathrm{IS}}$ as one aligned display — $\E[\widehat\mu_p^{\mathrm{IS}}]=\E_m[wR]=\sum_a m_\alpha\tfrac{p}{m_\alpha}R=\sum_a pR=\mu_p$ (the weight undoing the change of measure) — and noted $\widehat\mu_p^{\mathrm{SN}}$ is consistent ($\E_m[w]=1$) but finite-$K$ biased. Still 10 pp.

## 2026-06-21 (later 6)

### Trimmed I.4.3(iii-b) to peer length
- Q3(iii-b) is 2 marks but had grown to **6 boxed equations + a full score-identity derivation**. Cut to a peer-sized answer (~2 short paras + 1 display + 1 box, like (v-a)/(v-b)): the direct "No" with the sampling-distribution-vs-baseline reason; a **one-line** version of the population baseline-invariance remark ($\E_p[s_\theta]=0$ ⇒ harmless for the exact gradient; matters only finite-sample / $\sigma_r$-norm / clipping / multi-step — kept at the user's request); the IS baseline $\widehat\mu_p=\frac1K\sum w_jr_j$ (with its unbiasedness) and the SN variant inline (both kept); and the single boxed reweighted advantage $\widetilde A_i=w_i(r_i-\widehat\mu_p)/\widehat\sigma_p$. Removed the $A_m=A_p+(\mu_p-\mu_m)$ algebra, the boxed $(\mu_p-\mu_m)\E_p[s_\theta]$ identity, the full score-identity derivation, the separate $\widehat\sigma_p$ box, and the duplicate $\widehat g^{\mathrm{corr}}$ box. Clean build, doc **10 → 9 pp**.

## 2026-06-21 (later 5)

### Tightened I.4.3(iv)
- Removed the "A complementary calculation … mixture-variance identity" block ($\operatorname{Var}_{m_\alpha}(R)=(1-\alpha)\sigma_p^2+\alpha\sigma_u^2+\alpha(1-\alpha)(\mu_p-\mu_u)^2$ and its increase/decrease conditions) — it describes the *raw reward* variance, not the *advantage-estimate* variance Q3(iv) actually asks about.
- Removed the two "Intuitively, …" sentences: each explained why $w\gtrless1$ (already given by $w>1\iff u<p$), not why the *variance* moves (which $\mathbb{E}_p[(w-1)A_p^2]\gtrless0$ already establishes), so they were redundant/misleading. (iv) now = boxed Cov difference + scalar version + the two conditions on $R$ with their sign arguments — exactly what's asked. Clean build, doc **11 → 10 pp**.

## 2026-06-21 (later 4)

### Trimmed I.4.3(iii-b)
- Q3(iii-b) is 2 marks ("derive the reweighted advantage"). Kept the **IS** and **SN** baseline/variance estimators ($\widehat\mu_p^{\mathrm{IS}}$, $\widehat\mu_p^{\mathrm{SN}}$, $(\widehat\sigma_p^{\mathrm{SN}})^2$), the corrected estimator $\widehat g_{\mathrm{mix}}^{\mathrm{corr}}$, and the absorbed-advantage reformulation $\widetilde A_i=w_i(r_i-\widehat\mu_p)/\widehat\sigma_p$ (the answer). Removed everything after it: the "$\widehat\mu_p$ may be either…" remark, the leave-one-out unbiased construction ($\widehat\mu_{p,-i}^{\mathrm{IS}}$, $\widehat g_{\mathrm{LOO}}$), and the trailing $\widehat\sigma_p$-heuristic caveat (~970 chars). Clean build, still 11 pp.

## 2026-06-21 (later 3)

### Deleted `i4_theory_revised.tex`; simplified I.4.3(v) to the coursework's brevity
- **Deleted `i4_theory_revised.tex`** (the standalone I.4 fragment) — no longer needed; `report/report_part1.tex` is now the single I.4 source, so future I.4 edits touch only the report.
- **Simplified I.4.3(v)** to match the coursework's explicit brevity (Q3(v) a/b are 2 marks each, asking for behaviour + "one sentence" and "explain briefly" + "one concrete fix"). (v-a): now just the boxed $w_i=c^T\to0$ behaviour + the single sentence on why $c<1$ is typical when $\alpha>0$ (responses drawn from $\pi_{\mathrm{mix}}$ ⇒ sampled tokens have $\pi_{\mathrm{mix}}>\pi_{\mathrm{old}}$); cut the per-token $p_t<u_t$ analysis and the sequence-level KL-drift digression. (v-b): brief "why unreliable" (ESS collapses, variance explodes) + a *single* concrete fix ($\alpha_T=O(1/T)\Rightarrow w_i=c_T^T\to e^{-\kappa}>0$) + its trade-off (weaker exploration); dropped the second (clipping) fix since the question asks for one.
- Removed the now-unused `\KL` preamble macro (its only use was the cut KL-drift line; `\E`/`\Var`/`\Cov` remain in use). Clean build; doc **12 → 11 pp** (the trim saved a page; I.1–I.3 untouched on pp.1–3).

## 2026-06-21 (later 2)

### I.4.3 replaced again — newer mixture-proposal solution (notation kept verbatim)
- The user supplied a further-revised I.4.3 (`problem3_solution.tex`); replaced the previous I.4.3 in **both** `report/report_part1.tex` and `i4_theory_revised.tex`. Per request, **kept the author's notation as-is** (no macro expansion this time): added `\newcommand`s for `\E`/`\Var`/`\Cov`/`\KL` to the report preamble instead. Same structural fit as before (drop standalone preamble/`\title`/`\maketitle`; `\section*{(i) …}`/`\subsection*{(a) …}` → bare `\paragraph{(i)}`…`\paragraph{(v-b)}`; (iii)/(v) wrapper headers dropped). Title kept as "I.4.3 Mixture proposal and importance weighting".
- **Repaired a copy corruption in the source**: a stray TAB had replaced the `\t` of `\to` on the (ii) line (`\widehat g_{\mathrm{mix}}<TAB>o`), which would have rendered as a stray "o"; restored to `\to` (now renders $\widehat g_{\mathrm{mix}}\to\widehat g_{\mathrm{GRPO}}$).
- **vs the previous (2026-06-21 later) version**: this one *carries the $\sigma$ standardisation through* the practical estimator ($\widehat A_i=(r_i-\bar r)/\sigma_r$, $\widehat g^{\mathrm{corr}}$) rather than deferring it (resolves the earlier "defers σ" flag); (v-a) uses the sequence-level KL identity $\mathbb{E}_{\pi_{\mathrm{mix}}}[\log(\pi_{\mathrm{old}}/\pi_{\mathrm{mix}})]=-\operatorname{KL}(\pi_{\mathrm{mix}}\|\pi_{\mathrm{old}})\le0$; (v-b) adds the explicit weight-clipping alternative $\widetilde w_i=\min\{w_i,C\}$ with its bias note. The earlier I.4.3 entry below is fully superseded.
- Removed `problem3_solution.tex` from the repo after integration (per request; a copy remains in `~/Downloads`). Clean build; doc now **12 pp** (I.4 on pp.4–12; I.1–I.3 untouched on pp.1–3).

## 2026-06-21 (later)

### I.4.3 replaced wholesale with the user's revised mixture-proposal solution
- Replaced all of I.4.3 (intro + parts (i)–(v-b)) in **both** `report/report_part1.tex` and `i4_theory_revised.tex` with the user-supplied standalone solution (`~/Downloads/problem3_grpo_solution.tex`). Done programmatically (`/tmp/splice_i43.py`) — no hand-transcription of the math.
- **Adaptations to fit the report** (content unchanged): dropped the standalone preamble/`\title`/`\maketitle`/`\end{document}`; remapped `\section*{(i) …}`/`\subsection*{(a) …}` → the report's bare `\paragraph{(i)}`…`\paragraph{(v-b)}` labels (dropping the "(iii)"/"(v)" wrapper headers, matching I.4.1/I.4.2 style); expanded the author's `\E`/`\Var`/`\Cov`/`\KL` macros to the report's spelled-out `\mathbb{E}`/`\operatorname{…}`. Kept the author's `\widehat`/`\widetilde` verbatim.
- **What's new vs the previous I.4.3** (the revision is more thorough): (iii-b) cleanly separates population-unbiasedness (score identity $\mathbb{E}_p[s_\theta]=0$) from finite-sample, and gives the raw IS baseline $\hat\mu_p^{\mathrm{IS}}$ (unbiased), the self-normalised $\hat\mu_p^{\mathrm{SN}}$ + weighted variance, and the leave-one-out $\hat\mu_{p,-i}^{\mathrm{IS}}$ with an explicit unbiasedness proof; (v-a) adds the KL/Jensen drift argument ($\mathbb{E}_{m_t}[\log(p_t/m_t)]=-\operatorname{KL}(m_t\|p_t)\le0$, with the $\mathbb{E}_{m_t}[p_t/m_t]=1$ caveat) for why log-ratios drift negative; (v-b) swaps the old per-token log-ratio clamp for a length-dependent mixing schedule $\alpha_T=O(1/T)\Rightarrow(c_T)^T\to e^{-\kappa}>0$. Note: it defers the $\sigma$ standardisation explicitly (treats it as a variance-control heuristic) rather than carrying $\sigma_{r,w}$ through.
- Clean build. Doc grows **10 → 11 pp** (I.4 now pp.4–11; I.1–I.3 untouched on pp.1–3, 3-page limit still holds).

## 2026-06-21

### I.4.2 Q1(iii) — dropped the inaccurate "joint distribution" wording
- Per request, fixed only the KL-vs-clip contrast in I.4.2(iii) (both files). The KL constraint was called "a joint distribution-level constraint coupling all actions" / "the joint KL divergence" — but $\pi_\theta(\cdot\mid q)$ is a *single* distribution over the one action, not a joint distribution over several variables. Reworded to: a constraint on the **whole distribution** $\pi_\theta(\cdot\mid q)$ — a single distribution over the action, whose divergence $\sum_a\pi_\theta\log(\pi_\theta/\pi_{\mathrm{old}})$ depends on **all** actions' probabilities (incl. unsampled), versus clipping acting only on the **single sampled action**. Both "joint" occurrences removed.
- Side effect: the cumulative I.4 expansions tip the doc **9 → 10 pp** (only the I.4.3 (v-b) tail spills onto p.10). I.1–I.3 untouched on pp.1–3, so the 3-page limit still holds; left as-is (not tightening unrequested parts).

## 2026-06-20

### I.4.2 Q1(i) — explicit small-$\eta$ expansion
- Per request, expanded only the small-$\eta$ step of I.4.2(i) (both files): added the intermediate algebra — first-order exponential expansion, the denominator $\to 1+\eta\,\mathbb{E}_{\pi_t}[A_t]+O(\eta^2)$, the $\tfrac{1}{1+x}=1-x+O(x^2)$ step, then multiply-out — leading to the (unchanged) result $\pi_{t+1}=\pi_t[1+\eta(A_t-\mathbb{E}_{\pi_t}[A_t])]+O(\eta^2)$. Rest of (i) and the closed-form softmax box untouched. Clean build, 9 pp.

### I.4.1 Q1(i) — explicit steps + conditioning convention
- Per request, expanded only Q1(i) (both `report/report_part1.tex` and `i4_theory_revised.tex`): added the skipped intermediate steps for $\mathbb{E}[X_i]=h_i\mathbb{E}[\hat A_i]=0$ and $\mathbb{E}[\hat g]=\tfrac1K\sum\mathbb{E}[X_i]=0$, and made the explanation state the convention $\mathbb{E}[\,\cdot\,]\equiv\mathbb{E}[\,\cdot\mid a_1,\dots,a_K]$ (all expectations conditional on the fixed actions). The $\mathbb{E}[\hat A_i]$ line and (ii)/(iii) left untouched. Clean build, 9 pp.

### I.4.1 Q1(iii) replaced with the user-supplied trace-based version
- Swapped the Q1(iii) answer (effective sample size) in **both** `report/report_part1.tex` and `i4_theory_revised.tex` for the version the user supplied verbatim ("use exactly this"). Recompiled clean — still **9 pp**; I.1–I.3 untouched on pp.1–3; the new (iii) sits in I.4.1 on pp.4–5. No overfull boxes / missing chars.
- **What changed vs the previous (directional) (iii)**: it now adopts the **lecture scalar-variance convention** $\operatorname{Var}_{\mathrm{tr}}(Z):=\operatorname{tr}(\operatorname{Cov}(Z))$, giving a single *scalar* $K_{\mathrm{eff}}=K(K-1)\lVert h_1\rVert^2/\sum_i\lVert h_i-\bar h\rVert^2$ in place of the per-direction $K_{\mathrm{eff}}(u)$. Retains the full matrix $\operatorname{Cov}(\hat g)=K^{-2}\sum_i(h_i-\bar h)(h_i-\bar h)^\top$, the iid sanity-check ($\operatorname{Var}_{\mathrm{tr,iid}}=\tfrac1K\operatorname{Var}_{\mathrm{tr}}(X_1)$), the $\Theta(K)$ growth result, the collinear $h_i=c_iv$ case, and the $h_i=h\Rightarrow K_{\mathrm{eff}}=\infty$ degenerate case.
- This **reverses** the 2026-06-19 note below ("drops the redundant trace-based K_eff, keeps the directional one"): the user's newest version deliberately uses the trace/scalar form to match the lecture convention.
- The `Let $\bar h=\dots$` opening at the top of I.4.1 is kept; the duplicate "Let $\bar h$" that led the pasted block was dropped to avoid repetition.

## 2026-06-19 (later)

### Self-contained `report/` folder
- Moved `report_part1.{tex,pdf}` into **`report/`** and copied the 3 figures it uses (`F1_reward_kl.pdf`, `F4_2x2_GxB.pdf`, `S3_group_degeneracy.pdf`) in with their report names; dropped `\graphicspath{{figures/}}` so they resolve co-located. `report/` rebuilds standalone (`tectonic report_part1.tex`). The `report/` figures are snapshots of `figures/` — re-copy if the plots are regenerated.

### Dropped the first-time baseline run (`bdbugenj`); I.1 now uses R0
- Verified R0 satisfies **I.1(i–iii)** under the default config (β=0.08, G=2; confirmed from `launch_R0.sh` — no knob overrides). Rewrote I.1 around R0 (W&B `8c2785ut`, ~5 h 05 m, 3364 steps).
- I.1(ii) reports **both** the `config.py`-default $n{=}64$ (base 45.3%, R0 42.2%→0.0%) and the full $n{=}1319$ for completeness (base 47.4%, R0 46.7%→6.2%), HF split / seed 42; **I.3 standardises on the full $n{=}1319$**. TFDS / the 51.6% number are dropped (they lived only in `bdbugenj`); R0 was evaluated on HF, so n=64 base is 45.3%.
- Tidy-up: trimmed baseline-patch (4) to just the HF-fallback justification (dropped the unused `prepare_test_tfds.py` / TFDS-materialisation clause) and removed the now-stale Table 1 caption note about an "I.1→I.3 source switch" (both are HF now). Only one TFDS mention remains (patch 4).
- Added baseline-patch (5): raised checkpoint retention (`SAVE_INTERVAL` 500→100, `MAX_TO_KEEP` 4→40) — persistence, not training; it's what lets us report best-val (the pruning problem `bdbugenj` had).
- **Regenerated F1 without the `bdbugenj` line** (now R0–R5; peak-KL annotation auto-updated 41→≈33 = R1's max).
- Removed the now-moot "baseline re-run" labels and the I.1↔I.3 28%-vs-47% reconciliation. `bdbugenj` no longer appears in the report. Clean build, 9 pp (I.1–I.3 on pp.1–3).

## 2026-06-19

### I.4 swapped to the revised version; report is now a single self-contained `.tex`
- Replaced the original I.4 with **`i4_theory_revised.tex`** and **inlined** it into `report_part1.tex` (no more `\input` — one self-contained file). Added the top-level `\section*{I.4 …}` heading (the revised file starts directly at subsection I.4.1). Clean build; still **9 pp** (I.1–I.3 on 1–3, revised I.4 on 4–9).
- `i4_theory.tex` (original) and `i4_theory_revised.tex` remain in the repo (no longer referenced by the report; kept for history).
- Gotcha logged: a first build mangled `\end{document}` → `␛nd{document}` because **zsh `echo` turns `\e` into an ESC byte (0x1B)**; rebuilt with `printf '%s'` (no escape processing).

### Revised vs original I.4 — the revised is better
- **Structure**: numbered subsections I.4.1/2/3 (vs one section + "1./2./3." paragraphs).
- **Concision** (I.4 penalises verbosity): ~1.2 KB shorter; drops the redundant trace-based `K_eff` (keeps the directional one, which is what the question asks for).
- **More rigorous**: Q1(ii) cleaner `Cov` via bilinearity + sharper "sign" statement; Q1(iii) cleaner `K_eff(u)` ratio + iid sanity-check + Θ(K); Q2(i) adds the small-η first-order expansion (shows η as a step-size); Q2(iii) explicit clip case-form; **Q3(iii-b)** the big upgrade — proves a deterministic baseline shift is unbiased (`E_p[s_θ]=0`), explains why weights alone don't fix group normalisation, and adds the **literally-unbiased leave-one-out Horvitz–Thompson baseline**; Q3(iv) matrix-form variance difference + explicit "no universal ordering".
- **No correctness regressions** (spot-checked the new derivations).

## 2026-06-18

### Rename
- `report.tex` / `report.pdf` → **`report_part1.tex`** / **`report_part1.pdf`** (old `report.pdf` removed); recompiled with `tectonic`.

### Self-review round 1 — 11 fixes
Read-through of Part I against the data; numbers = review item. All in `report_part1.tex` unless noted.

**Substantive**
1. **I.3 motivation** — "lowers advantage-estimate variance" → "lowers the variance of the **gradient estimate** $\hat g$". The $O(1/K)$ result is $\mathrm{Var}(\hat g)$; the per-sample advantage variance is $1-1/K$ (≈flat), so the old wording mis-named the quantity and contradicted I.4 Q1.
2. **I.3(d)** — was "the $\exp(\log\text{-ratio})$ KL blow-up that destroys the small-$K$ runs" (conflated two failure modes). Now distinguishes: **R0** = reward-hacking / over-optimisation collapse; **R1** = the KL-estimator instability (only with $\beta>0$).
3. **"Baseline" disambiguated** — `bdbugenj` (I.1 shared run; the grey "baseline" line in Fig 1) vs **R0 = "baseline re-run"** (I.3 controlled run: same config + dense checkpoints). Relabelled R0 in the 2×2 list and Table 1.
4. **28% vs 46.7% reconciled** — added a sentence: R0's dense checkpointing recovers the best-val that `bdbugenj`'s `MAX_TO_KEEP=4` pruning hid; also tagged I.1's 28.1% as the *best surviving* (pruned) checkpoint.

**Clarity / notation**
5. **K ≡ G stated once** — "group size $G$ (the $K$ of I.2(b) and the theory)".
6. **I.2(d) "locate" made specific** — "Tunix's RL learner" → "Tunix's `GRPOLearner` (module `tunix.rl`), in its clipped policy-gradient loss". *(Still confirm the exact function against the installed Tunix before the viva.)*
7. **TFDS→HF split justified** — Table 1 caption now notes HF and TFDS differ only in seed-42 ordering, so at full $n{=}1319$ the I.1→I.3 source switch is immaterial.

**Minor / adjacent**
8. **`i3_results.md`** (not the report) — R1 peak KL "~21" → **"~32.5"** in two places (β-paradox + F3 frontier), verified from `tb_scalars_R1.csv` (peak 32.5 @ step 1648). Confirmed Fig 1's "peak KL≈41" annotation is correct: it's baseline `bdbugenj`'s peak (41.0), the overall max.
9. **Jointly-owned line** — "six GRPO runs R0–R5" → "seven shared GRPO runs (`bdbugenj` and R0–R5)".
10. **I.2(c)** — "becomes $0/0$" → "has no signal (an $\varepsilon$ in the denominator averts a literal $0/0$)".
11. **I.3(d)** — "DAPO … small-$K$/long-output regime" → "DAPO … for long-output RL" (DAPO is not small-$K$).

### Layout
- Shrank Fig 1 (`F1_reward_kl`) to `0.86\linewidth` to absorb the added text and keep **I.1–I.3 within the 3-page limit** (verified: I.4 still starts on p.4). Clean build: 0 overfull, 0 missing-character.

### Verified correct (no change needed)
- Every Table 1 number and paired-CI claim matches the data; I.4's boxed results spot-check as correct.

### Still open (flagged, not done here)
- **GitLab**: port the repo to GitLab and swap the report link (submission requires GitLab, not GitHub).
- **Part II** (adaptive planning, ≤2 pp) not yet written — the final submission is a single PDF covering Parts I *and* II.
- Confirm the exact Tunix file/function for the PPO clip (item 6) before the viva.
