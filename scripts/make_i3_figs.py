"""Export the I.3 comparison figures from the team's exported run scalars.

Reads the long-format TensorBoard CSV exports (tag,step,value) for the six I.3
runs from logs/i3_runs/ and writes three vector figures into report/figures/.
The runs form a 2x2 on (group size K, KL weight beta) plus two K=2 controls:

    R0  K=2, beta=0.08  (in-protocol baseline re-run)   R1  K=2, beta=0.30
    R4  K=2, beta=0.00                                   R2  K=2, +length penalty
    R5  K=8, beta=0.08                                   R3b K=8, beta=0.00

  * i3_reward.pdf  -- smoothed mean rollout reward; the 2x2 R0/R4/R5/R3b
                      (K=2 dashed, K=8 solid) so the K contrast is visible
  * i3_kl.pdf      -- smoothed KL(pi_theta || pi_ref) for the beta>0 runs
                      R0/R1/R5 (K-contrast at matched beta + the beta=0.3 drift)
  * i3_decomp.pdf  -- per-term reward decomposition, R0 (collapse) vs R3b
                      (the I.3(c) diagnostic): shaping vs correctness

Run:
    python make_i3_figs.py
    python make_i3_figs.py --logdir ../logs/i3_runs --outdir ../report/figures
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REWARD_TAG = "rewards/train/score/mean"
KL_TAG = "actor/train/kl"
ADVSTD_TAG = "diag/train/adv_std"            # logged for the K=8 runs only
DEGEN_TAG = "diag/train/degenerate_frac"     # ditto
TERM_TAGS = [  # per-reward-function means logged by the runs
    ("rewards/train/match_format_exactly", "format exact (max 3)", "tab:blue"),
    ("rewards/train/match_format_approximately", "format approx (max 2.5)", "tab:cyan"),
    ("rewards/train/check_answer", "check_answer (max 3)", "tab:red"),
    ("rewards/train/check_numbers", "check_numbers (max 1.5)", "tab:orange"),
]
MAX_REWARD = 10.0

# name -> display label, colour, best-val ckpt step (argmax held-out reward),
# group size K, line style (K=2 dashed, K=8 solid so the K contrast reads).
RUNS = {
    "R0":  dict(label=r"R0 ($K{=}2,\beta{=}.08$)", color="tab:blue",   best=700,  K=2, ls="--"),
    "R1":  dict(label=r"R1 ($K{=}2,\beta{=}.30$)", color="tab:purple", best=600,  K=2, ls="--"),
    "R2":  dict(label=r"R2 ($K{=}2$, +len)",       color="tab:gray",   best=400,  K=2, ls="--"),
    "R4":  dict(label=r"R4 ($K{=}2,\beta{=}0$)",   color="tab:orange", best=1300, K=2, ls="--"),
    "R5":  dict(label=r"R5 ($K{=}8,\beta{=}.08$)", color="tab:green",  best=1300, K=8, ls="-"),
    "R3b": dict(label=r"R3b ($K{=}8,\beta{=}0$)",  color="tab:red",    best=1000, K=8, ls="-"),
}
REWARD_RUNS = ["R0", "R4", "R5", "R3b"]   # the (K, beta) 2x2
KL_RUNS = ["R0", "R1", "R5"]              # beta>0: K-contrast (R0 vs R5) + beta=0.3 drift (R1)
ALL_KEYS = list(RUNS)


def load_csv(path):
    """tag -> (steps, values), sorted by step."""
    by_tag = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["tag"] == "tag":
                continue
            by_tag[row["tag"]].append((int(float(row["step"])), float(row["value"])))
    out = {}
    for tag, pairs in by_tag.items():
        pairs.sort()
        s, v = zip(*pairs)
        out[tag] = (np.array(s), np.array(v))
    return out


def rolling(y, w):
    """Centred moving average without zero-padding (no edge hook)."""
    n = len(y)
    if n < 2:
        return y
    h = max(1, w // 2)
    c = np.concatenate([[0.0], np.cumsum(y)])
    out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - h), min(n, i + h + 1)
        out[i] = (c[hi] - c[lo]) / (hi - lo)
    return out


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--logdir", default=os.path.join(here, "..", "logs", "i3_runs"))
    ap.add_argument("--outdir", default=os.path.join(here, "..", "report", "figures"))
    ap.add_argument("--window", type=int, default=100, help="rolling-mean window (steps)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    data = {key: load_csv(os.path.join(args.logdir, f"tb_scalars_{key}.csv"))
            for key in ALL_KEYS}

    # ---- reward overlay: the (K, beta) 2x2 ----
    fig, ax = plt.subplots(figsize=(4.2, 2.3))
    for key in REWARD_RUNS:
        r = RUNS[key]
        s, v = data[key][REWARD_TAG]
        sm = rolling(v, args.window)
        ax.plot(s, sm, color=r["color"], lw=1.6, ls=r["ls"], label=r["label"])
        i = int(np.searchsorted(s, r["best"]))
        ax.plot(s[i], sm[i], "o", color=r["color"], ms=4, mec="black", mew=0.4)
    ax.axhline(MAX_REWARD, color="grey", ls=":", lw=0.8)
    ax.text(3364, MAX_REWARD, " max = 10", va="center", ha="right", fontsize=7, color="grey")
    ax.set_xlabel("GRPO step")
    ax.set_ylabel(r"mean reward $\bar r$")
    ax.set_xlim(0, 3364)
    ax.set_ylim(-3.2, 10.6)
    ax.legend(fontsize=6.3, loc="lower left", framealpha=0.9, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(args.outdir, "i3_reward.pdf")
    fig.savefig(out); plt.close(fig)
    print("wrote", os.path.normpath(out))

    # ---- KL overlay (beta>0 runs; beta=0 runs skip the reference pass) ----
    fig, ax = plt.subplots(figsize=(4.2, 2.3))
    klmax = 0.0
    for key in KL_RUNS:
        r = RUNS[key]
        s, v = data[key][KL_TAG]
        ax.plot(s, v, color=r["color"], alpha=0.16, lw=0.6)
        ax.plot(s, rolling(v, args.window), color=r["color"], lw=1.6, ls=r["ls"], label=r["label"])
        klmax = max(klmax, v.max())
    ax.set_xlabel("GRPO step")
    ax.set_ylabel(r"$\mathrm{KL}(\pi_\theta \,\|\, \pi_{\mathrm{ref}})$")
    ax.set_xlim(0, 3364)
    ax.set_ylim(0, 6.0)
    ax.text(0.98, 0.04, r"$\beta{=}0$ runs (R4, R3b): KL not computed",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3, color="gray")
    ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25)
    if klmax > 3.0:  # full-range inset for late spikes, as in the I.1 figure
        axin = ax.inset_axes([0.55, 0.45, 0.42, 0.42])
        for key in KL_RUNS:
            r = RUNS[key]
            s, v = data[key][KL_TAG]
            axin.plot(s, v, color=r["color"], lw=0.5, ls=r["ls"])
        axin.set_xlim(0, 3364)
        axin.set_ylim(0, klmax * 1.08)
        axin.tick_params(labelsize=6)
        axin.set_title("full range", fontsize=6)
    fig.tight_layout()
    out = os.path.join(args.outdir, "i3_kl.pdf")
    fig.savefig(out); plt.close(fig)
    print("wrote", os.path.normpath(out))

    # ---- per-term reward decomposition: collapse (K=2) vs winner (K=8) ----
    panels = [("R0", r"R0 ($K{=}2,\beta{=}.08$): collapse"),
              ("R3b", r"R3b ($K{=}8,\beta{=}0$): stable")]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.1), sharey=True)
    for ax, (key, title) in zip(axes, panels):
        for tag, label, color in TERM_TAGS:
            s, v = data[key][tag]
            ax.plot(s, rolling(v, args.window), color=color, lw=1.4, label=label)
        ax.set_xlim(0, 3364)
        ax.set_xlabel("GRPO step")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("per-term reward")
    axes[0].legend(fontsize=6.5, loc="lower left", framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(args.outdir, "i3_decomp.pdf")
    fig.savefig(out); plt.close(fig)
    print("wrote", os.path.normpath(out))

    # ---- summary numbers for the write-up ----
    print("\n--- summary (window=%d) ---" % args.window)
    for key in ALL_KEYS:
        s, v = data[key][REWARD_TAG]
        sm = rolling(v, args.window)
        i = int(np.argmax(sm))
        line = (f"{key:4s} K={RUNS[key]['K']}: reward peak {sm[i]:.2f}@{s[i]}, "
                f"final(last200) {v[-200:].mean():.2f}")
        if KL_TAG in data[key] and data[key][KL_TAG][1].max() > 1e-6:
            ks, kv = data[key][KL_TAG]
            line += f" | KL peak {kv.max():.2f}@{ks[int(kv.argmax())]}, last200 {kv[-200:].mean():.3f}"
        print(line)
    # advantage std and degenerate fraction (K=8 runs only) -- the K_eff link
    print("\n--- diag (K=8 runs): adv_std should match sqrt((K-1)/K)=%.4f ---" % np.sqrt(7/8))
    for key in ALL_KEYS:
        if ADVSTD_TAG not in data[key]:
            continue
        _, av = data[key][ADVSTD_TAG]
        av = av[av > 1e-6]  # drop degenerate-batch zeros
        ds, dv = data[key][DEGEN_TAG]
        dsm = rolling(dv, args.window)
        print(f"{key:4s}: adv_std median {np.median(av):.3f} (mean {av.mean():.3f}); "
              f"degenerate_frac mean {dv.mean():.3f}, last200 {dv[-200:].mean():.3f}")
    # per-term last-200 means (reward hacking signature)
    print()
    for key in ("R0", "R4", "R5", "R3b"):
        terms = {tag.split('/')[-1]: data[key][tag] for tag, _, _ in TERM_TAGS if tag in data[key]}
        fin = {k: v[1][-200:].mean() for k, v in terms.items()}
        print(f"{key:4s} per-term last-200: " + ", ".join(f"{k}={x:.2f}" for k, x in fin.items()))
    for key in ("R0", "R4", "R5", "R3b"):
        s, v = data[key]["completions/train/mean_length"]
        print(f"{key:4s} completion mean length: start200 {v[:200].mean():.0f}, last200 {v[-200:].mean():.0f}, max {v.max():.0f}")


if __name__ == "__main__":
    main()
