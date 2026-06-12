"""Export the I.3 comparison figures from the team's exported run scalars.

Reads the long-format TensorBoard CSV exports (tag,step,value) for runs
R0 (baseline re-run), R1 (BETA=0.3), R2 (+length penalty), R4 (BETA=0)
from logs/i3_runs/ and writes three vector figures into report/figures/:

  * i3_reward.pdf  -- smoothed mean rollout reward, all four runs overlaid
  * i3_kl.pdf      -- smoothed KL(pi_theta || pi_ref) for R0/R1/R2
                      (R4 trains with beta=0, so Tunix never computes KL)
  * i3_decomp.pdf  -- per-term reward decomposition (the I.3(c) diagnostic)

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
TERM_TAGS = [  # per-reward-function means logged by the runs
    ("rewards/train/match_format_exactly", "format exact (max 3)", "tab:blue"),
    ("rewards/train/match_format_approximately", "format approx (max 2.5)", "tab:cyan"),
    ("rewards/train/check_answer", "check_answer (max 3)", "tab:red"),
    ("rewards/train/check_numbers", "check_numbers (max 1.5)", "tab:orange"),
]
MAX_REWARD = 10.0

RUNS = [  # name, csv suffix, colour, best-val ckpt step (argmax held-out reward)
    ("R0 ($\\beta{=}0.08$, baseline)", "R0", "tab:blue", 700),
    ("R1 ($\\beta{=}0.3$)", "R1", "tab:red", 600),
    ("R2 (+length penalty)", "R2", "tab:green", 400),
    ("R4 ($\\beta{=}0$)", "R4", "tab:orange", 1300),
]


def load_csv(path):
    """tag -> (steps, values), sorted by step."""
    by_tag = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
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
            for _, key, _, _ in RUNS}

    # ---- reward overlay ----
    fig, ax = plt.subplots(figsize=(4.2, 2.3))
    for label, key, color, best in RUNS:
        s, v = data[key][REWARD_TAG]
        sm = rolling(v, args.window)
        ax.plot(s, sm, color=color, lw=1.6, label=label)
        i = int(np.searchsorted(s, best))
        ax.plot(s[i], sm[i], "o", color=color, ms=4, mec="black", mew=0.4)
    ax.axhline(MAX_REWARD, color="grey", ls="--", lw=0.8)
    ax.text(3364, MAX_REWARD, " max = 10", va="center", ha="right", fontsize=7, color="grey")
    ax.set_xlabel("GRPO step")
    ax.set_ylabel(r"mean reward $\bar r$")
    ax.set_xlim(0, 3364)
    ax.set_ylim(-3.2, 10.6)
    ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(args.outdir, "i3_reward.pdf")
    fig.savefig(out); plt.close(fig)
    print("wrote", os.path.normpath(out))

    # ---- KL overlay (R4 has beta=0: Tunix skips the reference pass, log is 0) ----
    fig, ax = plt.subplots(figsize=(4.2, 2.3))
    klmax = 0.0
    for label, key, color, best in RUNS:
        if key == "R4":
            continue
        s, v = data[key][KL_TAG]
        ax.plot(s, v, color=color, alpha=0.18, lw=0.6)
        ax.plot(s, rolling(v, args.window), color=color, lw=1.6, label=label)
        klmax = max(klmax, v.max())
    ax.set_xlabel("GRPO step")
    ax.set_ylabel(r"$\mathrm{KL}(\pi_\theta \,\|\, \pi_{\mathrm{ref}})$")
    ax.set_xlim(0, 3364)
    ax.set_ylim(0, 6.0)
    ax.text(0.98, 0.04, "R4 ($\\beta{=}0$): KL unconstrained, not computed",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
            color="tab:orange")
    ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25)
    if klmax > 3.0:  # full-range inset for late spikes, as in the I.1 figure
        axin = ax.inset_axes([0.55, 0.45, 0.42, 0.42])
        for label, key, color, best in RUNS:
            if key == "R4":
                continue
            s, v = data[key][KL_TAG]
            axin.plot(s, v, color=color, lw=0.5)
        axin.set_xlim(0, 3364)
        axin.set_ylim(0, klmax * 1.08)
        axin.tick_params(labelsize=6)
        axin.set_title("full range", fontsize=6)
    fig.tight_layout()
    out = os.path.join(args.outdir, "i3_kl.pdf")
    fig.savefig(out); plt.close(fig)
    print("wrote", os.path.normpath(out))

    # ---- per-term reward decomposition: collapse run vs best run ----
    panels = [("R0", "R0 ($\\beta{=}0.08$): collapse"), ("R4", "R4 ($\\beta{=}0$): stable")]
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
    for label, key, color, best in RUNS:
        s, v = data[key][REWARD_TAG]
        sm = rolling(v, args.window)
        i = int(np.argmax(sm))
        line = (f"{key}: reward peak {sm[i]:.2f}@{s[i]}, final(last200) "
                f"{v[-200:].mean():.2f}")
        if KL_TAG in data[key]:
            ks, kv = data[key][KL_TAG]
            line += f" | KL peak {kv.max():.2f}@{ks[int(kv.argmax())]}, last200 {kv[-200:].mean():.3f}"
        es, ev = data[key].get("rewards/eval/score/mean", (None, None))
        if es is not None:
            line += f" | eval-reward argmax @{es[int(np.argmax(ev))]}"
        print(line)
    for key in ("R0", "R4", "R2", "R1"):
        terms = {tag.split('/')[-1]: data[key][tag] for tag, _, _ in TERM_TAGS if tag in data[key]}
        fin = {k: v[1][-200:].mean() for k, v in terms.items()}
        print(f"{key} per-term last-200 means: " +
              ", ".join(f"{k}={x:.2f}" for k, x in fin.items()))
    if "rewards/train/length_penalty" in data["R2"]:
        s, v = data["R2"]["rewards/train/length_penalty"]
        print(f"R2 length_penalty: mean {v.mean():.3f}, last200 {v[-200:].mean():.3f}, "
              f"min {v.min():.2f}, active on {(v < 0).mean()*100:.1f}% of steps")
    for key in ("R0", "R1", "R2", "R4"):
        s, v = data[key]["completions/train/mean_length"]
        print(f"{key} completion mean length: start(200) {v[:200].mean():.0f}, last200 {v[-200:].mean():.0f}, max {v.max():.0f}")


if __name__ == "__main__":
    main()
