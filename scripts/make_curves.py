"""Export the I.1(iii) baseline training curves from the TensorBoard event file.

Reads the GRPO run's scalar logs (written by train.py to TENSORBOARD_DIR) and
writes two vector figures into report/figures/:

  * baseline_reward.pdf  -- mean rollout reward r-bar vs GRPO step
  * baseline_kl.pdf      -- KL(pi_theta || pi_ref) vs GRPO step

"r-bar" is the per-rollout reward summed over the four reward functions in
rewards.py (Tunix logs this as rewards/train/score/mean); the maximum attainable
is 3 + 2.5 + 3 + 1.5 = 10.

Run:
    python make_curves.py            # uses TENSORBOARD_DIR from config.py
    python make_curves.py --logdir /path/to/eventdir --outdir ../report/figures
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from config import TENSORBOARD_DIR

REWARD_TAG = "rewards/train/score/mean"   # summed-over-reward-fns rollout reward
REWARD_EVAL_TAG = "rewards/eval/score/mean"
KL_TAG = "actor/train/kl"
MAX_REWARD = 10.0  # 3 (format exact) + 2.5 (format approx) + 3 (answer) + 1.5 (numbers)


def load(logdir):
    files = sorted(glob.glob(os.path.join(logdir, "events*")))
    if not files:
        raise FileNotFoundError(f"no event files under {logdir}")
    ea = EventAccumulator(files[-1], size_guidance={"scalars": 0})
    ea.Reload()
    return ea


def series(ea, tag):
    s = ea.Scalars(tag)
    return np.array([x.step for x in s]), np.array([x.value for x in s])


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
    ap.add_argument("--logdir", default=TENSORBOARD_DIR)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "..", "report", "figures"))
    ap.add_argument("--window", type=int, default=100, help="rolling-mean window (steps)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    ea = load(args.logdir)

    # ---- reward ----
    rstep, rval = series(ea, REWARD_TAG)
    fig, ax = plt.subplots(figsize=(4.2, 2.3))
    ax.plot(rstep, rval, color="tab:blue", alpha=0.20, lw=0.7)
    ax.plot(rstep, rolling(rval, args.window), color="tab:blue", lw=1.8,
            label=fr"train $\bar r$ ({args.window}-step mean)")
    try:
        estep, eval_ = series(ea, REWARD_EVAL_TAG)
        ax.plot(estep, eval_, color="tab:orange", lw=1.2, marker="o", ms=2.5,
                label="held-out (val) $\\bar r$")
    except KeyError:
        pass
    ax.axhline(MAX_REWARD, color="grey", ls="--", lw=0.8)
    ax.text(rstep[-1], MAX_REWARD, " max = 10", va="center", ha="right",
            fontsize=7, color="grey")
    ax.set_xlabel("GRPO step")
    ax.set_ylabel(r"mean rollout reward $\bar r$")
    ax.set_xlim(0, rstep[-1])
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = os.path.join(args.outdir, "baseline_reward.pdf")
    fig.savefig(out); plt.close(fig)
    print("wrote", os.path.normpath(out))

    # ---- KL (zoomed main axis + full-range inset for the spikes) ----
    kstep, kval = series(ea, KL_TAG)
    ksm = rolling(kval, args.window)
    fig, ax = plt.subplots(figsize=(4.2, 2.3))
    ax.plot(kstep, kval, color="tab:red", alpha=0.20, lw=0.7)
    ax.plot(kstep, ksm, color="tab:red", lw=1.8, label=fr"{args.window}-step mean")
    ax.set_xlabel("GRPO step")
    ax.set_ylabel(r"$\mathrm{KL}(\pi_\theta \,\|\, \pi_{\mathrm{ref}})$")
    ax.set_xlim(0, kstep[-1])
    ax.set_ylim(0, 2.0)   # the regime KL occupies >99% of the time
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    ax.grid(alpha=0.25)
    # inset: full range so the late-training spikes are visible
    axin = ax.inset_axes([0.50, 0.50, 0.47, 0.45])
    axin.plot(kstep, kval, color="tab:red", lw=0.6)
    axin.set_ylim(0, kval.max() * 1.08)
    axin.set_xlim(0, kstep[-1])
    axin.tick_params(labelsize=6)
    axin.set_title("full range", fontsize=6)
    kpeak = kval.max(); kpeak_step = kstep[int(kval.argmax())]
    axin.annotate(fr"peak $\approx${kpeak:.0f}", xy=(kpeak_step, kpeak),
                  xytext=(0.05, 0.8), textcoords="axes fraction", fontsize=6,
                  arrowprops=dict(arrowstyle="->", lw=0.6))
    fig.tight_layout()
    out = os.path.join(args.outdir, "baseline_kl.pdf")
    fig.savefig(out); plt.close(fig)
    print("wrote", os.path.normpath(out))

    # ---- summary numbers for the write-up ----
    print("\n--- summary ---")
    print(f"steps logged           : {len(rstep)} (first={rstep[0]}, last={rstep[-1]})")
    print(f"reward r-bar  final    : {rval[-1]:.3f}  (last-50 mean {rval[-50:].mean():.3f}, max-attainable {MAX_REWARD})")
    print(f"reward r-bar  peak      : {rval.max():.3f} at step {rstep[int(rval.argmax())]}")
    print(f"KL  final              : {kval[-1]:.3f}  (last-50 mean {kval[-50:].mean():.3f}, peak {kval.max():.3f})")


if __name__ == "__main__":
    main()
