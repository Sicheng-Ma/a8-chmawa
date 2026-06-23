#!/usr/bin/env python3
"""Paired bootstrap 95% CI for accuracy differences on the SAME n=1319 examples.
All n=1319 dumps are source=hf, seed 42 -> per_example_correct[i] is the same example
across models, so we resample example indices and compute acc(A)-acc(B) on each draw."""
import json, os, sys
import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else "evals"
SUF = "_n1319"

def load(tag):
    p = f"{D}/eval_{tag}{SUF}.json"
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None, None
    try:
        d = json.load(open(p))
    except Exception:
        return None, None
    return np.array(d["per_example_correct"]), d["acc"]

def paired(A, B, nb=10000, seed=0):
    a, aa = load(A); b, ba = load(B)
    if a is None or b is None:
        return None
    assert len(a) == len(b), f"{A}/{B} length mismatch {len(a)} vs {len(b)}"
    rng = np.random.default_rng(seed); n = len(a)
    idx = rng.integers(0, n, size=(nb, n))
    da = 100.0 * (a[idx].mean(1) - b[idx].mean(1))
    lo, hi = np.percentile(da, [2.5, 97.5])
    return aa, ba, aa - ba, lo, hi

COMPS = [
    ("R3b_final", "base"), ("R3b_best", "base"), ("R5_best", "base"), ("R5_final", "base"),
    ("R4_best", "base"), ("R4_final", "base"), ("R0_best", "base"), ("R0_final", "base"),
    ("R1_best", "base"), ("R1_final", "base"), ("R2_best", "base"), ("R2_final", "base"),
    ("R3b_final", "R4_final"), ("R3b_best", "R4_best"),    # G effect @ beta=0
    ("R5_best", "R0_best"),   ("R5_final", "R0_final"),    # G effect @ beta=.08
    ("R4_final", "R0_final"), ("R3b_final", "R5_final"),   # beta effect @ G2 / @ G8
]
print(f"{'comparison':<22}{'A%':>7}{'B%':>7}{'dpp':>8}   95% CI (paired)    sig")
print("-" * 70)
for A, B in COMPS:
    r = paired(A, B)
    if r is None:
        print(f"{A+' - '+B:<22}  (dump missing)"); continue
    aa, ba, d, lo, hi = r
    sig = "SIG" if (lo > 0 or hi < 0) else "ns"
    print(f"{A+' - '+B:<22}{aa:7.2f}{ba:7.2f}{d:+8.2f}   [{lo:+6.2f},{hi:+6.2f}]   {sig}")
