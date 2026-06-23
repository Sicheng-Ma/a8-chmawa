#!/usr/bin/env python3
"""Export a W&B run's full history to long-format CSV [tag, step, value].

Output matches the training_logs/tb_scalars_<run>.csv format so the CSV drops straight into analysis/plot_report.py.
Uses scan_history() (every logged step, no sampling). Stdlib csv only (no pandas),
so it runs anywhere wandb is installed + authed (e.g. the TPU VM, which already is).

Run on the VM (authed) then scp the CSV down, or locally after `pip install wandb`:
  python analysis/export_wandb.py 8c2785ut training_logs/tb_scalars_R0.csv
  python analysis/export_wandb.py <run_id> [out.csv] [--entity E --project P]
"""
import argparse
import csv

import wandb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("out", nargs="?", default=None)
    ap.add_argument("--entity", default="sichengma0514-university-of-cambridge")
    ap.add_argument("--project", default="a8-grpo")
    a = ap.parse_args()

    run = wandb.Api().run(f"{a.entity}/{a.project}/{a.run_id}")
    print(f"run: name={run.name} id={a.run_id} state={run.state}")
    out = a.out or f"tb_scalars_{run.name or a.run_id}.csv"

    n, tags = 0, set()
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "step", "value"])
        for row in run.scan_history():
            step = row.get("_step")
            if step is None:
                continue
            for k, v in row.items():
                if k.startswith("_") or v is None:
                    continue
                try:
                    w.writerow([k, int(step), float(v)])
                    n += 1
                    tags.add(k)
                except (TypeError, ValueError):
                    pass  # skip non-scalar columns
    print(f"wrote {out}  ({n} rows, {len(tags)} tags)")


if __name__ == "__main__":
    main()
