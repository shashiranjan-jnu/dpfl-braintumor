"""Flatten results/runs/*/{config.yaml,metrics.json} into one CSV table.

    python scripts/aggregate.py --runs results/runs --out results/tables/all_runs.csv
    python scripts/aggregate.py --runs results/runs --grid configs/grids/week02_lr.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def flatten(d: dict, prefix: str = "") -> dict:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def collect(runs_dir: str, only_ids: set | None = None) -> pd.DataFrame:
    rows = []
    for rid in sorted(os.listdir(runs_dir)):
        mpath = os.path.join(runs_dir, rid, "metrics.json")
        if not os.path.exists(mpath) or (only_ids and rid not in only_ids):
            continue
        cfg = yaml.safe_load(open(os.path.join(runs_dir, rid, "config.yaml")))
        m = json.load(open(mpath))
        row = {"run_id": rid}
        row.update({f"cfg.{k}": v for k, v in flatten(cfg).items()})
        row.update({"epsilon": m["privacy"]["epsilon"], "sigma": m["privacy"]["sigma"],
                    "steps": m["privacy"]["steps"], "best_round": m["best_round"],
                    "train_acc": m["train_acc_best"], "train_seconds": m["train_seconds"],
                    "best_val_loss": min(h["val_loss"] for h in m["history"])})
        row.update({f"test.{k}": v for k, v in m["test_best"].items()})
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--grid", default=None, help="restrict to the runs of this grid file")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ids = None
    if a.grid:
        from scripts.run_grid import expand
        from src.run import resolve, run_id
        ids = {run_id(resolve(c)) for c in expand(yaml.safe_load(open(a.grid)))}
    df = collect(a.runs, ids)
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        df.to_csv(a.out, index=False)
    cols = [c for c in ["run_id", "cfg.model.name", "cfg.data.scenario", "cfg.fl.algorithm",
                        "cfg.optim.lr", "cfg.train.seed", "epsilon", "sigma", "best_round",
                        "best_val_loss", "train_acc", "test.accuracy", "test.balanced_accuracy",
                        "test.ece", "train_seconds"] if c in df]
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(df[cols].round(4).to_string(index=False) if len(df) else "no finished runs")


if __name__ == "__main__":
    main()
