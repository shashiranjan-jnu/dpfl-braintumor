"""Seed-level summary with t-based 95% intervals (decision D34).

Groups the finished runs of one or more grids by everything except the seed,
then reports mean, standard deviation and a 95% interval, mean +- t(0.975, n-1) * sd / sqrt(n),
for the main test metrics. Seed 0 (lr selection only, D30) is always excluded.

    python scripts/summarise.py --runs results/runs --grids configs/grids/week03_central.yaml \
        --out results/tables/week03_central_summary.csv --md results/tables/week03_central_summary.md
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.aggregate import collect  # noqa: E402
from scripts.run_grid import expand  # noqa: E402
from src.run import resolve, run_id  # noqa: E402

METRICS = {"test.balanced_accuracy": "bal_acc", "test.accuracy": "acc", "test.macro_f1": "macro_f1",
           "test.macro_auc_ovr": "auc", "test.ece": "ece", "test.sens_glioma": "sens_G",
           "test.sens_meningioma": "sens_M", "test.sens_notumor": "sens_N",
           "test.sens_pituitary": "sens_P", "test.tumour_missed_as_normal": "tumour_missed"}


def regime(r: pd.Series) -> str:
    wn = r.get("cfg.fl.weight_noise")
    if wn is not None and not pd.isna(wn) and float(wn) > 0:
        return f"weight noise {float(wn):g}"
    if not r.get("cfg.dp.enabled", False):
        return "no DP"
    unit = "patient " if r.get("cfg.dp.unit") == "unit" else ""
    nm = r.get("cfg.dp.noise_multiplier")
    if nm is not None and not pd.isna(nm) and float(nm) == 0.0:
        return f"{unit}clip only"
    return f"{unit}eps {float(r['cfg.dp.target_epsilon']):g}"


def summarise(df: pd.DataFrame, metrics: dict | None = None) -> pd.DataFrame:
    """Seed means grouped by mode, scenario, local client, regime, lr and, when the frame
    holds more than one, the FL algorithm (so FedAvg and SCAFFOLD runs are never pooled, D53)."""
    metrics = METRICS if metrics is None else metrics
    df = df[df["cfg.train.seed"] != 0].copy()
    df["regime"] = df.apply(regime, axis=1)
    if "cfg.fl.local_client" not in df:
        df["cfg.fl.local_client"] = np.nan
    keys = ["cfg.fl.mode", "cfg.data.scenario", "cfg.fl.local_client", "regime", "cfg.optim.lr"]
    names = ["mode", "scenario", "local_client", "regime", "lr"]
    if "cfg.fl.algorithm" in df and df["cfg.fl.algorithm"].nunique() > 1:
        keys, names = keys + ["cfg.fl.algorithm"], names + ["algorithm"]
    rows = []
    for k, g in df.groupby(keys, dropna=False, sort=False):
        row = dict(zip(names, k))
        row["n_seeds"] = len(g)
        row["epsilon"] = g["epsilon"].iloc[0]
        row["diverged"] = int(g["diverged"].sum())
        for col, name in metrics.items():
            if col not in g:
                continue
            v = g[col].astype(float).dropna().values
            n = len(v)
            m, sd = (v.mean(), v.std(ddof=1)) if n > 1 else (v.mean() if n else np.nan, np.nan)
            half = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
            row[name] = m
            row[f"{name}_sd"] = sd
            # every metric here lives in [0, 1], so the interval is clipped to that range
            row[f"{name}_lo"], row[f"{name}_hi"] = max(0.0, m - half), min(1.0, m + half)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    order = {"no DP": 0, "clip only": 1, "eps 64": 1.5, "eps 32": 1.6, "eps 16": 1.7, "eps 8": 2, "eps 3": 3, "eps 1": 4}
    out["_r"] = out.regime.map(lambda r: order.get(r.replace("patient ", ""), 9) + (10 if r.startswith("patient") else 0))
    out["_m"] = out["mode"].map({"central": 0, "federated": 1, "local": 2})
    return out.sort_values(["_m", "scenario", "local_client", "_r"]).drop(columns=["_r", "_m"]).reset_index(drop=True)


def to_markdown(s: pd.DataFrame) -> str:
    cols = ["mode", "scenario", "local_client", "regime", "lr", "n_seeds"]
    head = cols + ["bal_acc (95% CI)", "sens_M (95% CI)", "ece"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in s.iterrows():
        cell = lambda n: f"{r[n]:.3f} ({r[n + '_lo']:.3f} to {r[n + '_hi']:.3f})" if not pd.isna(r.get(n + "_lo")) else f"{r[n]:.3f}"
        lc = "" if pd.isna(r["local_client"]) else str(int(r["local_client"]))
        vals = [r["mode"], r["scenario"], lc, r["regime"], f"{r['lr']:g}", str(r["n_seeds"]),
                cell("bal_acc"), cell("sens_M"), f"{r['ece']:.3f}"]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--grids", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    ids = set()
    for g in a.grids:
        ids |= {run_id(resolve(c)) for c in expand(yaml.safe_load(open(g)))}
    s = summarise(collect(a.runs, ids))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    s.to_csv(a.out, index=False)
    md = to_markdown(s)
    if a.md:
        open(a.md, "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
