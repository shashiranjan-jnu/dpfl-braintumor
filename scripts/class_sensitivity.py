"""Per-class sensitivity on the clean test set (Step 5.3, D59). CPU only.

    python scripts/class_sensitivity.py --results results --out results/tables/week07_class_sensitivity

Recomputes, from results/runs/<run_id>/predictions.csv of every reported run (seeds 1 to 5), the recall of each
class on the 747 clean test images and the share of tumour images predicted as no tumour (a missed tumour).
Runs and their regimes come from week07_ncc_sensitivity_per_run.csv, so the selection matches the main tables;
the mean of the four recalls is checked against clean.bal_acc of that file. Seed means with t-based 95% intervals.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
PROB = [f"p_{c}" for c in CLASSES]


def regime(r) -> str | None:
    mode, scen, alg = r["cfg.fl.mode"], r["cfg.data.scenario"], r["cfg.fl.algorithm"]
    if scen != "natural3" or alg != "fedavg" or not pd.isna(r.get("cfg.fl.local_client")):
        return None
    if not pd.isna(r["cfg.fl.weight_noise"]):
        return f"FedAvg, weight noise s={r['cfg.fl.weight_noise']:g}" if r["cfg.fl.weight_noise"] < 0.1 else None
    head = "Central" if mode == "central" else "FedAvg natural"
    unit = ", patient" if r["cfg.dp.unit"] == "unit" else ""
    if not r["cfg.dp.enabled"]:
        return f"{head}, no DP"
    if r["cfg.dp.noise_multiplier"] == 0:
        return f"{head}{unit}, clip only"
    return f"{head}{unit}, eps {r['cfg.dp.target_epsilon']:g}"


def t_ci(v):
    v = np.asarray(v, float)
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), v.mean() - h, v.mean() + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "tables", "week07_class_sensitivity"))
    a = ap.parse_args()
    runs = pd.read_csv(os.path.join(a.results, "tables", "week07_ncc_sensitivity_per_run.csv"))
    runs = runs[runs["cfg.train.seed"] >= 1]
    drop = set(open(os.path.join(a.results, "ncc", "contaminated_test_ids.txt")).read().splitlines()) - {""}
    recs = []
    for _, r in runs.iterrows():
        reg = regime(r)
        if reg is None:
            continue
        pred = pd.read_csv(os.path.join(a.results, "runs", r.run_id, "predictions.csv"))
        pred = pred[~pred.image_id.isin(drop)]
        y, p = pred.y.values, pred[PROB].values.argmax(1)
        rec = {"regime": reg, "seed": int(r["cfg.train.seed"]), "run_id": r.run_id}
        for c, name in enumerate(CLASSES):
            rec[f"sens_{name}"] = float((p[y == c] == c).mean())
        rec["tumour_missed"] = float((p[y != 2] == 2).mean())
        rec["bal_acc"] = float(np.mean([rec[f"sens_{n}"] for n in CLASSES]))
        assert abs(rec["bal_acc"] - r["clean.bal_acc"]) < 1e-9, (r.run_id, rec["bal_acc"], r["clean.bal_acc"])
        recs.append(rec)
    df = pd.DataFrame(recs)
    n = {name: int((pred.y.values == c).sum()) for c, name in enumerate(CLASSES)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out + "_per_run.csv", index=False)
    cols = [f"sens_{c}" for c in CLASSES] + ["tumour_missed", "bal_acc"]
    rows = []
    for reg, g in df.groupby("regime", sort=False):
        row = {"regime": reg, "n_seeds": len(g)}
        for c in cols:
            row[c], row[c + "_lo"], row[c + "_hi"] = t_ci(g[c].values)
        rows.append(row)
    s = pd.DataFrame(rows)
    s.to_csv(a.out + ".csv", index=False)
    lines = ["| Regime | n | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for _, r in s.iterrows():
        lines.append(f"| {r.regime} | {r.n_seeds} | " +
                     " | ".join(f"{r[c]:.3f} ({r[c + '_lo']:.3f} to {r[c + '_hi']:.3f})" for c in cols) + " |")
    md = "\n".join(lines) + ("\n\nClean test images per class: " + ", ".join(f"{k} {v}" for k, v in n.items()) +
                             ". tumour_missed is the share of tumour images predicted as no tumour.\n")
    open(a.out + ".md", "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
