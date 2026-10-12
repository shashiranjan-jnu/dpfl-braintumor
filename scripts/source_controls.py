"""Source-confound controls from saved test predictions (gap G5, Step 4.2). CPU only.

The no-tumour class comes from SARTAJ and Br35H, tumour classes mostly from
Figshare, so overall accuracy can partly reflect which collection an image came
from. For every run this reports
  * SARTAJ-only 4-class balanced accuracy (the one source holding all four classes),
  * Figshare-only 3-class balanced accuracy (glioma, meningioma, pituitary),
  * tumour against no-tumour sensitivity (all tumour images) and specificity,
    the latter separately for SARTAJ and Br35H no-tumour images.
A large gap between SARTAJ and Br35H specificity, or between pooled and
SARTAJ-only balanced accuracy, points to a source shortcut. Seed means with
t-based 95% intervals follow D34.

    python scripts/source_controls.py --results results --grids week03_central week03_fed_seeds_a \
        --out results/tables/week07_source_controls.csv --md results/tables/week07_source_controls.md
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.aggregate import collect  # noqa: E402
from scripts.run_audit import grid_ids  # noqa: E402
from scripts.summarise import summarise  # noqa: E402

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
PROB = ["p_glioma", "p_meningioma", "p_notumor", "p_pituitary"]
NOTUMOUR = 2
COLS = {"test.balanced_accuracy": "bal_acc", "src.sartaj_bal_acc4": "sartaj_bal_acc4",
        "src.figshare_bal_acc3": "figshare_bal_acc3", "src.tumour_sensitivity": "tumour_sens",
        "src.spec_sartaj": "spec_sartaj", "src.spec_br35h": "spec_br35h",
        "src.sartaj_tumour_sens": "sartaj_tumour_sens"}


def controls(pred: pd.DataFrame) -> dict:
    src = pred.image_id.str.split(":").str[0].values
    y = pred.y.values
    p = pred[PROB].values.argmax(1)
    tum, ptum = y != NOTUMOUR, p != NOTUMOUR
    out = {}
    s = src == "sartaj"
    if s.any():
        out["sartaj_bal_acc4"] = balanced_accuracy_score(y[s], p[s])
        out["sartaj_tumour_sens"] = float(ptum[s & tum].mean()) if (s & tum).any() else np.nan
    f = (src == "figshare") & tum
    if f.any():
        out["figshare_bal_acc3"] = balanced_accuracy_score(y[f], p[f])
    out["tumour_sensitivity"] = float(ptum[tum].mean())
    for name in ("sartaj", "br35h"):
        m = (src == name) & ~tum
        out[f"spec_{name}"] = float((~ptum[m]).mean()) if m.any() else np.nan
        out[f"n_notumour_{name}"] = int(m.sum())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--grids", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    runs = os.path.join(a.results, "runs")
    df = collect(runs, set(grid_ids(a.grids)))
    extra = [{"run_id": r, **{f"src.{k}": v for k, v in
                              controls(pd.read_csv(os.path.join(runs, r, "predictions.csv"))).items()}}
             for r in df.run_id]
    df = df.merge(pd.DataFrame(extra), on="run_id")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out.replace(".csv", "_per_run.csv"), index=False)
    s = summarise(df, COLS)
    s.to_csv(a.out, index=False)
    names = list(COLS.values())
    head = ["mode", "scenario", "regime", "n"] + names
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in s.iterrows():
        cells = [f"{r[n]:.3f} ({r[n + '_lo']:.3f} to {r[n + '_hi']:.3f})" if not pd.isna(r.get(n + "_lo"))
                 else f"{r[n]:.3f}" for n in names]
        lines.append("| " + " | ".join([str(r["mode"]), str(r["scenario"]), r["regime"], str(r["n_seeds"])]
                                       + cells) + " |")
    md = "\n".join(lines) + "\n"
    if a.md:
        open(a.md, "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
