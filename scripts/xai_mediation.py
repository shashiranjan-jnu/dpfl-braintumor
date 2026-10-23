"""Explanation quality against accuracy and noise, on the clean test set (Step 5.2, D57). CPU only.

    python scripts/xai_mediation.py --results results --out results/tables/week07_xai_mediation

For every explained model, Grad-CAM top-15% IoU with the expert mask (correctly classified Figshare test
images, results/tables/week07_audit_summary_per_run.csv) and cross-seed consistency IoU
(week07_xai_border_per_run.csv) are joined to clean-set balanced accuracy and the noise multiplier sigma
(week07_ncc_sensitivity_per_run.csv). Reports the Pearson correlation with accuracy and an ordinary least
squares fit metric = b0 + b1 * accuracy + b2 * sigma with two-sided t-test p-values, which asks whether
sigma explains anything once accuracy is known.

Seeds of one configuration are not independent (consistency is measured against the configuration's own mean map),
so the same statistics are also computed on configuration means (D59), one point per (mode, scenario, regime).
Central configurations have five seeds and federated ones three.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coefficients and two-sided p-values, X includes the intercept column."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = resid @ resid / dof
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    p = 2 * stats.t.sf(np.abs(beta / se), dof)
    return beta, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "tables", "week07_xai_mediation"))
    a = ap.parse_args()
    T = os.path.join(a.results, "tables")
    acc = pd.read_csv(os.path.join(T, "week07_ncc_sensitivity_per_run.csv"))[["run_id", "clean.bal_acc", "sigma"]]
    iou = pd.read_csv(os.path.join(T, "week07_audit_summary_per_run.csv"))[
        ["run_id", "xai.gradcam.correct.iou15.mean"]].rename(columns={"xai.gradcam.correct.iou15.mean": "iou15"})
    keys = ["mode", "scenario", "regime"]
    con = pd.read_csv(os.path.join(T, "week07_xai_border_per_run.csv"))[keys + ["run_id", "consistency_iou"]]
    iou = iou.merge(con[keys + ["run_id"]], on="run_id")
    rows = []
    for level in ("model", "configuration"):
        for name, df, col in (("Grad-CAM top-15% IoU", iou, "iou15"), ("Cross-seed consistency IoU", con, "consistency_iou")):
            m = df.merge(acc, on="run_id").dropna(subset=[col, "clean.bal_acc", "sigma"])
            if level == "configuration":
                m = m.groupby(keys, as_index=False)[[col, "clean.bal_acc", "sigma"]].mean()
            r, pr = stats.pearsonr(m["clean.bal_acc"], m[col])
            X = np.column_stack([np.ones(len(m)), m["clean.bal_acc"], m["sigma"]])
            beta, p = ols(m[col].values, X)
            rows.append({"level": level, "metric": name, "n_models": len(m), "pearson_r_accuracy": r, "p_r": pr,
                         "b_accuracy": beta[1], "p_accuracy": p[1], "b_sigma": beta[2], "p_sigma": p[2]})
    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    out.to_csv(a.out + ".csv", index=False)
    lines = ["| Level | Metric | Points | r with accuracy | p | b accuracy | p | b sigma | p sigma |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['level']} | {r['metric']} | {r['n_models']} | {r['pearson_r_accuracy']:.3f} | {r['p_r']:.2e} | "
                     f"{r['b_accuracy']:.3f} | {r['p_accuracy']:.2e} | {r['b_sigma']:.4f} | {r['p_sigma']:.3f} |")
    md = "\n".join(lines) + "\n\nAccuracy is clean-set balanced accuracy (747 images); sigma is the noise multiplier (0 without noise).\n"
    open(a.out + ".md", "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
