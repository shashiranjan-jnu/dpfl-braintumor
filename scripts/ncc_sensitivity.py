"""Sensitivity of reported results to the near-duplicates found by the NCC check (D50). CPU only.

    python scripts/ncc_sensitivity.py --results results --out results/tables/week07_ncc_sensitivity

A test image is contaminated if some image in another split reaches NCC >= 0.95
with it, plain or mirrored (results/ncc/curated_pairs_ncc090.csv). The list is
written to results/ncc/contaminated_test_ids.txt. For every run of the reported
grids, test metrics are recomputed from predictions.csv on the clean test images
only, and seed means with t-based intervals are reported next to the original.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.aggregate import collect  # noqa: E402
from scripts.run_audit import grid_ids  # noqa: E402
from scripts.summarise import summarise  # noqa: E402

warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROB = ["p_glioma", "p_meningioma", "p_notumor", "p_pituitary"]
GRIDS = ["week03_central", "week04_unit_central", "week04_unit_central_ext", "week03_fed_seeds_a",
         "week03_fed_seeds_b", "week04_unit_fed", "week05_client_count", "week05_client_count_lr",
         "week06_scaffold_size", "week07_wn_seeds"]
NEAR = 0.95


def contaminated(pairs: pd.DataFrame, thr: float = NEAR) -> set[str]:
    c = pairs[(~pairs.same_split) & (pairs.ncc_max >= thr)]
    return set(c.loc[c.split_i == "test", "image_id_i"]) | set(c.loc[c.split_j == "test", "image_id_j"])


def clean_metrics(pred: pd.DataFrame, drop: set[str]) -> dict:
    keep = ~pred.image_id.isin(drop).values
    y, p = pred.y.values, pred[PROB].values.argmax(1)
    yk, pk = y[keep], p[keep]
    return {"bal_acc": balanced_accuracy_score(yk, pk),
            "sens_notumor": recall_score(yk == 2, pk == 2),
            "tumour_sens": float((pk[yk != 2] != 2).mean()),
            "n_test": int(keep.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--grids", nargs="+", default=GRIDS)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "tables", "week07_ncc_sensitivity"))
    a = ap.parse_args()
    pairs = pd.read_csv(os.path.join(a.results, "ncc", "curated_pairs_ncc090.csv"))
    drop = contaminated(pairs)
    open(os.path.join(a.results, "ncc", "contaminated_test_ids.txt"), "w").write("\n".join(sorted(drop)) + "\n")
    grids = [g for g in a.grids if os.path.exists(os.path.join(ROOT, "configs", "grids", f"{g}.yaml"))]
    df = collect(os.path.join(a.results, "runs"), set(grid_ids(grids)))
    extra = []
    for rid in df.run_id:
        pred = pd.read_csv(os.path.join(a.results, "runs", rid, "predictions.csv"))
        full = clean_metrics(pred, set())
        cl = clean_metrics(pred, drop)
        extra.append({"run_id": rid, "full.sens_notumor": full["sens_notumor"],
                      **{f"clean.{k}": v for k, v in cl.items()}})
    df = df.merge(pd.DataFrame(extra), on="run_id")
    df["delta.bal_acc"] = df["clean.bal_acc"] - df["test.balanced_accuracy"]
    cols = {"test.balanced_accuracy": "bal_acc", "clean.bal_acc": "bal_acc_clean",
            "full.sens_notumor": "sens_N", "clean.sens_notumor": "sens_N_clean"}
    s = summarise(df, cols)
    d = df.groupby(["cfg.fl.mode", "cfg.data.scenario", "cfg.optim.lr"], dropna=False)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out + "_per_run.csv", index=False)
    s.to_csv(a.out + ".csv", index=False)
    lines = ["| mode | scenario | regime | lr | n | bal_acc | bal_acc clean | sens_N | sens_N clean |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        lines.append(f"| {r['mode']} | {r['scenario']} | {r['regime']} | {r['lr']:g} | {r['n_seeds']} | "
                     f"{r['bal_acc']:.3f} | {r['bal_acc_clean']:.3f} | {r['sens_N']:.3f} | {r['sens_N_clean']:.3f} |")
    md = "\n".join(lines) + (f"\n\nContaminated test images (NCC >= {NEAR} with another split): {len(drop)} of 813; "
                             f"clean test set {int(df['clean.n_test'].iloc[0])} images. Largest absolute change in "
                             f"balanced accuracy over all {len(df)} runs: {df['delta.bal_acc'].abs().max():.3f}.\n")
    open(a.out + ".md", "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
