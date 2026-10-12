"""Seed-level summary of the Phase 4 audit (Steps 4.1 and 4.2).

Joins results/audit/<run_id>/ with the run configs, groups by everything except
the seed, and reports mean and t-based 95% intervals over seeds (D34), exactly as
scripts/summarise.py does for utility. Per-model patient-bootstrap intervals for
the explanation metrics stay in each xai_summary.json.

    python scripts/summarise_audit.py --results results --grids week03_central week03_fed_seeds_a \
        --out results/tables/week07_audit_summary.csv --md results/tables/week07_audit_summary.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.aggregate import collect, flatten  # noqa: E402
from scripts.run_audit import grid_ids  # noqa: E402
from scripts.summarise import summarise  # noqa: E402

AUDIT_METRICS = {
    "test.balanced_accuracy": "bal_acc",
    "mia.image.auc": "mia_auc", "mia.image.advantage": "mia_adv",
    "mia.image.tpr_at_fpr_0.01": "mia_tpr1", "mia.patient_mean.auc": "mia_pat_mean_auc",
    "mia.patient_max.auc": "mia_pat_max_auc",
    "probe.tumour_figshare_vs_sartaj.balanced_accuracy": "probe_tumour",
    "probe.notumour_sartaj_vs_br35h.balanced_accuracy": "probe_notumour",
}
for _m in ("gradcam", "gradcampp", "ig"):
    for _k in ("pointing", "iou15", "energy", "deletion", "insertion"):
        AUDIT_METRICS[f"xai.{_m}.correct.{_k}.mean"] = f"{_m}_{_k}"


def load_audit(results: str, ids: list[str]) -> pd.DataFrame:
    base = collect(os.path.join(results, "runs"), set(ids))
    rows = []
    base = base.copy()
    for rid in base.run_id:
        adir = os.path.join(results, "audit", rid)
        row = {"run_id": rid}
        for part, fn in (("mia", "mia.json"), ("probe", "probe.json"), ("xai", "xai_summary.json"),
                         ("rand", "randomisation.json")):
            p = os.path.join(adir, fn)
            if os.path.exists(p):
                row.update({f"{part}.{k}": v for k, v in flatten(json.load(open(p))).items()})
        rows.append(row)
    return base.merge(pd.DataFrame(rows), on="run_id", how="left") if rows else base


def to_markdown(s: pd.DataFrame) -> str:
    def cell(r, n):
        if n not in r or pd.isna(r[n]):
            return ""
        if pd.isna(r.get(n + "_lo")) or r.get("n_seeds", 0) < 2:
            return f"{r[n]:.3f}"
        return f"{r[n]:.3f} ({r[n + '_lo']:.3f} to {r[n + '_hi']:.3f})"
    head = ["mode", "scenario", "regime", "n", "bal_acc", "MIA AUC image", "MIA AUC patient (mean)",
            "probe tumour", "probe no-tumour", "Grad-CAM pointing", "IG energy"]
    names = ["bal_acc", "mia_auc", "mia_pat_mean_auc", "probe_tumour", "probe_notumour",
             "gradcam_pointing", "ig_energy"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for _, r in s.iterrows():
        lines.append("| " + " | ".join([str(r["mode"]), str(r["scenario"]), r["regime"], str(r["n_seeds"])]
                                       + [cell(r, n) for n in names]) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--grids", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", default=None)
    a = ap.parse_args()
    df = load_audit(a.results, grid_ids(a.grids))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out.replace(".csv", "_per_run.csv"), index=False)
    s = summarise(df, {k: v for k, v in AUDIT_METRICS.items() if k in df})
    s.to_csv(a.out, index=False)
    md = to_markdown(s)
    if a.md:
        open(a.md, "w").write(md)
    print(md)
    rand = [c for c in df if c.startswith("rand.")]
    if rand:
        print("randomisation check (Spearman with trained-model maps, seed 1 central):")
        print(df.dropna(subset=rand)[["run_id"] + rand].to_string(index=False))


if __name__ == "__main__":
    main()
