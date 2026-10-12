"""Skull-border analysis, layer-3 Grad-CAM and cross-seed consistency (Step 4.2, D45). CPU only.

    python scripts/xai_tables.py --results results --out results/tables/week07_xai_border

Reads results/audit/<run_id>/border.json and cams8.npz for the explanation set.
Border energy is the share of Grad-CAM mass in the outer 8-pixel rim of the brain,
tumour pixels excluded; its chance level is the rim's area share of the brain.
Consistency follows Gupta et al. (2026) with seeds in place of clients: IoU between
each model's top-10% Grad-CAM pixels and those of the configuration's mean map.
Seed means with t-based 95% intervals (D34).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.mia_tables import label, summarise, to_md  # noqa: E402
from scripts.run_grid import expand  # noqa: E402
from src.run import resolve, run_id  # noqa: E402
from src.xai.metrics import consistency_iou  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRIDS = ["week03_central", "week04_unit_central", "week04_unit_central_ext", "week03_fed_seeds_a"]
COLS = {("gradcam", "border_energy"): "gc_border", ("gradcam", "border_area"): "border_chance",
        ("gradcam", "outside_energy"): "gc_outside", ("gradcam", "outside_area"): "outside_chance", ("gradcam_l3", "border_energy"): "l3_border",
        ("gradcam_l3", "pointing"): "l3_pointing", ("gradcam_l3", "iou15"): "l3_iou15",
        ("gradcam_l3", "energy"): "l3_energy"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--grids", nargs="+", default=GRIDS)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "tables", "week07_xai_border"))
    a = ap.parse_args()
    recs, cams = [], {}
    for g in a.grids:
        for c in expand(yaml.safe_load(open(os.path.join(ROOT, "configs", "grids", f"{g}.yaml")))):
            cfg = resolve(c)
            rid = run_id(cfg)
            p = os.path.join(a.results, "audit", rid, "border.json")
            if not os.path.exists(p):
                continue
            b = json.load(open(p))
            key = (cfg["fl"]["mode"], cfg["data"]["scenario"], label(cfg))
            r = {"mode": key[0], "scenario": key[1], "regime": key[2], "seed": cfg["train"]["seed"], "run_id": rid}
            for (m, k), name in COLS.items():
                r[name] = ((b.get(m) or {}).get("correct") or {}).get(k, {}).get("mean", np.nan)
            recs.append(r)
            cp = os.path.join(a.results, "audit", rid, "cams8.npz")
            if os.path.exists(cp):
                cams.setdefault(key, []).append((len(recs) - 1, np.load(cp)["cam"].astype(np.float32)))
    df = pd.DataFrame(recs)
    if df.empty:
        print("no border results yet")
        return
    df["consistency_iou"] = np.nan
    for key, items in cams.items():
        if len(items) < 2:
            continue
        iou = consistency_iou(np.stack([c for _, c in items]))
        for (i, _), v in zip(items, iou.mean(1)):
            df.loc[i, "consistency_iou"] = v
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out + "_per_run.csv", index=False)
    cols = list(COLS.values()) + ["consistency_iou"]
    s = summarise(df, ["mode", "scenario", "regime"], cols)
    s.to_csv(a.out + ".csv", index=False)
    md = to_md(s, ["mode", "scenario", "regime"], cols)
    open(a.out + ".md", "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
