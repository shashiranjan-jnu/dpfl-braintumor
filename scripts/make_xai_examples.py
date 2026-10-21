"""Qualitative Grad-CAM examples under privacy (Step 5.2, D57). CPU only, needs the curated images.

    python scripts/make_xai_examples.py --data-dir <folder with brain_mri_curated_v1.npz> --results results \
        --out figures

Draws, for one Figshare test image per tumour class, the MRI slice with the expert mask outline and the
Grad-CAM map of central record-level models at seed 1 without DP, with clipping only, and at epsilon 8, 3
and 1. Maps are the stored 8 x 8 Grad-CAM maps of the audit (results/audit/<run_id>/cams8.npz, last
residual block, target = true class), bilinearly upsampled to 128 x 128 and scaled to [0, 1] per image.

Images are chosen by a rule fixed before plotting, so no example is hand-picked: per class, among clean
Figshare test images that the no-DP model classifies correctly, the image whose mask area is closest to the
class median. Writes figures/fig6_gradcam_examples.{pdf,png} and figures/figure_data/fig6_gradcam_examples.csv.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["mathtext.fontset"] = "stix"
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.ndimage import zoom  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.aggregate import collect  # noqa: E402
from scripts.run_audit import grid_ids  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = {0: "glioma", 1: "meningioma", 3: "pituitary"}
REGIMES = [("No DP", None), ("Clip only", 0.0), ("ε = 8", 8.0), ("ε = 3", 3.0), ("ε = 1", 1.0)]


def central_models(results: str) -> dict:
    """Run id of each central record-level regime at seed 1."""
    df = collect(os.path.join(results, "runs"), set(grid_ids(["week03_central"])))
    df = df[df["cfg.train.seed"] == 1]
    out = {}
    for name, eps in REGIMES:
        if name == "No DP":
            m = ~df["cfg.dp.enabled"].astype(bool)
        elif name == "Clip only":
            m = df["cfg.dp.enabled"].astype(bool) & (df["cfg.dp.noise_multiplier"] == 0)
        else:
            m = df["cfg.dp.target_epsilon"] == eps
        ids = df.loc[m, "run_id"].tolist()
        if len(ids) != 1:
            raise SystemExit(f"expected one run for {name}, found {len(ids)}")
        out[name] = ids[0]
    return out


def find(root: str, name: str) -> str:
    for r, _, fs in os.walk(root):
        if name in fs:
            return os.path.join(r, name)
    raise SystemExit(f"{name} not found under {root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--out", default=os.path.join(ROOT, "figures"))
    a = ap.parse_args()

    sp = pd.read_csv(os.path.join(ROOT, "data_manifest", "splits_v1.csv"))
    d = np.load(find(a.data_dir, "brain_mri_curated_v1.npz"), allow_pickle=False)
    if not (d["image_ids"].astype(str) == sp.image_id.astype(str).values).all():
        raise SystemExit("curated archive rows are not aligned with splits_v1.csv")
    images, masks = d["images"], d["masks"]
    drop = set(open(os.path.join(a.results, "ncc", "contaminated_test_ids.txt")).read().splitlines())

    runs = central_models(a.results)
    cams, preds = {}, {}
    for name, rid in runs.items():
        c = np.load(os.path.join(a.results, "audit", rid, "cams8.npz"))
        cams[name] = dict(zip(c["row"].tolist(), c["cam"].astype(np.float32)))
        per = pd.read_csv(os.path.join(a.results, "audit", rid, "xai_per_image.csv"))
        per = per[per.method == "gradcam"].set_index("row")
        preds[name] = per

    base = preds["No DP"]
    chosen = []
    for cls, cname in CLASSES.items():
        cand = base[(base.label == cls) & (base.pred == cls)]
        cand = cand[[str(sp.image_id.values[r]) not in drop for r in cand.index]]
        med = cand.mask_area.median()
        row = int((cand.mask_area - med).abs().sort_values(kind="stable").index[0])
        chosen.append((cname, row))

    fig, axes = plt.subplots(len(chosen), len(REGIMES) + 1, figsize=(7.0, 3.75), constrained_layout=True)
    recs = []
    names = ["glioma", "meningioma", "notumor", "pituitary"]
    for i, (cname, row) in enumerate(chosen):
        img = images[row].astype(np.float32) / 255.0
        mk = masks[row] > 0
        ax = axes[i, 0]
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.contour(mk, levels=[0.5], colors=["#00e5ff"], linewidths=0.8)
        ax.set_ylabel(cname, fontsize=9)
        if i == 0:
            ax.set_title("MRI, mask", fontsize=9)
        for j, (name, _) in enumerate(REGIMES, start=1):
            cam = zoom(cams[name][row], img.shape[0] / 8, order=1)
            cam = (cam - cam.min()) / max(float(cam.max() - cam.min()), 1e-12)
            ax = axes[i, j]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.imshow(cam, cmap="inferno", alpha=0.5, vmin=0, vmax=1)
            ax.contour(mk, levels=[0.5], colors=["#00e5ff"], linewidths=0.6)
            p = preds[name].loc[row]
            ax.text(0.03, 0.04, names[int(p.pred)], transform=ax.transAxes, fontsize=6.5,
                    color="white" if int(p.pred) == int(p.label) else "#ff6b6b")
            if i == 0:
                ax.set_title(name, fontsize=9)
            recs.append({"class": cname, "row": row, "image_id": sp.image_id.values[row],
                         "mask_area": float(p.mask_area), "regime": name, "run_id": runs[name],
                         "pred": names[int(p.pred)], "pointing": float(p.pointing), "iou15": float(p.iou15)})
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    os.makedirs(os.path.join(a.out, "figure_data"), exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(os.path.join(a.out, f"fig6_gradcam_examples.{ext}"), **kw)
    pd.DataFrame(recs).to_csv(os.path.join(a.out, "figure_data", "fig6_gradcam_examples.csv"), index=False,
                              float_format="%.4f")
    print(pd.DataFrame(recs).to_string(index=False))


if __name__ == "__main__":
    main()
