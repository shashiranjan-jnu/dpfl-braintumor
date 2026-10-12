"""Per-model explanation audit on Figshare test tumour images with expert masks (Step 4.2).

For each image the explanation targets the true class. Localisation metrics
(pointing game, top-15% IoU, energy in mask) compare the map with the mask;
faithfulness metrics (deletion and insertion AUC) perturb the image in map
order. Summaries report the mean with a 95% interval from resampling the 40
test patients, on all images and on correctly classified images.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch

from src.xai import maps as M
from src.xai import metrics as X


def figshare_test_index(data) -> np.ndarray:
    sp = data.splits
    sel = (sp.split.values == "test") & (sp.source.values == "figshare") & (sp.label_id.values != 2)
    has_mask = data.masks.reshape(len(data.masks), -1).max(1) > 0
    return np.where(sel & has_mask)[0]


def black_value(data) -> float:
    return (0.0 - data.mean) / data.std


def explain(model, x, y, method: str, baseline: float, ig_steps: int = 32) -> torch.Tensor:
    if method == "ig":
        return M.integrated_gradients(model, x, y, baseline, ig_steps)
    return M.METHODS[method](model, x, y)


def audit(model, data, methods=("gradcam", "gradcampp", "ig"), batch: int = 32,
          del_steps: int = 20, ig_steps: int = 32) -> tuple[pd.DataFrame, dict]:
    idx = figshare_test_index(data)
    dev = data.x.device
    base = black_value(data)
    rows = []
    for s in range(0, len(idx), batch):
        b = idx[s:s + batch]
        x, y = data.x[b], data.y[b]
        mk = torch.as_tensor(data.masks[b], device=dev)
        with torch.no_grad():
            pred = model.eval()(x).argmax(1)
        for m in methods:
            mp = explain(model, x, y, m, base, ig_steps).detach()
            d, ins = X.deletion_insertion(model, x, y, mp, base, del_steps)
            vals = {"pointing": X.pointing_game(mp, mk), "iou15": X.topk_iou(mp, mk),
                    "energy": X.energy_in_mask(mp, mk), "deletion": d, "insertion": ins}
            area = (mk.flatten(1) > 0).float().mean(1)
            for j in range(len(b)):
                rows.append({"row": int(b[j]), "unit": int(data.splits.unit.values[b[j]]), "method": m,
                             "label": int(y[j]), "pred": int(pred[j]), "mask_area": float(area[j]),
                             **{k: float(v[j]) for k, v in vals.items()}})
    df = pd.DataFrame(rows)
    summ = {}
    for m, g in df.groupby("method"):
        summ[m] = {}
        for subset, gg in (("all", g), ("correct", g[g.label == g.pred])):
            if gg.empty:
                continue
            summ[m][subset] = {"n": int(len(gg))}
            for k in ("pointing", "iou15", "energy", "deletion", "insertion", "mask_area"):
                mean, lo, hi = X.patient_bootstrap_ci(gg[k].values, gg.unit.values)
                summ[m][subset][k] = {"mean": mean, "lo": lo, "hi": hi}
    return df, summ


def border_audit(model, data, methods=("gradcam", "gradcampp", "gradcam_l3"), batch: int = 64,
                 band: int = 8, cam_size: int = 8):
    """Skull-border analysis and layer-3 Grad-CAM localisation (D44 item 5, D45).

    For each method: share of saliency in the outer brain band (excluding tumour
    pixels), outside the brain, and the band's area share (chance level). Layer-3
    Grad-CAM also gets the mask metrics. Grad-CAM maps downsampled to cam_size x
    cam_size are returned for the cross-seed consistency metric.
    """
    idx = figshare_test_index(data)
    dev = data.x.device
    rows, cams = [], []
    for s in range(0, len(idx), batch):
        b = idx[s:s + batch]
        x, y = data.x[b], data.y[b]
        mk = torch.as_tensor(data.masks[b], device=dev) > 0
        raw = x * data.std + data.mean
        brain, rim, out = X.brain_regions(raw, band)
        rim = rim & ~mk
        with torch.no_grad():
            pred = model.eval()(x).argmax(1)
        for m in methods:
            mp = M.METHODS[m](model, x, y).detach()
            vals = {"border_energy": X.region_energy(mp, rim), "outside_energy": X.region_energy(mp, out),
                    "border_area": rim.flatten(1).float().mean(1) / brain.flatten(1).float().mean(1).clamp(min=1e-6)}
            if m == "gradcam_l3":
                vals.update({"pointing": X.pointing_game(mp, mk), "iou15": X.topk_iou(mp, mk),
                             "energy": X.energy_in_mask(mp, mk)})
            if m == "gradcam":
                cams.append(torch.nn.functional.adaptive_avg_pool2d(mp[:, None], cam_size)[:, 0].cpu())
            for j in range(len(b)):
                rows.append({"row": int(b[j]), "unit": int(data.splits.unit.values[b[j]]), "method": m,
                             "label": int(y[j]), "pred": int(pred[j]),
                             **{k: float(v[j]) for k, v in vals.items()}})
    df = pd.DataFrame(rows)
    summ = {}
    for m, g in df.groupby("method"):
        summ[m] = {}
        for subset, gg in (("all", g), ("correct", g[g.label == g.pred])):
            if gg.empty:
                continue
            summ[m][subset] = {"n": int(len(gg))}
            for k in [c for c in ("border_energy", "outside_energy", "border_area", "pointing", "iou15", "energy")
                      if c in gg and gg[c].notna().any()]:
                mean, lo, hi = X.patient_bootstrap_ci(gg[k].values, gg.unit.values, n_boot=1000)
                summ[m][subset][k] = {"mean": mean, "lo": lo, "hi": hi}
    cam = torch.cat(cams).numpy().astype(np.float16) if cams else np.zeros((0, cam_size, cam_size), np.float16)
    return summ, cam, idx


def randomisation_check(model, data, build_fn, n: int = 64, methods=("gradcam", "ig")) -> dict:
    """Spearman correlation between maps of the trained model and of randomised models.

    'full': every weight re-initialised; 'top': only the last block (cam layer) and
    head re-initialised. Low correlation means the maps depend on what was learnt
    (Adebayo et al., 2018).
    """
    idx = figshare_test_index(data)[:n]
    x, y = data.x[idx], data.y[idx]
    base = black_value(data)
    torch.manual_seed(1234)
    full = build_fn().to(x.device)
    top = copy.deepcopy(model)
    for mod in list(top.cam_layer.modules()) + [top.net.fc if hasattr(top, "net") else top.head]:
        if hasattr(mod, "reset_parameters"):
            mod.reset_parameters()
    out = {}
    for m in methods:
        ref = explain(model, x, y, m, base)
        out[m] = {name: float(X.spearman_rows(ref, explain(r, x, y, m, base)).mean())
                  for name, r in (("full", full), ("top", top))}
    return out
