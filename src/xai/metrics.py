"""Explanation metrics against expert tumour masks, and faithfulness curves (RQ3).

maps and masks are (B, H, W). Localisation metrics need masks; faithfulness
metrics need the model. All return one value per image.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def pointing_game(maps: torch.Tensor, masks: torch.Tensor, tol: int = 3) -> torch.Tensor:
    """1 if the arg-max pixel falls inside the mask dilated by `tol` pixels."""
    m = masks.float()[:, None]
    if tol > 0:
        m = F.max_pool2d(m, 2 * tol + 1, stride=1, padding=tol)
    m = m[:, 0].flatten(1)
    idx = maps.flatten(1).argmax(1)
    return m.gather(1, idx[:, None])[:, 0]


def topk_iou(maps: torch.Tensor, masks: torch.Tensor, frac: float = 0.15) -> torch.Tensor:
    """IoU between the top `frac` of map pixels and the mask."""
    flat = maps.flatten(1)
    k = max(1, int(round(frac * flat.shape[1])))
    thr = flat.topk(k, dim=1).values[:, -1:]
    hot = flat >= thr
    gt = masks.flatten(1) > 0
    inter = (hot & gt).sum(1).float()
    union = (hot | gt).sum(1).float()
    return inter / union.clamp(min=1)


def energy_in_mask(maps: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """Share of saliency mass inside the mask; chance level is the mask's area share."""
    flat, gt = maps.flatten(1), (masks.flatten(1) > 0).float()
    return (flat * gt).sum(1) / flat.sum(1).clamp(min=1e-12)


@torch.no_grad()
def deletion_insertion(model, x, target, maps, baseline: float, steps: int = 20):
    """Deletion and insertion AUC (Petsiuk et al., 2018) of the target-class probability.

    Pixels are ranked by the map. Deletion replaces the top pixels with the
    baseline value step by step; insertion starts from the baseline image and
    reveals them. AUC is the mean probability over steps 0..steps.
    Lower deletion AUC and higher insertion AUC mean a more faithful map.
    """
    model.eval()
    b, _, h, w = x.shape
    order = maps.flatten(1).argsort(1, descending=True)
    base = torch.full_like(x, baseline)
    dele, ins = [], []
    for s in range(steps + 1):
        n = int(round(s / steps * h * w))
        keep = torch.zeros(b, h * w, device=x.device)
        if n:
            keep.scatter_(1, order[:, :n], 1.0)
        keep = keep.view(b, 1, h, w)
        xd = x * (1 - keep) + base * keep
        xi = base * (1 - keep) + x * keep
        dele.append(F.softmax(model(xd), 1).gather(1, target[:, None])[:, 0])
        ins.append(F.softmax(model(xi), 1).gather(1, target[:, None])[:, 0])
    return torch.stack(dele, 1).mean(1), torch.stack(ins, 1).mean(1)


def spearman_rows(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Spearman rank correlation per row between two (B, N) tensors."""
    ra = a.flatten(1).argsort(1).argsort(1).float()
    rb = b.flatten(1).argsort(1).argsort(1).float()
    ra, rb = ra - ra.mean(1, keepdim=True), rb - rb.mean(1, keepdim=True)
    return (ra * rb).sum(1) / (ra.norm(dim=1) * rb.norm(dim=1)).clamp(min=1e-12)


def patient_bootstrap_ci(values: np.ndarray, groups: np.ndarray, n_boot: int = 2000, seed: int = 0):
    """Mean and 95% interval, resampling patients (groups) with replacement."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    per = {g: values[groups == g] for g in ug}
    means = []
    for _ in range(n_boot):
        pick = rng.choice(ug, len(ug), replace=True)
        means.append(np.concatenate([per[g] for g in pick]).mean())
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def brain_regions(x_raw: torch.Tensor, band: int = 8, thr: float = 0.05):
    """Brain mask, skull-border band and outside region from raw [0, 1] images (B, 1, H, W).

    The brain is every pixel brighter than `thr` after a 5-pixel closing; the band is
    the outer `band` pixels of that mask (the rim where the skull and scalp sit).
    Returns three boolean tensors (B, H, W).
    """
    m = (x_raw > thr).float()
    m = F.max_pool2d(m, 5, stride=1, padding=2)
    m = -F.max_pool2d(-m, 5, stride=1, padding=2)            # closing fills small holes
    inner = -F.max_pool2d(-m, 2 * band + 1, stride=1, padding=band)
    brain = m[:, 0] > 0
    return brain, brain & ~(inner[:, 0] > 0), ~brain


def region_energy(maps: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    """Share of saliency mass in a boolean region, per image."""
    flat = maps.flatten(1)
    return (flat * region.flatten(1).float()).sum(1) / flat.sum(1).clamp(min=1e-12)


def consistency_iou(maps: np.ndarray, q: float = 0.9) -> np.ndarray:
    """Gupta et al. (2026) consistency, adapted across seeds.

    maps (S, N, h, w) for S models of one configuration. For each image, each model's
    top (1 - q) pixels are compared with the top (1 - q) pixels of the mean map.
    Returns IoU (S, N).
    """
    mean = maps.mean(0)
    def top(a):
        flat = a.reshape(*a.shape[:-2], -1)
        thr = np.quantile(flat, q, axis=-1, keepdims=True)
        return flat >= thr
    ref, each = top(mean)[None], top(maps)
    inter = (each & ref).sum(-1)
    union = (each | ref).sum(-1)
    return inter / np.maximum(union, 1)
