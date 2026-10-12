"""Curated dataset loading and client partitions.

The curated archive (brain_mri_curated_v1.npz) and splits_v1.csv are produced
by notebooks/kaggle/02-dedup-splits and verified by 03-verify-curated.
Row k of every array in the archive corresponds to row k of splits_v1.csv.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]  # label ids 0..3 (D13)
SCENARIOS = ["natural3", "iid_k10", "dir0.1_k10", "dir0.5_k10", "dir5.0_k10",
             "dir0.5_k3", "dir0.5_k5", "dir0.5_k15"]
# Patient-respecting Dirichlet partitions, computed in code from the split units
# (decision D33), named udir<alpha>_k<K>, e.g. udir0.5_k10. No unit crosses clients.
UDIR_SEED = 20261006


def udir_partition(units: np.ndarray, labels: np.ndarray, alpha: float, k: int,
                   seed: int = UDIR_SEED, min_images: int = 20, max_tries: int = 1000) -> np.ndarray:
    """Client id per row, assigning whole units with per-class Dirichlet proportions.

    Each unit takes its majority label. For every class the units of that class
    are shuffled and cut into K consecutive blocks whose unit counts follow
    p ~ Dirichlet(alpha * 1_K). Draws repeat until every client holds at least
    min_images images. The seed is fixed, so the partition is the same for every
    training seed, as for the stored partitions.
    """
    df = pd.DataFrame({"unit": units, "label": labels})
    maj = df.groupby("unit").label.agg(lambda v: v.value_counts().index[0])
    size = df.groupby("unit").size()
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        owner = {}
        for c in sorted(maj.unique()):
            us = np.array(sorted(maj.index[maj.values == c]))
            rng.shuffle(us)
            p = rng.dirichlet(alpha * np.ones(k))
            cuts = (np.cumsum(p) * len(us)).round().astype(int)[:-1]
            for j, block in enumerate(np.split(us, cuts)):
                owner.update({u: j for u in block})
        load = np.bincount([owner[u] for u in size.index], weights=size.values, minlength=k)
        if load.min() >= min_images:
            return np.array([owner[u] for u in units])
    raise ValueError(f"No udir partition with {min_images} images per client after {max_tries} draws")


def find_file(root: str, name: str) -> str:
    """Locate a file anywhere below root (Kaggle input mounts vary)."""
    if os.path.isfile(os.path.join(root, name)):
        return os.path.join(root, name)
    hits = glob.glob(os.path.join(root, "**", name), recursive=True)
    if not hits:
        raise FileNotFoundError(f"{name} not found below {root}")
    return sorted(hits)[0]


@dataclass
class Curated:
    x: torch.Tensor          # (N, 1, H, W) float32, standardised
    y: torch.Tensor          # (N,) int64
    masks: np.ndarray        # (N, H, W) uint8, Figshare tumour masks, zeros elsewhere
    splits: pd.DataFrame     # splits_v1.csv, row aligned with x
    mean: float
    std: float

    def idx(self, split: str) -> np.ndarray:
        return np.where(self.splits.split.values == split)[0]

    def clients(self, scenario: str) -> list[np.ndarray]:
        """Indices (into the full arrays) of each client's training images."""
        if scenario == "central":
            return [self.idx("train")]
        tr = self.idx("train")
        if scenario.startswith("udir"):
            alpha, k = scenario[4:].split("_k")
            assign = udir_partition(self.splits.unit.values[tr], self.splits.label_id.values[tr],
                                    float(alpha), int(k))
            return [tr[assign == j] for j in range(int(k))]
        col = self.splits[scenario]
        assign = col.values[tr]
        if pd.isna(assign).any():
            raise ValueError(f"Scenario {scenario} has unassigned training rows")
        assign = assign.astype(int)
        return [tr[assign == k] for k in range(assign.max() + 1)]

    def shadow_clients(self, scenario: str, seed: int, frac: float = 0.5) -> list[np.ndarray]:
        """Training sets of a LiRA shadow model (Step 4.1, D45).

        The pool is train plus test (validation still picks the round, as for the
        targets). Whole units are kept with probability `frac`, so a patient is
        either in or out, matching the patient-disjoint split of the targets.
        natural3 clients follow the source mapping of the stored partition.
        """
        keep = np.flatnonzero(shadow_members(self.splits, seed, frac))
        if scenario == "central":
            return [keep]
        if scenario != "natural3":
            raise ValueError("shadow models support central and natural3 only")
        tr = self.idx("train")
        col = self.splits["natural3"].values
        src = self.splits.source.values
        cmap = {s: int(pd.Series(col[tr][src[tr] == s]).mode()[0]) for s in np.unique(src[tr])}
        assign = np.array([cmap[s] for s in src[keep]])
        return [c for c in (keep[assign == j] for j in range(max(cmap.values()) + 1)) if len(c)]

    def units_of(self, idx: np.ndarray) -> list[np.ndarray]:
        """Group image indices by split unit, units in sorted order (unit-level DP)."""
        u = self.splits.unit.values[idx]
        order = np.argsort(u, kind="stable")
        idx, u = np.asarray(idx)[order], u[order]
        cut = np.flatnonzero(np.diff(u)) + 1
        return np.split(idx, cut)


def shadow_members(splits: pd.DataFrame, seed: int, frac: float = 0.5) -> np.ndarray:
    """Boolean mask over all rows: rows of the train+test pool whose unit is kept."""
    pool = np.isin(splits.split.values, ["train", "test"])
    units = np.unique(splits.unit.values[pool])
    rng = np.random.default_rng(1_000_003 + int(seed))
    kept = units[rng.random(len(units)) < frac]
    return pool & np.isin(splits.unit.values, kept)


def load_curated(data_dir: str, device: str | torch.device = "cpu",
                 npz_name: str = "brain_mri_curated_v1.npz",
                 splits_name: str = "splits_v1.csv") -> Curated:
    d = np.load(find_file(data_dir, npz_name), allow_pickle=False)
    sp = pd.read_csv(find_file(data_dir, splits_name))
    ids = d["image_ids"].astype(str)
    if len(ids) != len(sp) or not (ids == np.asarray(sp.image_id.tolist(), dtype=str)).all():
        raise ValueError("Archive rows are not aligned with splits_v1.csv")
    labels = d["labels"].astype(np.int64)
    if not (labels == sp.label_id.values).all():
        raise ValueError("Archive labels disagree with splits_v1.csv")

    imgs = d["images"].astype(np.float32) / 255.0
    tr = sp.split.values == "train"
    # Normalisation constants from the training split only, so validation and
    # test pixels never influence preprocessing.
    mean, std = float(imgs[tr].mean()), float(imgs[tr].std())
    x = torch.from_numpy((imgs - mean) / std).unsqueeze(1).to(device)
    y = torch.from_numpy(labels).to(device)
    return Curated(x=x, y=y, masks=d["masks"], splits=sp, mean=mean, std=std)


def make_synthetic(n: int = 160, size: int = 128, seed: int = 0,
                   unit_size: int = 1) -> tuple[dict, pd.DataFrame]:
    """Tiny fake archive with the real schema, for CPU tests and smoke runs."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 4, n)
    imgs = rng.integers(0, 60, (n, size, size)).astype(np.uint8)
    boxes = []
    for k, c in enumerate(labels):           # class-dependent square size, learnable by a GAP model
        h = 8 + 14 * c
        r0, c0 = rng.integers(0, size - h, 2)
        imgs[k, r0:r0 + h, c0:c0 + h] = 200
        boxes.append((r0, c0, h))
    masks = np.zeros_like(imgs)
    split = rng.choice(["train", "val", "test"], n, p=[0.6, 0.2, 0.2])
    source = rng.choice(["figshare", "sartaj", "br35h"], n)
    for k, (r0, c0, h) in enumerate(boxes):  # Figshare tumour images carry a mask of their square
        if source[k] == "figshare" and labels[k] != 2:
            masks[k, r0:r0 + h, c0:c0 + h] = 1
    sp = pd.DataFrame({
        "image_id": [f"syn:{k}" for k in range(n)], "source": source,
        "relpath": [f"{k}.png" for k in range(n)], "label": [CLASSES[c] for c in labels],
        "label_id": labels, "patient_id": "", "unit": np.arange(n) // unit_size, "split": split})
    tr = np.where(split == "train")[0]
    for k, s in enumerate(SCENARIOS):
        col = np.full(n, np.nan)
        nc = {"natural3": 3, "dir0.5_k3": 3, "dir0.5_k5": 5, "dir0.5_k15": 15}.get(s, 10)
        nc = min(nc, len(tr))
        col[tr] = np.arange(len(tr)) % nc
        sp[s] = pd.array(col, dtype="Int64")
    arch = dict(images=imgs, masks=masks, labels=labels.astype(np.int8),
                image_ids=np.asarray(sp.image_id.tolist(), dtype=str))
    return arch, sp


def write_synthetic(out_dir: str, **kw) -> str:
    os.makedirs(out_dir, exist_ok=True)
    arch, sp = make_synthetic(**kw)
    np.savez_compressed(os.path.join(out_dir, "brain_mri_curated_v1.npz"), **arch)
    sp.to_csv(os.path.join(out_dir, "splits_v1.csv"), index=False)
    return out_dir
