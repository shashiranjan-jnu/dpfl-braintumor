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
        col = self.splits[scenario]
        tr = self.idx("train")
        assign = col.values[tr]
        if pd.isna(assign).any():
            raise ValueError(f"Scenario {scenario} has unassigned training rows")
        assign = assign.astype(int)
        return [tr[assign == k] for k in range(assign.max() + 1)]


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


def make_synthetic(n: int = 160, size: int = 128, seed: int = 0) -> tuple[dict, pd.DataFrame]:
    """Tiny fake archive with the real schema, for CPU tests and smoke runs."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 4, n)
    imgs = rng.integers(0, 60, (n, size, size)).astype(np.uint8)
    for k, c in enumerate(labels):           # class-dependent square size, learnable by a GAP model
        h = 8 + 14 * c
        r0, c0 = rng.integers(0, size - h, 2)
        imgs[k, r0:r0 + h, c0:c0 + h] = 200
    masks = np.zeros_like(imgs)
    split = rng.choice(["train", "val", "test"], n, p=[0.6, 0.2, 0.2])
    source = rng.choice(["figshare", "sartaj", "br35h"], n)
    sp = pd.DataFrame({
        "image_id": [f"syn:{k}" for k in range(n)], "source": source,
        "relpath": [f"{k}.png" for k in range(n)], "label": [CLASSES[c] for c in labels],
        "label_id": labels, "patient_id": "", "unit": np.arange(n), "split": split})
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
