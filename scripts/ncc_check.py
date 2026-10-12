"""Near-duplicate cross-check with normalised cross-correlation (D32 item 4, Step 4.3). CPU only.

Perceptual hashing (D10) decided which images were removed as duplicates and
which images were linked into one split unit. This script checks both decisions
with a second, independent measure: the normalised cross-correlation (NCC,
Pearson correlation of pixel values) of 64 x 64 greyscale versions.

Part A, curated set (needs only brain_mri_curated_v1.npz). All 3845 x 3845 pairs,
also against the left-right flip of the second image. Pairs at NCC >= 0.90 are
listed with their split relation, unit relation, label agreement and pHash
distance. A pair in different splits at NCC >= 0.95 is a candidate leak that
hashing missed; its images are shown for inspection.

Part B, removed duplicates (needs the raw sources, as in notebook 02). For every
row of data_manifest/duplicates.csv with a kept image, both images are rebuilt
with the preprocessing of notebook 02 and their NCC is computed. A removed pair
with low NCC would be a hashing false positive.

    python scripts/ncc_check.py curated --data-dir DATA --out results/ncc
    python scripts/ncc_check.py removed --raw-dirs /kaggle/input --out results/ncc
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.data import find_file  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRESHOLDS = (0.90, 0.95, 0.98)
NEAR = 0.95            # pre-specified near-duplicate threshold for the headline count


def to64(x: np.ndarray) -> np.ndarray:
    """(N, 128, 128) uint8 -> (N, 64, 64) float32 by 2 x 2 averaging (box filter)."""
    x = x.astype(np.float32)
    return x.reshape(len(x), 64, 2, 64, 2).mean((2, 4))


def standardise(x: np.ndarray) -> np.ndarray:
    """Rows of zero mean and unit norm, so a dot product is the NCC. Flat images give zeros."""
    f = x.reshape(len(x), -1).astype(np.float64)
    f -= f.mean(1, keepdims=True)
    n = np.linalg.norm(f, axis=1, keepdims=True)
    return (f / np.where(n > 0, n, 1.0)).astype(np.float32)


def phash_distance(a: str, b: str) -> int:
    if not isinstance(a, str) or not isinstance(b, str):
        return -1
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def ncc_pairs(x64: np.ndarray, thr: float = 0.90, chunk: int = 512) -> pd.DataFrame:
    """All pairs i < j with max(NCC, NCC against the flip of j) >= thr."""
    z = standardise(x64)
    zf = standardise(x64[:, :, ::-1])
    rows = []
    n = len(z)
    for s in range(0, n, chunk):
        a = z[s:s + chunk]
        c, cf = a @ z.T, a @ zf.T
        best = np.maximum(c, cf)
        ii, jj = np.nonzero(best >= thr)
        keep = jj > ii + s
        for i, j in zip(ii[keep], jj[keep]):
            rows.append((i + s, j, float(c[i, j]), float(cf[i, j])))
    df = pd.DataFrame(rows, columns=["i", "j", "ncc", "ncc_flip"])
    df["ncc_max"] = np.maximum(df.ncc.values, df.ncc_flip.values)
    return df


def nearest_other_split(x64: np.ndarray, split: np.ndarray, chunk: int = 512) -> np.ndarray:
    """For each image, its highest NCC (plain or flipped) with any image of another split."""
    split = np.asarray(split).astype(str)
    z, zf = standardise(x64), standardise(x64[:, :, ::-1])
    out = np.full(len(z), -1.0, np.float32)
    for s in range(0, len(z), chunk):
        best = np.maximum(z[s:s + chunk] @ z.T, z[s:s + chunk] @ zf.T)
        best[split[s:s + chunk, None] == split[None, :]] = -1.0
        out[s:s + chunk] = best.max(1)
    return out


def describe_pairs(df: pd.DataFrame, sp: pd.DataFrame, hashes: pd.DataFrame | None) -> pd.DataFrame:
    i, j = df.i.values, df.j.values
    df = df.copy()
    for col in ("image_id", "source", "label", "split", "unit"):
        df[f"{col}_i"], df[f"{col}_j"] = sp[col].values[i], sp[col].values[j]
    df["same_unit"] = df.unit_i == df.unit_j
    df["same_split"] = df.split_i == df.split_j
    df["label_match"] = df.label_i == df.label_j
    rel = np.where(df.same_split, "within " + df.split_i,
                   np.where(df.split_i < df.split_j, df.split_i + "-" + df.split_j, df.split_j + "-" + df.split_i))
    df["split_relation"] = rel
    if hashes is not None:
        ph = hashes.set_index("image_id").phash.to_dict()
        df["phash_d"] = [phash_distance(ph.get(a), ph.get(b)) for a, b in zip(df.image_id_i, df.image_id_j)]
    return df.sort_values("ncc_max", ascending=False).reset_index(drop=True)


def curated_summary(df: pd.DataFrame, nearest: np.ndarray, sp: pd.DataFrame) -> dict:
    out = {"n_images": int(len(sp)), "thresholds": {}}
    for t in THRESHOLDS:
        d = df[df.ncc_max >= t]
        cross = d[~d.same_split]
        out["thresholds"][str(t)] = {
            "pairs": int(len(d)),
            "pairs_same_unit": int(d.same_unit.sum()),
            "pairs_cross_split": int(len(cross)),
            "cross_split_by_relation": cross.split_relation.value_counts().to_dict(),
            "cross_split_label_conflicts": int((~cross.label_match).sum()),
            "cross_split_images": int(len(set(cross.i) | set(cross.j))),
            "cross_split_flip_only": int(((cross.ncc < t) & (cross.ncc_flip >= t)).sum()),
        }
    te = sp.split.values == "test"
    out["test_nearest_other_split"] = {q: float(np.quantile(nearest[te], float(q))) for q in ("0.5", "0.9", "0.99")}
    out["test_nearest_other_split"]["max"] = float(nearest[te].max())
    out["test_images_at_or_above_near"] = int((nearest[te] >= NEAR).sum())
    return out


# ---------- Part B: rebuild raw images exactly as notebook 02 did ----------

def to_uint8(img):
    lo, hi = np.percentile(img, (0.5, 99.5))
    return (np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def pad_square(a):
    h, w = a.shape
    s = max(h, w)
    out = np.zeros((s, s), a.dtype)
    out[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = a
    return out


def resize128(a8):
    from PIL import Image
    return np.asarray(Image.fromarray(a8).resize((128, 128), Image.BOX))


def locate_raw(raw_dirs: list[str], work: str) -> tuple[dict, dict]:
    mat = {}
    for root in raw_dirs:
        for z in glob.glob(os.path.join(root, "**", "brainTumorDataPublic_*.zip"), recursive=True):
            with zipfile.ZipFile(z) as zf:
                zf.extractall(os.path.join(work, "mat"))
    for root in raw_dirs + [os.path.join(work, "mat")]:
        for p in glob.glob(os.path.join(root, "**", "*.mat"), recursive=True):
            if os.path.basename(p) != "cvind.mat":
                mat.setdefault(os.path.basename(p), p)
    dirs = {}
    for name, key in (("brain-tumor-classification-mri", "sartaj"), ("brain-tumor-detection", "br35h")):
        for root in raw_dirs:
            for r, _, _ in os.walk(root):
                if os.path.basename(r) == name:
                    dirs.setdefault(key, r)
    return mat, dirs


def load_raw(image_id: str, raw_index: pd.DataFrame, mat: dict, dirs: dict) -> np.ndarray:
    src, rel = image_id.split(":", 1)
    if src == "figshare":
        import h5py
        with h5py.File(mat[os.path.basename(rel)], "r") as f:
            img = np.array(f["cjdata"]["image"], dtype=np.float32).T
    else:
        from PIL import Image
        with Image.open(os.path.join(dirs[src], rel)) as im:
            img = np.asarray(im.convert("L"), dtype=np.float32)
    return resize128(pad_square(to_uint8(img)))


def removed_check(dups: pd.DataFrame, loader) -> pd.DataFrame:
    rows = []
    d = dups[dups.kept_id.notna() & (dups.kept_id != "")]
    for r in d.itertuples():
        try:
            a, b = loader(r.image_id), loader(r.kept_id)
        except (KeyError, FileNotFoundError, OSError) as e:
            rows.append({"image_id": r.image_id, "kept_id": r.kept_id, "match_d": r.match_d,
                         "action": r.action, "ncc": np.nan, "ncc_flip": np.nan, "error": type(e).__name__})
            continue
        x = to64(np.stack([a, b]))
        z, zf = standardise(x), standardise(x[:, :, ::-1])
        rows.append({"image_id": r.image_id, "kept_id": r.kept_id, "match_d": r.match_d, "action": r.action,
                     "ncc": float(z[0] @ z[1]), "ncc_flip": float(z[0] @ zf[1]), "error": ""})
    out = pd.DataFrame(rows)
    if len(out):
        out["ncc_max"] = out[["ncc", "ncc_flip"]].max(axis=1)
    return out


def removed_summary(df: pd.DataFrame) -> dict:
    ok = df[df.error == ""]
    res = {"rows": int(len(df)), "computed": int(len(ok)), "errors": int((df.error != "").sum()), "by_phash_d": {}}
    for d, g in ok.groupby("match_d"):
        res["by_phash_d"][str(int(d))] = {
            "n": int(len(g)), "ncc_median": float(g.ncc_max.median()), "ncc_min": float(g.ncc_max.min()),
            **{f"below_{t}": int((g.ncc_max < t).sum()) for t in THRESHOLDS}}
    return res


def montage(pairs: pd.DataFrame, x: np.ndarray, path: str, n: int = 12) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = pairs.head(n)
    if p.empty:
        return
    fig, ax = plt.subplots(2, len(p), figsize=(2.0 * len(p), 4.6))
    ax = np.asarray(ax).reshape(2, -1)
    for k, r in enumerate(p.itertuples()):
        for row, (idx, tag) in enumerate(((r.i, r.split_i), (r.j, r.split_j))):
            ax[row, k].imshow(x[idx], cmap="gray")
            ax[row, k].axis("off")
            lab = r.label_i if row == 0 else r.label_j
            ax[row, k].set_title(f"{tag} {lab[:4]}\nNCC {r.ncc_max:.3f}" if row == 0 else f"{tag} {lab[:4]}",
                                 fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=["curated", "removed"])
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--raw-dirs", nargs="*", default=["/kaggle/input"])
    ap.add_argument("--work", default="/tmp/ncc_work")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "ncc"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    man = os.path.join(ROOT, "data_manifest")
    if a.part == "curated":
        d = np.load(find_file(a.data_dir, "brain_mri_curated_v1.npz"))
        sp = pd.read_csv(os.path.join(man, "splits_v1.csv"))
        assert (d["image_ids"].astype(str) == sp.image_id.astype(str).values).all(), "rows not aligned"
        x = d["images"]
        x64 = to64(x)
        pairs = describe_pairs(ncc_pairs(x64, min(THRESHOLDS)), sp,
                               pd.read_csv(os.path.join(man, "hashes.csv"), dtype=str))
        nearest = nearest_other_split(x64, sp.split.values)
        pairs.to_csv(os.path.join(a.out, "curated_pairs_ncc090.csv"), index=False, float_format="%.4f")
        summ = curated_summary(pairs, nearest, sp)
        json.dump(summ, open(os.path.join(a.out, "curated_summary.json"), "w"), indent=1)
        cross = pairs[~pairs.same_split]
        montage(cross, x, os.path.join(a.out, "top_cross_split_pairs.png"))
        print(json.dumps(summ, indent=1))
        print("\nTop cross-split pairs:")
        cols = ["image_id_i", "image_id_j", "split_relation", "label_i", "label_j", "ncc", "ncc_flip", "phash_d"]
        print(cross[cols].head(15).to_string(index=False))
    else:
        mat, dirs = locate_raw(a.raw_dirs, a.work)
        print(f"raw sources: {len(mat)} Figshare .mat files, folders {sorted(dirs)}")
        if len(mat) < 3064 or set(dirs) != {"sartaj", "br35h"}:
            raise SystemExit("Raw sources incomplete. Attach the same inputs as notebook 02: the Figshare "
                             "zips (output of 01-data-acquisition) and the SARTAJ and Br35H Kaggle datasets.")
        raw_index = pd.read_csv(os.path.join(man, "raw_index.csv"), dtype=str)
        cache = {}

        def loader(image_id):
            if image_id not in cache:
                cache[image_id] = load_raw(image_id, raw_index, mat, dirs)
            return cache[image_id]
        res = removed_check(pd.read_csv(os.path.join(man, "duplicates.csv"), dtype={"kept_id": str}), loader)
        res.to_csv(os.path.join(a.out, "removed_pairs_ncc.csv"), index=False, float_format="%.4f")
        summ = removed_summary(res)
        json.dump(summ, open(os.path.join(a.out, "removed_summary.json"), "w"), indent=1)
        print(json.dumps(summ, indent=1))
        low = res[res.error == ""].sort_values("ncc_max").head(10)
        print("\nLowest-NCC removed pairs:")
        print(low[["image_id", "kept_id", "match_d", "action", "ncc", "ncc_flip"]].to_string(index=False))


if __name__ == "__main__":
    main()
