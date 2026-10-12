"""NCC near-duplicate cross-check (scripts/ncc_check.py)."""
import numpy as np
import pandas as pd

from scripts.ncc_check import (curated_summary, describe_pairs, ncc_pairs, nearest_other_split,
                               phash_distance, removed_check, removed_summary, standardise, to64)


def _set(n=40, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 255, (n, 128, 128)).astype(np.uint8)
    x[5] = x[3]                                            # exact copy, train vs test
    x[7] = x[2][:, ::-1]                                   # flipped copy, train vs val
    x[9] = np.clip(x[1].astype(int) + rng.integers(-3, 4, (128, 128)), 0, 255).astype(np.uint8)  # same unit
    sp = pd.DataFrame({"image_id": [f"s:{k}" for k in range(n)], "source": "s", "label": "a",
                       "split": ["train"] * 4 + ["test", "test", "val", "val", "train", "train"] + ["train"] * (n - 10),
                       "unit": np.arange(n)})
    sp.loc[9, "unit"] = 1
    return x, sp


def test_standardised_dot_is_pearson():
    a = np.random.default_rng(1).random((2, 64, 64)).astype(np.float32)
    z = standardise(a)
    assert abs(float(z[0] @ z[1]) - np.corrcoef(a[0].ravel(), a[1].ravel())[0, 1]) < 1e-5
    assert to64(np.zeros((3, 128, 128), np.uint8)).shape == (3, 64, 64)


def test_planted_duplicates_found_and_classified():
    x, sp = _set()
    pairs = describe_pairs(ncc_pairs(to64(x), 0.9), sp, None)
    found = {(int(r.i), int(r.j)) for r in pairs.itertuples()}
    assert {(3, 5), (2, 7), (1, 9)} <= found and len(found) == 3
    r = pairs.set_index(["i", "j"])
    assert r.loc[(2, 7), "ncc"] < 0.5 < 0.99 < r.loc[(2, 7), "ncc_flip"]
    assert r.loc[(3, 5), "split_relation"] == "test-train" and r.loc[(1, 9), "same_unit"]
    near = nearest_other_split(to64(x), sp.split.values)
    s = curated_summary(pairs, near, sp)
    assert s["thresholds"]["0.95"]["pairs_cross_split"] == 2
    assert s["thresholds"]["0.95"]["cross_split_flip_only"] == 1
    assert s["test_images_at_or_above_near"] == 1


def test_removed_check_and_phash():
    x, _ = _set()
    imgs = {f"s:{k}": x[k] for k in range(len(x))}
    dups = pd.DataFrame({"image_id": ["s:5", "s:7", "s:11", "s:12"], "kept_id": ["s:3", "s:2", "s:13", None],
                         "match_d": [0, 2, 4, -1], "action": "removed_duplicate"})
    res = removed_check(dups, lambda k: imgs[k])
    assert len(res) == 3 and res.ncc_max.iloc[0] > 0.99 and res.ncc_max.iloc[1] > 0.99
    assert res.ncc_max.iloc[2] < 0.5
    s = removed_summary(res)
    assert s["by_phash_d"]["4"]["below_0.9"] == 1
    assert phash_distance("ff", "0f") == 4 and phash_distance(None, "0f") == -1


def test_contaminated_and_clean_metrics():
    from scripts.ncc_sensitivity import clean_metrics, contaminated
    pairs = pd.DataFrame({"same_split": [False, False, True], "ncc_max": [0.99, 0.93, 0.99],
                          "split_i": ["test", "test", "test"], "split_j": ["train", "train", "test"],
                          "image_id_i": ["a", "b", "c"], "image_id_j": ["x", "y", "z"]})
    assert contaminated(pairs) == {"a"}
    pred = pd.DataFrame({"image_id": ["a", "b", "c", "d"], "y": [2, 2, 0, 1],
                         "p_glioma": [0, 0, 1, 0], "p_meningioma": [0, 1, 0, 1],
                         "p_notumor": [1, 0, 0, 0], "p_pituitary": [0, 0, 0, 0]})
    m = clean_metrics(pred, {"a"})
    assert m["n_test"] == 3 and m["sens_notumor"] == 0.0 and m["tumour_sens"] == 1.0


def test_summarise_keeps_algorithms_apart():
    from scripts.summarise import summarise
    rows = []
    for alg, v in (("fedavg", 0.8), ("scaffold", 0.6)):
        for seed in (1, 2):
            rows.append({"cfg.fl.mode": "federated", "cfg.data.scenario": "iid_k10", "cfg.fl.local_client": None,
                         "cfg.optim.lr": 0.01, "cfg.dp.enabled": False, "cfg.train.seed": seed,
                         "cfg.fl.algorithm": alg, "epsilon": float("inf"), "diverged": False,
                         "test.balanced_accuracy": v})
    s = summarise(pd.DataFrame(rows))
    assert len(s) == 2 and sorted(s.bal_acc.round(2)) == [0.6, 0.8]
