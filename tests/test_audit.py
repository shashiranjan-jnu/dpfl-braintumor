"""Phase 4 audit code: explanation maps and metrics, membership inference, source probe."""
import json
import os

import numpy as np
import torch

from scripts.run_audit import audit_one, load_model
from src.attacks import mia, probe
from src.data.data import load_curated, write_synthetic
from src.models.models import build_model
from src.run import main as run_one
from src.xai import audit as xai
from src.xai import maps as M
from src.xai import metrics as X


def _data(tmp_path, n=160):
    return load_curated(write_synthetic(str(tmp_path / "d"), n=n, unit_size=2))


def test_map_equal_to_mask_scores_perfectly():
    masks = torch.zeros(2, 32, 32)
    masks[0, 4:12, 4:12] = 1
    masks[1, 20:30, 10:20] = 1
    maps = masks.clone()
    assert torch.all(X.pointing_game(maps, masks) == 1)
    assert torch.allclose(X.energy_in_mask(maps, masks), torch.ones(2))
    off = torch.roll(masks, shifts=(16, 16), dims=(1, 2))
    assert torch.all(X.pointing_game(off, masks, tol=0) == 0)
    assert torch.all(X.energy_in_mask(off, masks) == 0)
    iou = X.topk_iou(maps, masks, frac=64 / 1024)  # top-64 pixels equal the first mask exactly
    assert abs(float(iou[0]) - 1.0) < 1e-6


def test_spearman_and_bootstrap():
    a = torch.rand(3, 50)
    assert torch.allclose(X.spearman_rows(a, a), torch.ones(3), atol=1e-5)
    assert torch.allclose(X.spearman_rows(a, -a), -torch.ones(3), atol=1e-5)
    v = np.array([1.0, 1.0, 0.0, 0.0])
    mean, lo, hi = X.patient_bootstrap_ci(v, np.array([0, 0, 1, 1]), n_boot=200)
    assert mean == 0.5 and 0.0 <= lo <= mean <= hi <= 1.0


def test_maps_shapes_ranges_and_faithfulness_bounds():
    torch.manual_seed(0)
    m = build_model("small_cnn", 4, False, 8)
    x, y = torch.randn(3, 1, 64, 64), torch.tensor([0, 1, 2])
    for f in (M.gradcam, M.gradcampp):
        out = f(m, x, y)
        assert out.shape == (3, 64, 64) and out.min() >= 0 and out.max() <= 1 + 1e-6
    ig = M.integrated_gradients(m, x, y, baseline=-1.0, steps=4)
    assert ig.shape == (3, 64, 64) and torch.isfinite(ig).all()
    d, i = X.deletion_insertion(m, x, y, ig, baseline=-1.0, steps=4)
    assert d.shape == i.shape == (3,) and ((d >= 0) & (d <= 1) & (i >= 0) & (i <= 1)).all()


def test_attack_summary_extremes():
    s = mia.attack_summary(np.array([2.0, 3.0, 4.0]), np.array([-1.0, 0.0, 1.0]))
    assert s["auc"] == 1.0 and s["advantage"] == 1.0
    s = mia.attack_summary(np.zeros(50), np.zeros(50))
    assert s["auc"] == 0.5 and s["advantage"] == 0.0


def test_audit_parts_on_trained_checkpoint(tmp_path):
    ddir = write_synthetic(str(tmp_path / "d"), n=160, unit_size=2)
    data = load_curated(ddir)
    cfg = {"fl": {"mode": "central", "rounds": 2}, "model": {"name": "small_cnn", "pretrained": False},
           "optim": {"lr": 0.1}, "train": {"seed": 1}}
    rid = run_one(cfg, ddir, str(tmp_path / "runs"), data=data, device="cpu")
    rdir = os.path.join(tmp_path, "runs", rid)
    import yaml
    rcfg = yaml.safe_load(open(os.path.join(rdir, "config.yaml")))
    model = load_model(rcfg, os.path.join(rdir, "model_best.pt"), "cpu")
    out = str(tmp_path / "audit")
    os.makedirs(out)
    audit_one(rid, rcfg, model, data, out, True, True, print)
    done = json.load(open(os.path.join(out, "audit.json")))
    assert done["parts"] == ["mia", "probe", "xai", "randomisation"]
    m = json.load(open(os.path.join(out, "mia.json")))
    assert 0 <= m["image"]["auc"] <= 1 and m["patient_mean"]["n_members"] > 0
    p = json.load(open(os.path.join(out, "probe.json")))
    assert set(p) == set(probe.TASKS)
    s = json.load(open(os.path.join(out, "xai_summary.json")))
    assert set(s) == {"gradcam", "gradcampp", "ig"}
    n_mask = len(xai.figshare_test_index(data))
    assert s["ig"]["all"]["n"] == n_mask > 0
    r = json.load(open(os.path.join(out, "randomisation.json")))
    assert set(r) == {"gradcam", "ig"} and all(-1 <= v <= 1 for d in r.values() for v in d.values())
    audit_one(rid, rcfg, model, data, out, True, True, print)  # rerun skips every part
    assert json.load(open(os.path.join(out, "audit.json")))["parts"] == done["parts"]
