"""Launcher 15 code: shadow subsets, weight-noise comparator, LiRA, stratified attack, border audit."""
import json
import os

import numpy as np
import pandas as pd
import torch

from scripts.mia_tables import lira_fit, lira_scores, phi, within_stratum_auc
from scripts.run_audit import audit_one, load_model
from src.data.data import load_curated, shadow_members, write_synthetic
from src.run import main as run_one, resolve, run_id
from src.xai.metrics import brain_regions, consistency_iou, region_energy

BASE = {"fl": {"rounds": 2}, "model": {"name": "small_cnn", "pretrained": False},
        "optim": {"lr": 0.1}, "dp": {"sample_rate": 0.25}}


def _cfg(**over):
    import copy
    from src.run import deep_update
    return deep_update(copy.deepcopy(BASE), over)


def test_shadow_members_whole_units_and_pool_only(tmp_path):
    d = load_curated(write_synthetic(str(tmp_path / "d"), n=200, unit_size=4))
    sp = d.splits
    a, b = shadow_members(sp, 0), shadow_members(sp, 1)
    assert not a[sp.split.values == "val"].any()
    assert (a == shadow_members(sp, 0)).all() and (a != b).any()
    for u in np.unique(sp.unit.values[sp.split.values != "val"]):
        m = (sp.unit.values == u) & (sp.split.values != "val")
        assert a[m].all() or not a[m].any()
    pool = np.isin(sp.split.values, ["train", "test"])
    assert 0.25 < a[pool].mean() < 0.75


def test_shadow_and_weight_noise_runs(tmp_path):
    ddir = write_synthetic(str(tmp_path / "d"), n=160, unit_size=2)
    data = load_curated(ddir)
    out = str(tmp_path / "runs")
    base = run_one(_cfg(), ddir, out, data=data, device="cpu")
    sh = run_one(_cfg(data={"shadow": {"seed": 3, "frac": 0.5}}), ddir, out, data=data, device="cpu")
    shc = run_one(_cfg(fl={"mode": "central"}, data={"shadow": {"seed": 3, "frac": 0.5}}), ddir, out,
                  data=data, device="cpu")
    assert len({base, sh, shc}) == 3
    env = json.load(open(os.path.join(out, shc, "env.json")))
    assert env["client_sizes"] == [int(shadow_members(data.splits, 3).sum())]
    wn = run_one(_cfg(fl={"weight_noise": 0.5}), ddir, out, data=data, device="cpu")
    m = json.load(open(os.path.join(out, wn, "metrics.json")))
    assert m["privacy"]["weight_noise"] == 0.5 and m["privacy"]["epsilon"] == "inf"
    # absent keys leave earlier run ids untouched
    assert run_id(resolve(_cfg())) == base
    try:
        run_one(_cfg(fl={"weight_noise": 0.1}, dp={"enabled": True, "target_epsilon": 8.0}), ddir, out,
                data=data, device="cpu")
        raise AssertionError("weight noise with DP must be refused")
    except ValueError:
        pass


def test_phi_and_lira_separate_members():
    z = np.array([[2.0, 0.0, 0.0, 0.0]])
    assert abs(phi(z, np.array([0]))[0] - (2.0 - np.log(3.0))) < 1e-9
    rng = np.random.default_rng(0)
    S, N = 16, 400
    sh_in = rng.random((S, N)) < 0.5
    base = rng.normal(0, 1, N)
    sh_phi = (base[None] + 2.0 * sh_in + rng.normal(0, 0.3, (S, N)))[..., None].repeat(2, -1)
    fit = lira_fit(sh_phi, sh_in)
    t_in = rng.random(N) < 0.5
    t_phi = (base + 2.0 * t_in + rng.normal(0, 0.3, N))[:, None].repeat(2, -1)
    on, off, valid = lira_scores(t_phi, fit)
    from sklearn.metrics import roc_auc_score
    assert valid.all() and roc_auc_score(t_in, on) > 0.95 and roc_auc_score(t_in, off) > 0.95


def test_within_stratum_removes_composition_effect():
    # members come mostly from an easy stratum; inside each stratum scores are uninformative
    rng = np.random.default_rng(1)
    strata = np.array(["a"] * 200 + ["b"] * 200)
    member = np.r_[rng.random(200) < 0.8, rng.random(200) < 0.2]
    scores = np.r_[rng.normal(2, 1, 200), rng.normal(0, 1, 200)]
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(member, scores) > 0.7
    assert abs(within_stratum_auc(scores, member, strata) - 0.5) < 0.1


def test_border_regions_and_consistency():
    x = torch.zeros(1, 1, 64, 64)
    x[0, 0, 8:56, 8:56] = 0.8
    brain, rim, out = brain_regions(x, band=4)
    assert brain.sum() == 48 * 48 and rim.sum() == 48 * 48 - 40 * 40
    maps = torch.zeros(1, 64, 64)
    maps[0, 10, 10] = 1.0
    assert float(region_energy(maps, rim)[0]) == 1.0
    a = np.random.default_rng(0).random((3, 5, 8, 8))
    same = np.repeat(a[:1], 3, 0)
    assert np.allclose(consistency_iou(same), 1.0) and consistency_iou(a).mean() < 1.0


def test_audit_logits_and_border_parts(tmp_path):
    ddir = write_synthetic(str(tmp_path / "d"), n=160, unit_size=2)
    data = load_curated(ddir)
    rid = run_one(_cfg(fl={"mode": "central"}), ddir, str(tmp_path / "runs"), data=data, device="cpu")
    import yaml
    rdir = os.path.join(tmp_path, "runs", rid)
    cfg = yaml.safe_load(open(os.path.join(rdir, "config.yaml")))
    model = load_model(cfg, os.path.join(rdir, "model_best.pt"), "cpu")
    out = str(tmp_path / "audit")
    os.makedirs(out)
    audit_one(rid, cfg, model, data, out, True, False, print, parts=("logits", "border"))
    assert json.load(open(os.path.join(out, "audit.json")))["parts"] == ["logits", "border"]
    z = np.load(os.path.join(out, "logits.npz"))
    assert z["z"].shape == z["zf"].shape == (len(data.y), 4) and z["z"].dtype == np.float16
    b = json.load(open(os.path.join(out, "border.json")))
    assert set(b) == {"gradcam", "gradcampp", "gradcam_l3"}
    assert "pointing" in b["gradcam_l3"]["all"] and "border_energy" in b["gradcam"]["all"]
    c = np.load(os.path.join(out, "cams8.npz"))
    assert c["cam"].shape[1:] == (8, 8) and len(c["cam"]) == len(c["row"])


def test_weight_noise_sigma_rule_and_grids(tmp_path):
    from scripts.make_shadow_grid import choose_sigma, shadow_grid, write
    from scripts.run_grid import expand
    assert choose_sigma({0.5: 1.4, 0.01: 0.7, 0.003: 0.6, 0.001: 0.5}, 0.65) == 0.003
    assert choose_sigma({0.5: 1.4, 0.01: 0.9}, 0.65) == 0.01
    g = shadow_grid("t", [{"data": {"scenario": "natural3"}}], "")
    cfgs = expand(g)
    assert len(cfgs) == 16 and {c["train"]["seed"] for c in cfgs} == set(range(1000, 1016))
    assert {c["data"]["shadow"]["seed"] for c in cfgs} == set(range(16))
    assert os.path.exists(write(g, str(tmp_path)))
