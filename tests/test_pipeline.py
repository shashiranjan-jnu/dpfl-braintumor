"""Pipeline sanity tests on a tiny synthetic archive with the real schema."""
import json
import os

import numpy as np
import torch

from src.data.data import load_curated, write_synthetic
from src.run import main as run_one, resolve, run_id


def _cfg(**over):
    # Tests pin the small CNN without pretrained weights, so they run offline on CPU.
    c = {"fl": {"rounds": 3, "local_epochs": 1.0}, "dp": {"sample_rate": 0.25},
         "model": {"name": "small_cnn", "pretrained": False}, "optim": {"lr": 0.1}}
    for k, v in over.items():
        c.setdefault(k, {}).update(v)
    return c


def test_clients_are_disjoint_and_cover_train(tmp_path):
    d = load_curated(write_synthetic(str(tmp_path / "d"), n=120))
    tr = set(d.idx("train"))
    for s in ["natural3", "dir0.5_k10", "dir0.5_k15"]:
        parts = d.clients(s)
        allidx = np.concatenate(parts)
        assert len(allidx) == len(set(allidx)) == len(tr)
        assert set(allidx) == tr


def test_run_id_is_stable_and_config_sensitive():
    a = run_id(resolve(_cfg()))
    assert a == run_id(resolve(_cfg()))
    assert a != run_id(resolve(_cfg(train={"seed": 1})))


def test_end_to_end_runs_write_complete_folders(tmp_path):
    ddir = write_synthetic(str(tmp_path / "d"), n=120)
    data = load_curated(ddir)
    out = str(tmp_path / "runs")
    for extra in [{}, {"dp": {"enabled": True, "target_epsilon": 8.0, "sample_rate": 0.25}},
                  {"fl": {"algorithm": "fedprox", "mu": 0.01, "rounds": 2}},
                  {"fl": {"algorithm": "scaffold", "rounds": 2}},
                  {"fl": {"mode": "central", "rounds": 2}},
                  {"fl": {"mode": "local", "local_client": 1, "rounds": 2}}]:
        cfg = _cfg()
        for k, v in extra.items():
            cfg.setdefault(k, {}).update(v)
        rid = run_one(cfg, ddir, out, data=data, device="cpu")
        rdir = os.path.join(out, rid)
        for f in ["config.yaml", "env.json", "log.txt", "predictions.csv", "metrics.json"]:
            assert os.path.exists(os.path.join(rdir, f)), f
        m = json.load(open(os.path.join(rdir, "metrics.json")))
        assert 0 <= m["test_best"]["accuracy"] <= 1
        if cfg.get("dp", {}).get("enabled"):
            assert 7.5 < m["privacy"]["epsilon"] <= 8.0


def test_finished_run_is_skipped(tmp_path):
    ddir = write_synthetic(str(tmp_path / "d"), n=80)
    data = load_curated(ddir)
    out = str(tmp_path / "runs")
    rid = run_one(_cfg(fl={"rounds": 1}), ddir, out, data=data, device="cpu")
    stamp = os.path.getmtime(os.path.join(out, rid, "metrics.json"))
    run_one(_cfg(fl={"rounds": 1}), ddir, out, data=data, device="cpu")
    assert os.path.getmtime(os.path.join(out, rid, "metrics.json")) == stamp


def test_same_seed_same_result(tmp_path):
    ddir = write_synthetic(str(tmp_path / "d"), n=80)
    data = load_curated(ddir)
    a = run_one(_cfg(dp={"enabled": True, "noise_multiplier": 1.0}), ddir,
                str(tmp_path / "a"), data=data, device="cpu")
    b = run_one(_cfg(dp={"enabled": True, "noise_multiplier": 1.0}), ddir,
                str(tmp_path / "b"), data=data, device="cpu")
    pa = open(os.path.join(tmp_path, "a", a, "predictions.csv")).read()
    pb = open(os.path.join(tmp_path, "b", b, "predictions.csv")).read()
    assert pa == pb


def test_divergence_is_caught_not_crashing(tmp_path):
    """A run whose loss becomes non-finite stops cleanly and is flagged (bug found in Step 2.4)."""
    ddir = write_synthetic(str(tmp_path / "d"), n=80)
    data = load_curated(ddir)
    rid = run_one(_cfg(optim={"lr": 1e6, "momentum": 0.9, "grad_clip": None}, fl={"rounds": 4}), ddir,
                  str(tmp_path / "r"), data=data, device="cpu")
    m = json.load(open(os.path.join(tmp_path, "r", rid, "metrics.json")))
    assert m["diverged"] is True
    assert len(m["history"]) <= 4


def test_zero_noise_multiplier_is_honoured(tmp_path):
    ddir = write_synthetic(str(tmp_path / "d"), n=80)
    data = load_curated(ddir)
    rid = run_one(_cfg(dp={"enabled": True, "noise_multiplier": 0.0, "target_epsilon": 8.0}),
                  ddir, str(tmp_path / "r"), data=data, device="cpu")
    m = json.load(open(os.path.join(tmp_path, "r", rid, "metrics.json")))
    assert m["privacy"]["sigma"] == 0.0 and m["privacy"]["epsilon"] == "inf"


def test_cosine_schedule_and_grad_clip(tmp_path):
    from src.fl.federated import Cfg, round_lr
    c = Cfg(lr=0.1, rounds=10, schedule="cosine")
    assert abs(round_lr(c, 0) - 0.1) < 1e-12 and round_lr(c, 5) < 0.06 and round_lr(c, 9) < 0.01
    assert round_lr(Cfg(lr=0.1), 7) == 0.1
    ddir = write_synthetic(str(tmp_path / "d"), n=80)
    data = load_curated(ddir)
    rid = run_one(_cfg(optim={"grad_clip": 1.0, "schedule": "cosine", "momentum": 0.9}),
                  ddir, str(tmp_path / "r"), data=data, device="cpu")
    assert json.load(open(os.path.join(tmp_path, "r", rid, "metrics.json")))["diverged"] is False


def test_lr_selection_protocol(tmp_path):
    """select_lr picks the lowest-validation-loss lr per configuration and emits a seeds grid."""
    import yaml
    from scripts.select_lr import select
    from scripts.run_grid import expand
    ddir = write_synthetic(str(tmp_path / "d"), n=80)
    data = load_curated(ddir)
    out = str(tmp_path / "runs")
    grid = {"name": "t", "fixed": _cfg(fl={"rounds": 2}),
            "presets": [{}, {"dp": {"enabled": True, "target_epsilon": 8.0}}],
            "axes": {"optim.lr": [0.01, 0.1]}}
    for c in expand(grid):
        run_one(c, ddir, out, data=data, device="cpu")
    presets, table = select(grid, out)
    assert len(presets) == 2 and table.selected.sum() == 2
    for p in presets:
        assert p["optim"]["lr"] in (0.01, 0.1)
    follow = {"presets": presets, "axes": {"train.seed": [1, 2]}}
    assert len(expand(follow)) == 4
    # D33: unit-level presets inherit the record-level lr at the same scenario and epsilon
    from scripts.make_unit_grid import unit_presets
    grid["fixed"]["train"] = {"seed": 0}
    for c in expand(grid):
        run_one(c, ddir, out, data=data, device="cpu")
    up = unit_presets([grid], out, ["natural3"], [8.0])
    rec = [p for p in select(grid, out)[0] if p.get("dp", {}).get("enabled")][0]
    assert len(up) == 1 and up[0]["dp"]["unit"] == "unit"
    assert up[0]["optim"]["lr"] == rec["optim"]["lr"] and "seed" not in up[0].get("train", {})


def test_udir_partition_keeps_units_whole_and_is_fixed(tmp_path):
    d = load_curated(write_synthetic(str(tmp_path / "d"), n=200, unit_size=4))
    a, b = d.clients("udir0.5_k3"), d.clients("udir0.5_k3")
    assert all((x == y).all() for x, y in zip(a, b))
    tr = d.idx("train")
    assert sorted(np.concatenate(a).tolist()) == sorted(tr.tolist())
    owner = {}
    for j, part in enumerate(a):
        for u in d.splits.unit.values[part]:
            assert owner.setdefault(u, j) == j


def test_unit_level_run_reports_patient_epsilon(tmp_path):
    ddir = write_synthetic(str(tmp_path / "d"), n=160, unit_size=4)
    data = load_curated(ddir)
    out = str(tmp_path / "runs")
    base = {"dp": {"enabled": True, "target_epsilon": 8.0, "sample_rate": 0.25}}
    rec = run_one(_cfg(**base), ddir, out, data=data, device="cpu")
    uni = run_one(_cfg(dp={**base["dp"], "unit": "unit"}), ddir, out, data=data, device="cpu")
    assert rec != uni
    mr = json.load(open(os.path.join(out, rec, "metrics.json")))["privacy"]
    mu = json.load(open(os.path.join(out, uni, "metrics.json")))["privacy"]
    assert mr["unit"] == "record" and mu["unit"] == "unit"
    assert abs(mr["epsilon"] - mu["epsilon"]) < 1e-9          # same sigma, q, steps
    # synthetic natural3 splits units across clients, so the worst unit composes
    assert mu["units_max_clients"] > 1 and mu["unit_epsilon_worst"] > mu["epsilon"]
    assert mr["epsilon_if_k_runs"]["2"] > mr["epsilon"]
    uc = run_one(_cfg(dp={**base["dp"], "unit": "unit"}, fl={"mode": "central", "rounds": 2}),
                 ddir, out, data=data, device="cpu")
    mc = json.load(open(os.path.join(out, uc, "metrics.json")))["privacy"]
    assert mc["units_max_clients"] == 1 and abs(mc["unit_epsilon_worst"] - mc["epsilon"]) < 1e-9


def test_record_level_run_id_unchanged_by_unit_key():
    """dp.unit is optional and absent from base.yaml, so old run ids survive."""
    assert "unit" not in resolve(_cfg())["dp"]
