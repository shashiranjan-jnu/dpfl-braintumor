"""Pipeline sanity tests on a tiny synthetic archive with the real schema."""
import json
import os

import numpy as np
import torch

from src.data.data import load_curated, write_synthetic
from src.run import main as run_one, resolve, run_id


def _cfg(**over):
    c = {"fl": {"rounds": 3, "local_epochs": 1.0}, "dp": {"sample_rate": 0.25}}
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
