"""Phase 7.4 slice cap and k_max trim (D68, D69, D70)."""
import json
import os

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.data.data import load_curated, write_synthetic
from src.fl.federated import cap_per_unit
from src.run import main as run_one, resolve, run_id

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cfg(**over):
    c = {"fl": {"rounds": 2, "local_epochs": 1.0, "mode": "central"}, "dp": {"sample_rate": 0.25},
         "model": {"name": "small_cnn", "pretrained": False}, "optim": {"lr": 0.1}}
    for k, v in over.items():
        c.setdefault(k, {}).update(v)
    return c


@pytest.mark.parametrize("m", [1, 2, 3, 10])
def test_cap_keeps_at_most_m_per_unit_and_all_when_small(m):
    g = torch.Generator().manual_seed(0)
    for trial in range(50):
        n = int(torch.randint(0, 40, (1,), generator=g))
        b = torch.sort(torch.randperm(200, generator=g)[:n]).values
        units = torch.randint(0, 6, (n,), generator=g)
        kept = cap_per_unit(b, units, m, g)
        assert set(kept.tolist()) <= set(b.tolist())
        assert torch.all(kept[1:] > kept[:-1])                         # sorted, no duplicates
        ku = units[torch.isin(b, kept)]
        for u in units.unique():
            have = int((units == u).sum())
            assert int((ku == u).sum()) == min(have, m)


def test_cap_choice_is_uniform_within_unit():
    g = torch.Generator().manual_seed(1)
    b = torch.arange(8)
    units = torch.zeros(8, dtype=torch.long)                          # one unit with 8 sampled slices
    counts = torch.zeros(8)
    for _ in range(4000):
        counts[cap_per_unit(b, units, 2, g)] += 1
    assert torch.allclose(counts / 4000, torch.full((8,), 2 / 8), atol=0.03)


def test_old_run_ids_are_unchanged():
    """Optional Phase 7 keys are absent from base.yaml, so every frozen run keeps its id."""
    runs = os.path.join(ROOT, "results", "runs")
    some = sorted(d for d in os.listdir(runs) if os.path.isfile(os.path.join(runs, d, "config.yaml")))[:20] \
        if os.path.isdir(runs) else []
    if not some:
        pytest.skip("no stored runs")
    for rid in some:
        cfg = yaml.safe_load(open(os.path.join(runs, rid, "config.yaml")))
        assert run_id(cfg) == rid                                     # stored configs are already resolved
        assert "slice_cap" not in cfg["dp"] and "train_drop" not in cfg["data"]
    assert "slice_cap" not in resolve(_cfg())["dp"] and "train_drop" not in resolve(_cfg())["data"]


def test_capped_run_trims_and_reports_patient_epsilon(tmp_path):
    ddir = write_synthetic(str(tmp_path / "d"), n=160, unit_size=4)
    data = load_curated(ddir)
    tr = data.idx("train")
    u = data.splits.unit.values[tr]
    big = pd.Series(u).value_counts().index[0]
    drop = data.splits.image_id.values[tr][u == big][:2]
    dfile = str(tmp_path / "drop.csv")
    pd.DataFrame({"image_id": drop}).to_csv(dfile, index=False)
    out = str(tmp_path / "runs")
    dp = {"enabled": True, "noise_multiplier": 3.0, "sample_rate": 0.25}
    rec = run_one(_cfg(dp=dp, data={"train_drop": dfile}), ddir, out, data=data, device="cpu")
    cap = run_one(_cfg(dp={**dp, "slice_cap": 1}, data={"train_drop": dfile}), ddir, out, data=data, device="cpu")
    assert rec != cap
    env = json.load(open(os.path.join(out, cap, "env.json")))
    assert env["client_sizes"] == [len(tr) - 2]
    pr = json.load(open(os.path.join(out, rec, "metrics.json")))["privacy"]
    pc = json.load(open(os.path.join(out, cap, "metrics.json")))["privacy"]
    assert pc["slice_cap"] == 1 and pr["slice_cap"] is None
    assert pc["patient_epsilon_worst"] < pr["patient_epsilon_worst"]   # capping can only help a patient
    assert pr["patient_epsilon_worst"] > pr["epsilon"]                  # a multi-slice patient pays more
    with pytest.raises(ValueError):
        run_one(_cfg(dp={**dp, "slice_cap": 1, "unit": "unit"}), ddir, out, data=data, device="cpu")
