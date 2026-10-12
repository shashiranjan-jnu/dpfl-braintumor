"""Phase 4 audit of saved checkpoints: membership inference, source probe, explanations.

    python scripts/run_audit.py --grids week03_central week03_fed_seeds_a ... \
        --xai-grids week03_central week03_fed_seeds_a \
        --data-dir DATA --ckpt-root /kaggle/input --results-repo RES --shard 0 --nshards 2 --push

For every finished run of the listed grids, the checkpoint <run_id>.pt is found
anywhere below --ckpt-root (the launchers saved them to /kaggle/working/checkpoints,
so earlier launcher notebooks are attached as inputs). Results go to
results/audit/<run_id>/: mia.json and probe.json for every run, xai_summary.json
and xai_per_image.csv for runs of --xai-grids, randomisation.json for seed-1
central runs. audit.json is written last and lists the parts done, so finished
parts are skipped on a rerun. No training happens here.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_grid import expand, push_run  # noqa: E402
from src.attacks import mia, probe  # noqa: E402
from src.data.data import load_curated  # noqa: E402
from src.models.models import build_model  # noqa: E402
from src.run import clean, resolve, run_id  # noqa: E402
from src.xai import audit as xai  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def grid_ids(names):
    ids = []
    for g in names:
        for c in expand(yaml.safe_load(open(os.path.join(ROOT, "configs", "grids", f"{g}.yaml")))):
            rid = run_id(resolve(c))
            if rid not in ids:
                ids.append(rid)
    return ids


def find_ckpt(root: str, rid: str, cache: dict) -> str | None:
    if not cache:
        for p in glob.glob(os.path.join(root, "**", "*.pt"), recursive=True):
            cache[os.path.splitext(os.path.basename(p))[0]] = p
    return cache.get(rid)


def load_model(cfg: dict, path: str, device) -> torch.nn.Module:
    m = build_model(cfg["model"]["name"], 4, False, cfg["model"].get("width", 16))
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.to(device).eval()


def audit_one(rid, cfg, model, data, rdir, want_xai, want_rand, log):
    done_path = os.path.join(rdir, "audit.json")
    done = json.load(open(done_path)) if os.path.exists(done_path) else {"parts": []}
    t0 = time.time()
    if "mia" not in done["parts"]:
        json.dump(clean(mia.audit(model, data, data.splits)), open(os.path.join(rdir, "mia.json"), "w"), indent=1)
        done["parts"].append("mia")
    if "probe" not in done["parts"]:
        json.dump(clean(probe.probe(model, data, data.splits)), open(os.path.join(rdir, "probe.json"), "w"), indent=1)
        done["parts"].append("probe")
    if want_xai and "xai" not in done["parts"]:
        df, summ = xai.audit(model, data)
        df.to_csv(os.path.join(rdir, "xai_per_image.csv"), index=False, float_format="%.5f")
        json.dump(clean(summ), open(os.path.join(rdir, "xai_summary.json"), "w"), indent=1)
        done["parts"].append("xai")
    if want_rand and "randomisation" not in done["parts"]:
        r = xai.randomisation_check(model, data, lambda: build_model(cfg["model"]["name"], 4, False,
                                                                   cfg["model"].get("width", 16)))
        json.dump(clean(r), open(os.path.join(rdir, "randomisation.json"), "w"), indent=1)
        done["parts"].append("randomisation")
    done["seconds"] = round(time.time() - t0, 1)
    json.dump(done, open(done_path, "w"), indent=1)
    log(f"audit {rid} parts {done['parts']} {done['seconds']}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", nargs="+", required=True)
    ap.add_argument("--xai-grids", nargs="*", default=[])
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--ckpt-root", required=True)
    ap.add_argument("--results-repo", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_curated(a.data_dir, device)
    ids = grid_ids(a.grids)
    xai_ids = set(grid_ids(a.xai_grids))
    mine = ids[a.shard::a.nshards]
    cache, missing, failed = {}, [], []
    for i, rid in enumerate(mine, 1):
        run_dir = os.path.join(a.results_repo, "results", "runs", rid)
        if not os.path.exists(os.path.join(run_dir, "metrics.json")):
            missing.append(rid)
            continue
        path = find_ckpt(a.ckpt_root, rid, cache)
        if path is None:
            missing.append(rid)
            print(f"[{i}/{len(mine)}] {rid} no checkpoint")
            continue
        cfg = yaml.safe_load(open(os.path.join(run_dir, "config.yaml")))
        rdir = os.path.join(a.results_repo, "results", "audit", rid)
        os.makedirs(rdir, exist_ok=True)
        want_rand = cfg["fl"]["mode"] == "central" and cfg["train"]["seed"] == 1
        try:
            model = load_model(cfg, path, device)
            audit_one(rid, cfg, model, data, rdir, rid in xai_ids, want_rand,
                      lambda s: print(f"[{i}/{len(mine)}] {s}", flush=True))
        except Exception as e:  # keep going, report at the end
            failed.append(rid)
            print(f"[{i}/{len(mine)}] {rid} FAILED {type(e).__name__}: {e}", flush=True)
            continue
        if a.push:
            print("[push]", rid, "ok" if push_run(a.results_repo, f"results/audit/{rid}", f"Audit {rid}") else "FAILED")
    print(f"shard {a.shard} finished, {len(mine)} runs, missing {len(missing)}, failed {failed}")
    if missing:
        print("missing (no metrics or no checkpoint):", missing[:20])


if __name__ == "__main__":
    main()
