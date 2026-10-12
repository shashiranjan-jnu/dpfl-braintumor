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

--parts limits the work (launcher 15 runs mia probe logits border): logits.npz
holds float32 logits of every row, plain and flipped, for stratified attacks
and LiRA (D45); border.json and cams8.npz hold the skull-border analysis,
layer-3 Grad-CAM and 8 x 8 Grad-CAM maps for cross-seed consistency, for runs
of --xai-grids. --probe-baselines also probes the ImageNet initialisation and a
random network once (shard 0). Several --ckpt-root folders may be given.
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
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


def find_ckpt(roots, rid: str, cache: dict) -> str | None:
    if not cache:
        for root in ([roots] if isinstance(roots, str) else roots):
            for p in glob.glob(os.path.join(root, "**", "*.pt"), recursive=True):
                cache[os.path.splitext(os.path.basename(p))[0]] = p
    return cache.get(rid)


def load_model(cfg: dict, path: str, device) -> torch.nn.Module:
    m = build_model(cfg["model"]["name"], 4, False, cfg["model"].get("width", 16))
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.to(device).eval()


ALL_PARTS = ("mia", "probe", "xai", "randomisation", "logits", "border")
# A part counts as done only if audit.json lists it and its files exist. Launcher 15 listed
# logits and border as done while .gitignore kept the .npz files out of the repository (D47).
PART_FILES = {"mia": ["mia.json"], "probe": ["probe.json"], "xai": ["xai_summary.json"],
              "randomisation": ["randomisation.json"], "logits": ["logits.npz"],
              "border": ["border.json", "cams8.npz"]}


def audit_one(rid, cfg, model, data, rdir, want_xai, want_rand, log, parts=ALL_PARTS[:4]):
    done_path = os.path.join(rdir, "audit.json")
    done = json.load(open(done_path)) if os.path.exists(done_path) else {"parts": []}
    done["parts"] = [p for p in done["parts"]
                     if all(os.path.exists(os.path.join(rdir, f)) for f in PART_FILES.get(p, []))]
    done.setdefault("errors", {})
    t0 = time.time()
    todo = lambda p: p in parts and p not in done["parts"]
    if todo("logits"):
        z, zf = mia.all_logits(model, data.x)
        np.savez_compressed(os.path.join(rdir, "logits.npz"), z=z.astype(np.float32), zf=zf.astype(np.float32))
        done["parts"].append("logits")
    if want_xai and todo("border"):
        summ, cam, idx = xai.border_audit(model, data)
        json.dump(clean(summ), open(os.path.join(rdir, "border.json"), "w"), indent=1)
        np.savez_compressed(os.path.join(rdir, "cams8.npz"), cam=cam, row=idx.astype(np.int32))
        done["parts"].append("border")
    for part, fn in (("mia", lambda: mia.audit(model, data, data.splits)),
                     ("probe", lambda: probe.probe(model, data, data.splits))):
        if not todo(part):
            continue
        try:
            json.dump(clean(fn()), open(os.path.join(rdir, f"{part}.json"), "w"), indent=1)
            done["parts"].append(part)
            done["errors"].pop(part, None)
        except ValueError as e:      # non-finite outputs of a diverged model
            done["errors"][part] = str(e)[:200]
    if want_xai and todo("xai"):
        df, summ = xai.audit(model, data)
        df.to_csv(os.path.join(rdir, "xai_per_image.csv"), index=False, float_format="%.5f")
        json.dump(clean(summ), open(os.path.join(rdir, "xai_summary.json"), "w"), indent=1)
        done["parts"].append("xai")
    if want_rand and todo("randomisation"):
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
    ap.add_argument("--ckpt-root", nargs="+", required=True)
    ap.add_argument("--parts", nargs="+", default=list(ALL_PARTS[:4]), choices=ALL_PARTS)
    ap.add_argument("--probe-baselines", action="store_true")
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
    if a.probe_baselines and a.shard == 0:
        bdir = os.path.join(a.results_repo, "results", "audit", "_baselines")
        os.makedirs(bdir, exist_ok=True)
        for tag, pre in (("pretrained", True), ("random", False)):
            torch.manual_seed(0)
            m = build_model("resnet18_frozenbn", 4, pre).to(device).eval()
            json.dump(clean(probe.probe(m, data, data.splits)), open(os.path.join(bdir, f"probe_{tag}.json"), "w"), indent=1)
            print("probe baseline", tag, open(os.path.join(bdir, f"probe_{tag}.json")).read().replace("\n", " "))
        if a.push:
            print("[push] baselines", push_run(a.results_repo, "results/audit/_baselines", "Probe baselines"))
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
                      lambda s: print(f"[{i}/{len(mine)}] {s}", flush=True), parts=a.parts)
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
