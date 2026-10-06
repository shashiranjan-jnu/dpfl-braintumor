"""Expand a grid file into runs, shard them across GPUs, skip finished runs,
and optionally push each finished run folder to GitHub.

Grid file format (configs/grids/*.yaml):

    name: week03_fedavg
    fixed:                      # overrides applied to every run
      fl: {rounds: 60}
    presets:                    # optional list of whole override dicts, crossed with axes
      - {model: {name: small_cnn, width: 32}}
      - {model: {name: resnet18_frozenbn, pretrained: true}}
    axes:                       # cartesian product; dotted keys
      data.scenario: [natural3, dir0.5_k10]
      dp.target_epsilon: [null, 8]
      train.seed: [0, 1, 2]
    rules:                      # optional, per-value extra overrides
      dp.target_epsilon:
        8: {dp: {enabled: true}}

Usage on Kaggle (one process per GPU):

    CUDA_VISIBLE_DEVICES=0 python scripts/run_grid.py --grid configs/grids/x.yaml \
        --data-dir /kaggle/input/brain-mri-curated-v1 --results-repo /tmp/results \
        --shard 0 --nshards 2 --push &

With --push, the git remote URL is read from the environment variable
GIT_PUSH_URL, so the token never touches .git/config or notebook output.
A file lock serialises pushes from parallel shards.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import itertools
import os
import shutil
import subprocess
import sys
import time
import traceback

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.run import main as run_one, resolve, run_id  # noqa: E402


def set_dotted(d: dict, key: str, val):
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = val


def expand(grid: dict) -> list[dict]:
    """fixed, then one preset (a whole override dict), then the axes product."""
    from src.run import deep_update
    axes = grid.get("axes", {}) or {}
    keys = list(axes)
    presets = grid.get("presets") or [{}]
    out = []
    for preset, combo in itertools.product(
            presets, itertools.product(*[axes[k] for k in keys]) if keys else [()]):
        cfg = deep_update(copy.deepcopy(grid.get("fixed", {}) or {}), preset)
        for k, v in zip(keys, combo):
            set_dotted(cfg, k, v)
            extra = ((grid.get("rules") or {}).get(k) or {}).get(v)
            if extra:
                cfg = deep_update(cfg, extra)
        out.append(cfg)
    return out


def _git(args, cwd, token=""):
    r = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True)
    msg = (r.stdout + r.stderr)
    if token:
        msg = msg.replace(token, "***")
    return r.returncode, msg


def push_run(repo: str, rel_path: str, message: str, branch: str = "main") -> bool:
    url = os.environ.get("GIT_PUSH_URL", "")
    token = url.split("@")[0].split("//")[-1] if "@" in url else ""
    lock = open(os.path.join(repo, ".git", "push.lock"), "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        _git(["add", rel_path], repo)
        code, _ = _git(["diff", "--cached", "--quiet"], repo)
        if code == 0:
            return True
        _git(["commit", "-q", "-m", message], repo)
        for attempt in range(1, 4):
            c1, m1 = _git(["pull", "-q", "--rebase", "--autostash", url or "origin", branch], repo, token)
            c2, m2 = _git(["push", "-q", url or "origin", f"HEAD:{branch}"], repo, token)
            if c1 == 0 and c2 == 0:
                return True
            print(f"[push] attempt {attempt} failed: {(m1 + m2)[-300:]}")
            time.sleep(10 * attempt)
        return False
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--results-repo", required=True,
                    help="working copy that receives results/runs/<id> and is pushed")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--ckpt-dir", default=None,
                    help="copy model_best.pt here as <run_id>.pt, e.g. /kaggle/working/checkpoints")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    grid = yaml.safe_load(open(a.grid))
    cfgs = expand(grid)
    ids = [run_id(resolve(c)) for c in cfgs]
    if len(set(ids)) != len(ids):
        raise SystemExit("Duplicate configurations in grid")
    out_root = os.path.join(a.results_repo, "results", "runs")
    mine = [(c, i) for k, (c, i) in enumerate(zip(cfgs, ids)) if k % a.nshards == a.shard]
    todo = [(c, i) for c, i in mine if not os.path.exists(os.path.join(out_root, i, "metrics.json"))]
    print(f"grid {grid.get('name')} total {len(cfgs)} shard {a.shard}/{a.nshards} "
          f"mine {len(mine)} todo {len(todo)}")
    if a.dry_run:
        for c, i in todo:
            print(i, c)
        return

    import torch
    from src.data.data import load_curated
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_curated(a.data_dir, dev)
    failed = []
    for n, (c, i) in enumerate(todo, 1):
        print(f"=== [{n}/{len(todo)}] {i}")
        try:
            run_one(c, a.data_dir, out_root, data=data, device=dev)
            if a.ckpt_dir:
                os.makedirs(a.ckpt_dir, exist_ok=True)
                src_pt = os.path.join(out_root, i, "model_best.pt")
                if os.path.exists(src_pt):
                    shutil.copy(src_pt, os.path.join(a.ckpt_dir, f"{i}.pt"))
            if a.push:
                ok = push_run(a.results_repo, os.path.join("results", "runs", i),
                              f"Run {i} ({grid.get('name')})")
                print(f"[push] {i} {'ok' if ok else 'FAILED'}")
        except Exception:
            failed.append(i)
            traceback.print_exc()
            with open(os.path.join(out_root, f"FAILED_{i}.txt"), "w") as f:
                f.write(traceback.format_exc())
    print(f"shard {a.shard} finished, failed {failed}")


if __name__ == "__main__":
    main()
