"""Results freeze manifest (Step 4.3). CPU only.

    python scripts/freeze.py            # writes results/FROZEN_MANIFEST.{json,md}
    python scripts/freeze.py --check    # verifies every frozen file against its recorded hash

Lists every grid the paper reports with its run ids and how many are finished,
and every table, figure and audit summary with a SHA-256 hash, so that the
manuscript (Step 5.1) reads only frozen files and any later change is detected.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_audit import grid_ids  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTED_GRIDS = {
    "utility, central record and patient level": ["week03_central", "week04_unit_central", "week04_unit_central_ext"],
    "utility, FedAvg partitions": ["week03_fed_seeds_a", "week03_fed_seeds_b", "week04_unit_fed", "week04_udir_seeds"],
    "local-only baselines": ["week03_local"],
    "H5 client count": ["week05_client_count", "week05_client_count_lr"],
    "FedProx and SCAFFOLD": ["week06_select_fedprox", "week06_select_fedprox_ext", "week06_scaffold_seeds_a",
                             "week06_scaffold_size"],
    "weight-noise comparator": ["week07_wn_select", "week07_wn_seeds"],
    "LiRA shadows": ["week07_lira_shadows", "week07_wn_shadows"],
    "learning-rate selection (seed 0)": ["week03_select_fed", "week04_select_udir", "week04_select_unit"],
}
FROZEN_GLOBS = ["results/tables/*", "figures/*.png", "figures/*.pdf", "figures/CAPTIONS.md",
                "figures/figure_data/*", "results/ncc/*", "results/audit/_baselines/*",
                "data_manifest/splits_v1.csv", "data_manifest/duplicates.csv", "data_manifest/hashes.csv"]


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build() -> dict:
    grids = {}
    for group, names in REPORTED_GRIDS.items():
        for g in names:
            if not os.path.exists(os.path.join(ROOT, "configs", "grids", f"{g}.yaml")):
                continue
            ids = grid_ids([g])
            done = [r for r in ids if os.path.exists(os.path.join(ROOT, "results", "runs", r, "metrics.json"))]
            grids[g] = {"group": group, "n_runs": len(ids), "n_finished": len(done), "run_ids": ids}
    files = {}
    for pat in FROZEN_GLOBS:
        for p in sorted(glob.glob(os.path.join(ROOT, pat))):
            if os.path.isfile(p):
                files[os.path.relpath(p, ROOT)] = sha(p)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"frozen_from_commit": commit, "test_set": "clean, 747 images (D51); full 813 in the supplement",
            "grids": grids, "files": files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    path = os.path.join(ROOT, "results", "FROZEN_MANIFEST.json")
    if a.check:
        m = json.load(open(path))
        bad = [f for f, h in m["files"].items() if not os.path.exists(os.path.join(ROOT, f))
               or sha(os.path.join(ROOT, f)) != h]
        print(f"{len(m['files'])} frozen files, {len(bad)} changed or missing")
        for f in bad:
            print("  changed:", f)
        sys.exit(1 if bad else 0)
    m = build()
    json.dump(m, open(path, "w"), indent=1)
    lines = ["# Results freeze manifest", "",
             f"Built from commit {m['frozen_from_commit'][:10]}. Test set: {m['test_set']}. "
             "Verify with `python scripts/freeze.py --check`.", "",
             "| Group | Grid | Runs | Finished |", "|---|---|---|---|"]
    for g, v in m["grids"].items():
        lines.append(f"| {v['group']} | {g} | {v['n_runs']} | {v['n_finished']} |")
    tot = sum(v["n_runs"] for v in m["grids"].values())
    fin = sum(v["n_finished"] for v in m["grids"].values())
    lines += ["", f"Total {tot} runs, {fin} finished. {len(m['files'])} files hashed (tables, figures, figure data, "
              "NCC results, probe baselines, split and duplicate manifests)."]
    open(os.path.join(ROOT, "results", "FROZEN_MANIFEST.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
