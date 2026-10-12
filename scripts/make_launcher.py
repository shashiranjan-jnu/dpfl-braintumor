"""Generate a Kaggle launcher notebook for a code tag and a list of grids.

    python scripts/make_launcher.py --name 05-step2-diag --ref v0.2-rc2 \
        --grids week02_diag_central week02_diag_fed --title "Step 2.4 utility diagnostics"

Every launcher has the same five cells: setup, tests, smoke run, grids on two
GPUs, summary tables. Only the tag and the grid list differ, so each step's
notebook is reproducible from this script.

    python scripts/make_launcher.py --kind audit --name 14-audit --ref v0.5-audit \
        --grids week03_central ... --xai-grids week03_central ...

builds a Phase 4 audit notebook instead: setup with a checkpoint check, tests,
scripts/run_audit.py on two GPUs over checkpoints of earlier launchers attached
as inputs, and the audit summary. No training happens in it.
"""
from __future__ import annotations

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SETUP = r'''
import os, sys, glob, json, shutil, subprocess, datetime
from kaggle_secrets import UserSecretsClient

USER, EMAIL, REPO = "shashiranjan-jnu", "shashi17_scs@jnu.ac.in", "dpfl-braintumor"
CODE_REF = "{ref}"                 # tag of the code to run
GRIDS = {grids}
token = UserSecretsClient().get_secret("kaggle_dpfl-braintumor")
os.environ["GIT_PUSH_URL"] = f"https://{{token}}@github.com/{{USER}}/{{REPO}}.git"
CLEAN = f"https://github.com/{{USER}}/{{REPO}}.git"
CODE, RES = "/tmp/code", "/tmp/results"
TODAY = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")

def run(cmd, cwd=None, check=True, env=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)
    out = (r.stdout + r.stderr).replace(token, "***")
    if check and r.returncode != 0:
        raise RuntimeError(out[-3000:])
    return out

for d in (CODE, RES):
    shutil.rmtree(d, ignore_errors=True)
    run(["git", "clone", "-q", os.environ["GIT_PUSH_URL"], d])
    run(["git", "remote", "set-url", "origin", CLEAN], cwd=d)
run(["git", "checkout", "-q", CODE_REF], cwd=CODE)
run(["git", "config", "user.name", USER], cwd=RES)
run(["git", "config", "user.email", EMAIL], cwd=RES)
COMMIT = run(["git", "rev-parse", "HEAD"], cwd=CODE).strip()
print(run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{{CODE}}/requirements.txt"])[-500:])

hits = glob.glob("/kaggle/input/**/brain_mri_curated_v1.npz", recursive=True)
if not hits:
    raise SystemExit("Dataset not attached. In the right panel use Add Input, Your Datasets, "
                     "brain-mri-curated-v1, wait for it to mount, then rerun this cell.")
DATA = os.path.dirname(hits[0])
import torch
print("code", CODE_REF, COMMIT[:10], "| data", DATA)
print("torch", torch.__version__, "| GPUs", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
'''

TESTS = r'''
# Test suite on the Kaggle image, report pushed to results/tests/.
sys.path.insert(0, CODE)
from scripts.run_grid import push_run
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests", "-p", "no:cacheprovider"],
                   cwd=CODE, capture_output=True, text=True)
report = f"date {TODAY}\ncode {CODE_REF} {COMMIT}\ntorch {torch.__version__}\nreturncode {r.returncode}\n\n" + r.stdout[-6000:]
print(report[-2500:])
os.makedirs(f"{RES}/results/tests", exist_ok=True)
open(f"{RES}/results/tests/{TODAY}_{CODE_REF}_pytest.txt", "w").write(report)
print("push", push_run(RES, "results/tests", f"Test report {CODE_REF}"))
assert r.returncode == 0, "Tests failed, stop here and report"
'''

SMOKE = r'''
# Smoke run on real data, two rounds with DP at epsilon 8.
out = run([sys.executable, "-m", "src.run", "--config", "configs/experiments/smoke.yaml",
           "--data-dir", DATA, "--out", "/tmp/smoke_runs"], cwd=CODE)
print(out[-800:])
'''

GRID = r'''
# Grids, one process per GPU, each runs its shard of every grid in order.
os.makedirs("/kaggle/working/logs", exist_ok=True)
procs = []
for shard in (0, 1):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard))
    script = " && ".join(
        f"{sys.executable} scripts/run_grid.py --grid configs/grids/{g}.yaml --data-dir {DATA} "
        f"--results-repo {RES} --shard {shard} --nshards 2 --push --ckpt-dir /kaggle/working/checkpoints"
        for g in GRIDS)
    log = open(f"/kaggle/working/logs/shard{shard}.log", "w")
    procs.append(subprocess.Popen(["bash", "-c", script], cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT))
codes = [p.wait() for p in procs]
print("exit codes", codes)
for shard in (0, 1):
    print(f"----- shard {shard} tail")
    print(open(f"/kaggle/working/logs/shard{shard}.log").read().replace(token, "***")[-1500:])
'''

SUMMARY = r'''
# Summary tables, pushed to results/tables/.
for g in GRIDS:
    print(f"===== {g}")
    print(run([sys.executable, "scripts/aggregate.py", "--runs", f"{RES}/results/runs",
               "--grid", f"configs/grids/{g}.yaml", "--out", f"{RES}/results/tables/{g}.csv"], cwd=CODE))
print("push", push_run(RES, "results/tables", f"Summary tables {CODE_REF}"))
'''


CKPT_CHECK = r'''
# Checkpoints come from earlier launchers' outputs attached as inputs (Add Input, Notebook Output).
XAI_GRIDS = {xai}
pts = glob.glob("/kaggle/input/**/checkpoints/*.pt", recursive=True)
by_dir = {{}}
for p in pts:
    by_dir.setdefault(p.split("/checkpoints/")[0], []).append(p)
for d, v in sorted(by_dir.items()):
    print(f"{{len(v):4d}} checkpoints in {{d}}")
if not pts:
    raise SystemExit("No checkpoints found. Add Input, Notebook Output, and attach the outputs of "
                     "{needs}, then rerun this cell.")
'''

AUDIT = r'''
# Audit on two GPUs, each process takes every second run of the listed grids.
os.makedirs("/kaggle/working/logs", exist_ok=True)
procs = []
for shard in (0, 1):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard))
    cmd = ([sys.executable, "scripts/run_audit.py", "--grids", *GRIDS, "--xai-grids", *XAI_GRIDS,
            "--data-dir", DATA, "--ckpt-root", "/kaggle/input", "--results-repo", RES,
            "--shard", str(shard), "--nshards", "2", "--push"])
    log = open(f"/kaggle/working/logs/audit_shard{shard}.log", "w")
    procs.append(subprocess.Popen(cmd, cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT))
codes = [p.wait() for p in procs]
print("exit codes", codes)
for shard in (0, 1):
    print(f"----- shard {shard} tail")
    print(open(f"/kaggle/working/logs/audit_shard{shard}.log").read().replace(token, "***")[-2000:])
'''

AUDIT_SUMMARY = r'''
# Audit summary over seeds, pushed to results/tables/.
# no git pull here: origin is the token-free URL, and every audit push already rebased RES
print(run([sys.executable, "scripts/summarise_audit.py", "--results", f"{RES}/results", "--grids", *GRIDS,
           "--out", f"{RES}/results/tables/{NAME}_audit_summary.csv",
           "--md", f"{RES}/results/tables/{NAME}_audit_summary.md"], cwd=CODE))
print("push", push_run(RES, "results/tables", f"Audit summary {CODE_REF}"))
'''


def build_audit(name: str, ref: str, grids: list[str], xai: list[str], needs: str, title: str,
                table: str = "week07") -> str:
    def code(s):
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": s.strip("\n")}
    intro = (f"# {name}, {title}\n\nSettings: **Accelerator GPU T4 x2**, **Internet On**, secret "
             "`kaggle_dpfl-braintumor` attached, input `brain-mri-curated-v1` added, and the outputs of "
             f"{needs} added through Add Input, Notebook Output.\n\n"
             f"Runs code tag `{ref}`. Audits saved checkpoints of grids {', '.join(grids)} "
             f"(membership inference and source probe), with explanations for {', '.join(xai)}. "
             "No training happens here. Results go to `results/audit/<run_id>/`.\n\n"
             "Generated by `scripts/make_launcher.py --kind audit`, edit that script rather than this notebook.")
    setup = SETUP.format(ref=ref, grids=json.dumps(grids)) + f"NAME = {json.dumps(table)}\n"
    cells = [{"cell_type": "markdown", "metadata": {}, "source": intro},
             code(setup + CKPT_CHECK.format(xai=json.dumps(xai), needs=needs)), code(TESTS),
             code(AUDIT), code(AUDIT_SUMMARY)]
    return _write(name, cells)


def _write(name: str, cells: list) -> str:
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                      "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 4}
    path = os.path.join(ROOT, "notebooks", "kaggle", f"{name}.ipynb")
    json.dump(nb, open(path, "w"), indent=1)
    return path


def build(name: str, ref: str, grids: list[str], title: str) -> str:
    def code(s):
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": s.strip("\n")}
    intro = (f"# {name}, {title}\n\nSettings: **Accelerator GPU T4 x2**, **Internet On**, secret "
             "`kaggle_dpfl-braintumor` attached, input `brain-mri-curated-v1` added.\n\n"
             f"Runs code tag `{ref}` on grids {', '.join(grids)}. Code and results working copies live "
             "in `/tmp`, so neither is saved in the notebook output, and the token exists only in an "
             "environment variable. Checkpoints are copied to `/kaggle/working/checkpoints`.\n\n"
             "Generated by `scripts/make_launcher.py`, edit that script rather than this notebook.")
    cells = [{"cell_type": "markdown", "metadata": {}, "source": intro},
             code(SETUP.format(ref=ref, grids=json.dumps(grids))), code(TESTS), code(SMOKE),
             code(GRID), code(SUMMARY)]
    return _write(name, cells)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--grids", nargs="+", required=True)
    ap.add_argument("--title", default="grid run")
    ap.add_argument("--kind", choices=["grid", "audit"], default="grid")
    ap.add_argument("--xai-grids", nargs="*", default=[])
    ap.add_argument("--needs", default="the earlier launchers", help="launcher outputs to attach (audit)")
    a = ap.parse_args()
    for g in a.grids + a.xai_grids:
        assert os.path.exists(os.path.join(ROOT, "configs", "grids", f"{g}.yaml")), g
    if a.kind == "audit":
        print(build_audit(a.name, a.ref, a.grids, a.xai_grids, a.needs, a.title))
    else:
        print(build(a.name, a.ref, a.grids, a.title))
