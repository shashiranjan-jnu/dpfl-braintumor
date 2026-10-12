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


OLD_AUDIT = ["week03_central", "week04_unit_central", "week04_unit_central_ext", "week03_fed_seeds_a",
             "week03_fed_seeds_b", "week04_unit_fed"]
XAI_SET = ["week03_central", "week04_unit_central", "week04_unit_central_ext", "week03_fed_seeds_a"]

LIRA_SELECT = r'''
# Weight-noise sigma selection at seed 0 (5 runs), then the seeds and shadow grids by the fixed rule (D45).
os.makedirs("/kaggle/working/logs", exist_ok=True)
def shards(grids, tag):
    procs = []
    for shard in (0, 1):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard))
        script = " && ".join(
            f"{sys.executable} scripts/run_grid.py --grid configs/grids/{g}.yaml --data-dir {DATA} "
            f"--results-repo {RES} --shard {shard} --nshards 2 --push --ckpt-dir /kaggle/working/checkpoints"
            for g in grids)
        log = open(f"/kaggle/working/logs/{tag}_shard{shard}.log", "w")
        procs.append(subprocess.Popen(["bash", "-c", script], cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT))
    codes = [p.wait() for p in procs]
    print(tag, "exit codes", codes)
    for shard in (0, 1):
        print(f"----- {tag} shard {shard} tail")
        print(open(f"/kaggle/working/logs/{tag}_shard{shard}.log").read().replace(token, "***")[-1500:])

shards(["week07_wn_select"], "select")
print(run([sys.executable, "scripts/make_shadow_grid.py", "wn", "--results", f"{RES}/results", "--out", RES], cwd=CODE))
for g in ("week07_wn_seeds", "week07_wn_shadows"):
    shutil.copy(f"{RES}/configs/grids/{g}.yaml", f"{CODE}/configs/grids/{g}.yaml")
print("push", push_run(RES, "configs/grids", "Weight-noise grids from week07_wn_select"))
'''

LIRA_TRAIN = r'''
# Training: weight-noise seeds (5 per sigma), LiRA shadows (64), weight-noise shadows (16 per sigma).
# About 4.8 hours on two GPUs (DP shadows about 8 min each). Finished runs are skipped if this cell is rerun.
shards(["week07_wn_seeds", "week07_lira_shadows", "week07_wn_shadows"], "train")
'''

LIRA_AUDIT = r'''
# Audit: logits for every target and shadow, border analysis for the explanation set,
# loss attack and probe for the new targets, probe baselines once. No training.
AUDIT_TARGETS = OLD_AUDIT + ["week07_wn_select", "week07_wn_seeds"]
os.makedirs("/kaggle/working/logs", exist_ok=True)
procs = []
for shard in (0, 1):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard))
    common = ["--data-dir", DATA, "--ckpt-root", "/kaggle/input", "/kaggle/working/checkpoints",
              "--results-repo", RES, "--shard", str(shard), "--nshards", "2", "--push"]
    a = [sys.executable, "scripts/run_audit.py", "--grids", *AUDIT_TARGETS, "--xai-grids", *XAI_GRIDS,
         "--parts", "mia", "probe", "logits", "border", "--probe-baselines", *common]
    b = [sys.executable, "scripts/run_audit.py", "--grids", "week07_lira_shadows", "week07_wn_shadows",
         "--parts", "logits", *common]
    script = " ".join(map(shlex.quote, a)) + " && " + " ".join(map(shlex.quote, b))
    log = open(f"/kaggle/working/logs/audit_shard{shard}.log", "w")
    procs.append(subprocess.Popen(["bash", "-c", script], cwd=CODE, env=env, stdout=log, stderr=subprocess.STDOUT))
codes = [p.wait() for p in procs]
print("exit codes", codes)
for shard in (0, 1):
    txt = open(f"/kaggle/working/logs/audit_shard{shard}.log").read().replace(token, "***")
    print(f"----- audit shard {shard}")
    print("\n".join(l for l in txt.splitlines() if "finished" in l or "missing" in l or "FAILED" in l or "baseline" in l))
'''

LIRA_SUMMARY = r'''
# Tables: weight-noise utility, stratified loss attack, LiRA, border and consistency. Pushed to results/tables/.
print(run([sys.executable, "scripts/summarise.py", "--runs", f"{RES}/results/runs",
           "--grids", "configs/grids/week07_wn_seeds.yaml", "--out", f"{RES}/results/tables/week07_wn_seeds_summary.csv",
           "--md", f"{RES}/results/tables/week07_wn_seeds_summary.md"], cwd=CODE))
print(run([sys.executable, "scripts/mia_tables.py", "--results", f"{RES}/results",
           "--out", f"{RES}/results/tables/week07"], cwd=CODE)[-6000:])
print(run([sys.executable, "scripts/xai_tables.py", "--results", f"{RES}/results",
           "--out", f"{RES}/results/tables/week07_xai_border"], cwd=CODE)[-4000:])
for t in ("pretrained", "random"):
    p = f"{RES}/results/audit/_baselines/probe_{t}.json"
    print("probe baseline", t, open(p).read() if os.path.exists(p) else "missing")
print("push", push_run(RES, "results/tables", f"Tables {CODE_REF}"))
'''


def build_lira(name: str, ref: str, title: str) -> str:
    def code(s):
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": s.strip("\n")}
    needs = "07-week3-core, 08-fed-seeds-a, 09-fed-seeds-b-unit and 10-unit-udir"
    intro = (f"# {name}, {title}\n\nSettings: **Accelerator GPU T4 x2**, **Internet On**, secret "
             "`kaggle_dpfl-braintumor` attached, input `brain-mri-curated-v1` added, and the outputs of "
             f"{needs} added through Add Input, Notebook Output.\n\n"
             f"Runs code tag `{ref}`. Trains the weight-noise comparator (5 selection runs, then seeds and "
             "shadows by a fixed rule) and 64 LiRA shadow models, then saves logits of every audited model, "
             "the skull-border analysis and probe baselines. About 5.5 hours on two GPUs.\n\n"
             "Generated by `scripts/make_launcher.py --kind lira`, edit that script rather than this notebook.")
    setup = (SETUP.format(ref=ref, grids=json.dumps(OLD_AUDIT)) + "import shlex\n"
             + f"OLD_AUDIT = {json.dumps(OLD_AUDIT)}\n")
    cells = [{"cell_type": "markdown", "metadata": {}, "source": intro},
             code(setup + CKPT_CHECK.format(xai=json.dumps(XAI_SET), needs=needs)), code(TESTS),
             code(LIRA_SELECT), code(LIRA_TRAIN), code(LIRA_AUDIT), code(LIRA_SUMMARY)]
    return _write(name, cells)


RECOVER_AUDIT = LIRA_AUDIT.replace(' "--probe-baselines",', '').replace(
    "# loss attack and probe for the new targets, probe baselines once. No training.",
    "# loss attack and probe for the new targets. No training. Parts whose files are missing are redone (D47).")


def build_recover(name: str, ref: str, title: str) -> str:
    def code(s):
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": s.strip("\n")}
    needs = "07-week3-core, 08-fed-seeds-a, 09-fed-seeds-b-unit, 10-unit-udir and 15-lira-wn"
    intro = (f"# {name}, {title}\n\nSettings: **Accelerator GPU T4 x2**, **Internet On**, secret "
             "`kaggle_dpfl-braintumor` attached, input `brain-mri-curated-v1` added, and the outputs of "
             f"{needs} added through Add Input, Notebook Output.\n\n"
             f"Runs code tag `{ref}`. No training. Recomputes the logits of every audited and shadow model and "
             "the border analysis with saved Grad-CAM maps, which launcher 15 computed but could not push "
             "because .gitignore excluded .npz files (D47), then builds the LiRA, stratified-attack and border "
             "tables. About 40 minutes.\n\n"
             "Generated by `scripts/make_launcher.py --kind recover`, edit that script rather than this notebook.")
    setup = (SETUP.format(ref=ref, grids=json.dumps(OLD_AUDIT)) + "import shlex\n"
             + f"OLD_AUDIT = {json.dumps(OLD_AUDIT)}\n")
    cells = [{"cell_type": "markdown", "metadata": {}, "source": intro},
             code(setup + CKPT_CHECK.format(xai=json.dumps(XAI_SET), needs=needs)), code(TESTS),
             code(RECOVER_AUDIT), code(LIRA_SUMMARY)]
    return _write(name, cells)


NCC_SETUP = r'''
import os, sys, glob, subprocess, datetime, shutil
from kaggle_secrets import UserSecretsClient

USER, EMAIL, REPO = "shashiranjan-jnu", "shashi17_scs@jnu.ac.in", "dpfl-braintumor"
CODE_REF = "{ref}"
token = UserSecretsClient().get_secret("kaggle_dpfl-braintumor")
os.environ["GIT_PUSH_URL"] = f"https://{{token}}@github.com/{{USER}}/{{REPO}}.git"
CLEAN = f"https://github.com/{{USER}}/{{REPO}}.git"
CODE, RES = "/tmp/code", "/tmp/results"

def run(cmd, cwd=None, check=True):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
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
sys.path.insert(0, CODE)
from scripts.run_grid import push_run

hits = glob.glob("/kaggle/input/**/brain_mri_curated_v1.npz", recursive=True)
if not hits:
    raise SystemExit("Attach the dataset brain-mri-curated-v1 through Add Input, then rerun this cell.")
DATA = os.path.dirname(hits[0])
mats = glob.glob("/kaggle/input/**/brainTumorDataPublic_*.zip", recursive=True) + \
       glob.glob("/kaggle/input/**/*.mat", recursive=True)
dirs = [r for r, _, _ in os.walk("/kaggle/input")
        if os.path.basename(r) in ("brain-tumor-classification-mri", "brain-tumor-detection")]
print("code", CODE_REF, run(["git", "rev-parse", "--short", "HEAD"], cwd=CODE).strip(), "| curated", DATA)
print("raw Figshare files found:", len(mats), "| raw folders found:", sorted({{os.path.basename(d) for d in dirs}}))
print("Part B needs Figshare zips or .mat files and both folders; Part A needs only the curated dataset.")
'''

NCC_A = r'''
# Part A: all pairs of the 3845 curated images, NCC on 64 x 64, plain and flipped. A few minutes on CPU.
print(run([sys.executable, "scripts/ncc_check.py", "curated", "--data-dir", DATA,
           "--out", f"{RES}/results/ncc"], cwd=CODE)[-6000:])
print("push", push_run(RES, "results/ncc", f"NCC check, curated set ({CODE_REF})"))
from IPython.display import Image, display
p = f"{RES}/results/ncc/top_cross_split_pairs.png"
if os.path.exists(p):
    display(Image(p))
'''

NCC_B = r'''
# Part B: every removed duplicate against the image it was merged into, rebuilt from the raw sources.
out = run([sys.executable, "scripts/ncc_check.py", "removed", "--raw-dirs", "/kaggle/input",
           "--out", f"{RES}/results/ncc"], cwd=CODE, check=False)
print(out[-6000:])
print("push", push_run(RES, "results/ncc", f"NCC check, removed duplicates ({CODE_REF})"))
'''


FIG_SETUP = NCC_SETUP.split("hits = glob.glob")[0] + r'''
print(run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{{CODE}}/requirements.txt"])[-500:])  # opacus etc.
import torch
TODAY = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
COMMIT = run(["git", "rev-parse", "HEAD"], cwd=CODE).strip()
for f in ("results/ncc/curated_pairs_ncc090.csv", "results/tables/week05_h5_mechanism.csv",
          "configs/grids/week07_wn_seeds.yaml"):
    assert os.path.exists(f"{{RES}}/{{f}}"), f"missing {{f}} in the results repository"
n_logits = len(glob.glob(f"{{RES}}/results/audit/*/logits.npz"))
print("code", CODE_REF, run(["git", "rev-parse", "--short", "HEAD"], cwd=CODE).strip(), "| logits files", n_logits)
'''

FIG_TABLES = r'''
# Tables the figures read, recomputed from the run and audit files on GitHub (CPU, a few minutes).
R = f"{RES}/results"
T = f"{R}/tables"
print(run([sys.executable, "scripts/ncc_sensitivity.py", "--results", R, "--out", f"{T}/week07_ncc_sensitivity"], cwd=CODE)[-1500:])
print(run([sys.executable, "scripts/mia_tables.py", "--results", R, "--grids-dir", f"{RES}/configs/grids",
           "--exclude", f"{R}/ncc/contaminated_test_ids.txt", "--out", f"{T}/week07_clean"], cwd=CODE)[-2500:])
print(run([sys.executable, "scripts/xai_tables.py", "--results", R, "--out", f"{T}/week07_xai_border"], cwd=CODE)[-800:])
'''

FIG_MAKE = r'''
# Figures 1 to 4, then a check against the copies already on GitHub, then push.
import pandas as pd, numpy as np
old = {f: pd.read_csv(f"{RES}/figures/figure_data/{f}") for f in
       ("fig1_privacy_utility.csv", "fig2_lira_roc.csv", "fig3_explanations.csv", "fig4_h5_noise.csv")
       if os.path.exists(f"{RES}/figures/figure_data/{f}")}
print(run([sys.executable, "scripts/make_figures.py", "--results", R, "--out", f"{RES}/figures"], cwd=CODE))
for f, o in old.items():
    n = pd.read_csv(f"{RES}/figures/figure_data/{f}")
    num = [c for c in o.columns if o[c].dtype.kind == "f" and c in n]
    diff = max(float(np.nanmax(np.abs(o[c].values - n[c].values))) for c in num) if len(o) == len(n) else float("inf")
    print(f"{f}: {len(n)} rows, largest difference from the committed version {diff:.2e}")
from IPython.display import Image, display
for f in ("fig1_privacy_utility", "fig2_lira_roc", "fig3_explanations", "fig4_h5_noise"):
    display(Image(f"{RES}/figures/{f}.png", width=900))
print("push tables", push_run(RES, "results/tables", f"Tables regenerated on Kaggle ({CODE_REF})"))
print("push figures", push_run(RES, "figures", f"Figures regenerated on Kaggle ({CODE_REF})"))
'''


def build_figures(name: str, ref: str, title: str) -> str:
    def code(s):
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": s.strip("\n")}
    intro = (f"# {name}, {title}\n\nSettings: **Accelerator None (CPU)**, **Internet On**, secret "
             "`kaggle_dpfl-braintumor` attached. No dataset input is needed: everything is computed from the run, "
             "audit and NCC files already on GitHub.\n\n"
             f"Runs code tag `{ref}`: tests, then the tables behind the figures (clean test set, D51), then "
             "figures 1 to 4 with a comparison against the committed versions, then pushes tables and figures.\n\n"
             "Generated by `scripts/make_launcher.py --kind figures`, edit that script rather than this notebook.")
    cells = [{"cell_type": "markdown", "metadata": {}, "source": intro},
             code(FIG_SETUP.format(ref=ref)), code(TESTS),
             code(FIG_TABLES), code(FIG_MAKE)]
    return _write(name, cells)


def build_ncc(name: str, ref: str, title: str) -> str:
    def code(s):
        return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                "source": s.strip("\n")}
    intro = (f"# {name}, {title}\n\nSettings: **Accelerator None (CPU)**, **Internet On**, secret "
             "`kaggle_dpfl-braintumor` attached. Inputs: the dataset `brain-mri-curated-v1` (Part A), and for "
             "Part B the same raw inputs as notebook 02: the Figshare zips (output of 01-data-acquisition) and "
             "the Kaggle datasets sartajbhuvaji/brain-tumor-classification-mri and "
             "ahmedhamada0/brain-tumor-detection.\n\n"
             f"Runs code tag `{ref}`. Uses no GPU quota. Results go to `results/ncc/`.\n\n"
             "Generated by `scripts/make_launcher.py --kind ncc`, edit that script rather than this notebook.")
    cells = [{"cell_type": "markdown", "metadata": {}, "source": intro},
             code(NCC_SETUP.format(ref=ref)), code(NCC_A), code(NCC_B)]
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
    ap.add_argument("--kind", choices=["grid", "audit", "lira", "recover", "ncc", "figures"], default="grid")
    ap.add_argument("--xai-grids", nargs="*", default=[])
    ap.add_argument("--needs", default="the earlier launchers", help="launcher outputs to attach (audit)")
    a = ap.parse_args()
    for g in a.grids + a.xai_grids:
        assert os.path.exists(os.path.join(ROOT, "configs", "grids", f"{g}.yaml")), g
    if a.kind == "figures":
        print(build_figures(a.name, a.ref, a.title))
    elif a.kind == "ncc":
        print(build_ncc(a.name, a.ref, a.title))
    elif a.kind == "recover":
        print(build_recover(a.name, a.ref, a.title))
    elif a.kind == "lira":
        print(build_lira(a.name, a.ref, a.title))
    elif a.kind == "audit":
        print(build_audit(a.name, a.ref, a.grids, a.xai_grids, a.needs, a.title))
    else:
        print(build(a.name, a.ref, a.grids, a.title))
