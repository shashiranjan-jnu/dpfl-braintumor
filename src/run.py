"""Single run: one resolved config -> one folder results/runs/<run_id>/.

    python -m src.run --config configs/experiments/smoke.yaml \
        --data-dir /kaggle/input/brain-mri-curated-v1 --out results/runs

Files written, metrics.json last so that its presence marks a finished run:
    config.yaml       resolved configuration
    env.json          git commit, versions, device, timing, privacy parameters
    log.txt           progress log
    predictions.csv   test-set softmax outputs of the selected model
    metrics.json      history, test metrics (best-val and last round), privacy
The selected model weights go to <run_dir>/model_best.pt, which .gitignore excludes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml

from src.data.data import load_curated
from src.fl.federated import Cfg, train
from src.privacy import dp as dpm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def resolve(cfg: dict) -> dict:
    base = yaml.safe_load(open(os.path.join(ROOT, "configs", "base.yaml")))
    return deep_update(base, cfg)


def run_id(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def clean(o):
    """Strict JSON: NaN -> null, infinity -> the string "inf"."""
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        o = float(o)
        if np.isnan(o):
            return None
        if np.isinf(o):
            return "inf" if o > 0 else "-inf"
    if isinstance(o, np.integer):
        return int(o)
    return o


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def to_cfg(c: dict, sigma: float) -> Cfg:
    return Cfg(algorithm=c["fl"]["algorithm"], rounds=c["fl"]["rounds"],
               local_epochs=c["fl"]["local_epochs"], mu=c["fl"]["mu"],
               lr=c["optim"]["lr"], momentum=c["optim"]["momentum"],
               weight_decay=c["optim"]["weight_decay"], sample_rate=c["dp"]["sample_rate"],
               hflip=c["data"]["hflip"], dp=c["dp"]["enabled"], sigma=sigma,
               clip_norm=c["dp"]["clip_norm"], max_physical_batch=c["dp"]["max_physical_batch"],
               seed=c["train"]["seed"], model=c["model"]["name"],
               pretrained=c["model"]["pretrained"], width=c["model"].get("width", 16),
               grad_clip=c["optim"].get("grad_clip"), schedule=c["optim"].get("schedule", "none"),
               unit_level=c["dp"].get("unit", "record") == "unit")


def main(cfg_in: dict, data_dir: str, out_root: str, data=None, device=None, force=False) -> str:
    cfg = resolve(cfg_in)
    rid = run_id(cfg)
    rdir = os.path.join(out_root, rid)
    if os.path.exists(os.path.join(rdir, "metrics.json")) and not force:
        print(f"[skip] {rid} already finished")
        return rid
    os.makedirs(rdir, exist_ok=True)
    logf = open(os.path.join(rdir, "log.txt"), "w")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line)
        logf.write(line + "\n")
        logf.flush()

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if cfg["train"].get("deterministic", True):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    if data is None:
        data = load_curated(data_dir, device)

    mode = cfg["fl"]["mode"]
    if mode == "central":
        clients = data.clients("central")
    elif mode == "local":
        clients = [data.clients(cfg["data"]["scenario"])[cfg["fl"]["local_client"]]]
    else:
        clients = data.clients(cfg["data"]["scenario"])

    # Unit-level DP (D33). The key is optional and absent from base.yaml, so the
    # run_id of every record-level run stays unchanged.
    unit = cfg["dp"].get("unit", "record")
    if unit not in ("record", "unit"):
        raise ValueError(f"dp.unit must be record or unit, got {unit}")
    scen = cfg["data"]["scenario"]
    if unit == "unit" and cfg["dp"]["enabled"] and mode == "federated" and scen.startswith(("iid", "dir")):
        raise ValueError(f"Unit-level DP is not allowed on {scen}, whose stored partition splits "
                         "patients across clients (D33). Use natural3 or a udir partition.")
    client_units = [data.units_of(k) for k in clients] if unit == "unit" else None
    # A unit split across m clients is released by m mechanisms, so its
    # privacy loss composes over m clients (same sigma, q, steps).
    owners = {}
    for j, k in enumerate(clients):
        for u in np.unique(data.splits.unit.values[k]):
            owners[u] = owners.get(u, 0) + 1
    m_max = max(owners.values()) if owners else 1

    tmp = to_cfg(cfg, 0.0)
    steps = cfg["fl"]["rounds"] * tmp.local_steps
    q, delta = cfg["dp"]["sample_rate"], cfg["dp"]["delta"]
    sigma, eps = 0.0, float("inf")
    if cfg["dp"]["enabled"]:
        nm = cfg["dp"]["noise_multiplier"]
        sigma = nm if nm is not None else dpm.calibrate_sigma(
            cfg["dp"]["target_epsilon"], q, steps, delta, cfg["dp"]["accountant"])
        eps = dpm.epsilon_opacus(sigma, q, steps, delta, cfg["dp"]["accountant"])
    c = to_cfg(cfg, sigma)
    acc = cfg["dp"]["accountant"]
    # Epsilon of the guarantee actually stated in the paper:
    #  unit_epsilon_worst  unit-level runs, worst unit, composed over the m_max clients holding it
    #  epsilon_if_k_runs   cost of choosing among k candidate runs on validation data (D33),
    #                      since every candidate was trained on the same private records
    extra = {"unit": unit, "units_max_clients": int(m_max),
             "n_units": [int(len(u)) for u in client_units] if client_units else None}
    if cfg["dp"]["enabled"] and sigma > 0:
        extra["unit_epsilon_worst"] = (dpm.epsilon_opacus(sigma, q, steps * m_max, delta, acc)
                                       if unit == "unit" else None)
        m_eff = m_max if unit == "unit" else 1
        extra["epsilon_if_k_runs"] = {str(k): dpm.epsilon_opacus(sigma, q, steps * m_eff * k, delta, acc)
                                      for k in (2, 3)}

    yaml.safe_dump(cfg, open(os.path.join(rdir, "config.yaml"), "w"), sort_keys=False)
    env = {"run_id": rid, "git_commit": git_commit(), "python": platform.python_version(),
           "torch": torch.__version__, "device": str(device),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "kaggle_image": os.environ.get("KAGGLE_DOCKER_IMAGE"), "argv": sys.argv,
           "n_clients": len(clients), "client_sizes": [int(len(k)) for k in clients],
           "local_steps": c.local_steps, "total_steps_per_client": steps,
           "sample_rate": q, "sigma": sigma, "delta": delta, "epsilon_spent": eps,
           "expected_batch_per_client": [round(q * len(k), 2) for k in clients], **extra}
    log(f"run {rid} mode {mode} algo {c.algorithm} clients {env['client_sizes']} "
        f"steps/client {steps} sigma {sigma:.4f} eps {eps:.3f} unit {unit} m_max {m_max}")

    t0 = time.time()
    res = train(data, clients, c, log, client_units=client_units)
    env["train_seconds"] = round(time.time() - t0, 1)
    json.dump(clean(env), open(os.path.join(rdir, "env.json"), "w"), indent=2)

    ti = res["test_index"]
    pred = pd.DataFrame(res["pred_best"].round(5), columns=["p_glioma", "p_meningioma",
                                                             "p_notumor", "p_pituitary"])
    pred.insert(0, "y", data.y[torch.as_tensor(ti, device=data.y.device)].cpu().numpy())
    pred.insert(0, "image_id", data.splits.image_id.values[ti])
    pred.to_csv(os.path.join(rdir, "predictions.csv"), index=False)
    torch.save(res["model_state_best"], os.path.join(rdir, "model_best.pt"))

    metrics = {"run_id": rid, "best_round": res["best_round"], "diverged": res["diverged"],
               "train_acc_best": res["train_acc_best"],
               "test_best": res["test_best"], "test_last": res["test_last"],
               "test_best_by_source": res["test_best_by_source"],
               "privacy": {"enabled": cfg["dp"]["enabled"], "epsilon": eps, "delta": delta,
                           "sigma": sigma, "q": q, "steps": steps,
                           "clip_norm": cfg["dp"]["clip_norm"], **extra},
               "history": res["history"], "train_seconds": env["train_seconds"]}
    json.dump(clean(metrics), open(os.path.join(rdir, "metrics.json"), "w"), indent=2)
    tb = res["test_best"]
    log(f"done {rid} best_round {res['best_round']} test_acc {tb['accuracy']:.4f} "
        f"bal_acc {tb['balanced_accuracy']:.4f} ece {tb['ece']:.4f} "
        f"train_acc {res['train_acc_best']:.4f} {env['train_seconds']}s")
    logf.close()
    return rid


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "runs"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(yaml.safe_load(open(a.config)) or {}, a.data_dir, a.out, force=a.force)
