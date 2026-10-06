"""Federated training with FedAvg, FedProx, and SCAFFOLD, with or without DP.

Every client uses Poisson sampling with a fixed rate q and runs
K = round(local_epochs / q) local steps per round. With DP enabled, each step
releases a privatised gradient (src/privacy/dp.py). The FedProx proximal term
and the SCAFFOLD control-variate correction depend only on model weights that
are themselves outputs of the DP mechanism, so adding them after privatisation
is post-processing and costs no privacy.

Participation is full in every round. Aggregation weights are client sizes,
which are treated as public metadata in the threat model (D19).
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.eval.metrics import classification_report
from src.models.models import build_model
from src.privacy import dp as dpm


@dataclass
class Cfg:
    algorithm: str = "fedavg"        # fedavg | fedprox | scaffold
    rounds: int = 60
    local_epochs: float = 1.0
    mu: float = 0.0                  # FedProx proximal weight
    lr: float = 0.1
    momentum: float = 0.0
    weight_decay: float = 0.0
    sample_rate: float = 0.1         # q, fixed for every client
    hflip: bool = True
    dp: bool = False
    sigma: float = 0.0
    clip_norm: float = 1.0
    max_physical_batch: int = 64
    seed: int = 0
    model: str = "small_cnn"
    pretrained: bool = False
    width: int = 16
    grad_clip: float | None = None   # global-norm clip for the non-DP path only (D28)
    schedule: str = "none"           # none | cosine, learning rate decayed across rounds
    eval_batch: int = 256

    @property
    def local_steps(self) -> int:
        return max(1, int(round(self.local_epochs / self.sample_rate)))


def _params(model: nn.Module) -> dict:
    return {k: v.detach().clone() for k, v in model.named_parameters()}


def _load(model: nn.Module, params: dict) -> None:
    with torch.no_grad():
        for k, v in model.named_parameters():
            v.copy_(params[k])


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, batch: int = 256) -> np.ndarray:
    model.eval()
    out = [F.softmax(model(x[s:s + batch]).float(), 1) for s in range(0, len(x), batch)]
    model.train()
    return torch.cat(out).cpu().numpy() if out else np.zeros((0, 4))


def round_lr(cfg: Cfg, t: int) -> float:
    """Learning rate for round t (0-based). Data-independent, so free under DP."""
    if cfg.schedule == "cosine":
        return cfg.lr * 0.5 * (1 + math.cos(math.pi * t / max(1, cfg.rounds)))
    return cfg.lr


def local_update(model, gparams, x, y, idx, cfg: Cfg, gen, c_glob=None, c_loc=None, lr=None):
    lr = cfg.lr if lr is None else lr
    _load(model, gparams)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.momentum,
                          weight_decay=cfg.weight_decay)
    n = len(idx)
    expected = cfg.sample_rate * n
    batch_sizes, clip_frac = [], []
    for _ in range(cfg.local_steps):
        b = idx[dpm.poisson_mask(n, cfg.sample_rate, gen, x.device)]
        xb, yb = x[b], y[b]
        if cfg.hflip and len(b):
            flip = torch.rand(len(b), generator=gen, device=x.device) < 0.5
            xb = torch.where(flip[:, None, None, None], xb.flip(-1), xb)
        batch_sizes.append(len(b))
        if cfg.dp:
            if len(b):
                gsum, norms = dpm.clipped_grad_sum(model, F.cross_entropy, xb, yb,
                                                   cfg.clip_norm, cfg.max_physical_batch,
                                                   return_norms=True)
                clip_frac.append(float((norms >= cfg.clip_norm * 0.999).float().mean()))
            else:
                gsum = {k: torch.zeros_like(v) for k, v in model.named_parameters()}
            g = dpm.add_noise(gsum, cfg.sigma, cfg.clip_norm, expected, gen)
        else:
            if not len(b):
                continue
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(xb), yb).backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            g = {k: v.grad.detach().clone() for k, v in model.named_parameters()}
        for k, p in model.named_parameters():
            gk = g[k]
            if cfg.algorithm == "fedprox" and cfg.mu > 0:
                gk = gk + cfg.mu * (p.detach() - gparams[k])
            if cfg.algorithm == "scaffold":
                gk = gk + c_glob[k] - c_loc[k]
            p.grad = gk
        opt.step()
    new = _params(model)
    dc = None
    if cfg.algorithm == "scaffold":
        # Option II of Karimireddy et al. (2020): c_i+ = c_i - c + (x - y_i) / (K * lr)
        scale = 1.0 / (cfg.local_steps * lr)
        dc = {k: (-c_glob[k] + (gparams[k] - new[k]) * scale) for k in new}
    stats = {"mean_batch": float(np.mean(batch_sizes)) if batch_sizes else 0.0,
             "clip_frac": float(np.mean(clip_frac)) if clip_frac else float("nan")}
    return new, dc, stats


def train(data, clients: list[np.ndarray], cfg: Cfg, log=print) -> dict:
    """Runs federated training and returns history, best/last test reports, predictions."""
    dev = data.x.device
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = build_model(cfg.model, 4, cfg.pretrained, cfg.width).to(dev)
    gparams = _params(model)
    idx_t = [torch.as_tensor(c, device=dev, dtype=torch.long) for c in clients]
    sizes = np.array([len(c) for c in clients], dtype=float)
    w = sizes / sizes.sum()
    gens = [torch.Generator(device=dev).manual_seed(cfg.seed * 100003 + j) for j in range(len(clients))]
    zeros = {k: torch.zeros_like(v) for k, v in gparams.items()}
    c_glob = {k: v.clone() for k, v in zeros.items()}
    c_loc = [{k: v.clone() for k, v in zeros.items()} for _ in clients]

    val_i, test_i = data.idx("val"), data.idx("test")
    xv, yv = data.x[val_i], data.y[val_i].cpu().numpy()
    best = {"val_loss": float("inf"), "round": -1, "params": {k: v.clone() for k, v in gparams.items()}}
    hist = []
    diverged = False
    for t in range(cfg.rounds):
        t0 = time.time()
        news, dcs, st = [], [], []
        for j, idx in enumerate(idx_t):
            p, dc, s = local_update(model, gparams, data.x, data.y, idx, cfg, gens[j],
                                    c_glob, c_loc[j], lr=round_lr(cfg, t))
            news.append(p); dcs.append(dc); st.append(s)
        gparams = {k: sum(w[j] * news[j][k] for j in range(len(news))) for k in gparams}
        if cfg.algorithm == "scaffold":
            for j in range(len(clients)):
                for k in c_loc[j]:
                    c_loc[j][k] += dcs[j][k]
            for k in c_glob:
                c_glob[k] += sum(dcs[j][k] for j in range(len(clients))) / len(clients)
        _load(model, gparams)
        pv = predict(model, xv, cfg.eval_batch)
        rep = classification_report(pv, yv)
        row = {"round": t + 1, "val_loss": rep["nll"], "val_acc": rep["accuracy"],
               "val_bal_acc": rep["balanced_accuracy"],
               "mean_batch": float(np.mean([s["mean_batch"] for s in st])),
               "clip_frac": float(np.nanmean([s["clip_frac"] for s in st])) if cfg.dp else None,
               "secs": round(time.time() - t0, 2)}
        hist.append(row)
        if not all(torch.isfinite(v).all() for v in gparams.values()) or not math.isfinite(rep["nll"]):
            diverged = True
            log(f"round {t + 1} diverged (non-finite weights or loss), stopping run")
            break
        if rep["nll"] < best["val_loss"]:
            best = {"val_loss": rep["nll"], "round": t + 1,
                    "params": {k: v.clone() for k, v in gparams.items()}}
        if (t + 1) % max(1, cfg.rounds // 10) == 0 or t == 0:
            log(f"round {t + 1:3d} val_loss {rep['nll']:.4f} val_acc {rep['accuracy']:.4f} "
                f"bal_acc {rep['balanced_accuracy']:.4f} batch {row['mean_batch']:.1f} {row['secs']}s")

    if diverged and best["round"] < 0:
        gparams = best["params"]                     # initial weights, still finite
    out = {"history": hist, "best_round": best["round"], "diverged": diverged}
    xt, yt = data.x[test_i], data.y[test_i].cpu().numpy()
    src = data.splits.source.values[test_i]
    for tag, params in (("best", best["params"]), ("last", gparams)):
        _load(model, params)
        pt = predict(model, xt, cfg.eval_batch)
        out[f"test_{tag}"] = classification_report(pt, yt)
        out[f"test_{tag}_by_source"] = {s: classification_report(pt[src == s], yt[src == s])
                                        for s in np.unique(src)}
        if tag == "best":
            out["pred_best"] = pt
            tr = torch.as_tensor(np.concatenate(clients), device=dev)
            out["train_acc_best"] = float((predict(model, data.x[tr], cfg.eval_batch).argmax(1)
                                           == data.y[tr].cpu().numpy()).mean())
            out["model_state_best"] = {k: v.cpu() for k, v in model.state_dict().items()}
    out["test_index"] = test_i
    return out
