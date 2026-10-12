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
    unit_level: bool = False         # DP over units (patients) instead of images (D33)
    scaffold_c: str = "uniform"      # uniform (historical runs) | size: weight control variates like the model (D40)
    schedule: str = "none"           # none | cosine, learning rate decayed across rounds
    weight_noise: float = 0.0        # Gupta-style comparator: N(0, s^2) added to client weights, no clipping (D34)
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


def _sample(idx, units, cfg: Cfg, gen, dev):
    """One Poisson batch. Returns image indices and, at unit level, the unit sizes.

    Record level samples images with rate q. Unit level samples units with
    rate q and takes every image of each sampled unit, in unit order.
    """
    if units is None:
        return idx[dpm.poisson_mask(len(idx), cfg.sample_rate, gen, dev)], None
    keep = dpm.poisson_mask(len(units), cfg.sample_rate, gen, dev).nonzero().flatten().tolist()
    if not keep:
        return idx[:0], []
    return torch.cat([units[i] for i in keep]), [len(units[i]) for i in keep]


def control_weights(cfg: Cfg, size_weights: np.ndarray) -> np.ndarray:
    """Weights for averaging SCAFFOLD control variates into the server's c (D40).

    The server model is a size-weighted average, so it descends sum_j w_j f_j. The
    server control variate must estimate the gradient of that same objective,
    c = sum_j w_j c_j. With the original uniform average, every client is pulled
    towards the unweighted mean gradient, which up-weights small clients. "uniform"
    is kept only so that runs made before D40 stay reproducible.
    """
    if cfg.scaffold_c == "size":
        return np.asarray(size_weights, dtype=float)
    if cfg.scaffold_c == "uniform":
        return np.full(len(size_weights), 1.0 / len(size_weights))
    raise ValueError(f"scaffold_c must be uniform or size, got {cfg.scaffold_c}")


def momentum_step_sum(steps: int, beta: float) -> float:
    """Total displacement, in units of lr * g, of `steps` heavy-ball SGD steps on a constant gradient g.

    torch SGD (dampening 0, no Nesterov) keeps buf_k = beta * buf_{k-1} + g and steps
    p -= lr * buf_k, so buf_k = g (1 - beta^k) / (1 - beta) and the displacement after
    K steps is lr * g * sum_{k=1..K} (1 - beta^k) / (1 - beta). With beta = 0 this is K.
    The momentum buffer starts at zero every round because the optimiser is rebuilt.
    """
    if beta == 0:
        return float(steps)
    return float(sum((1 - beta ** k) / (1 - beta) for k in range(1, steps + 1)))


def local_update(model, gparams, x, y, idx, cfg: Cfg, gen, c_glob=None, c_loc=None, lr=None,
                 units=None):
    lr = cfg.lr if lr is None else lr
    units = units if (cfg.unit_level and cfg.dp) else None
    _load(model, gparams)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.momentum,
                          weight_decay=cfg.weight_decay)
    n = len(idx) if units is None else len(units)
    expected = cfg.sample_rate * n      # q * n_j images, or q * U_j units at unit level
    batch_sizes, clip_frac = [], []
    taken = 0                           # optimiser steps actually applied (non-DP skips empty batches)
    for _ in range(cfg.local_steps):
        b, sizes = _sample(idx, units, cfg, gen, x.device)
        xb, yb = x[b], y[b]
        if cfg.hflip and len(b):
            flip = torch.rand(len(b), generator=gen, device=x.device) < 0.5
            xb = torch.where(flip[:, None, None, None], xb.flip(-1), xb)
        batch_sizes.append(len(b))
        if cfg.dp:
            if len(b) and sizes is not None:
                gsum, norms = dpm.clipped_unit_grad_sum(model, F.cross_entropy, xb, yb, sizes,
                                                        cfg.clip_norm, cfg.max_physical_batch,
                                                        return_norms=True)
                clip_frac.append(float((norms >= cfg.clip_norm * 0.999).float().mean()))
            elif len(b):
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
        taken += 1
    new = _params(model)
    dc = None
    if cfg.algorithm == "scaffold":
        # Option II of Karimireddy et al. (2020): c_i+ = c_i - c + (x - y_i) / (K * lr).
        # K * lr is the displacement per unit gradient under plain SGD. With heavy-ball
        # momentum the same gradient moves the weights by lr * S(K, beta) instead
        # (S = 41.4 for K = 10, beta = 0.9), so K * lr would inflate c_i about 4.1 times
        # (D38). S is exact for a constant gradient and equals K at beta = 0.
        denom = lr * momentum_step_sum(taken, cfg.momentum)
        scale = 1.0 / denom if denom > 0 else 0.0
        dc = {k: (-c_glob[k] + (gparams[k] - new[k]) * scale) for k in new}
    stats = {"mean_batch": float(np.mean(batch_sizes)) if batch_sizes else 0.0,
             "clip_frac": float(np.mean(clip_frac)) if clip_frac else float("nan")}
    return new, dc, stats


def train(data, clients: list[np.ndarray], cfg: Cfg, log=print, client_units=None) -> dict:
    """Runs federated training and returns history, best/last test reports, predictions."""
    dev = data.x.device
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = build_model(cfg.model, 4, cfg.pretrained, cfg.width).to(dev)
    gparams = _params(model)
    idx_t = [torch.as_tensor(c, device=dev, dtype=torch.long) for c in clients]
    units_t = [None] * len(clients)
    if cfg.unit_level and cfg.dp:
        if client_units is None:
            raise ValueError("unit_level DP needs client_units")
        units_t = [[torch.as_tensor(u, device=dev, dtype=torch.long) for u in cu] for cu in client_units]
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
                                    c_glob, c_loc[j], lr=round_lr(cfg, t), units=units_t[j])
            if cfg.weight_noise > 0:
                # Unclipped weight noise on the released client model, as in Gupta et al. (2026).
                # Sensitivity is unbounded, so no epsilon can be derived for this mechanism.
                p = {k: v + cfg.weight_noise * torch.randn(v.shape, generator=gens[j], device=dev)
                     for k, v in p.items()}
            news.append(p); dcs.append(dc); st.append(s)
        gparams = {k: sum(w[j] * news[j][k] for j in range(len(news))) for k in gparams}
        if cfg.algorithm == "scaffold":
            for j in range(len(clients)):
                for k in c_loc[j]:
                    c_loc[j][k] += dcs[j][k]
            cw = control_weights(cfg, w)
            for k in c_glob:
                c_glob[k] += sum(cw[j] * dcs[j][k] for j in range(len(clients)))
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
