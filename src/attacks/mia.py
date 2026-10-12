"""Membership inference audit, loss-threshold attack (RQ4, Step 4.1).

Members are the training split, non-members the test split; the validation
split is excluded because it chose the reported round. Scores are -loss per
image, so a higher score means "more likely a member". Patient-level scores
average the image scores of each Figshare patient (D32), comparing training
patients with test patients, which are disjoint by construction (Step 1.4).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve


@torch.no_grad()
def per_example_loss(model, x, y, batch: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(x), batch):
        out.append(F.cross_entropy(model(x[s:s + batch]), y[s:s + batch], reduction="none"))
    return torch.cat(out).cpu().numpy()


def attack_summary(member_scores: np.ndarray, nonmember_scores: np.ndarray) -> dict:
    s = np.concatenate([member_scores, nonmember_scores])
    lab = np.concatenate([np.ones(len(member_scores)), np.zeros(len(nonmember_scores))])
    fpr, tpr, _ = roc_curve(lab, s)
    out = {"auc": float(roc_auc_score(lab, s)), "n_members": int(len(member_scores)),
           "n_nonmembers": int(len(nonmember_scores)),
           "advantage": float(np.max(tpr - fpr))}
    for f in (0.001, 0.01, 0.1):
        out[f"tpr_at_fpr_{f}"] = float(np.interp(f, fpr, tpr))
    out["fpr_resolution"] = 1.0 / max(1, len(nonmember_scores))
    return out


def audit(model, data, splits) -> dict:
    """Image-level and Figshare patient-level loss-threshold attacks for one model."""
    tr = np.where(splits.split.values == "train")[0]
    te = np.where(splits.split.values == "test")[0]
    idx = np.concatenate([tr, te])
    loss = per_example_loss(model, data.x[idx], data.y[idx])
    score = -loss
    s_tr, s_te = score[:len(tr)], score[len(tr):]
    res = {"image": attack_summary(s_tr, s_te),
           "train_loss_mean": float(loss[:len(tr)].mean()), "test_loss_mean": float(loss[len(tr):].mean())}
    src = splits.source.values
    unit = splits.unit.values
    for agg in ("mean", "max"):
        f = np.mean if agg == "mean" else np.max
        m = [f(s_tr[(unit[tr] == u)]) for u in np.unique(unit[tr][src[tr] == "figshare"])]
        n = [f(s_te[(unit[te] == u)]) for u in np.unique(unit[te][src[te] == "figshare"])]
        res[f"patient_{agg}"] = attack_summary(np.array(m), np.array(n))
    sizes = np.array([np.sum(unit[tr] == u) for u in np.unique(unit[tr][src[tr] == "figshare"])])
    res["figshare_train_slices_per_patient"] = {"median": float(np.median(sizes)), "max": int(sizes.max())}
    return res
