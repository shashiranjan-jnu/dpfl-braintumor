"""Classification, calibration, and clinical metrics."""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import (balanced_accuracy_score, f1_score, roc_auc_score)

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(out)


def classification_report(probs: np.ndarray, y: np.ndarray) -> dict:
    """Per-source subsets can hold a single class (Br35H is all no-tumour), where
    AUC is undefined. Those fields become NaN and the sklearn warnings are muted."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _report(probs, y)


def _report(probs: np.ndarray, y: np.ndarray) -> dict:
    pred = probs.argmax(1)
    k = probs.shape[1]
    onehot = np.eye(k)[y]
    res = {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(k)), zero_division=0)),
        "nll": float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)).mean()),
        "brier": float(((probs - onehot) ** 2).sum(1).mean()),
        "ece": ece(probs, y),
    }
    try:
        res["macro_auc_ovr"] = float(roc_auc_score(y, probs, multi_class="ovr", labels=list(range(k))))
    except ValueError:
        res["macro_auc_ovr"] = float("nan")
    for c in range(k):
        pos, neg = y == c, y != c
        res[f"sens_{CLASSES[c]}"] = float((pred[pos] == c).mean()) if pos.any() else float("nan")
        res[f"spec_{CLASSES[c]}"] = float((pred[neg] != c).mean()) if neg.any() else float("nan")
    # Clinically decisive error: a tumour called normal.
    tumour = y != CLASSES.index("notumor")
    res["tumour_missed_as_normal"] = float((pred[tumour] == CLASSES.index("notumor")).mean()) if tumour.any() else float("nan")
    return res
