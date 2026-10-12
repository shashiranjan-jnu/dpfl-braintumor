"""Source-prediction probe on penultimate features (D32, gap G5).

If a linear probe can tell which collection an image came from, using only
images of the same class, the representation encodes acquisition source, a
shortcut that labels alone would not reveal. Two label-controlled tasks:
tumour images (glioma, meningioma, pituitary) Figshare against SARTAJ, and
no-tumour images SARTAJ against Br35H. The probe is fitted on the training
split and scored on the test split with balanced accuracy (chance 0.5).
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@torch.no_grad()
def penultimate(model, x, batch: int = 256) -> np.ndarray:
    head = model.net.fc if hasattr(model, "net") else model.head
    feats = []
    h = head.register_forward_hook(lambda m, inp, out: feats.append(inp[0].detach().cpu()))
    try:
        model.eval()
        for s in range(0, len(x), batch):
            model(x[s:s + batch])
    finally:
        h.remove()
    return torch.cat(feats).numpy()


TASKS = {"tumour_figshare_vs_sartaj": ([0, 1, 3], ["figshare", "sartaj"]),
         "notumour_sartaj_vs_br35h": ([2], ["sartaj", "br35h"])}


def probe(model, data, splits) -> dict:
    lab = splits.label_id.values
    src = splits.source.values
    sp = splits.split.values
    feats = penultimate(model, data.x)
    out = {}
    for name, (classes, sources) in TASKS.items():
        sel = np.isin(lab, classes) & np.isin(src, sources)
        tr, te = sel & (sp == "train"), sel & (sp == "test")
        ytr, yte = (src[tr] == sources[1]).astype(int), (src[te] == sources[1]).astype(int)
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            out[name] = None
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(feats[tr], ytr)
        out[name] = {"balanced_accuracy": float(balanced_accuracy_score(yte, clf.predict(feats[te]))),
                     "n_train": int(tr.sum()), "n_test": int(te.sum())}
    return out
