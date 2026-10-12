"""Stratified loss attack and LiRA from saved logits (Step 4.1, D45). CPU only.

    python scripts/mia_tables.py --results results --out results/tables/week07

Reads results/audit/<run_id>/logits.npz (float32 logits of every row, plain and
flipped). Members of a target are its training rows, non-members its test rows.

1. Within-stratum loss attack for every audited target: the AUC over member and
   non-member pairs from the same (source, class) cell, so a model that is simply
   worse on one collection cannot look like a leak. Also Figshare-only AUC.
2. LiRA (Carlini et al., 2022) on natural3 FedAvg at no DP, epsilon 8, 3, 1 and on
   the weight-noise comparator. Each target configuration has 16 shadow models on
   random halves of train plus test (whole units). The statistic is the logit of the
   true-class probability, phi = z_y - logsumexp(z_other), for the image and its flip.
   Online LiRA compares Gaussians fitted to IN and OUT shadow statistics of each
   example with one pooled variance per side (the fixed-variance variant, advised for
   few shadows); offline LiRA uses OUT shadows only. Patient scores sum (online) or
   average (offline) image scores over a Figshare patient.
Seed means with t-based 95% intervals (D34) go to <out>_lira.{csv,md} and
<out>_mia_stratified.{csv,md}.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import make_shadow_grid as msg  # noqa: E402
from scripts.run_grid import expand  # noqa: E402
from src.attacks.mia import attack_summary  # noqa: E402
from src.data.data import shadow_members  # noqa: E402
from src.run import deep_update, resolve, run_id  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def phi(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Logit-scaled confidence of the true class, z (N, C) float, y (N,)."""
    z = z.astype(np.float64)
    zy = z[np.arange(len(y)), y]
    other = z.copy()
    other[np.arange(len(y)), y] = -np.inf
    return zy - logsumexp(other, axis=1)


def load_phi(results: str, rid: str, y: np.ndarray) -> np.ndarray | None:
    p = os.path.join(results, "audit", rid, "logits.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    if not (np.isfinite(d["z"]).all() and np.isfinite(d["zf"]).all()):
        return None                                                     # diverged model, no usable outputs
    return np.stack([phi(d["z"], y), phi(d["zf"], y)], 1)            # (N, 2)


def _logpdf(x, mu, sd):
    return -0.5 * ((x - mu) / sd) ** 2 - np.log(sd)


def lira_fit(sh_phi: np.ndarray, sh_in: np.ndarray):
    """sh_phi (S, N, A) shadow statistics, sh_in (S, N) membership. Returns per-example
    IN/OUT means (N, A) and pooled standard deviations (A,)."""
    w_in = sh_in[..., None].astype(float)
    w_out = 1.0 - w_in
    n_in, n_out = w_in.sum(0), w_out.sum(0)
    mu_in = (sh_phi * w_in).sum(0) / np.maximum(n_in, 1)
    mu_out = (sh_phi * w_out).sum(0) / np.maximum(n_out, 1)
    sd_in = np.sqrt(((sh_phi - mu_in) ** 2 * w_in).sum((0, 1)) / np.maximum(w_in.sum((0, 1)) - 1, 1))
    sd_out = np.sqrt(((sh_phi - mu_out) ** 2 * w_out).sum((0, 1)) / np.maximum(w_out.sum((0, 1)) - 1, 1))
    valid = (n_in[:, 0] > 0) & (n_out[:, 0] > 0)
    return mu_in, mu_out, np.maximum(sd_in, 1e-6), np.maximum(sd_out, 1e-6), valid


def lira_scores(t_phi, fit):
    mu_in, mu_out, sd_in, sd_out, valid = fit
    online = (_logpdf(t_phi, mu_in, sd_in) - _logpdf(t_phi, mu_out, sd_out)).sum(1)
    offline = ((t_phi - mu_out) / sd_out).sum(1)
    return online, offline, valid


def within_stratum_auc(scores, member, strata, min_n: int = 5) -> float:
    num = den = 0.0
    for s in np.unique(strata):
        m = strata == s
        n1, n0 = int((member & m).sum()), int((~member & m).sum())
        if n1 < min_n or n0 < min_n:
            continue
        auc = roc_auc_score(member[m].astype(int), scores[m])
        num += auc * n1 * n0
        den += n1 * n0
    return num / den if den else float("nan")


def patient_summary(scores, rows_tr, rows_te, units, sources, agg):
    def per(rows):
        f = rows[sources[rows] == "figshare"]
        return np.array([agg(scores[f[units[f] == u]]) for u in np.unique(units[f])])
    return attack_summary(per(rows_tr), per(rows_te))


def t_ci(v):
    v = np.asarray([x for x in v if x is not None and not np.isnan(x)], float)
    if len(v) < 2:
        return (v.mean() if len(v) else np.nan), np.nan, np.nan
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), v.mean() - h, v.mean() + h


def label(cfg: dict) -> str:
    wn = cfg["fl"].get("weight_noise")
    if wn:
        return f"weight noise {wn:g}"
    if not cfg["dp"]["enabled"]:
        return "no DP"
    unit = "patient " if cfg["dp"].get("unit") == "unit" else ""
    nm = cfg["dp"].get("noise_multiplier")
    if nm is not None and float(nm) == 0.0:
        return f"{unit}clip only"
    return f"{unit}eps {float(cfg['dp']['target_epsilon']):g}"


def summarise(df: pd.DataFrame, keys: list[str], cols: list[str]) -> pd.DataFrame:
    rows = []
    for k, g in df.groupby(keys, sort=False):
        r = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        r["n_seeds"] = len(g)
        for c in cols:
            m, lo, hi = t_ci(g[c].values)
            r[c], r[c + "_lo"], r[c + "_hi"] = m, lo, hi
        rows.append(r)
    return pd.DataFrame(rows)


def to_md(s: pd.DataFrame, keys: list[str], cols: list[str]) -> str:
    lines = ["| " + " | ".join(keys + ["n"] + cols) + " |", "|" + "---|" * (len(keys) + 1 + len(cols))]
    for _, r in s.iterrows():
        cells = [f"{r[c]:.3f} ({r[c + '_lo']:.3f} to {r[c + '_hi']:.3f})" if not np.isnan(r[c + "_lo"])
                 else ("" if np.isnan(r[c]) else f"{r[c]:.3f}") for c in cols]
        lines.append("| " + " | ".join([str(r[k]) for k in keys] + [str(r["n_seeds"])] + cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--splits", default=os.path.join(ROOT, "data_manifest", "splits_v1.csv"))
    ap.add_argument("--grids-dir", default=os.path.join(ROOT, "configs", "grids"),
                    help="where week07_wn_seeds.yaml lives (generated in the launcher)")
    ap.add_argument("--target-grids", nargs="+", default=[
        "week03_central", "week04_unit_central", "week04_unit_central_ext", "week03_fed_seeds_a",
        "week03_fed_seeds_b", "week04_unit_fed", "week07_wn_seeds"])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "tables", "week07"))
    ap.add_argument("--exclude", default=None,
                    help="text file of image_ids left out of every attack evaluation (D50 sensitivity)")
    a = ap.parse_args()
    sp = pd.read_csv(a.splits)
    y = sp.label_id.values.astype(int)
    src, units, split = sp.source.values, sp.unit.values, sp.split.values
    drop = np.zeros(len(sp), bool)
    if a.exclude:
        drop = sp.image_id.isin(set(open(a.exclude).read().split())).values
        print(f"excluding {int(drop.sum())} images from evaluation")
    tr = np.flatnonzero((split == "train") & ~drop)
    te = np.flatnonzero((split == "test") & ~drop)
    member = np.zeros(len(sp), bool)
    member[tr] = True
    rows_eval = np.concatenate([tr, te])
    strata = np.array([f"{s}:{c}" for s, c in zip(src, y)])

    # 1. within-stratum loss attack for every target with logits
    recs = []
    for g in a.target_grids:
        path = os.path.join(a.grids_dir, f"{g}.yaml")
        if not os.path.exists(path):
            print("skip grid (not found)", g)
            continue
        for c in expand(yaml.safe_load(open(path))):
            cfg = resolve(c)
            rid = run_id(cfg)
            p = load_phi(a.results, rid, y)
            if p is None:
                continue
            d = np.load(os.path.join(a.results, "audit", rid, "logits.npz"))
            z = d["z"].astype(np.float64)
            loss = logsumexp(z, 1) - z[np.arange(len(y)), y]
            sc = -loss
            fig = rows_eval[src[rows_eval] == "figshare"]
            recs.append({"grid": g, "mode": cfg["fl"]["mode"], "scenario": cfg["data"]["scenario"],
                         "regime": label(cfg), "seed": cfg["train"]["seed"], "run_id": rid,
                         "auc_pooled": roc_auc_score(member[rows_eval].astype(int), sc[rows_eval]),
                         "auc_within_stratum": within_stratum_auc(sc[rows_eval], member[rows_eval], strata[rows_eval]),
                         "auc_figshare": roc_auc_score(member[fig].astype(int), sc[fig])})
    strat = pd.DataFrame(recs)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    if not strat.empty:
        strat.to_csv(a.out + "_mia_stratified_per_run.csv", index=False)
        cols = ["auc_pooled", "auc_within_stratum", "auc_figshare"]
        s = summarise(strat, ["mode", "scenario", "regime"], cols)
        s.to_csv(a.out + "_mia_stratified.csv", index=False)
        md = to_md(s, ["mode", "scenario", "regime"], cols)
        open(a.out + "_mia_stratified.md", "w").write(md)
        print(md)

    # 2. LiRA per target configuration
    targets = list(msg.LIRA_TARGETS)
    wn_path = os.path.join(a.grids_dir, "week07_wn_seeds.yaml")
    if os.path.exists(wn_path):
        targets += yaml.safe_load(open(wn_path))["presets"]
    lrecs = []
    for t in targets:
        sh_cfgs = expand(msg.shadow_grid("x", msg.shadow_presets([t], a.results), ""))
        sh_phi, sh_in = [], []
        for c in sh_cfgs:
            p = load_phi(a.results, run_id(resolve(c)), y)
            if p is not None:
                sh_phi.append(p)
                sh_in.append(shadow_members(sp, c["data"]["shadow"]["seed"], c["data"]["shadow"]["frac"]))
        lab = label(resolve(t))
        if len(sh_phi) < 4:
            print(f"LiRA {lab}: only {len(sh_phi)} shadows with logits, skipped")
            continue
        fit = lira_fit(np.stack(sh_phi), np.stack(sh_in))
        for seed in (1, 2, 3, 4, 5):
            rid = run_id(resolve(deep_update(t, {"train": {"seed": seed}})))
            tp = load_phi(a.results, rid, y)
            if tp is None:
                continue
            on, off, valid = lira_scores(tp, fit)
            r_tr, r_te = tr[valid[tr]], te[valid[te]]
            rec = {"regime": lab, "seed": seed, "run_id": rid, "n_shadows": len(sh_phi)}
            for name, sc, agg in (("online", on, np.sum), ("offline", off, np.mean)):
                im = attack_summary(sc[r_tr], sc[r_te])
                pa = patient_summary(sc, r_tr, r_te, units, src, agg)
                rec.update({f"{name}_auc": im["auc"], f"{name}_tpr_at_fpr_0.01": im["tpr_at_fpr_0.01"],
                            f"{name}_tpr_at_fpr_0.001": im["tpr_at_fpr_0.001"], f"{name}_adv": im["advantage"],
                            f"{name}_patient_auc": pa["auc"], f"{name}_patient_tpr_at_fpr_0.1": pa["tpr_at_fpr_0.1"]})
            lrecs.append(rec)
    lira = pd.DataFrame(lrecs)
    if not lira.empty:
        lira.to_csv(a.out + "_lira_per_run.csv", index=False)
        cols = ["online_auc", "online_tpr_at_fpr_0.01", "online_tpr_at_fpr_0.001", "online_patient_auc",
                "offline_auc", "offline_tpr_at_fpr_0.01", "offline_patient_auc"]
        s = summarise(lira, ["regime"], cols)
        s.to_csv(a.out + "_lira.csv", index=False)
        md = to_md(s, ["regime"], cols)
        md += (f"\nImage level: {len(tr)} members, {len(te)} non-members, FPR resolution {1 / len(te):.4f}, "
               "so TPR at 0.1% FPR rests on the single most confident non-member. Patient level: "
               "Figshare only, 40 test patients, FPR resolution 0.025.\n")
        open(a.out + "_lira.md", "w").write(md)
        print(md)


if __name__ == "__main__":
    main()
