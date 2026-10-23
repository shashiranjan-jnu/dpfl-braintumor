"""LiRA with composition-matched non-members and an empirical floor (Step 5.3, D59). CPU only.

    python scripts/mia_matched.py --results results --exclude results/ncc/contaminated_test_ids.txt \
        --out results/tables/week07_mia_matched

Why. Removing the 66 contaminated test images left a clean non-member set whose mix of sources and classes
differs from the members (Br35H 8.8% of members against 0.8% of clean non-members, no tumour 16.6% against
8.0%). A score that only recognises the source or class of an image can then separate members from
non-members, so the nominal chance level of TPR = FPR is not the right floor.

What. The script reuses the LiRA fit of scripts/mia_tables.py (same shadows, same statistic, same online
score) and evaluates every target model in three ways.
  pooled     all members against all clean non-members (reproduces week07_clean_lira).
  matched    non-members reweighted so that their (source, class) distribution equals that of the members.
             Members in a stratum without non-members are left out. Weighted ROC, TPR at 1% weighted FPR.
  figshare   Figshare members against Figshare clean non-members only, one source, the three tumour classes,
             reweighted by class in the same way.
The diverged weight-noise models (s = 0.5, test balanced accuracy 0.250 to 0.295) learned nothing about their
members, so their attack values give an empirical floor. Each regime is compared with the nominal 0.01 by a
one-sample t-test over seeds and with the floor by Welch's t-test (both two-sided, seeds as replicates).
Writes <out>_per_run.csv, <out>.csv and <out>.md.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import make_shadow_grid as msg  # noqa: E402
from scripts.mia_tables import label, lira_fit, lira_scores, load_phi, t_ci  # noqa: E402
from scripts.run_grid import expand  # noqa: E402
from src.data.data import shadow_members  # noqa: E402
from src.run import deep_update, resolve, run_id  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLOOR = "weight noise 0.5"


def matched_weights(strata_mem: np.ndarray, strata_non: np.ndarray):
    """Keep members whose stratum has non-members, weight non-members to the members' stratum mix."""
    keep_strata = np.intersect1d(np.unique(strata_mem), np.unique(strata_non))
    keep_mem = np.isin(strata_mem, keep_strata)
    w = np.zeros(len(strata_non))
    n_mem = keep_mem.sum()
    for s in keep_strata:
        m = strata_non == s
        w[m] = ((strata_mem[keep_mem] == s).sum() / n_mem) / m.sum()
    return keep_mem, w / w.sum() * len(w)


def weighted_summary(s_mem: np.ndarray, s_non: np.ndarray, w_non: np.ndarray | None = None) -> dict:
    w_non = np.ones(len(s_non)) if w_non is None else w_non
    lab = np.r_[np.ones(len(s_mem)), np.zeros(len(s_non))]
    sc = np.r_[s_mem, s_non]
    sw = np.r_[np.ones(len(s_mem)), w_non]
    fpr, tpr, _ = roc_curve(lab, sc, sample_weight=sw)
    return {"auc": float(roc_auc_score(lab, sc, sample_weight=sw)),
            "tpr_at_fpr_0.01": float(np.interp(0.01, fpr, tpr)),
            "n_mem": int(len(s_mem)), "n_non": int(len(s_non)),
            "ess_non": float(w_non.sum() ** 2 / (w_non ** 2).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--splits", default=os.path.join(ROOT, "data_manifest", "splits_v1.csv"))
    ap.add_argument("--grids-dir", default=os.path.join(ROOT, "configs", "grids"))
    ap.add_argument("--exclude", default=os.path.join(ROOT, "results", "ncc", "contaminated_test_ids.txt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "tables", "week07_mia_matched"))
    a = ap.parse_args()
    sp = pd.read_csv(a.splits)
    y = sp.label_id.values.astype(int)
    src, split = sp.source.values, sp.split.values
    drop = sp.image_id.isin(set(open(a.exclude).read().splitlines())).values if a.exclude else np.zeros(len(sp), bool)
    tr = np.flatnonzero((split == "train") & ~drop)
    te = np.flatnonzero((split == "test") & ~drop)
    strata = np.array([f"{s}:{c}" for s, c in zip(src, y)])

    comp = []
    for name, rows in (("members", tr), ("clean non-members", te)):
        comp.append(f"{name}: {len(rows)} images, Br35H {np.mean(src[rows] == 'br35h'):.3f}, "
                    f"no tumour {np.mean(sp.label.values[rows] == 'notumor'):.3f}")
    print("\n".join(comp))

    targets = list(msg.LIRA_TARGETS)
    wn_path = os.path.join(a.grids_dir, "week07_wn_seeds.yaml")
    if os.path.exists(wn_path):
        targets += yaml.safe_load(open(wn_path))["presets"]
    recs = []
    for t in targets:
        sh_phi, sh_in = [], []
        for c in expand(msg.shadow_grid("x", msg.shadow_presets([t], a.results), "")):
            p = load_phi(a.results, run_id(resolve(c)), y)
            if p is not None:
                sh_phi.append(p)
                sh_in.append(shadow_members(sp, c["data"]["shadow"]["seed"], c["data"]["shadow"]["frac"]))
        lab = label(resolve(t))
        if len(sh_phi) < 4:
            print(f"{lab}: only {len(sh_phi)} shadows, skipped")
            continue
        fit = lira_fit(np.stack(sh_phi), np.stack(sh_in))
        for seed in (1, 2, 3, 4, 5):
            rid = run_id(resolve(deep_update(t, {"train": {"seed": seed}})))
            tp = load_phi(a.results, rid, y)
            if tp is None:
                continue
            on, _, valid = lira_scores(tp, fit)
            m, n = tr[valid[tr]], te[valid[te]]
            rec = {"regime": lab, "seed": seed, "run_id": rid, "n_shadows": len(sh_phi)}
            variants = {"pooled": (m, n)}
            variants["matched"] = (m, n)
            fm, fn = m[src[m] == "figshare"], n[src[n] == "figshare"]
            variants["figshare"] = (fm, fn)
            for v, (mm, nn) in variants.items():
                if v == "pooled":
                    s = weighted_summary(on[mm], on[nn])
                else:
                    keep, w = matched_weights(strata[mm], strata[nn])
                    s = weighted_summary(on[mm][keep], on[nn], w)
                for k, val in s.items():
                    rec[f"{v}_{k}"] = val
            recs.append(rec)
    df = pd.DataFrame(recs)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out + "_per_run.csv", index=False)

    rows = []
    floor = df[df.regime == FLOOR]
    for reg, g in df.groupby("regime", sort=False):
        for v in ("pooled", "matched", "figshare"):
            for metric in ("tpr_at_fpr_0.01", "auc"):
                col = f"{v}_{metric}"
                mean, lo, hi = t_ci(g[col].values)
                nominal = 0.01 if metric.startswith("tpr") else 0.5
                p_nom = stats.ttest_1samp(g[col].values, nominal).pvalue
                p_floor = (stats.ttest_ind(g[col].values, floor[col].values, equal_var=False).pvalue
                           if reg != FLOOR and len(floor) > 1 else np.nan)
                rows.append({"regime": reg, "variant": v, "metric": metric, "n_seeds": len(g), "mean": mean,
                             "lo": lo, "hi": hi, "nominal": nominal, "p_vs_nominal": p_nom,
                             "floor_mean": floor[col].mean() if len(floor) else np.nan, "p_vs_floor": p_floor,
                             "n_mem": int(g[f"{v}_n_mem"].iloc[0]), "n_non": int(g[f"{v}_n_non"].iloc[0]),
                             "ess_non": float(g[f"{v}_ess_non"].mean())})
    s = pd.DataFrame(rows)
    s.to_csv(a.out + ".csv", index=False)
    lines = ["| Regime | Variant | Metric | n | Mean (95% CI) | p vs nominal | Floor mean | p vs floor | Members | Non-members (ESS) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        lines.append(f"| {r.regime} | {r.variant} | {r.metric} | {r.n_seeds} | {r['mean']:.4f} ({r.lo:.4f} to {r.hi:.4f}) | "
                     f"{r.p_vs_nominal:.3g} | {r.floor_mean:.4f} | {r.p_vs_floor:.3g} | {r.n_mem} | {r.n_non} ({r.ess_non:.0f}) |")
    md = ("\n".join(comp) + "\n\n" + "\n".join(lines) +
          "\n\nOnline LiRA, 16 shadow models per target, clean test set as non-members. Nominal level 0.01 for TPR at 1% "
          "FPR and 0.5 for AUC, one-sample t-test over seeds. Floor: diverged weight-noise models (s = 0.5), Welch t-test. "
          "Matched and figshare variants reweight non-members to the members' (source, class) mix; ESS is the effective "
          "number of non-members after weighting.\n")
    open(a.out + ".md", "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
