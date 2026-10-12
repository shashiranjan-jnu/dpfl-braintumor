"""Paper figures on the clean test set (Step 4.3, D51). CPU only.

    python scripts/make_figures.py

Writes figures/fig{1..4}_*.{png,pdf} and figures/figure_data/*.csv (the numbers
behind every plotted point, for the supplement). Inputs are files already in the
repository: results/tables/week07_ncc_sensitivity_per_run.csv (clean balanced
accuracy per run), results/audit/<run_id>/logits.npz (LiRA), the explanation table
week07_xai_border_per_run.csv, and week05_h5_mechanism.csv (noise scale per K).

Fig 1  privacy-utility: clean balanced accuracy against epsilon.
Fig 2  LiRA ROC on log axes, image level and Figshare patient level.
Fig 3  explanation consistency against epsilon, and against accuracy.
Fig 4  H5: clean balanced accuracy against the aggregated noise std.
Intervals are t-based 95% over seeds 1 to 5 (D34).
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import make_shadow_grid as msg  # noqa: E402
from scripts.mia_tables import lira_fit, lira_scores, load_phi  # noqa: E402
from scripts.run_grid import expand  # noqa: E402
from scripts.summarise import regime  # noqa: E402
from src.data.data import shadow_members  # noqa: E402
from src.run import deep_update, resolve, run_id  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
DATA = os.path.join(FIG, "figure_data")
# reference categorical palette, fixed order (dataviz skill); validated for light surfaces
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, GRID, FAINT = "#0b0b0b", "#52514e", "#e4e3dc", "#9a998f"
INF_X = 200.0          # where "no DP" sits on the log epsilon axis


def style(ax):
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"figures/{name}.png")


def tci(v):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (v.mean() if len(v) else np.nan), np.nan, np.nan
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), v.mean() - h, v.mean() + h


def eps_x(reg: str) -> float:
    if reg.endswith("no DP"):
        return INF_X
    return float(reg.split("eps ")[1]) if "eps " in reg else np.nan


def runs() -> pd.DataFrame:
    d = pd.read_csv(os.path.join(RES, "tables", "week07_ncc_sensitivity_per_run.csv"))
    d = d[d["cfg.train.seed"] != 0].copy()
    d["regime"] = d.apply(regime, axis=1)
    return d


# ---------------------------------------------------------------- Fig 1
def fig_privacy_utility(d: pd.DataFrame):
    series = [
        ("Central, record-level DP", S1, (d["cfg.fl.mode"] == "central") & ~d.regime.str.startswith("patient")),
        ("Central, patient-level DP", S2, (d["cfg.fl.mode"] == "central")
         & (d.regime.str.startswith("patient") | (d.regime == "no DP"))),
        ("FedAvg, natural3 (3 sources)", S3, (d["cfg.fl.mode"] == "federated") & (d["cfg.data.scenario"] == "natural3")
         & d["cfg.fl.weight_noise"].isna() & ~d.regime.str.startswith("patient")
         & ~((d.regime == "eps 8") & (d["cfg.optim.lr"] != 0.01))),
        ("FedAvg, Dirichlet α 0.5, K 10", S4, (d["cfg.fl.mode"] == "federated") & (d["cfg.data.scenario"] == "dir0.5_k10")
         & (d["cfg.optim.lr"] == np.where(d.regime == "no DP", 0.01, 0.003))),
    ]
    rows = []
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    for name, col, m in series:
        g = d[m & ~d.regime.str.contains("clip")]
        pts = []
        for reg, gg in g.groupby("regime"):
            mu, lo, hi = tci(gg["clean.bal_acc"])
            x = eps_x(reg.replace("patient ", ""))
            pts.append((x, mu, lo, hi, len(gg)))
            rows.append({"series": name, "regime": reg, "x_epsilon": x, "bal_acc_clean": mu, "lo": lo, "hi": hi,
                         "n_seeds": len(gg)})
        pts.sort()
        x, mu, lo, hi, _ = map(np.array, zip(*pts))
        ax.plot(x, mu, color=col, lw=2, label=name, zorder=3)
        ax.errorbar(x, mu, yerr=[mu - lo, hi - mu], color=col, ls="none", marker="o", ms=6, mec="white",
                    mew=1.5, elinewidth=1.2, capsize=0, zorder=4)
    wn = d[(d["cfg.fl.weight_noise"] == 0.003)]
    mu, lo, hi = tci(wn["clean.bal_acc"])
    ax.errorbar([INF_X], [mu], yerr=[[mu - lo], [hi - mu]], color=S5, marker="D", ms=7, mec="white",
                mew=1.5, ls="none", label="FedAvg natural3, weight noise σ 0.003 (no guarantee)", zorder=5)
    ax.annotate("weight noise\nσ 0.003", (INF_X, mu), textcoords="offset points", xytext=(-58, -14),
                fontsize=7.5, color=INK)
    rows.append({"series": "weight noise 0.003", "regime": "weight noise 0.003", "x_epsilon": np.nan,
                 "bal_acc_clean": mu, "lo": lo, "hi": hi, "n_seeds": len(wn)})
    ax.axhline(0.25, color=FAINT, lw=1, ls=":")
    ax.text(1.05, 0.262, "chance (4 classes)", color=MUTED, fontsize=7.5)
    ax.set_xscale("log")
    ticks = [1, 3, 8, 16, 32, 64, INF_X]
    ax.set_xticks(ticks)
    ax.set_xticklabels(["1", "3", "8", "16", "32", "64", "no DP"])
    ax.minorticks_off()
    ax.set_xlabel("privacy budget ε  (δ = 10⁻⁵, larger is weaker privacy)", color=INK, fontsize=9)
    ax.set_ylabel("balanced accuracy, clean test set (747 images)", color=INK, fontsize=9)
    ax.set_ylim(0.2, 0.95)
    style(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", labelcolor=INK)
    save(fig, "fig1_privacy_utility")
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "fig1_privacy_utility.csv"), index=False, float_format="%.4f")


# ---------------------------------------------------------------- Fig 2
def lira_curves():
    sp = pd.read_csv(os.path.join(ROOT, "data_manifest", "splits_v1.csv"))
    y = sp.label_id.values.astype(int)
    drop = sp.image_id.isin(set(open(os.path.join(RES, "ncc", "contaminated_test_ids.txt")).read().splitlines())).values
    tr = np.flatnonzero((sp.split.values == "train") & ~drop)
    te = np.flatnonzero((sp.split.values == "test") & ~drop)
    src, units = sp.source.values, sp.unit.values
    wn = yaml_presets(os.path.join(ROOT, "configs", "grids", "week07_wn_seeds.yaml"))
    targets = [("No DP", S1, msg.LIRA_TARGETS[0]), ("ε 8", S2, msg.LIRA_TARGETS[1]),
               ("ε 3", S3, msg.LIRA_TARGETS[2]), ("ε 1", S4, msg.LIRA_TARGETS[3]),
               ("Weight noise σ 0.003", S5, [p for p in wn if p["fl"]["weight_noise"] == 0.003][0])]
    grid = np.logspace(-3, 0, 200)
    out = {}
    for name, col, t in targets:
        sh = expand(msg.shadow_grid("x", msg.shadow_presets([t], RES), ""))
        phis, ins = [], []
        for c in sh:
            p = load_phi(RES, run_id(resolve(c)), y)
            if p is not None:
                phis.append(p)
                ins.append(shadow_members(sp, c["data"]["shadow"]["seed"], c["data"]["shadow"]["frac"]))
        fit = lira_fit(np.stack(phis), np.stack(ins))
        img, pat = [], []
        for seed in (1, 2, 3, 4, 5):
            tp = load_phi(RES, run_id(resolve(deep_update(t, {"train": {"seed": seed}}))), y)
            on, _, valid = lira_scores(tp, fit)
            a, b = tr[valid[tr]], te[valid[te]]
            f, r, _ = roc_curve(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[on[a], on[b]])
            img.append(np.interp(grid, f, r))
            per = lambda rows: np.array([on[rows[(src[rows] == "figshare") & (units[rows] == u)]].sum()
                                         for u in np.unique(units[rows][src[rows] == "figshare"])])
            pa, pb = per(a), per(b)
            f, r, _ = roc_curve(np.r_[np.ones(len(pa)), np.zeros(len(pb))], np.r_[pa, pb])
            pat.append(np.interp(grid, f, r))
        out[name] = (col, np.mean(img, 0), np.mean(pat, 0))
    return grid, out, len(te)


def yaml_presets(path):
    import yaml
    return yaml.safe_load(open(path))["presets"]


def fig_lira():
    grid, curves, n_te = lira_curves()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.8), constrained_layout=True)
    rows = []
    for ax, k, title, floor in ((axes[0], 1, f"Image level ({n_te} test images)", 1 / n_te),
                                (axes[1], 2, "Patient level (Figshare, 40 test patients)", 1 / 40)):
        ax.plot(grid, grid, color=FAINT, lw=1, ls="--", label="random guess")
        for name, v in curves.items():
            ok = grid >= floor
            ax.plot(grid[ok], np.maximum(v[k][ok], 1e-3), color=v[0], lw=2, label=name)
            for g, t in zip(grid, v[k]):
                rows.append({"level": "image" if k == 1 else "patient", "attack_target": name, "fpr": g, "tpr": t})
        ax.axvspan(1e-3, floor, color=GRID, alpha=0.6, lw=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-3, 1)
        ax.set_ylim(1e-3, 1)
        ax.set_xlabel("false positive rate", color=INK, fontsize=9)
        ax.set_title(title, loc="left", fontsize=9, color=INK)
        style(ax)
    axes[0].set_ylabel("true positive rate (online LiRA, mean of 5 targets)", color=INK, fontsize=9)
    axes[1].legend(frameon=False, fontsize=7.5, loc="lower right", labelcolor=INK)
    save(fig, "fig2_lira_roc")
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "fig2_lira_roc.csv"), index=False, float_format="%.5f")


# ---------------------------------------------------------------- Fig 3
def fig_explanations(d: pd.DataFrame):
    x = pd.read_csv(os.path.join(RES, "tables", "week07_xai_border_per_run.csv"))
    x = x.merge(d[["run_id", "clean.bal_acc"]], on="run_id", how="left")
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.7), constrained_layout=True)
    ax = axes[0]
    rows = []
    for name, col, m in (("Central, record-level", S1, (x["mode"] == "central") & ~x.regime.str.startswith("patient")),
                         ("Central, patient-level", S2, (x["mode"] == "central")
                          & (x.regime.str.startswith("patient") | (x.regime == "no DP")))):
        g = x[m & ~x.regime.str.contains("clip")]
        pts = []
        for reg, gg in g.groupby("regime"):
            mu, lo, hi = tci(gg.consistency_iou)
            pts.append((eps_x(reg.replace("patient ", "")), mu, lo, hi))
            rows.append({"panel": "a", "series": name, "regime": reg, "consistency_iou": mu, "lo": lo, "hi": hi})
        pts.sort()
        xx, mu, lo, hi = map(np.array, zip(*pts))
        ax.plot(xx, mu, color=col, lw=2, label=name)
        ax.errorbar(xx, mu, yerr=[mu - lo, hi - mu], color=col, ls="none", marker="o", ms=6, mec="white", mew=1.5,
                    elinewidth=1.2)
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 8, 16, 32, 64, INF_X])
    ax.set_xticklabels(["1", "3", "8", "16", "32", "64", "no DP"])
    ax.minorticks_off()
    ax.set_xlabel("privacy budget ε", color=INK, fontsize=9)
    ax.set_ylabel("Grad-CAM consistency across seeds (IoU)", color=INK, fontsize=9)
    ax.set_title("a  Consistency falls as ε shrinks", loc="left", fontsize=9, color=INK)
    ax.set_ylim(0, 0.8)
    style(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", labelcolor=INK)
    ax = axes[1]
    groups = (("Central, record-level", S1, (x["mode"] == "central") & ~x.regime.str.startswith("patient")),
              ("Central, patient-level", S2, (x["mode"] == "central") & x.regime.str.startswith("patient")),
              ("FedAvg, 5 partitions", S3, x["mode"] == "federated"))
    for name, col, m in groups:
        g = x[m].dropna(subset=["consistency_iou", "clean.bal_acc"])
        ax.scatter(g["clean.bal_acc"], g.consistency_iou, s=22, color=col, edgecolor="white", lw=0.8, label=name)
        for rid, acc, ci in zip(g.run_id, g["clean.bal_acc"], g.consistency_iou):
            rows.append({"panel": "b", "series": name, "run_id": rid, "bal_acc_clean": acc, "consistency_iou": ci})
    g = x.dropna(subset=["consistency_iou", "clean.bal_acc"])
    rr = stats.pearsonr(g["clean.bal_acc"], g.consistency_iou)[0]
    ax.text(0.03, 0.95, f"r = {rr:.2f}, {len(g)} models", transform=ax.transAxes, color=MUTED, fontsize=8, va="top")
    ax.set_xlabel("balanced accuracy, clean test set", color=INK, fontsize=9)
    ax.set_title("b  …and tracks accuracy", loc="left", fontsize=9, color=INK)
    ax.set_ylim(0, 0.8)
    style(ax)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right", labelcolor=INK)
    save(fig, "fig3_explanations")
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "fig3_explanations.csv"), index=False, float_format="%.4f")


# ---------------------------------------------------------------- Fig 4
def fig_h5(d: pd.DataFrame):
    h = pd.read_csv(os.path.join(RES, "tables", "week05_h5_mechanism.csv"))
    noise = {(int(r.K), r.regime): r.agg_noise_std for r in h.itertuples()}
    f = d[(d["cfg.fl.mode"] == "federated") & d["cfg.data.scenario"].str.match(r"dir0\.5_k\d+$")
          & d.regime.isin(["eps 8", "eps 3"])].copy()
    f["K"] = f["cfg.data.scenario"].str.extract(r"_k(\d+)").astype(int)
    rows = []
    fig, ax = plt.subplots(figsize=(5.8, 4.0), constrained_layout=True)
    for (reg, lr), col, ls in ((("eps 8", 0.01), S1, "-"), (("eps 8", 0.003), S1, "--"),
                               (("eps 3", 0.003), S2, "-")):
        g = f[(f.regime == reg) & (f["cfg.optim.lr"] == lr)]
        pts = []
        for k, gg in g.groupby("K"):
            mu, lo, hi = tci(gg["clean.bal_acc"])
            pts.append((noise[(k, reg)], mu, lo, hi, k))
            rows.append({"regime": reg, "lr": lr, "K": k, "agg_noise_std": noise[(k, reg)],
                         "bal_acc_clean": mu, "lo": lo, "hi": hi, "n_seeds": len(gg)})
        pts.sort()
        xx, mu, lo, hi, ks = map(np.array, zip(*pts))
        lab = f"{reg.replace('eps', 'ε')}, lr {lr:g}"
        ax.plot(xx, mu, color=col, lw=2, ls=ls, label=lab)
        ax.errorbar(xx, mu, yerr=[mu - lo, hi - mu], color=col, ls="none", marker="o", ms=6, mec="white", mew=1.5,
                    elinewidth=1.2)
        if lr == 0.01 or reg == "eps 3":
            for a, b, k in zip(xx, mu, ks):
                ax.annotate(f"K={k}", (a, b), textcoords="offset points", xytext=(6, 4), fontsize=7.5, color=INK)
    ax.set_xlabel("aggregated noise std  σC√K / (qN)", color=INK, fontsize=9)
    ax.set_ylabel("balanced accuracy, clean test set", color=INK, fontsize=9)
    ax.set_title("Dirichlet α 0.5, the same 2,610 images over K clients", loc="left", fontsize=9, color=INK)
    ax.set_ylim(0.4, 0.85)
    style(ax)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK)
    save(fig, "fig4_h5_noise")
    pd.DataFrame(rows).to_csv(os.path.join(DATA, "fig4_h5_noise.csv"), index=False, float_format="%.5f")


def main():
    os.makedirs(DATA, exist_ok=True)
    d = runs()
    fig_privacy_utility(d)
    fig_lira()
    fig_explanations(d)
    fig_h5(d)
    json.dump({"test_set": "clean, 747 images (D51)", "intervals": "t-based 95% over seeds 1 to 5"},
              open(os.path.join(DATA, "README.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
