"""Privacy-utility figures from the seed-level summaries (Step 3.2).

    python scripts/plot_privacy_utility.py

Reads results/tables/week04_fed_all.csv and week04_privacy_unit_central.csv
(written by scripts/summarise.py) and writes figures/fig_privacy_utility.{png,pdf}:
(a) federated balanced accuracy against epsilon for each partition,
(b) central record-level against patient-level DP.
Points are means over seeds 1 to 5, bars are t-based 95% intervals.
The no-DP result is drawn at a separate "inf" position, never on the log axis.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "results", "tables")
OUT = os.path.join(ROOT, "figures")

# Categorical slots in fixed order (validated reference palette, light mode).
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"
INF_X = 256.0            # where "no DP" is drawn, beyond the largest epsilon


def eps_of(regime: str) -> float:
    r = regime.replace("patient ", "")
    if r in ("no DP", "clip only"):
        return np.inf
    return float(r.split()[1])


def style(ax, xticks):
    ax.set_xscale("log", base=2)
    ax.set_xticks(xticks)
    ax.set_xticklabels([("no DP" if x == INF_X else f"{x:g}") for x in xticks])
    ax.minorticks_off()
    ax.set_ylim(0.2, 1.0)
    ax.axhline(0.25, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.text(xticks[0], 0.262, "chance", color=MUTED, fontsize=8, va="bottom")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=9)
    ax.axvline(np.sqrt(64 * INF_X) if xticks[-2] == 64 else np.sqrt(8 * INF_X), color=GRID, lw=6, zorder=0)


def series(ax, d, color, label, marker="o", ls="-"):
    d = d.assign(x=d.regime.map(eps_of)).replace({"x": {np.inf: INF_X}}).sort_values("x")
    y, lo, hi = d.bal_acc.values, d.bal_acc_lo.values, d.bal_acc_hi.values
    fin = d.x.values < INF_X
    # finite epsilons are joined; the no-DP point stands alone beyond the axis break
    ax.plot(d.x[fin], y[fin], color=color, lw=2, ls=ls, label=label)
    ax.errorbar(d.x, y, yerr=[y - lo, hi - y], color=color, ls="none", marker=marker, ms=6,
                mec="white", mew=1.2, capsize=0, elinewidth=1.5)


def main():
    fed = pd.read_csv(os.path.join(TAB, "week04_fed_all.csv"))
    cen = pd.read_csv(os.path.join(TAB, "week04_privacy_unit_central.csv"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)

    ax = axes[0]
    order = [("natural3", "natural3 (3 clients)"), ("iid_k10", "IID, 10 clients"),
             ("dir5.0_k10", "Dirichlet α 5, 10 clients"), ("dir0.5_k10", "Dirichlet α 0.5, 10 clients"),
             ("dir0.1_k10", "Dirichlet α 0.1, 10 clients")]
    for i, (sc, lab) in enumerate(order):
        d = fed[(fed.scenario == sc) & ~fed.regime.str.startswith("patient")]
        series(ax, d, SLOTS[i], lab)
    style(ax, [1, 3, 8, INF_X])
    ax.set_title("a  Federated FedAvg, record-level DP", loc="left", color=INK, fontsize=10, fontweight="bold")
    ax.set_xlabel("privacy budget ε (δ = 1e-5)", color=INK)
    ax.set_ylabel("balanced accuracy (test, 4 classes)", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)

    ax = axes[1]
    rec = cen[cen.regime.isin(["eps 1", "eps 3", "eps 8"])]
    pat = cen[cen.regime.str.startswith("patient") & (cen.regime != "patient clip only")]
    series(ax, rec, SLOTS[0], "record-level (one image)")
    series(ax, pat, SLOTS[1], "patient-level (whole patient)", marker="s")
    n = cen[cen.regime == "no DP"].iloc[0]
    ax.errorbar([INF_X], [n.bal_acc], yerr=[[n.bal_acc - n.bal_acc_lo], [n.bal_acc_hi - n.bal_acc]],
                color=MUTED, marker="D", ms=6, mec="white", mew=1.2, ls="none", elinewidth=1.5,
                label="no DP (shared)")
    style(ax, [1, 3, 8, 16, 32, 64, INF_X])
    ax.set_title("b  Central training, privacy unit", loc="left", color=INK, fontsize=10, fontweight="bold")
    ax.set_xlabel("privacy budget ε (δ = 1e-5)", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)

    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_privacy_utility.{ext}"), dpi=300)
    print(os.path.join(OUT, "fig_privacy_utility.png"))


if __name__ == "__main__":
    main()
