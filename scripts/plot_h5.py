"""H5 figure: balanced accuracy against the aggregated noise scale (Step 3.4).

    python scripts/plot_h5.py

With full participation and size-weighted FedAvg, the noise added to the averaged
gradient has standard deviation sigma C sqrt(K) / (q N) whatever the client sizes,
so epsilon and client count K act through one number. Reads
results/tables/week05_h5_mechanism.csv and writes figures/fig_h5_noise.{png,pdf}.
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLOTS = {"eps 8": "#2a78d6", "eps 3": "#eb6834"}
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e4e3dc"


def main():
    t = pd.read_csv(os.path.join(ROOT, "results", "tables", "week05_h5_mechanism.csv"))
    t = t[t.regime != "no DP"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(figsize=(5.5, 4), constrained_layout=True)
    for reg, d in t.groupby("regime"):
        d = d.sort_values("K")
        ax.plot(d.agg_noise_std, d.bal_acc, color=SLOTS[reg], lw=2, label=reg.replace("eps", "ε"))
        ax.errorbar(d.agg_noise_std, d.bal_acc, yerr=[d.bal_acc - d.lo, d.hi - d.bal_acc], color=SLOTS[reg],
                    ls="none", marker="o", ms=6, mec="white", mew=1.2, elinewidth=1.5)
        for _, r in d.iterrows():
            ax.annotate(f"K={r.K}", (r.agg_noise_std, r.bal_acc), textcoords="offset points", xytext=(6, 4),
                        fontsize=8, color=INK)
    ax.set_xlabel("aggregated noise std  σC√K / (qN)", color=INK)
    ax.set_ylabel("balanced accuracy (test)", color=INK)
    ax.set_title("Dirichlet α 0.5, same 2,610 images split over K clients", loc="left", fontsize=10, color=INK)
    ax.set_ylim(0.4, 0.85)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.legend(frameon=False, labelcolor=INK)
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(ROOT, "figures", f"fig_h5_noise.{ext}"), dpi=300)
    print("figures/fig_h5_noise.png")


if __name__ == "__main__":
    main()
