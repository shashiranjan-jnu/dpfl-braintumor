"""Per-patient epsilon of the record-level runs already trained (Step 7.1, E2.2, D68). CPU only.

    python scripts/patient_accounting.py --results results --out results/tables/week13_patient_eps

For every training unit (a Figshare patient or a group of near-identical SARTAJ or Br35H images) and every
record-level budget (epsilon 1, 3, 8; sigma read from the frozen runs), the script computes the per-patient epsilon
of record-level DP-SGD with src/privacy/patient_accounting.py:
  central      one client holding all k slices of the unit
  natural3     the unit's slices at each of the three source clients, composed over the clients that hold it
  dir0.5_k10   the same for the ten-client Dirichlet partition, where units are split across clients
with the PLD accountant (primary) and, for central, the numerical RDP accountant (check) and the two group-privacy
baselines. q = 0.1, T x S = 1000 steps per client, delta = 1e-5, as in the frozen runs.
Writes <out>_units.csv (one row per unit, partition and budget), <out>_by_k.csv (central, every k up to 40, every fifth
k above, and the largest unit) and <out>.csv / .md (summaries).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.privacy import patient_accounting as pa  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELTA, Q, STEPS = 1e-5, 0.1, 1000


def sigmas(results: str) -> dict[float, float]:
    d = pd.read_csv(os.path.join(results, "tables", "week07_ncc_sensitivity_per_run.csv"))
    r = d[(d["cfg.dp.enabled"] == True) & (d["cfg.dp.unit"] != "unit") & d["cfg.fl.weight_noise"].isna()  # noqa: E712
          & d["cfg.dp.target_epsilon"].isin([1.0, 3.0, 8.0])]
    out = {}
    for e, g in r.groupby("cfg.dp.target_epsilon"):
        v = g["sigma"].unique()
        assert len(v) == 1, (e, v)
        out[float(e)] = float(v[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--splits", default=os.path.join(ROOT, "data_manifest", "splits_v1.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "tables", "week13_patient_eps"))
    a = ap.parse_args()
    sig = sigmas(a.results)
    sp = pd.read_csv(a.splits)
    tr = sp[sp.split == "train"]
    src = tr.groupby("unit").source.first()

    # 1. central, every k up to 40, every fifth k above, and the largest unit
    kmax = int(tr.groupby("unit").size().max())
    rows = []
    for e, s in sig.items():
        for k in sorted(set(range(1, 41)) | set(range(45, kmax, 5)) | {kmax}):
            pld = pa.pld_epsilon(s, k, None, Q, STEPS, DELTA)
            rdp, order = pa.rdp_epsilon(s, k, None, Q, STEPS, DELTA) if k <= 40 or k % 20 == 0 or k == kmax \
                else (np.nan, np.nan)
            ge, gd = pa.approx_group_delta(e, DELTA, k)
            rows.append({"record_eps": e, "sigma": s, "k": k, "eps_pld": pld, "eps_rdp": rdp, "rdp_order": order,
                         "eps_group_rdp": pa.rdp_group_epsilon(s, k, Q, STEPS, DELTA),
                         "group_eps": ge, "group_delta": gd})
        print(f"central eps {e:g}: done up to k = {kmax}", flush=True)
    byk = pd.DataFrame(rows)
    byk["tightening_rdp_group"] = byk.eps_group_rdp / byk.eps_pld

    # 2. every unit in each partition (same accountant, cached by client composition), composed over the clients that hold it
    cache: dict = {}
    urows = []
    for part in ("central", "natural3", "dir0.5_k10"):
        for unit, g in tr.groupby("unit"):
            ks = [len(g)] if part == "central" else sorted(g.groupby(part).size().tolist(), reverse=True)
            for e, s in sig.items():
                key = (tuple(ks), s)
                if key not in cache:
                    cache[key] = pa.pld_epsilon_multi(s, ks, None, Q, STEPS, DELTA)
                urows.append({"partition": part, "unit": unit, "source": src[unit], "k": len(g),
                              "n_clients": len(ks), "client_ks": " ".join(map(str, ks)), "record_eps": e,
                              "sigma": s, "eps_pld": cache[key]})
        print(f"{part}: {tr.unit.nunique()} units done", flush=True)
    units = pd.DataFrame(urows)

    summ = []
    for (part, e), g in units.groupby(["partition", "record_eps"], sort=False):
        for name, h in (("all units", g), ("Figshare patients", g[g.source == "figshare"])):
            summ.append({"partition": part, "record_eps": e, "units": name, "n": len(h),
                         "median_k": h.k.median(), "median_eps": h.eps_pld.median(),
                         "p90_eps": h.eps_pld.quantile(0.9), "max_eps": h.eps_pld.max(),
                         "share_eps_at_most_2x": float((h.eps_pld <= 2 * e).mean()),
                         "units_split": int((h.n_clients > 1).sum())})
    summ = pd.DataFrame(summ)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    byk.to_csv(a.out + "_by_k.csv", index=False)
    units.to_csv(a.out + "_units.csv", index=False)
    summ.to_csv(a.out + ".csv", index=False)
    lines = ["| Partition | Record eps | Units | n | Median k | Median eps | 90th pct eps | Max eps | Share <= 2x | Split |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in summ.iterrows():
        lines.append(f"| {r.partition} | {r.record_eps:g} | {r.units} | {r.n} | {r.median_k:g} | {r.median_eps:.2f} | "
                     f"{r.p90_eps:.2f} | {r.max_eps:.2f} | {r.share_eps_at_most_2x:.3f} | {r.units_split} |")
    sel = byk[byk.k.isin([1, 2, 5, 13, 24, 32, kmax])]
    lines += ["", "Central, by patient size k (PLD primary, RDP check, Renyi group bound, (eps, delta) group delta):", "",
              "| Record eps | k | PLD eps | RDP eps | Renyi group eps | Group delta at k eps |", "|---|---|---|---|---|---|"]
    for _, r in sel.iterrows():
        lines.append(f"| {r.record_eps:g} | {r.k} | {r.eps_pld:.2f} | {r.eps_rdp:.2f} | {r.eps_group_rdp:.2f} | "
                     f"{r.group_delta:.2e} |")
    md = "\n".join(lines) + (f"\n\nq = {Q}, {STEPS} steps per client, delta = {DELTA:g}; sigma from the frozen runs "
                             f"({', '.join(f'eps {e:g}: {s:.4f}' for e, s in sig.items())}). Worst-case aligned "
                             "gradients; accounting after Ganesh (2024) and Schuchardt et al. (2024).\n")
    open(a.out + ".md", "w").write(md)
    print(md)


if __name__ == "__main__":
    main()
