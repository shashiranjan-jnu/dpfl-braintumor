"""Supplementary tables from frozen files only (Step 5.2). CPU only.

    python scripts/make_supplement.py --out paper/supplement/tables.tex

Reads results/tables/*, results/ncc/*, results/audit/_baselines/* and the run
metrics of the reported grids, and writes LaTeX table environments (booktabs,
longtable) that paper/supplement/supplement.tex inputs. Every number in the
supplement therefore comes from a file listed in results/FROZEN_MANIFEST.json
or from a run of a frozen grid.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.aggregate import collect  # noqa: E402
from scripts.ncc_sensitivity import clean_metrics  # noqa: E402
from scripts.run_audit import grid_ids  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = os.path.join(ROOT, "results", "tables")
SCEN = {"natural3": "natural", "iid_k10": "IID, $K{=}10$", "udir0.5_k10": "unit Dir. 0.5, $K{=}10$"}


def scen(s: str) -> str:
    if s in SCEN:
        return SCEN[s]
    m = re.match(r"dir([\d.]+)_k(\d+)", str(s))
    return f"Dir. {m.group(1)}, $K{{=}}{m.group(2)}$" if m else str(s)


def regime(r: str) -> str:
    r = str(r)
    m = re.match(r"(patient )?eps ([\d.]+)", r)
    if m:
        return f"{'patient ' if m.group(1) else ''}$\\varepsilon={m.group(2)}$"
    m = re.match(r"weight noise ([\d.]+)", r)
    if m:
        return f"weight noise $s={m.group(1)}$"
    return r


def ORDER(label: str) -> tuple:
    m = re.search(r"([\d.]+)\$", label)
    eps = float(m.group(1)) if m else float("inf")
    return ("patient" in label, "weight" in label, -eps if "clip" not in label else -1e9)


def f3(x) -> str:
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"


def ci(r, n) -> str:
    if pd.isna(r.get(n)):
        return "--"
    lo, hi = r.get(n + "_lo"), r.get(n + "_hi")
    if lo is None or pd.isna(lo):
        return f3(r[n])
    return f"{r[n]:.3f} ({lo:.3f}, {hi:.3f})"


def table(label, caption, head, rows, align=None, long=False, small=True, note=None, resize=False) -> str:
    align = align or "l" * (len(head) - len(rows[0]) if False else 0) or "l" + "r" * (len(head) - 1)
    h = " & ".join(head) + r" \\"
    body = "\n".join(" & ".join(map(str, r)) + r" \\" for r in rows)
    size = (r"\scriptsize" if len(head) >= 8 else r"\footnotesize") if small else ""
    nt = f"\n\\par\\smallskip{{\\footnotesize {note}}}" if note else ""
    if long:
        return (f"{{{size}\n\\begin{{longtable}}{{{align}}}\n\\caption{{{caption}}}\\label{{{label}}}\\\\\n\\toprule\n{h}\n"
                f"\\midrule\n\\endfirsthead\n\\toprule\n{h}\n\\midrule\n\\endhead\n\\bottomrule\n\\endlastfoot\n"
                f"{body}\n\\end{{longtable}}}}{nt}\n\n")
    return (f"\\begin{{table}}[H]\n\\centering{size}\n\\caption{{{caption}}}\\label{{{label}}}\n"
            + (r"\resizebox{\textwidth}{!}{" if len(head) >= 8 or resize else "{")
            + f"\\begin{{tabular}}{{{align}}}\n\\toprule\n{h}\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}}}{nt}\n"
            f"\\end{{table}}\n\n")


def cfg_label(r) -> str:
    alg = r.get("cfg.fl.algorithm", "fedavg")
    mode = r["cfg.fl.mode"]
    dp = bool(r.get("cfg.dp.enabled")) and r.get("cfg.dp.target_epsilon") not in (None, np.nan)
    eps = r.get("cfg.dp.target_epsilon")
    unit = "patient " if str(r.get("cfg.dp.unit", "")) in ("unit", "True", "true", "1") else ""
    if r.get("cfg.dp.enabled") and pd.notna(eps):
        reg = f"{unit}$\\varepsilon={eps:g}$"
    elif r.get("cfg.dp.enabled"):
        reg = f"{unit}clip only"
    else:
        reg = "no DP"
    if pd.notna(r.get("cfg.fl.weight_noise")) and r.get("cfg.fl.weight_noise"):
        reg = f"weight noise $s={r['cfg.fl.weight_noise']:g}$"
    return mode, scen(r["cfg.data.scenario"]), reg, alg, dp


# ---------------------------------------------------------------- tables
def s_ncc() -> str:
    c = json.load(open(os.path.join(ROOT, "results", "ncc", "curated_summary.json")))
    rows = []
    for thr, v in c["thresholds"].items():
        rel = v["cross_split_by_relation"]
        rows.append([thr, v["pairs"], v["pairs_same_unit"], v["pairs_cross_split"], rel.get("test-train", 0),
                     rel.get("test-val", 0), rel.get("train-val", 0), v["cross_split_flip_only"],
                     v["cross_split_label_conflicts"]])
    a = table("tab:s_ncc", "Part A of the NCC check, curated pairs at or above each threshold.",
              ["NCC $\\geq$", "Pairs", "Same unit", "Cross-split", "Test--train", "Test--val", "Train--val",
               "Mirror only", "Label conflict"], rows)
    q = c["test_nearest_other_split"]
    note = (f"Across the 813 test images, the NCC to the nearest image in another split has median {q['0.5']:.3f}, "
            f"90th percentile {q['0.9']:.3f} and maximum {q['max']:.4f}. "
            f"{c['test_images_at_or_above_near']} test images reach 0.95 and form the contaminated set.")
    r = json.load(open(os.path.join(ROOT, "results", "ncc", "removed_summary.json")))
    rows = [[d, v["n"], f"{v['ncc_median']:.4f}", f"{v['ncc_min']:.3f}", v["below_0.9"], v["below_0.95"],
             v["below_0.98"]] for d, v in r["by_phash_d"].items()]
    b = table("tab:s_ncc_b", "Part B of the NCC check, removed duplicates against their kept partner.",
              ["$d$", "Pairs", "Median NCC", "Min NCC", "$<0.90$", "$<0.95$", "$<0.98$"], rows)
    return a + note + "\n\n" + b


def s_full_clean() -> str:
    d = pd.read_csv(os.path.join(T, "week07_ncc_sensitivity.csv"))
    d = d.assign(_o=[ORDER(regime(x)) for x in d.regime]).sort_values(["mode", "scenario", "algorithm", "_o", "lr"])
    rows = [["central" if r["mode"] == "central" else scen(r["scenario"]),
             "SCAFFOLD" if r["algorithm"] == "scaffold" else "FedAvg" if r["mode"] == "federated" else "--",
             regime(r["regime"]), f"{r['lr']:g}",
             ci(r, "bal_acc"), ci(r, "bal_acc_clean"), f3(r["sens_N"]), f3(r["sens_N_clean"])]
            for _, r in d.iterrows()]
    return table("tab:s_full_clean", "Balanced accuracy on the full (813) and clean (747) test sets, mean (95\\% CI).",
                 ["Training", "Algorithm", "Regime", "lr", "Bal. acc. (full)", "Bal. acc. (clean)",
                  "sens$_N$ full", "sens$_N$ clean"], rows, align="llllrrrr", long=True)


def s_local() -> str:
    df = collect(os.path.join(ROOT, "results", "runs"), set(grid_ids(["week03_local"])))
    drop = set(open(os.path.join(ROOT, "results", "ncc", "contaminated_test_ids.txt")).read().splitlines())
    names = {0: "Figshare", 1: "SARTAJ", 2: "Br35H"}
    out = []
    for _, r in df.iterrows():
        if r["cfg.train.seed"] == 0:
            continue
        pred = pd.read_csv(os.path.join(ROOT, "results", "runs", r["run_id"], "predictions.csv"))
        out.append({"client": names[int(r["cfg.fl.local_client"])], "dp": "$\\varepsilon=8$" if r["cfg.dp.enabled"]
                    else "no DP", "full": r["test.balanced_accuracy"],
                    "clean": clean_metrics(pred, drop)["bal_acc"]})
    g = pd.DataFrame(out).groupby(["client", "dp"], sort=True).agg(n=("full", "size"), full=("full", "mean"),
                                                                     clean=("clean", "mean")).reset_index()
    rows = [[r["client"], r["dp"], r["n"], f3(r["full"]), f3(r["clean"])] for _, r in g.iterrows()]
    return table("tab:s_local", "Local-only baselines of the natural partition, seed means.",
                 ["Client", "Regime", "Seeds", "Bal. acc. (full)", "Bal. acc. (clean)"], rows)


def s_sources() -> str:
    d = pd.read_csv(os.path.join(T, "week07_source_controls.csv"))
    rows = [[r["mode"], scen(r["scenario"]), regime(r["regime"]), f3(r["bal_acc"]), f3(r["sartaj_bal_acc4"]),
             f3(r["figshare_bal_acc3"]), f3(r["tumour_sens"]), f3(r["spec_sartaj"]), f3(r["spec_br35h"])]
            for _, r in d.iterrows()]
    return table("tab:s_sources", "Per-source controls on the full test set, seed means.",
                 ["Mode", "Partition", "Regime", "Bal. acc.", "SARTAJ (4 cl.)", "Figshare (3 cl.)",
                  "Tumour sens.", "Spec. SARTAJ", "Spec. Br35H"], rows, align="lllrrrrrr", long=True)


def s_probe() -> str:
    d = pd.read_csv(os.path.join(T, "week07_audit_summary.csv"))
    rows = [[r["mode"], scen(r["scenario"]), regime(r["regime"]), ci(r, "probe_tumour"), ci(r, "probe_notumour")]
            for _, r in d.iterrows()]
    for name, fn in (("ImageNet initialisation", "probe_pretrained.json"), ("random network", "probe_random.json")):
        b = json.load(open(os.path.join(ROOT, "results", "audit", "_baselines", fn)))
        rows.append(["baseline", "--", name, f3(b["tumour_figshare_vs_sartaj"]["balanced_accuracy"]),
                     f3(b["notumour_sartaj_vs_br35h"]["balanced_accuracy"])])
    return table("tab:s_probe", "Source probe on penultimate features, balanced accuracy.",
                 ["Mode", "Partition", "Regime", "Tumour probe", "No-tumour probe"], rows, align="lllrr", long=True)


def s_loss_mia() -> str:
    full = pd.read_csv(os.path.join(T, "week07_mia_stratified.csv"))
    clean = pd.read_csv(os.path.join(T, "week07_clean_mia_stratified.csv"))
    m = full.merge(clean, on=["mode", "scenario", "regime"], suffixes=("", "_c"))
    rows = [[r["mode"], scen(r["scenario"]), regime(r["regime"]), f3(r["auc_pooled"]), f3(r["auc_within_stratum"]),
             f3(r["auc_pooled_c"]), f3(r["auc_within_stratum_c"]), f3(r["auc_figshare_c"])] for _, r in m.iterrows()]
    return table("tab:s_loss_mia", "Loss-threshold membership attack, AUC.",
                 ["Mode", "Partition", "Regime", "Pooled (full)", "Stratum (full)", "Pooled (clean)",
                  "Stratum (clean)", "Figshare (clean)"], rows, align="lllrrrrr", long=True)


def s_lira() -> str:
    out = ""
    for f, name in (("week07_clean_lira", "clean test set (747 images)"),
                    ("week07_lira", "full test set (813 images)")):
        d = pd.read_csv(os.path.join(T, f + ".csv"))
        rows = [[regime(r["regime"]), r["n_seeds"], ci(r, "online_auc"), ci(r, "online_tpr_at_fpr_0.01"),
                 ci(r, "online_tpr_at_fpr_0.001"), ci(r, "online_patient_auc"), f3(r["offline_auc"]),
                 f3(r["offline_tpr_at_fpr_0.01"]), f3(r["offline_patient_auc"])] for _, r in d.iterrows()]
        out += table(f"tab:s_{f}", f"LiRA with the {name}, mean (95\\% CI) over five targets.",
                     ["Target", "$n$", "AUC", "TPR@1\\%", "TPR@0.1\\%", "Patient AUC", "Off. AUC", "Off. TPR@1\\%",
                      "Off. patient AUC"], rows, align="lrrrrrrrr")
    return out


def s_scaffold() -> str:
    d = pd.read_csv(os.path.join(T, "week06_scaffold_vs_fedavg.csv"))
    rows = []
    for _, r in d.iterrows():
        p, reg = [x.strip() for x in r["config"].split("|")]
        rows.append([scen(p), regime(reg), f3(r["fedavg"]), f3(r["scaffold_size"]), f"{r['diff']:+.3f}",
                     f"{r['p']:.4f}", f"{r['p_holm']:.3f}", f3(r["scaffold_uniform_seeds1to3"])])
    return table("tab:s_scaffold", "SCAFFOLD against FedAvg on the full test set.",
                 ["Partition", "Regime", "FedAvg", "SCAFFOLD (size)", "Diff.", "$p$", "$p$ Holm",
                  "SCAFFOLD (uniform)"], rows, align="llrrrrrr")


def s_fedprox() -> str:
    df = collect(os.path.join(ROOT, "results", "runs"),
                 set(grid_ids(["week06_select_fedprox", "week06_select_fedprox_ext"])))
    fa = collect(os.path.join(ROOT, "results", "runs"), set(grid_ids(["week03_select_fed"])))
    fa = fa[fa["cfg.train.seed"] == 0]
    key = ["cfg.data.scenario", "cfg.dp.target_epsilon", "cfg.optim.lr"]
    rows = []
    for k, g in df.groupby(key, dropna=False):
        base = fa[(fa["cfg.data.scenario"] == k[0]) & (fa["cfg.optim.lr"] == k[2]) &
                  ((fa["cfg.dp.target_epsilon"] == k[1]) | (fa["cfg.dp.target_epsilon"].isna() & pd.isna(k[1])))
                  & (fa["cfg.fl.algorithm"] == "fedavg")]
        b = base["best_val_loss"].min() if len(base) else np.nan
        vals = {m: v for m, v in zip(g["cfg.fl.mu"], g["best_val_loss"])}
        reg = "no DP" if pd.isna(k[1]) else f"$\\varepsilon={k[1]:g}$"
        rows.append([scen(k[0]), reg, f"{k[2]:g}", f3(b)] + [f3(vals.get(m, np.nan)) for m in (0.01, 0.1, 1.0, 10.0)])
    return table("tab:s_fedprox", "FedProx screening at seed 0, best validation loss.",
                 ["Partition", "Regime", "lr", "$\\mu=0$", "$\\mu=0.01$", "$\\mu=0.1$", "$\\mu=1$", "$\\mu=10$"],
                 rows, align="lllrrrrr", long=True)


def s_wn() -> str:
    df = collect(os.path.join(ROOT, "results", "runs"), set(grid_ids(["week07_wn_select"])))
    ref = collect(os.path.join(ROOT, "results", "runs"), set(grid_ids(["week03_fed_seeds_a", "week03_fed_seeds_b",
                                                                      "week03_select_fed"])))
    ref = ref[(ref["cfg.data.scenario"] == "natural3") & (ref["cfg.dp.target_epsilon"] == 8) &
              (ref["cfg.train.seed"] == 0) & (ref["cfg.optim.lr"] == 0.01) & (ref["cfg.fl.algorithm"] == "fedavg")]
    thr = ref["best_val_loss"].min()
    rows = []
    for _, r in df.sort_values("cfg.fl.weight_noise", ascending=False).iterrows():
        v = r["best_val_loss"]
        rows.append([f"{r['cfg.fl.weight_noise']:g}", "yes" if r["diverged"] else "no", f3(v),
                     "yes" if (not r["diverged"]) and pd.notna(v) and v <= thr else "no"])
    s = pd.read_csv(os.path.join(T, "week07_wn_seeds_summary.csv"))
    note = (f"The reference is the seed-0 validation loss of record-level $\\varepsilon=8$ FedAvg on the natural partition, "
            f"{thr:.3f}. The rule takes the largest $s$ that meets it. Over seeds 1 to 5 on the full test set, "
            + " and ".join(f"{regime(r['regime'])} reaches balanced accuracy {ci(r, 'bal_acc')}" for _, r in s.iterrows())
            + ".")
    return table("tab:s_wn", "Selection of the weight-noise level $s$ at seed 0.",
                 ["$s$", "Diverged", "Best val. loss", "Meets reference"], rows, align="rrrr", note=note)


def s_lr() -> str:
    rows = []
    for f, grid in (("week03_lr_selection", "week03_select_fed"), ("week04_lr_selection_unit", "week04_select_unit"),
                    ("week04_lr_selection_udir", "week04_select_udir")):
        d = pd.read_csv(os.path.join(T, f + ".csv"))
        meta = collect(os.path.join(ROOT, "results", "runs"), set(d.run_id))
        keep = ["run_id", "cfg.fl.mode", "cfg.data.scenario", "cfg.dp.enabled", "cfg.dp.target_epsilon",
                "cfg.fl.algorithm"] + [c for c in ("cfg.dp.unit", "cfg.fl.weight_noise") if c in meta]
        d = d.merge(meta[keep], on="run_id", how="left")
        for _, r in d.iterrows():
            mode, sc, reg, alg, _ = cfg_label(r)
            rows.append([sc, reg, f"{r['lr']:g}", f3(r["best_val_loss"]), f3(r["val_bal_acc_at_best"]),
                         "\\checkmark" if r["selected"] else ""])
    return table("tab:s_lr", "Learning-rate selection at seed 0, federated models.",
                 ["Partition", "Regime", "lr", "Best val. loss", "Val. bal. acc.", "Selected"], rows,
                 align="llrrrc", long=True)


def s_h5() -> str:
    d = pd.read_csv(os.path.join(T, "week05_h5_mechanism.csv"))
    rows = [[r["K"], regime(r["regime"]), f"{r['n_min']}/{r['n_median']}/{r['n_max']}", f"{r['sigma']:.3f}",
             f"{r['agg_noise_std']:.4f}", f"{r['bal_acc']:.3f} ({r['lo']:.3f}, {r['hi']:.3f})", f3(r["sens_M"])]
            for _, r in d.sort_values(["regime", "K"]).iterrows()]
    return table("tab:s_h5", "Client count at fixed data on the full test set.",
                 ["$K$", "Regime", "Client size min/med./max", "$\\sigma$", "Agg. noise s.d.", "Bal. acc.",
                  "sens$_M$"], rows, align="rlrrrrr")


def s_xai() -> str:
    d = pd.read_csv(os.path.join(T, "week07_audit_summary.csv"))
    d = d[d["mode"] == "central"]
    rows = []
    for _, r in d.iterrows():
        for m, name in (("gradcam", "Grad-CAM"), ("gradcampp", "Grad-CAM++"), ("ig", "Int. Gradients")):
            rows.append([regime(r["regime"]), name] + [f3(r[f"{m}_{k}"]) for k in
                                                       ("pointing", "iou15", "energy", "deletion", "insertion")])
    return table("tab:s_xai", "Localisation and faithfulness of three attribution methods, central models.",
                 ["Regime", "Method", "Pointing", "IoU top 15\\%", "Energy in mask", "Deletion", "Insertion"], rows,
                 align="llrrrrr", long=True)


def s_rand() -> str:
    d = pd.read_csv(os.path.join(T, "week07_audit_summary_per_run.csv"))
    rc = [c for c in d.columns if c.startswith("rand.")]
    d = d.dropna(subset=rc, how="all")
    meta = collect(os.path.join(ROOT, "results", "runs"), set(d.run_id))
    d = d.merge(meta[["run_id", "cfg.fl.mode", "cfg.data.scenario", "cfg.dp.enabled", "cfg.dp.target_epsilon",
                      "cfg.fl.algorithm"] + [c for c in ("cfg.dp.unit", "cfg.fl.weight_noise") if c in meta]],
                on="run_id", how="left", suffixes=("_x", ""))
    lab = {"rand.gradcam.full": "Grad-CAM, all layers", "rand.gradcam.top": "Grad-CAM, last block",
           "rand.ig.full": "IG, all layers", "rand.ig.top": "IG, last block"}
    short = [lab.get(c, c) for c in rc]
    d["reg"] = [cfg_label(r)[2] for _, r in d.iterrows()]
    d = d.iloc[sorted(range(len(d)), key=lambda i: ORDER(d["reg"].iloc[i]))]
    rows = [[r["reg"]] + [f3(r[c]) for c in rc] for _, r in d.iterrows()]
    return table("tab:s_rand", "Randomisation check, Spearman correlation with re-initialised networks.", ["Regime"] + short, rows,
                 align="l" + "r" * len(rc))


def s_border() -> str:
    d = pd.read_csv(os.path.join(T, "week07_xai_border.csv"))
    rows = [[r["mode"], scen(r["scenario"]), regime(r["regime"]), f3(r["gc_border"]), f3(r["border_chance"]),
             f3(r["gc_outside"]), f3(r["outside_chance"]), f3(r["l3_pointing"]), f3(r["l3_iou15"]),
             f3(r["consistency_iou"])] for _, r in d.iterrows()]
    return table("tab:s_border", "Border rim, layer-3 Grad-CAM and cross-seed consistency.",
                 ["Mode", "Partition", "Regime", "Rim", "Rim chance", "Outside", "Outside chance", "L3 pointing",
                  "L3 IoU", "Consistency"], rows, align="lllrrrrrrr", long=True)


def s_eps() -> str:
    grids = ["week03_central", "week04_unit_central", "week04_unit_central_ext", "week03_fed_seeds_a",
             "week03_fed_seeds_b", "week04_unit_fed", "week04_udir_seeds", "week05_client_count",
             "week05_client_count_lr"]
    df = collect(os.path.join(ROOT, "results", "runs"), set(grid_ids(grids)))
    df = df[df["cfg.dp.enabled"].astype(bool) & df["cfg.dp.target_epsilon"].notna()]
    rows = []
    for _, g in df.groupby(["cfg.fl.mode", "cfg.data.scenario", "cfg.dp.target_epsilon",
                            df.get("cfg.dp.unit", pd.Series("", index=df.index)).fillna("")], sort=True):
        r = g.iloc[0]
        unit = str(r.get("cfg.dp.unit", "")) not in ("", "nan", "None", "False")
        rows.append([r["cfg.fl.mode"], scen(r["cfg.data.scenario"]), "patient" if unit else "image",
                     f"{r['cfg.dp.target_epsilon']:g}", f"{r['sigma']:.3f}", f"{g['epsilon'].mean():.2f}",
                     f"{g['epsilon_k2'].mean():.2f}" if r["cfg.fl.mode"] != "central" and g["epsilon_k2"].notna().any()
                     else "--",
                     f"{g['unit_epsilon_worst'].mean():.2f}" if unit and g["unit_epsilon_worst"].notna().any() else "--"])
    return table("tab:s_eps", "Accounted privacy loss of every private configuration.",
                 ["Mode", "Partition", "Unit", "Target $\\varepsilon$", "$\\sigma$", "$\\varepsilon$",
                  "$\\varepsilon_{k=2}$", "Worst-unit $\\varepsilon$"], rows, align="lllrrrrr", long=True,
                 note="Central models use the learning rate 0.02 fixed in pilot runs on the same training data, so "
                      "their selection cost is disclosed but not composed. Natural3 splits 11 units across two "
                      "clients, which raises the worst-unit guarantee of that partition.")


def s_mediation() -> str:
    d = pd.read_csv(os.path.join(T, "week07_xai_mediation.csv"))
    if "level" not in d:
        d["level"] = "model"
    rows = [[r["level"], r["metric"].replace("%", "\\%"), r["n_models"], f3(r["pearson_r_accuracy"]),
             "$<0.001$" if r["p_r"] < 0.001 else f3(r["p_r"]), f3(r["b_accuracy"]), f"{r['b_sigma']:.4f}",
             f3(r["p_sigma"])] for _, r in d.iterrows()]
    return table("tab:s_mediation", "Explanation quality against accuracy and noise multiplier.",
                 ["Level", "Metric", "Points", "$r$", "$p$", "$b$ accuracy", "$b$ $\\sigma$", "$p$ ($\\sigma$)"], rows,
                 align="llrrrrrr")


def s_matched() -> str:
    d = pd.read_csv(os.path.join(T, "week07_mia_matched.csv"))
    d = d[d.metric == "tpr_at_fpr_0.01"]
    names = {"pooled": "all", "matched": "matched", "figshare": "Figshare"}
    rows = []
    for _, r in d.iterrows():
        pv = lambda p: "--" if pd.isna(p) else ("$<0.001$" if p < 0.001 else f3(p))
        rows.append([regime(r["regime"]).replace("weight noise", "w.\\ noise"), names[r["variant"]],
                     f"{r['mean']:.3f} ({r['lo']:.3f}, {r['hi']:.3f})", pv(r["p_vs_nominal"]), pv(r["p_vs_floor"]),
                     f"{int(r['n_mem'])}", f"{int(r['n_non'])} ({r['ess_non']:.0f})"])
    return table("tab:s_matched", "LiRA TPR at 1\\% FPR with composition-matched non-members.",
                 ["Target", "Non-members", "TPR@1\\% (95\\% CI)", "$p$ vs 0.01", "$p$ vs $s=0.5$", "Members",
                  "Non-members (ESS)"], rows, align="llrrrrr")


def s_class() -> str:
    d = pd.read_csv(os.path.join(T, "week07_class_sensitivity.csv"))
    cols = ["sens_glioma", "sens_meningioma", "sens_notumor", "sens_pituitary", "tumour_missed"]
    order = ["Central, no DP", "Central, clip only", "Central, eps 8", "Central, eps 3", "Central, eps 1",
             "Central, patient, clip only", "Central, patient, eps 64", "Central, patient, eps 32",
             "Central, patient, eps 16", "Central, patient, eps 8", "Central, patient, eps 3", "FedAvg natural, no DP",
             "FedAvg natural, eps 8", "FedAvg natural, eps 3", "FedAvg natural, eps 1", "FedAvg natural, patient, eps 8",
             "FedAvg, weight noise s=0.003"]
    d = d.set_index("regime").loc[[o for o in order if o in set(d.regime)]].reset_index()
    lab = lambda x: re.sub(r"eps ([0-9.]+)", lambda m: "$\\varepsilon=" + m.group(1) + "$", x).replace("s=0.003", "$s=0.003$")
    rows = [[lab(r["regime"])] + [ci(r, c) for c in cols] for _, r in d.iterrows()]
    return table("tab:s_class", "Per-class sensitivity on the clean test set, mean (95\\% CI) over five seeds.",
                 ["Setting", "Glioma", "Meningioma", "No tumour", "Pituitary", "Missed tumour"], rows,
                 align="lrrrrr", note="Clean test images per class are 291 glioma, 188 meningioma, 60 no tumour and "
                 "208 pituitary. Missed tumour is the share of tumour images predicted as no tumour. The $t$-intervals "
                 "are not truncated to $[0, 1]$.", resize=True)


SECTIONS = [("ncc", s_ncc), ("full_clean", s_full_clean), ("local", s_local), ("sources", s_sources),
            ("probe", s_probe), ("eps", s_eps), ("lr", s_lr), ("h5", s_h5), ("fedprox", s_fedprox),
            ("scaffold", s_scaffold), ("wn", s_wn), ("lira", s_lira), ("loss_mia", s_loss_mia),
            ("xai", s_xai), ("rand", s_rand), ("border", s_border), ("mediation", s_mediation),
            ("matched", s_matched), ("class", s_class)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "paper", "supplement", "tables"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name, fn in SECTIONS:
        tex = fn()
        open(os.path.join(a.out, f"{name}.tex"), "w").write(tex)
        print(f"{name}: {tex.count(chr(10))} lines")


if __name__ == "__main__":
    main()
