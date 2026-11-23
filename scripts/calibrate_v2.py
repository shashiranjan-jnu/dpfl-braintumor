"""Noise multipliers for the Phase 7.4 slice-cap arms (protocol v2, D69). CPU only.

    python scripts/calibrate_v2.py --out configs/sigma_v2.csv

For each per-patient target epsilon and slice cap m (None = record-level DP-SGD, no cap), the smallest sigma whose
per-patient epsilon for the largest admissible patient (k_max slices at one client) is at most the target, with
q = 0.1, 1000 steps and delta = 1e-5 (src/privacy/patient_accounting.calibrate_sigma). The achieved epsilon is
recomputed with both accountants as a check.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.privacy import patient_accounting as pa  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default=os.path.join(ROOT, "configs", "protocol_v2.yaml"))
    ap.add_argument("--out", default=os.path.join(ROOT, "configs", "sigma_v2.csv"))
    a = ap.parse_args()
    pr = yaml.safe_load(open(a.protocol))
    sc, fx = pr["slice_cap_mechanism"], pr["fixed_from_v1"]
    kmax, q, steps, delta = int(sc["k_max"]), fx["sample_rate_q"], fx["rounds_T"] * fx["local_steps_S"], fx["delta"]
    rows = []
    for e in sc["per_patient_targets"]:
        for m in list(sc["m_values"]) + [None]:
            s = pa.calibrate_sigma(float(e), kmax, m, q, steps, delta)
            eps, method = pa.epsilon(s, [kmax], m, q, steps, delta)
            rows.append({"per_patient_eps": e, "m": "record" if m is None else m, "k_max": kmax, "sigma": s,
                         "eps_achieved": eps, "eps_method": method,
                         "eps_rdp_check": pa.rdp_epsilon(s, kmax, m, q, steps, delta)[0]})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
