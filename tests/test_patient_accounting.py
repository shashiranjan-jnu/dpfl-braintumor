"""Checks of the per-patient accountant (Phase 7, D68)."""
import math

import numpy as np
import pytest

from src.privacy import patient_accounting as pa


def test_shift_distribution_sums_to_one_and_caps():
    j, p = pa.shift_distribution(13, None, 0.1)
    assert len(j) == 14 and abs(p.sum() - 1) < 1e-12
    j, p = pa.shift_distribution(13, 2, 0.1)
    assert list(j) == [0.0, 1.0, 2.0]
    assert abs(p[2] - (1 - 0.9 ** 13 - 13 * 0.1 * 0.9 ** 12)) < 1e-12


def test_rdp_of_one_slice_matches_opacus():
    # the numerical mixture with k = 1 is the subsampled Gaussian, whose RDP Opacus computes in closed form
    sigma = 2.179
    ours = np.array([pa._rdp_step(sigma, 1, None, 0.1, a, 200001) for a in (2.0, 4.0, 8.0)])
    ref = pa.record_rdp(sigma, 0.1, 1, (2.0, 4.0, 8.0))
    assert np.allclose(ours, ref, rtol=2e-3)


@pytest.mark.parametrize("sigma,eps", [(12.885, 1.0), (4.832, 3.0), (2.179, 8.0)])
def test_one_slice_reproduces_paper_budgets(sigma, eps):
    e, _ = pa.rdp_epsilon(sigma, 1)
    assert abs(e - eps) < 0.02                     # RDP conversion used by Opacus 1.6
    assert pa.pld_epsilon(sigma, 1) <= e + 1e-6     # PLD is at least as tight


def test_epsilon_grows_with_patient_size_and_implementations_agree():
    sigma = 4.832
    pld = [pa.pld_epsilon(sigma, k) for k in (1, 2, 5, 13)]
    rdp = [pa.rdp_epsilon(sigma, k)[0] for k in (1, 2, 5, 13)]
    assert all(a < b for a, b in zip(pld, pld[1:]))
    assert all(p <= r * 1.001 for p, r in zip(pld, rdp))        # PLD never looser than RDP
    assert all(r / p < 1.25 for p, r in zip(pld, rdp))           # and close to it


def test_cap_bounds_epsilon_and_m_at_least_k_is_record_level():
    sigma = 4.832
    assert pa.pld_epsilon(sigma, 13, m=1) < pa.pld_epsilon(sigma, 13, m=4) < pa.pld_epsilon(sigma, 13)
    assert math.isclose(pa.pld_epsilon(sigma, 5, m=5), pa.pld_epsilon(sigma, 5), rel_tol=1e-9)


def test_composition_over_clients():
    sigma = 4.832
    one = pa.pld_epsilon(sigma, 4)
    two = pa.pld_epsilon_multi(sigma, [4, 4])
    assert two > one


def test_group_baselines_are_looser():
    sigma, k = 4.832, 13
    tight = pa.pld_epsilon(sigma, k)
    assert pa.rdp_group_epsilon(sigma, k) > tight
    _, d = pa.approx_group_delta(3.0, 1e-5, k)
    assert d > 1                                                 # vacuous at k = 13


def test_calibration_hits_target():
    s = pa.calibrate_sigma(16.0, 13, 2)
    assert pa.pld_epsilon(s, 13, 2) <= 16.0 < pa.pld_epsilon(s * 0.98, 13, 2)


def test_overflow_case_falls_back_to_rdp():
    # dp_accounting 0.6.0 overflows here (found in Step 7.1); the RDP bound must take over and lie between neighbours
    e, method = pa.epsilon(2.179296875, [22])
    assert math.isfinite(e) and method in ("pld", "rdp")
    assert pa.epsilon(2.179296875, [21])[0] < e < pa.rdp_epsilon(2.179296875, 23)[0]
