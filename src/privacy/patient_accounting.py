"""Per-patient privacy accounting of DP-SGD with Poisson sampling of slices (Phase 7, D67, D68).

A patient (unit) with k slices is added to or removed from a client's data. Under Poisson sampling at rate q, the
number of the patient's slices in one batch is J ~ Binomial(k, q). With per-slice clipping to C and worst-case
alignment of the clipped gradients, the patient shifts the noisy sum by J*C, so one step is a mixture of Gaussians
with shifts j*C, j = 0..k, against N(0, sigma^2 C^2). Ganesh (2024, arXiv:2401.10294) shows that accounting this
mixture gives tight group-level guarantees for DP-SGD; Schuchardt et al. (2024, arXiv:2403.04867) treat the same
problem in a general framework. This module applies that accounting; it is not a new accountant.

The slice cap m (mechanism A of D68) keeps at most m of a patient's sampled slices per step, chosen uniformly at
random, so the shift becomes min(J, m) * C. m = 1 bounds every patient by C per step; m >= k is plain record-level
DP-SGD.

Two independent implementations are provided and must agree:
  pld_epsilon   privacy loss distribution of the mixture (Google dp_accounting, connect-the-dots, pessimistic)
  rdp_epsilon   numerical Renyi divergence of the mixture on a grid, both directions, converted with the
                conversion Opacus 1.6 uses (Balle et al. 2020)
plus the two group-privacy baselines the paper compares against:
  approx_group_delta   (k eps, k e^{(k-1) eps} delta) for an (eps, delta) guarantee (Dwork and Roth 2014)
  rdp_group_epsilon    Renyi group privacy for groups of 2^c (Mironov 2017, Proposition 2)
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom

ORDERS = (1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0, 24.0,
          28.0, 32.0, 40.0, 48.0, 56.0, 64.0, 80.0, 96.0, 128.0, 160.0, 192.0, 256.0)


def shift_distribution(k: int, m: int | None, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Support (in units of C) and probabilities of the per-step shift min(J, m), J ~ Binomial(k, q)."""
    top = k if m is None else min(k, m)
    j = np.arange(top + 1)
    p = binom.pmf(j, k, q)
    if m is not None and k > m:
        p[-1] = binom.sf(m - 1, k, q)          # P(J >= m)
    return j.astype(float), p / p.sum()


def _to_eps(rdp: np.ndarray, orders: np.ndarray, delta: float) -> tuple[float, float]:
    eps = rdp - (np.log(delta) + np.log(orders)) / (orders - 1) + np.log((orders - 1) / orders)
    i = int(np.nanargmin(eps))
    return float(eps[i]), float(orders[i])


def default_interval(k: int, m: int | None) -> float:
    """Discretisation of the privacy loss: 1e-3 up to 20 shifts, 1e-2 above, where epsilon is in the hundreds.
    The pessimistic PLD stays an upper bound at either interval."""
    return 1e-3 if (k if m is None else min(k, m)) <= 20 else 1e-2


def pld_epsilon(sigma: float, k: int, m: int | None = None, q: float = 0.1, steps: int = 1000,
                delta: float = 1e-5, interval: float | None = None) -> float:
    """Per-patient epsilon after `steps` Poisson-sampled steps, from the privacy loss distribution."""
    from dp_accounting.pld import privacy_loss_distribution as pldm
    interval = interval or default_interval(k, m)
    j, p = shift_distribution(k, m, q)
    pld = pldm.from_mixture_gaussian_mechanism(standard_deviation=sigma, sensitivities=list(j),
                                               sampling_probs=list(p), value_discretization_interval=interval)
    return float(pld.self_compose(steps).get_epsilon_for_delta(delta))


def pld_epsilon_multi(sigma: float, ks: list[int], m: int | None = None, q: float = 0.1, steps: int = 1000,
                      delta: float = 1e-5, interval: float | None = None) -> float:
    """Patient spread over several clients with ks[c] slices at client c: compose the per-client guarantees."""
    from dp_accounting.pld import privacy_loss_distribution as pldm
    total = None
    interval = interval or default_interval(max(ks), m)
    for k in ks:
        j, p = shift_distribution(k, m, q)
        pld = pldm.from_mixture_gaussian_mechanism(standard_deviation=sigma, sensitivities=list(j),
                                                   sampling_probs=list(p),
                                                   value_discretization_interval=interval).self_compose(steps)
        total = pld if total is None else total.compose(pld)
    return float(total.get_epsilon_for_delta(delta))


@lru_cache(maxsize=4096)
def _rdp_step(sigma: float, k: int, m: int | None, q: float, alpha: float, n_grid: int) -> float:
    j, p = shift_distribution(k, m, q)
    x = np.linspace(-14 * sigma, j[-1] + 14 * sigma, n_grid)
    dx = x[1] - x[0]
    logq = -x ** 2 / (2 * sigma ** 2)
    logp = logsumexp(np.log(np.maximum(p, 1e-300))[:, None] - (x[None, :] - j[:, None]) ** 2 / (2 * sigma ** 2), axis=0)
    c = -0.5 * np.log(2 * np.pi * sigma ** 2)
    d_pq = logsumexp(alpha * (logp + c) + (1 - alpha) * (logq + c)) + np.log(dx)
    d_qp = logsumexp(alpha * (logq + c) + (1 - alpha) * (logp + c)) + np.log(dx)
    return float(max(d_pq, d_qp) / (alpha - 1))


def rdp_epsilon(sigma: float, k: int, m: int | None = None, q: float = 0.1, steps: int = 1000,
                delta: float = 1e-5, n_grid: int = 200001) -> tuple[float, float]:
    """Per-patient epsilon from the numerical Renyi divergence of the mixture (both directions), with its order."""
    orders = np.array(ORDERS)
    rdp = np.array([_rdp_step(sigma, k, m, q, a, n_grid) for a in orders]) * steps
    return _to_eps(rdp, orders, delta)


def record_rdp(sigma: float, q: float = 0.1, steps: int = 1000, orders=ORDERS) -> np.ndarray:
    """Opacus RDP of the Poisson-subsampled Gaussian mechanism for one record."""
    from opacus.accountants.analysis.rdp import compute_rdp
    return np.asarray(compute_rdp(q=q, noise_multiplier=sigma, steps=steps, orders=list(orders)), float)


def approx_group_delta(eps: float, delta: float, k: int) -> tuple[float, float]:
    """Classical group privacy of an (eps, delta) guarantee: (k eps, k e^{(k-1) eps} delta)."""
    return k * eps, float(k * math.exp(min((k - 1) * eps, 700.0)) * delta)


def rdp_group_epsilon(sigma: float, k: int, q: float = 0.1, steps: int = 1000, delta: float = 1e-5) -> float:
    """Renyi group privacy (Mironov 2017, Prop. 2): an (a, e)-RDP mechanism is (a / 2^c, 3^c e)-RDP for groups of
    2^c. k is rounded up to the next power of two; orders a with a / 2^c <= 1 are skipped."""
    c = max(0, math.ceil(math.log2(k))) if k > 1 else 0
    big = np.array([2 ** c * a for a in ORDERS if 2 ** c * a <= 4096], float)
    if len(big) == 0:
        return float("inf")
    rdp = record_rdp(sigma, q, steps, tuple(big)) * 3 ** c
    small = big / 2 ** c
    keep = small > 1
    if not keep.any():
        return float("inf")
    return _to_eps(rdp[keep], small[keep], delta)[0]


def calibrate_sigma(target_eps: float, k: int, m: int | None, q: float = 0.1, steps: int = 1000,
                    delta: float = 1e-5, lo: float = 0.3, hi: float = 500.0, iters: int = 24) -> float:
    """Smallest sigma (to bisection precision, rounded up) whose PLD per-patient epsilon is at most target_eps."""
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        if pld_epsilon(mid, k, m, q, steps, delta) > target_eps:
            lo = mid
        else:
            hi = mid
    return hi
