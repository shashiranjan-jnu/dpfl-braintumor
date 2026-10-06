"""DP-SGD building blocks.

Mechanism per local step at one client j (record-level DP, decision D19):

    B      ~ Poisson sample of D_j, each record kept independently with rate q
    g_i    = grad of loss(x_i, y_i)                       for every i in B
    g_i   <- g_i * min(1, C / ||g_i||_2)                  per-example clipping
    g_hat  = (sum_i g_i + N(0, sigma^2 C^2 I)) / (q * n_j)

The sum of clipped gradients has L2 sensitivity C to adding or removing one
record, so each step is a subsampled Gaussian mechanism with noise multiplier
sigma and sampling rate q. Dividing by the expected batch size q * n_j, never
by the realised |B|, keeps the denominator data-independent.

With a fixed q, every client runs the same number of steps, so every client
spends the same epsilon regardless of how many images it holds.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap


def per_sample_grad_fn(model: nn.Module, loss_fn):
    """Returns f(params, buffers, x, y) -> dict of per-example gradients."""
    def loss_one(params, buffers, x, y):
        out = functional_call(model, (params, buffers), (x.unsqueeze(0),))
        return loss_fn(out, y.unsqueeze(0))
    return vmap(grad(loss_one), in_dims=(None, None, 0, 0))


def clipped_grad_sum(model: nn.Module, loss_fn, x: torch.Tensor, y: torch.Tensor,
                     clip_norm: float, max_physical_batch: int = 64,
                     return_norms: bool = False):
    """Sum over the batch of per-example gradients clipped to L2 norm clip_norm.

    The batch is processed in chunks of max_physical_batch so that per-example
    gradients of large models fit in GPU memory. Chunking changes nothing
    mathematically, because clipping happens per example before summation.
    """
    params = {k: v.detach() for k, v in model.named_parameters() if v.requires_grad}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    fn = per_sample_grad_fn(model, loss_fn)
    total = {k: torch.zeros_like(v) for k, v in params.items()}
    norms = []
    for s in range(0, x.shape[0], max_physical_batch):
        g = fn(params, buffers, x[s:s + max_physical_batch], y[s:s + max_physical_batch])
        sq = sum(t.flatten(1).pow(2).sum(1) for t in g.values())
        nrm = sq.sqrt()
        factor = (clip_norm / (nrm + 1e-6)).clamp(max=1.0)
        for k, t in g.items():
            total[k] += torch.einsum("i,i...->...", factor, t)
        if return_norms:
            norms.append(nrm * factor)
    if return_norms:
        return total, (torch.cat(norms) if norms else torch.zeros(0, device=x.device))
    return total


def add_noise(grad_sum: dict, sigma: float, clip_norm: float, expected_batch: float,
              generator: torch.Generator | None = None) -> dict:
    """Gaussian mechanism on the clipped sum, then division by q * n_j."""
    out = {}
    for k, g in grad_sum.items():
        noise = torch.randn(g.shape, generator=generator, device=g.device, dtype=g.dtype)
        out[k] = (g + sigma * clip_norm * noise) / expected_batch
    return out


def poisson_mask(n: int, q: float, generator: torch.Generator, device) -> torch.Tensor:
    return torch.rand(n, generator=generator, device=device) < q


# ---------------------------------------------------------------- accounting
def epsilon_opacus(sigma: float, q: float, steps: int, delta: float,
                   accountant: str = "rdp") -> float:
    if sigma <= 0:
        return float("inf")
    from opacus.accountants import create_accountant
    acc = create_accountant(accountant)
    acc.history.append((sigma, q, steps))
    return float(acc.get_epsilon(delta))


def epsilon_dp_accounting(sigma: float, q: float, steps: int, delta: float,
                          method: str = "rdp") -> float:
    """Independent second accountant (Google dp_accounting), used in tests."""
    if sigma <= 0:
        return float("inf")
    import dp_accounting
    event = dp_accounting.SelfComposedDpEvent(
        dp_accounting.PoissonSampledDpEvent(q, dp_accounting.GaussianDpEvent(sigma)), steps)
    if method == "rdp":
        from dp_accounting.rdp import rdp_privacy_accountant
        acc = rdp_privacy_accountant.RdpAccountant()
    else:
        from dp_accounting.pld import pld_privacy_accountant
        acc = pld_privacy_accountant.PLDAccountant()
    acc.compose(event)
    return float(acc.get_epsilon(delta))


def calibrate_sigma(target_epsilon: float, q: float, steps: int, delta: float,
                    accountant: str = "rdp", tol: float = 0.005) -> float:
    """Smallest sigma whose epsilon does not exceed target_epsilon (bisection)."""
    lo, hi = 0.05, 1.0
    while epsilon_opacus(hi, q, steps, delta, accountant) > target_epsilon:
        hi *= 2
        if hi > 1e4:
            raise ValueError("Target epsilon unreachable")
    while hi - lo > tol * hi:
        mid = (lo + hi) / 2
        if epsilon_opacus(mid, q, steps, delta, accountant) > target_epsilon:
            lo = mid
        else:
            hi = mid
    return hi
