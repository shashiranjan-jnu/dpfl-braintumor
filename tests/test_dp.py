"""Step 2.2 DP correctness tests. Run with:  python -m pytest -q tests"""
import math

import pytest
import torch
import torch.nn.functional as F

from src.models.models import SmallCNN, n_params
from src.privacy import dp as dpm

torch.manual_seed(0)


def _batch(n=12, size=128):
    x = torch.randn(n, 1, size, size)
    y = torch.randint(0, 4, (n,))
    return x, y


def test_model_has_no_batchnorm_and_expected_size():
    m = SmallCNN()
    assert not any(isinstance(k, torch.nn.BatchNorm2d) for k in m.modules())
    assert 90_000 < n_params(m) < 110_000


def test_per_example_clipping_bound():
    """Test 1. Every clipped per-example gradient has L2 norm <= C."""
    m = SmallCNN()
    x, y = _batch()
    for C in (0.01, 0.1, 1.0):
        _, norms = dpm.clipped_grad_sum(m, F.cross_entropy, x, y, C, 5, return_norms=True)
        assert norms.max().item() <= C * (1 + 1e-4)


def test_zero_noise_large_clip_matches_autograd():
    """Test 3. sigma = 0 and C = inf reproduces the non-private gradient exactly."""
    m = SmallCNN()
    x, y = _batch()
    gsum = dpm.clipped_grad_sum(m, F.cross_entropy, x, y, clip_norm=1e9, max_physical_batch=5)
    g = dpm.add_noise(gsum, sigma=0.0, clip_norm=1e9, expected_batch=len(y))
    m.zero_grad()
    F.cross_entropy(m(x), y, reduction="mean").backward()
    for k, p in m.named_parameters():
        assert torch.allclose(g[k], p.grad, atol=1e-5, rtol=1e-4), k


def test_microbatching_is_exact():
    m = SmallCNN()
    x, y = _batch(10)
    a = dpm.clipped_grad_sum(m, F.cross_entropy, x, y, 0.5, max_physical_batch=10)
    b = dpm.clipped_grad_sum(m, F.cross_entropy, x, y, 0.5, max_physical_batch=3)
    for k in a:
        assert torch.allclose(a[k], b[k], atol=1e-6)


def test_sensitivity_of_clipped_sum():
    """Removing one record changes the clipped sum by at most C in L2 norm."""
    m = SmallCNN()
    x, y = _batch(8)
    C = 0.3
    full = dpm.clipped_grad_sum(m, F.cross_entropy, x, y, C)
    drop = dpm.clipped_grad_sum(m, F.cross_entropy, x[1:], y[1:], C)
    diff = math.sqrt(sum((full[k] - drop[k]).pow(2).sum().item() for k in full))
    assert diff <= C * (1 + 1e-4)


def test_noise_scale():
    g = {"w": torch.zeros(200_000)}
    out = dpm.add_noise(g, sigma=1.3, clip_norm=0.7, expected_batch=10.0,
                        generator=torch.Generator().manual_seed(1))
    assert abs(out["w"].std().item() - 1.3 * 0.7 / 10.0) < 0.002


@pytest.mark.parametrize("sigma,steps", [(0.7, 600), (1.0, 600), (1.5, 1200), (2.5, 600)])
def test_two_accountants_agree(sigma, steps):
    """Test 2. Opacus RDP and Google dp_accounting RDP agree within 5 percent."""
    q, delta = 0.1, 1e-5
    a = dpm.epsilon_opacus(sigma, q, steps, delta)
    b = dpm.epsilon_dp_accounting(sigma, q, steps, delta, "rdp")
    assert abs(a - b) / b < 0.05, (a, b)
    # RDP is an upper bound, so the tighter PLD accountant must not exceed it.
    c = dpm.epsilon_dp_accounting(sigma, q, steps, delta, "pld")
    assert c <= a * 1.01


@pytest.mark.parametrize("target", [1.0, 3.0, 8.0])
def test_calibration_hits_target(target):
    q, steps, delta = 0.1, 600, 1e-5
    s = dpm.calibrate_sigma(target, q, steps, delta)
    eps = dpm.epsilon_opacus(s, q, steps, delta)
    assert eps <= target and eps > target * 0.97


def test_poisson_rate():
    gen = torch.Generator().manual_seed(0)
    m = dpm.poisson_mask(200_000, 0.1, gen, "cpu")
    assert abs(m.float().mean().item() - 0.1) < 0.003


def test_frozen_bn_resnet_is_per_example_and_stats_never_change():
    """Frozen BatchNorm must act per example, so DP sensitivity holds (D25)."""
    from src.models.models import ResNet18FrozenBN
    m = ResNet18FrozenBN(pretrained=False)
    m.train()
    assert all(not k.training for k in m.modules() if isinstance(k, torch.nn.BatchNorm2d))
    x, y = _batch(6, 64)
    with torch.no_grad():
        alone = m(x[:1])
        mixed = m(x)[:1]
    assert torch.allclose(alone, mixed, atol=1e-5)       # no cross-example mixing
    before = {k: v.clone() for k, v in m.named_buffers()}
    dpm.clipped_grad_sum(m, F.cross_entropy, x, y, 1.0, 3)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    F.cross_entropy(m(x), y).backward()
    opt.step()
    for k, v in m.named_buffers():
        assert torch.equal(v, before[k]), k


def test_frozen_bn_resnet_clipping_bound():
    from src.models.models import ResNet18FrozenBN
    m = ResNet18FrozenBN(pretrained=False)
    x, y = _batch(4, 64)
    _, norms = dpm.clipped_grad_sum(m, F.cross_entropy, x, y, 0.5, 2, return_norms=True)
    assert norms.max().item() <= 0.5 * (1 + 1e-4)
