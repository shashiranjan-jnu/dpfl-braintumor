"""Saliency maps for RQ3: Grad-CAM, Grad-CAM++ and Integrated Gradients.

Every function takes a batch x (B, 1, H, W) and target class indices (B,) and
returns non-negative maps (B, H, W) at input resolution, each scaled to [0, 1].
Models are put in eval mode; BatchNorm in resnet18_frozenbn is eval anyway.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _normalise(m: torch.Tensor) -> torch.Tensor:
    m = m.clamp(min=0)
    flat = m.flatten(1)
    lo, hi = flat.min(1).values, flat.max(1).values
    return ((flat - lo[:, None]) / (hi - lo + 1e-12)[:, None]).view_as(m)


def _cam_parts(model, x, target):
    """Activations A (B, C, h, w) of model.cam_layer and dy_c/dA."""
    acts = {}

    def hook(_, __, out):
        out.retain_grad()
        acts["a"] = out

    h = model.cam_layer.register_forward_hook(hook)
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        logits = model(x)
        logits.gather(1, target[:, None]).sum().backward()
    finally:
        h.remove()
    a = acts["a"]
    return a.detach(), a.grad.detach(), logits.detach()


def gradcam(model, x, target) -> torch.Tensor:
    a, g, _ = _cam_parts(model, x, target)
    w = g.mean((2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    return _normalise(cam)


def gradcampp(model, x, target) -> torch.Tensor:
    """Grad-CAM++ (Chattopadhay et al., 2018) with the usual exponential closed form."""
    a, g, _ = _cam_parts(model, x, target)
    g2, g3 = g ** 2, g ** 3
    denom = 2 * g2 + (a * g3).sum((2, 3), keepdim=True)
    alpha = g2 / torch.where(denom != 0, denom, torch.ones_like(denom))
    w = (alpha * F.relu(g)).sum((2, 3), keepdim=True)
    cam = F.relu((w * a).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
    return _normalise(cam)


def integrated_gradients(model, x, target, baseline: float, steps: int = 32) -> torch.Tensor:
    """|IG| with a constant baseline image (the standardised value of a black pixel).

    Uses the right Riemann sum over `steps` points. Returns |attribution| scaled to [0, 1].
    """
    model.eval()
    out = torch.zeros_like(x)
    base = torch.full_like(x, baseline)
    for k in range(1, steps + 1):
        xi = (base + (k / steps) * (x - base)).requires_grad_(True)
        logit = model(xi).gather(1, target[:, None]).sum()
        out += torch.autograd.grad(logit, xi)[0]
    attr = ((x - base) * out / steps).abs()[:, 0]
    return _normalise(attr)


METHODS = {"gradcam": gradcam, "gradcampp": gradcampp}
