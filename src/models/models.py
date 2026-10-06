"""Models without BatchNorm.

BatchNorm mixes statistics across the examples of a batch, which breaks the
per-example sensitivity bound that DP-SGD relies on. Both models therefore use
GroupNorm, which normalises each example independently.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _block(cin: int, cout: int, groups: int = 8) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.GroupNorm(min(groups, cout), cout),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """Four conv blocks and a global-average-pooling head, about 98k parameters.

    128 x 128 input -> 8 x 8 x 128 feature map -> GAP -> linear(4).
    `self.features[-1]` is the last conv block, used as the Grad-CAM target.
    """

    def __init__(self, n_classes: int = 4, width: int = 16):
        super().__init__()
        w = width
        self.features = nn.Sequential(_block(1, w), _block(w, 2 * w),
                                      _block(2 * w, 4 * w), _block(4 * w, 8 * w))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(8 * w, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x)).flatten(1))

    @property
    def cam_layer(self) -> nn.Module:
        return self.features[-1]


def _bn_to_gn(module: nn.Module, groups: int = 32) -> nn.Module:
    """Replace every BatchNorm2d with GroupNorm, keeping the learned affine terms."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            gn = nn.GroupNorm(min(groups, child.num_features), child.num_features)
            with torch.no_grad():
                gn.weight.copy_(child.weight)
                gn.bias.copy_(child.bias)
            setattr(module, name, gn)
        else:
            _bn_to_gn(child, groups)
    return module


class ResNet18GN(nn.Module):
    """torchvision ResNet-18 with GroupNorm and a single-channel stem.

    With pretrained=True, ImageNet weights are loaded first. The RGB stem is
    collapsed to one channel by summing over input channels, and BatchNorm
    layers are swapped for GroupNorm with the affine parameters copied across.
    The BatchNorm running statistics are discarded, so the pretrained features
    are a starting point rather than an exact transfer (decision D20).
    """

    def __init__(self, n_classes: int = 4, pretrained: bool = False):
        super().__init__()
        from torchvision.models import resnet18, ResNet18_Weights
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        w = net.conv1.weight.detach().sum(1, keepdim=True)
        net.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            net.conv1.weight.copy_(w)
        _bn_to_gn(net)
        net.fc = nn.Linear(net.fc.in_features, n_classes)
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @property
    def cam_layer(self) -> nn.Module:
        return self.net.layer4


def build_model(name: str, n_classes: int = 4, pretrained: bool = False) -> nn.Module:
    if name == "small_cnn":
        return SmallCNN(n_classes)
    if name == "resnet18_gn":
        return ResNet18GN(n_classes, pretrained)
    raise ValueError(f"Unknown model {name}")


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
