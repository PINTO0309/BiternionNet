from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .losses import normalize_biternion


@dataclass(frozen=True)
class ModelConfig:
    output_dim: int
    head: str
    variant: str = "standard"
    input_size: tuple[int, int] = (46, 46)
    backbone_activation: str = "relu"


class BiternionHead(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return normalize_biternion(x)


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "swish":
        return nn.SiLU(inplace=True)
    raise ValueError(f"Unsupported backbone activation: {name}")


def _conv_block(in_channels: int, out_channels: int, activation: str) -> list[nn.Module]:
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=3),
        nn.BatchNorm2d(out_channels),
        _activation(activation),
    ]


def standard_backbone(extra_pool: bool = False, activation: str = "relu") -> nn.Sequential:
    layers: list[nn.Module] = []
    layers += _conv_block(3, 24, activation)
    layers += [nn.Conv2d(24, 24, kernel_size=3), nn.BatchNorm2d(24), nn.MaxPool2d(2), _activation(activation)]
    layers += _conv_block(24, 48, activation)
    layers += [nn.Conv2d(48, 48, kernel_size=3), nn.BatchNorm2d(48), nn.MaxPool2d(2), _activation(activation)]
    layers += _conv_block(48, 64, activation)
    layers += _conv_block(64, 64, activation)
    if extra_pool:
        layers += [nn.MaxPool2d(2)]
    return nn.Sequential(*layers)


def hoc_backbone(activation: str = "relu") -> nn.Sequential:
    layers: list[nn.Module] = []
    layers += _conv_block(3, 24, activation)
    layers += _conv_block(24, 24, activation)
    layers += [nn.Conv2d(24, 24, kernel_size=3), nn.BatchNorm2d(24), nn.MaxPool2d((3, 2)), _activation(activation)]
    layers += _conv_block(24, 48, activation)
    layers += _conv_block(48, 48, activation)
    layers += [nn.Conv2d(48, 48, kernel_size=3), nn.BatchNorm2d(48), nn.MaxPool2d(3), _activation(activation)]
    layers += _conv_block(48, 64, activation)
    layers += _conv_block(64, 64, activation)
    return nn.Sequential(*layers)


class HeadPoseNet(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        if config.variant == "hoc":
            self.backbone = hoc_backbone(config.backbone_activation)
        elif config.variant == "idiap":
            self.backbone = standard_backbone(extra_pool=True, activation=config.backbone_activation)
        elif config.variant == "standard":
            self.backbone = standard_backbone(extra_pool=False, activation=config.backbone_activation)
        else:
            raise ValueError(f"Unsupported model variant: {config.variant}")

        feature_dim = self._feature_dim(config.input_size)
        modules: list[nn.Module] = [
            nn.Dropout(0.2),
            nn.Flatten(),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, config.output_dim),
        ]
        if config.head == "biternion":
            modules.append(BiternionHead())
        self.head = nn.Sequential(*modules)
        self._init_last_layer(config.head)

    def _feature_dim(self, input_size: tuple[int, int]) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, input_size[0], input_size[1])
            return int(self.backbone(dummy).numel())

    def _init_last_layer(self, head: str) -> None:
        last_linear = next(module for module in reversed(self.head) if isinstance(module, nn.Linear))
        if head == "biternion":
            nn.init.normal_(last_linear.weight, mean=0.0, std=0.01)
        else:
            nn.init.constant_(last_linear.weight, 0.0)
        nn.init.constant_(last_linear.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


def build_model(config: ModelConfig) -> HeadPoseNet:
    return HeadPoseNet(config)
