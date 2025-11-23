
from typing import Sequence

import torch
import torch.nn as nn

from .convnext_step1_patchify import LayerNormChannels

__all__ = ["DeepBaselineNetBN3ResidualConvNeXt"]


class ConvNeXtResidualBlock(nn.Module):

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 7,
        mlp_ratio: int = 4,
        drop_prob: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        hidden_channels = channels * mlp_ratio

        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
        )
        self.norm = LayerNormChannels(channels)
        self.pwconv1 = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(
            drop_prob) if drop_prob > 0 else nn.Identity()

        if layer_scale_init_value > 0:
            gamma = layer_scale_init_value * torch.ones(channels)
            self.gamma = nn.Parameter(gamma.view(1, -1, 1, 1))
        else:
            self.gamma = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out = self.dwconv(x)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        out = self.dropout(out)

        if self.gamma is not None:
            out = self.gamma * out

        return residual + out

class StridedDownsample(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            LayerNormChannels(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
        )


class ConvNeXtStage(nn.Sequential):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        *,
        kernel_size: int,
        mlp_ratio: int,
        drop_prob: float,
        layer_scale_init_value: float,
        downsample: bool,
    ):
        layers = []
        if downsample:
            layers.append(StridedDownsample(in_channels, out_channels))
        elif in_channels != out_channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))

        for _ in range(num_blocks):
            layers.append(
                ConvNeXtResidualBlock(
                    out_channels,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    drop_prob=drop_prob,
                    layer_scale_init_value=layer_scale_init_value,
                )
            )

        super().__init__(*layers)


class DeepBaselineNetBN3ResidualConvNeXt(nn.Module):

    def __init__(
        self,
        *,
        num_classes: int = 10,
        channels: Sequence[int] = (64, 128, 256, 512),
        blocks_per_stage: Sequence[int] = (3, 3, 6, 3),
        kernel_size: int = 7,
        mlp_ratio: int = 4,
        drop_prob: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        init_weights: bool = False,
    ):
        super().__init__()
        if len(channels) != len(blocks_per_stage):
            raise ValueError(
                "channels와 blocks_per_stage의 길이는 동일해야 합니다."
            )

        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        stages = []
        in_channels = channels[0]
        for idx, (out_channels, num_blocks) in enumerate(
            zip(channels, blocks_per_stage)
        ):
            stage = ConvNeXtStage(
                in_channels,
                out_channels,
                num_blocks,
                kernel_size=kernel_size,
                mlp_ratio=mlp_ratio,
                drop_prob=drop_prob,
                layer_scale_init_value=layer_scale_init_value,
                downsample=idx != 0,
            )
            stages.append(stage)
            in_channels = out_channels

        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(channels[-1], num_classes),
        )

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.head(x)
        return x
