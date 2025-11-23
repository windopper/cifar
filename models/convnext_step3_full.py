from typing import Sequence

import torch
import torch.nn as nn

from .convnext_step1_patchify import ConvNeXtHead, ConvNeXtStem, LayerNormChannels
from .convnext_step2_local_block import ConvNeXtBlock, ResidualBranch

__all__ = [
    "DownsampleBlock",
    "ConvNeXtStage",
    "ConvNeXtBody",
    "ConvNeXtCIFAR",
    "convnext_tiny",
]


class DownsampleBlock(nn.Sequential):

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__(
            LayerNormChannels(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=stride, stride=stride),
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
        p_drop: float,
    ):
        layers = []
        if in_channels != out_channels:
            layers.append(DownsampleBlock(in_channels, out_channels))
        layers.extend(
            ConvNeXtBlock(
                out_channels,
                kernel_size=kernel_size,
                mlp_ratio=mlp_ratio,
                p_drop=p_drop,
            )
            for _ in range(num_blocks)
        )
        super().__init__(*layers)


class ConvNeXtBody(nn.Sequential):

    def __init__(
        self,
        in_channels: int,
        channel_list: Sequence[int],
        num_blocks_list: Sequence[int],
        *,
        kernel_size: int,
        mlp_ratio: int,
        p_drop: float,
    ):
        layers = []
        for out_channels, num_blocks in zip(channel_list, num_blocks_list):
            layers.append(
                ConvNeXtStage(
                    in_channels,
                    out_channels,
                    num_blocks,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    p_drop=p_drop,
                )
            )
            in_channels = out_channels
        super().__init__(*layers)


class ConvNeXtCIFAR(nn.Module):

    def __init__(
        self,
        *,
        num_classes: int = 10,
        in_channels: int = 3,
        patch_size: int = 4,
        channel_list: Sequence[int] = (96, 192, 384, 768),
        num_blocks_list: Sequence[int] = (3, 3, 9, 3),
        kernel_size: int = 7,
        mlp_ratio: int = 4,
        p_drop: float = 0.0,
        init_weights: bool = False,
    ):
        super().__init__()
        self.stem = ConvNeXtStem(in_channels, channel_list[0], patch_size)
        self.body = ConvNeXtBody(
            channel_list[0],
            channel_list,
            num_blocks_list,
            kernel_size=kernel_size,
            mlp_ratio=mlp_ratio,
            p_drop=p_drop,
        )
        self.head = ConvNeXtHead(channel_list[-1], num_classes)

        if init_weights:
            self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.zeros_(module.bias)
            elif isinstance(module, ResidualBranch):
                nn.init.zeros_(module.gamma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.body(x)
        return self.head(x)


def convnext_tiny(
    *,
    num_classes: int = 10,
    init_weights: bool = False,
    **kwargs,
) -> ConvNeXtCIFAR:
    """ConvNeXt Tiny 구성을 반환하는 헬퍼."""

    return ConvNeXtCIFAR(
        num_classes=num_classes,
        channel_list=(64, 128, 256, 512),
        num_blocks_list=(2, 2, 2, 2),
        kernel_size=7,
        patch_size=1,
        init_weights=init_weights,
        **kwargs,
    )



