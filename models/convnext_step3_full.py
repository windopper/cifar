"""
ConvNeXt Step 3 - 풀 스택 계층형 ConvNeXt
=========================================

이 파일은 ConvNeXt의 Stage/Downsample 구조까지 포함한 완제품 버전을 제공한다.
Julius Ruseckas의 CIFAR10 ConvNeXt 노트북
(https://juliusruseckas.github.io/ml/convnext-cifar10.html)을 토대로,
각 Stage에 DownsampleBlock을 배치하고 다단계 채널 폭/블록 깊이를 구성한다.

의도:
- DownsampleBlock으로 해상도를 절반씩 줄이며 계층형 표현 학습
- Stage별 블록 수/채널 수를 설정할 수 있도록 파라미터화
- Stem/Body/Head를 결합한 ConvNeXt Tiny(CIFAR 맞춤) 모델 제공
"""

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
    """LayerNorm + 스트라이드 Conv로 해상도를 절반으로 줄인다."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__(
            LayerNormChannels(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=stride, stride=stride),
        )


class ConvNeXtStage(nn.Sequential):
    """Downsample(필요 시) + ConvNeXtBlock * N."""

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
    """다중 Stage를 순차적으로 연결."""

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
    """CIFAR용 ConvNeXt 전체 모델."""

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
        channel_list=(96, 192, 384, 768),
        num_blocks_list=(3, 3, 9, 3),
        kernel_size=7,
        mlp_ratio=4,
        patch_size=4,
        init_weights=init_weights,
        **kwargs,
    )



