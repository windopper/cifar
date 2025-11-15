"""
ConvNeXt Step 2 - Depthwise ConvNeXt Block
==========================================

이 단계에서는 Stem/Head 외에 ConvNeXt의 핵심인 Depthwise Separable Conv 블록과
Residual γ 스케일링을 추가한다. Julius Ruseckas의 CIFAR10 ConvNeXt 예제
(https://juliusruseckas.github.io/ml/convnext-cifar10.html)를 바탕으로, 블록을 반복하여
로컬 토큰 상호작용을 학습하는 소형 네트워크를 구성한다.

의도:
- Depthwise Conv를 통해 공간적 혼합, 1x1 Conv를 통해 채널 혼합을 분리
- Residual 경로에 학습 가능한 γ(초기 0) 스칼라를 두어 안정적인 학습 유도
- Stage/Downsample 없이 동일 해상도에서 ConvNeXt Block만 누적
"""

import torch
import torch.nn as nn

from .convnext_step1_patchify import (
    ConvNeXtHead,
    ConvNeXtStem,
    LayerNormChannels,
)

__all__ = [
    "ResidualBranch",
    "ConvNeXtBlock",
    "ConvNeXtLocalBlockClassifier",
]


class ResidualBranch(nn.Module):
    """γ 파라미터가 있는 Residual 래퍼."""

    def __init__(self, *layers: nn.Module):
        super().__init__()
        self.residual = nn.Sequential(*layers)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.gamma * self.residual(x)


class ConvNeXtBlock(ResidualBranch):
    """ConvNeXt의 Depthwise Conv + 채널 혼합 블록."""

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 7,
        mlp_ratio: int = 4,
        p_drop: float = 0.0,
    ):
        padding = (kernel_size - 1) // 2
        hidden_channels = channels * mlp_ratio
        super().__init__(
            nn.Conv2d(
                channels,
                channels,
                kernel_size,
                padding=padding,
                groups=channels,
            ),
            LayerNormChannels(channels),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.Dropout(p_drop),
        )


class ConvNeXtLocalBlockClassifier(nn.Module):
    """ConvNeXt Stem + 반복 블록 + Head 구조."""

    def __init__(
        self,
        *,
        patch_size: int = 4,
        embed_dim: int = 96,
        num_blocks: int = 4,
        kernel_size: int = 7,
        mlp_ratio: int = 4,
        p_drop: float = 0.0,
        num_classes: int = 10,
        init_weights: bool = False,
    ):
        super().__init__()
        self.stem = ConvNeXtStem(3, embed_dim, patch_size)
        self.blocks = nn.Sequential(
            *[
                ConvNeXtBlock(
                    embed_dim,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    p_drop=p_drop,
                )
                for _ in range(num_blocks)
            ]
        )
        self.head = ConvNeXtHead(embed_dim, num_classes)

        if init_weights:
            self._reset_parameters()

    def _reset_parameters(self):
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
        x = self.blocks(x)
        return self.head(x)



