"""
ConvNeXt Step 1 - Patchify Stem
================================

ConvNeXt는 ConvNet을 Transformer 스타일의 디자인으로 재해석하면서, 입력 이미지를
`patch_size x patch_size` 컨볼루션으로 임베딩한 뒤 LayerNorm을 적용해 채널 분포를
안정화합니다. 이 파일은 Julius Ruseckas의 "ConvNeXt on CIFAR10"
(https://juliusruseckas.github.io/ml/convnext-cifar10.html) 구현을 참고하여
패치 임베딩(Stem)과 분류 Head만을 구성한 가장 단순한 형태를 제공합니다.

의도:
- 패치 처리 과정을 별도 클래스로 분리하여 ConvNeXt의 입력부 의도를 설명
- 채널 차원 기준 LayerNorm을 적용하는 보조 모듈 도입
- Global Average Pooling + LayerNorm + Linear 조합의 Head 구조 정리
"""

import torch
import torch.nn as nn

__all__ = [
    "LayerNormChannels",
    "ConvNeXtStem",
    "ConvNeXtHead",
    "ConvNeXtPatchifyClassifier",
]


class LayerNormChannels(nn.Module):
    """채널 차원 기준 LayerNorm.

    ConvNeXt는 채널 우선(C, H, W) 텐서를 다루므로, PyTorch 기본 LayerNorm을 활용하려면
    (B, H, W, C) 형태로 전치했다가 되돌리는 보조 모듈이 필요하다.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = self.norm(x)
        return x.transpose(-1, 1)


class ConvNeXtStem(nn.Sequential):
    """패치 임베딩 전용 Stem (Conv -> LayerNorm)."""

    def __init__(self, in_channels: int = 3, embed_dim: int = 96, patch_size: int = 4):
        super().__init__(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            LayerNormChannels(embed_dim),
        )


class ConvNeXtHead(nn.Sequential):
    """Global Average Pooling 기반 분류 Head."""

    def __init__(self, in_channels: int, num_classes: int = 10):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, num_classes),
        )


class ConvNeXtPatchifyClassifier(nn.Module):
    """ConvNeXt의 Stem+Head만을 사용한 최소 구성 네트워크."""

    def __init__(
        self,
        *,
        patch_size: int = 4,
        embed_dim: int = 96,
        num_classes: int = 10,
        init_weights: bool = False,
    ):
        super().__init__()
        self.stem = ConvNeXtStem(3, embed_dim, patch_size)
        self.head = ConvNeXtHead(embed_dim, num_classes)

        if init_weights:
            self._reset_parameters()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.head(x)



