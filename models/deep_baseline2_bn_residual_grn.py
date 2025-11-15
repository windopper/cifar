"""
DeepBaselineNetBN2ResidualGRN
============================

DeepBaselineNetBN2Residual에 2024/2025년 CNN 이미지 분류 SOTA 계열(ConvNeXt V2, EfficientViT 등)
모델에서 핵심적으로 사용하는 Global Response Normalization(GRN)과
대규모 depthwise-separable Conv 블록을 결합한 변형을 도입한 버전입니다.

도입된 최신 기술: ConvNeXt V2 스타일의 GRN + LayerScale + DropPath
- Depthwise Conv (k=7)로 지역적 문맥을 넓게 포착
- GRN(Global Response Normalization)은 ConvNeXt V2 (Meta AI, CVPR 2023)에서 제안되어
  2024/2025년 ImageNet Top-1 챔피언 CNN 계열에서 표준으로 사용됨
- LayerScale과 DropPath(Stochastic Depth)는 깊은 네트워크에서도 안정적인 학습을 제공
"""

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """
    Stochastic Depth 구현 (ConvNeXt, EfficientNet, DeiT 등에서 사용)
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class LayerNorm2d(nn.Module):
    """
    ConvNeXt 계열이 즐겨 사용하는 Channel-first LayerNorm 구현
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        weight = self.weight.view(1, -1, 1, 1)
        bias = self.bias.view(1, -1, 1, 1)
        return x * weight + bias


class GlobalResponseNorm(nn.Module):
    """
    ConvNeXt V2의 핵심인 GRN(Global Response Normalization) 모듈
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2ResidualBlock(nn.Module):
    """
    ConvNeXt V2 스타일 Residual Block
    - DW Conv(k=7) + LayerNorm2d
    - 1x1 Conv 확장 -> GELU -> GRN -> 1x1 Conv 축소
    - LayerScale + DropPath + Skip 연결
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 4,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-5,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=7,
            padding=3,
            stride=stride,
            groups=in_channels,
            bias=True,
        )
        self.norm = LayerNorm2d(in_channels)

        hidden_dim = int(out_channels * expansion)
        self.pwconv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.grn = GlobalResponseNorm(hidden_dim)
        self.pwconv2 = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

        if layer_scale_init_value > 0:
            self.layer_scale = nn.Parameter(
                layer_scale_init_value * torch.ones(out_channels)
            )
        else:
            self.layer_scale = None

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        else:
            self.shortcut = nn.Identity()

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)

        out = self.dwconv(x)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.grn(out)
        out = self.pwconv2(out)

        if self.layer_scale is not None:
            out = out * self.layer_scale.view(1, -1, 1, 1)

        out = shortcut + self.drop_path(out)
        return out


class DeepBaselineNetBN2ResidualGRN(nn.Module):
    """
    DeepBaselineNetBN2Residual + ConvNeXt V2 블록 융합 모델
    - GRN, LayerScale, DropPath 기반 Residual Block으로 고급 표현력 확보
    - 4개의 Stage(64/128/256/512 채널)로 구성, 각 Stage는 2개의 블록
    - AdaptiveAvgPool + LayerNorm 기반 헤드로 안정적 분류
    """

    def __init__(
        self,
        init_weights: bool = False,
        drop_path_rate: float = 0.1,
        layer_scale_init_value: float = 1e-5,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        stage_specs = [
            (64, 2, 1),
            (128, 2, 2),
            (256, 2, 2),
            (512, 2, 2),
        ]

        total_blocks = sum(num_blocks for _, num_blocks, _ in stage_specs)
        if drop_path_rate > 0:
            drop_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        else:
            drop_rates = [0.0] * total_blocks

        stages = []
        in_channels = 64
        dp_idx = 0

        for out_channels, num_blocks, stride in stage_specs:
            blocks = []
            for block_idx in range(num_blocks):
                block_stride = stride if block_idx == 0 else 1
                blocks.append(
                    ConvNeXtV2ResidualBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        stride=block_stride,
                        drop_path=drop_rates[dp_idx],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                )
                dp_idx += 1
                in_channels = out_channels
            stages.append(nn.Sequential(*blocks))

        self.stages = nn.ModuleList(stages)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 10),
        )

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, LayerNorm2d)):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

