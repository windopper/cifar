"""
Attention Module 모듈 (GELU 활성화 함수 사용)

Residual Attention Network의 CIFAR-10(32x32) 입력을 위한 Attention 모듈들을 제공합니다.
각 stage별로 다른 해상도에 최적화된 attention 모듈을 제공합니다.
GELU(Gaussian Error Linear Unit) 활성화 함수를 사용합니다.

참고 자료:
- Residual Attention Network 논문
- https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch
"""

from typing import Tuple

import torch
import torch.nn as nn

from .residual_block_gelu import ResidualBlock


class AttentionModuleStage1CIFAR_GELU(nn.Module):
    """32x32 해상도(또는 사용자 지정 size1) 입력을 위한 CIFAR 전용 Stage-1 Attention (GELU 활성화 함수 사용)."""

    def __init__(self, in_channels: int, out_channels: int,
                 size1: Tuple[int, int] = (32, 32), size2: Tuple[int, int] = (16, 16)):
        super().__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            ResidualBlock(out_channels, out_channels),
            ResidualBlock(out_channels, out_channels)
        )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.down_residual_blocks1 = ResidualBlock(out_channels, out_channels)
        self.skip1_connection = ResidualBlock(out_channels, out_channels)

        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.middle_2r_blocks = nn.Sequential(
            ResidualBlock(out_channels, out_channels),
            ResidualBlock(out_channels, out_channels)
        )

        self.interpolation1 = nn.UpsamplingBilinear2d(size=size2)
        self.up_residual_blocks1 = ResidualBlock(out_channels, out_channels)
        self.interpolation2 = nn.UpsamplingBilinear2d(size=size1)

        self.conv1_1_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

        self.last_blocks = ResidualBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)

        out_mpool1 = self.mpool1(x)
        out_down = self.down_residual_blocks1(out_mpool1)
        out_skip1 = self.skip1_connection(out_down)

        out_mpool2 = self.mpool2(out_down)
        out_middle = self.middle_2r_blocks(out_mpool2)

        out_interp = self.interpolation1(out_middle) + out_down
        out = out_interp + out_skip1
        out = self.up_residual_blocks1(out)

        out_interp2 = self.interpolation2(out) + out_trunk
        out_mask = self.conv1_1_blocks(out_interp2)

        out = (1 + out_mask) * out_trunk
        out_last = self.last_blocks(out)
        return out_last


class AttentionModuleStage2CIFAR_GELU(nn.Module):
    """16x16 해상도 입력을 위한 CIFAR 전용 Stage-2 Attention (GELU 활성화 함수 사용)."""

    def __init__(self, in_channels: int, out_channels: int,
                 size: Tuple[int, int] = (16, 16)):
        super().__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            ResidualBlock(out_channels, out_channels),
            ResidualBlock(out_channels, out_channels)
        )

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.middle_2r_blocks = nn.Sequential(
            ResidualBlock(out_channels, out_channels),
            ResidualBlock(out_channels, out_channels)
        )

        self.interpolation = nn.UpsamplingBilinear2d(size=size)
        self.conv1_1_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

        self.last_blocks = ResidualBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)

        out_mpool1 = self.mpool1(x)
        out_middle = self.middle_2r_blocks(out_mpool1)
        out_interp = self.interpolation(out_middle) + out_trunk

        out_mask = self.conv1_1_blocks(out_interp)
        out = (1 + out_mask) * out_trunk
        out_last = self.last_blocks(out)
        return out_last


class AttentionModuleStage3CIFAR_GELU(nn.Module):
    """8x8 이하 해상도 입력을 위한 CIFAR 전용 Stage-3 Attention (GELU 활성화 함수 사용)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.first_residual_blocks = ResidualBlock(in_channels, out_channels)

        self.trunk_branches = nn.Sequential(
            ResidualBlock(out_channels, out_channels),
            ResidualBlock(out_channels, out_channels)
        )

        self.middle_2r_blocks = nn.Sequential(
            ResidualBlock(out_channels, out_channels),
            ResidualBlock(out_channels, out_channels)
        )

        self.conv1_1_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

        self.last_blocks = ResidualBlock(out_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.first_residual_blocks(x)
        out_trunk = self.trunk_branches(x)
        out_middle = self.middle_2r_blocks(x)

        out_mask = self.conv1_1_blocks(out_middle)
        out = (1 + out_mask) * out_trunk
        out_last = self.last_blocks(out)
        return out_last

