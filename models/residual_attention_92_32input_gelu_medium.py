"""
ResidualAttentionModel_92_32input_GELU_Medium
==============================================

Residual Attention Network의 중간 크기 모델 (~20M 파라미터, GELU 활성화 함수 사용)
Tiny 모델을 기반으로 residual 층을 늘려서 파라미터 크기를 약 2배로 만든 버전입니다.

네트워크 구조:
- conv1: 초기 feature extraction (3 -> 64, 32x32 유지)
- residual_block1: 64 -> 64 (stride=1, 32x32 유지)
- residual_block1_2: 64 -> 64 (stride=1, 32x32 유지) [추가]
- attention_module1: Stage1 Attention (64 -> 64, 32x32)
- residual_block2: 64 -> 128 (stride=2, 16x16)
- residual_block2_2: 128 -> 128 (stride=1, 16x16 유지) [추가]
- attention_module2: Stage2 Attention (128 -> 128, 16x16) - 1개
- residual_block3: 128 -> 256 (stride=2, 8x8)
- residual_block3_2: 256 -> 256 (stride=1, 8x8 유지) [추가]
- attention_module3: Stage3 Attention (256 -> 256, 8x8) - 1개
- residual_block4: 256 -> 512 (stride=1, 8x8 유지)
- residual_block5: 512 -> 512 (stride=1, 8x8 유지)
- residual_block6: 512 -> 512 (stride=1, 8x8 유지) [추가]
- mpool2: AvgPool2d(8) -> 1x1
- fc: 512 -> 10

참고 자료:
- Residual Attention Network 논문
- https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.residual_block_gelu import ResidualBlock
from .modules import (
    AttentionModuleStage1CIFAR_GELU,
    AttentionModuleStage2CIFAR_GELU,
    AttentionModuleStage3CIFAR_GELU,
)


class ResidualAttentionModel_92_32input_GELU_Medium(nn.Module):
    """
    Residual Attention Network의 중간 크기 모델 (~20M 파라미터, GELU 활성화 함수 사용)
    
    Tiny 모델을 기반으로 residual 층을 늘려서 파라미터 크기를 약 2배로 만든 버전입니다.
    GELU 활성화 함수를 사용합니다.
    
    네트워크 구조:
    - conv1: 초기 feature extraction (3 -> 64, 32x32 유지)
    - residual_block1: 64 -> 64 (stride=1, 32x32 유지)
    - residual_block1_2: 64 -> 64 (stride=1, 32x32 유지) [추가]
    - attention_module1: Stage1 Attention (64 -> 64, 32x32)
    - residual_block2: 64 -> 128 (stride=2, 16x16)
    - residual_block2_2: 128 -> 128 (stride=1, 16x16 유지) [추가]
    - attention_module2: Stage2 Attention (128 -> 128, 16x16) - 1개
    - residual_block3: 128 -> 256 (stride=2, 8x8)
    - residual_block3_2: 256 -> 256 (stride=1, 8x8 유지) [추가]
    - attention_module3: Stage3 Attention (256 -> 256, 8x8) - 1개
    - residual_block4: 256 -> 512 (stride=1, 8x8 유지)
    - residual_block5: 512 -> 512 (stride=1, 8x8 유지)
    - residual_block6: 512 -> 512 (stride=1, 8x8 유지) [추가]
    - mpool2: AvgPool2d(8) -> 1x1
    - fc: 512 -> 10
    """

    def __init__(self, init_weights: bool = False):
        super(ResidualAttentionModel_92_32input_GELU_Medium, self).__init__()

        # 초기 feature extraction: 3 -> 64, 32x32 유지
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )  # 32*32

        # Stage 1: 64 -> 64, 32x32 유지
        self.residual_block1 = ResidualBlock(64, 64)  # 32*32
        self.residual_block1_2 = ResidualBlock(64, 64)  # 32*32 (추가)
        self.attention_module1 = AttentionModuleStage1CIFAR_GELU(
            64, 64, size1=(32, 32), size2=(16, 16)
        )  # 32*32

        # Stage 2: 64 -> 128, 16x16
        self.residual_block2 = ResidualBlock(64, 128, stride=2)  # 16*16
        self.residual_block2_2 = ResidualBlock(128, 128)  # 16*16 (추가)
        self.attention_module2 = AttentionModuleStage2CIFAR_GELU(
            128, 128, size=(16, 16)
        )  # 16*16

        # Stage 3: 128 -> 256, 8x8
        self.residual_block3 = ResidualBlock(128, 256, stride=2)  # 8*8
        self.residual_block3_2 = ResidualBlock(256, 256)  # 8*8 (추가)
        self.attention_module3 = AttentionModuleStage3CIFAR_GELU(256, 256)  # 8*8

        # Stage 4: 256 -> 512, 8x8 유지
        self.residual_block4 = ResidualBlock(256, 512)  # 8*8
        self.residual_block5 = ResidualBlock(512, 512)  # 8*8
        self.residual_block6 = ResidualBlock(512, 512)  # 8*8 (추가)

        # Global Average Pooling 및 분류기
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=8)
        )

        self.fc = nn.Linear(512, 10)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        """가중치 초기화 - Kaiming initialization 사용 (GELU용)"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: 입력 이미지 [batch, 3, 32, 32] (CIFAR-10)

        Returns:
            out: 분류 로짓 [batch, 10]
        """
        # 초기 feature extraction
        out = self.conv1(x)  # [batch, 64, 32, 32]

        # Stage 1: 32x32
        out = self.residual_block1(out)  # [batch, 64, 32, 32]
        out = self.residual_block1_2(out)  # [batch, 64, 32, 32]
        out = self.attention_module1(out)  # [batch, 64, 32, 32]

        # Stage 2: 16x16
        out = self.residual_block2(out)  # [batch, 128, 16, 16]
        out = self.residual_block2_2(out)  # [batch, 128, 16, 16]
        out = self.attention_module2(out)  # [batch, 128, 16, 16]

        # Stage 3: 8x8
        out = self.residual_block3(out)  # [batch, 256, 8, 8]
        out = self.residual_block3_2(out)  # [batch, 256, 8, 8]
        out = self.attention_module3(out)  # [batch, 256, 8, 8]

        # Stage 4: 8x8 유지
        out = self.residual_block4(out)  # [batch, 512, 8, 8]
        out = self.residual_block5(out)  # [batch, 512, 8, 8]
        out = self.residual_block6(out)  # [batch, 512, 8, 8]

        # Global Average Pooling
        out = self.mpool2(out)  # [batch, 512, 1, 1]

        # Flatten 및 분류
        out = out.view(out.size(0), -1)  # [batch, 512]
        out = self.fc(out)  # [batch, 10]

        return out


def make_residual_attention_92_32input_gelu_medium(init_weights: bool = False) -> ResidualAttentionModel_92_32input_GELU_Medium:
    """
    ~20M 파라미터 수준의 중간 크기 Residual Attention 모델을 생성 (GELU 활성화 함수 사용).
    Tiny 모델을 기반으로 residual 층을 늘려서 파라미터 크기를 약 2배로 만든 버전입니다.
    """
    return ResidualAttentionModel_92_32input_GELU_Medium(init_weights=init_weights)


__all__ = [
    "ResidualAttentionModel_92_32input_GELU_Medium",
    "make_residual_attention_92_32input_gelu_medium",
]

