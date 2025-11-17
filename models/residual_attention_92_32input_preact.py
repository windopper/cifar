"""
ResidualAttentionModel_92_32input_PreAct
=========================================

Residual Attention Network의 92-layer 변형 모델로, 32x32 입력 크기(CIFAR-10)에 최적화된 구조입니다.
Pre-activation Residual Block을 사용하여 그래디언트 흐름을 개선한 버전입니다.

네트워크 구조:
- conv1: 초기 feature extraction (3 -> 32, 32x32 유지)
- preact_residual_block1: 32 -> 128 (stride=1, 32x32 유지)
- attention_module1: Stage1 Attention (128 -> 128, 32x32)
- preact_residual_block2: 128 -> 256 (stride=2, 16x16)
- attention_module2: Stage2 Attention (256 -> 256, 16x16) - 2개
- preact_residual_block3: 256 -> 512 (stride=2, 8x8)
- attention_module3: Stage3 Attention (512 -> 512, 8x8) - 3개
- preact_residual_block4: 512 -> 1024 (stride=1, 8x8 유지)
- preact_residual_block5: 1024 -> 1024 (stride=1, 8x8 유지)
- preact_residual_block6: 1024 -> 1024 (stride=1, 8x8 유지)
- mpool2: AvgPool2d(8) -> 1x1
- fc: 1024 -> 10

Pre-activation 구조의 장점:
- BN -> ReLU -> Conv 순서로 배치하여 그래디언트 흐름 개선
- 깊은 네트워크에서 더 안정적인 학습
- Identity mapping을 더 쉽게 학습

참고 자료:
- Residual Attention Network 논문
- Pre-activation ResNet 논문: "Identity Mappings in Deep Residual Networks" (He et al., 2016)
- https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import (
    PreActivationResidualBlock,
    AttentionModuleStage1CIFAR_PreAct,
    AttentionModuleStage2CIFAR_PreAct,
    AttentionModuleStage3CIFAR_PreAct,
)


class ResidualAttentionModel_92_32input_PreAct(nn.Module):
    """
    Residual Attention Network의 92-layer 변형 모델 (32x32 입력용)
    Pre-activation Residual Block을 사용한 버전
    
    CIFAR-10 해상도(32x32)에 맞춰 설계된 Residual Attention Network 구조입니다.
    각 stage에서 attention module을 여러 개 배치하여 더 깊은 attention 구조를 구현했습니다.
    Pre-activation 구조를 사용하여 그래디언트 흐름을 개선했습니다.
    """

    def __init__(self, init_weights: bool = False):
        super(ResidualAttentionModel_92_32input_PreAct, self).__init__()

        # 초기 feature extraction: 3 -> 32, 32x32 유지
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )  # 32*32

        # Stage 1: 32 -> 128, 32x32 유지
        self.residual_block1 = PreActivationResidualBlock(32, 128)  # 32*32
        self.attention_module1 = AttentionModuleStage1CIFAR_PreAct(
            128, 128, size1=(32, 32), size2=(16, 16)
        )  # 32*32

        # Stage 2: 128 -> 256, 16x16
        self.residual_block2 = PreActivationResidualBlock(128, 256, stride=2)  # 16*16
        self.attention_module2 = AttentionModuleStage2CIFAR_PreAct(
            256, 256, size=(16, 16)
        )  # 16*16
        self.attention_module2_2 = AttentionModuleStage2CIFAR_PreAct(
            256, 256, size=(16, 16)
        )  # 16*16

        # Stage 3: 256 -> 512, 8x8
        self.residual_block3 = PreActivationResidualBlock(256, 512, stride=2)  # 8*8
        self.attention_module3 = AttentionModuleStage3CIFAR_PreAct(512, 512)  # 8*8
        self.attention_module3_2 = AttentionModuleStage3CIFAR_PreAct(512, 512)  # 8*8
        self.attention_module3_3 = AttentionModuleStage3CIFAR_PreAct(512, 512)  # 8*8

        # Stage 4: 512 -> 1024, 8x8 유지
        self.residual_block4 = PreActivationResidualBlock(512, 1024)  # 8*8
        self.residual_block5 = PreActivationResidualBlock(1024, 1024)  # 8*8
        self.residual_block6 = PreActivationResidualBlock(1024, 1024)  # 8*8

        # Global Average Pooling 및 분류기
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8)
        )

        self.fc = nn.Linear(1024, 10)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        """가중치 초기화 - Kaiming initialization 사용"""
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
        out = self.conv1(x)  # [batch, 32, 32, 32]

        # Stage 1: 32x32
        out = self.residual_block1(out)  # [batch, 128, 32, 32]
        out = self.attention_module1(out)  # [batch, 128, 32, 32]

        # Stage 2: 16x16
        out = self.residual_block2(out)  # [batch, 256, 16, 16]
        out = self.attention_module2(out)  # [batch, 256, 16, 16]
        out = self.attention_module2_2(out)  # [batch, 256, 16, 16]

        # Stage 3: 8x8
        out = self.residual_block3(out)  # [batch, 512, 8, 8]
        out = self.attention_module3(out)  # [batch, 512, 8, 8]
        out = self.attention_module3_2(out)  # [batch, 512, 8, 8]
        out = self.attention_module3_3(out)  # [batch, 512, 8, 8]

        # Stage 4: 8x8 유지
        out = self.residual_block4(out)  # [batch, 1024, 8, 8]
        out = self.residual_block5(out)  # [batch, 1024, 8, 8]
        out = self.residual_block6(out)  # [batch, 1024, 8, 8]

        # Global Average Pooling
        out = self.mpool2(out)  # [batch, 1024, 1, 1]

        # Flatten 및 분류
        out = out.view(out.size(0), -1)  # [batch, 1024]
        out = self.fc(out)  # [batch, 10]

        return out


class ResidualAttentionModel_92_32input_PreAct_Tiny(nn.Module):
    """
    Residual Attention Network의 경량 버전 (~10M 파라미터)
    Pre-activation Residual Block을 사용한 버전
    
    채널 수를 조정하여 파라미터를 10M 수준으로 맞춘 버전입니다.
    
    네트워크 구조:
    - conv1: 초기 feature extraction (3 -> 32, 32x32 유지)
    - preact_residual_block1: 32 -> 64 (stride=1, 32x32 유지)
    - attention_module1: Stage1 Attention (64 -> 64, 32x32)
    - preact_residual_block2: 64 -> 128 (stride=2, 16x16)
    - attention_module2: Stage2 Attention (128 -> 128, 16x16) - 1개
    - preact_residual_block3: 128 -> 256 (stride=2, 8x8)
    - attention_module3: Stage3 Attention (256 -> 256, 8x8) - 1개
    - preact_residual_block4: 256 -> 512 (stride=1, 8x8 유지)
    - mpool2: AvgPool2d(8) -> 1x1
    - fc: 512 -> 10
    """

    def __init__(self, init_weights: bool = False):
        super(ResidualAttentionModel_92_32input_PreAct_Tiny, self).__init__()

        # 초기 feature extraction: 3 -> 32, 32x32 유지
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )  # 32*32

        # Stage 1: 32 -> 64, 32x32 유지
        self.residual_block1 = PreActivationResidualBlock(32, 64)  # 32*32
        self.attention_module1 = AttentionModuleStage1CIFAR_PreAct(
            64, 64, size1=(32, 32), size2=(16, 16)
        )  # 32*32

        # Stage 2: 64 -> 128, 16x16
        self.residual_block2 = PreActivationResidualBlock(64, 128, stride=2)  # 16*16
        self.attention_module2 = AttentionModuleStage2CIFAR_PreAct(
            128, 128, size=(16, 16)
        )  # 16*16

        # Stage 3: 128 -> 256, 8x8
        self.residual_block3 = PreActivationResidualBlock(128, 256, stride=2)  # 8*8
        self.attention_module3 = AttentionModuleStage3CIFAR_PreAct(256, 256)  # 8*8

        # Stage 4: 256 -> 512, 8x8 유지
        self.residual_block4 = PreActivationResidualBlock(256, 512)  # 8*8

        # Global Average Pooling 및 분류기
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8)
        )

        self.fc = nn.Linear(512, 10)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        """가중치 초기화 - Kaiming initialization 사용"""
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
        out = self.conv1(x)  # [batch, 32, 32, 32]

        # Stage 1: 32x32
        out = self.residual_block1(out)  # [batch, 64, 32, 32]
        out = self.attention_module1(out)  # [batch, 64, 32, 32]

        # Stage 2: 16x16
        out = self.residual_block2(out)  # [batch, 128, 16, 16]
        out = self.attention_module2(out)  # [batch, 128, 16, 16]

        # Stage 3: 8x8
        out = self.residual_block3(out)  # [batch, 256, 8, 8]
        out = self.attention_module3(out)  # [batch, 256, 8, 8]

        # Stage 4: 8x8 유지
        out = self.residual_block4(out)  # [batch, 512, 8, 8]

        # Global Average Pooling
        out = self.mpool2(out)  # [batch, 512, 1, 1]

        # Flatten 및 분류
        out = out.view(out.size(0), -1)  # [batch, 512]
        out = self.fc(out)  # [batch, 10]

        return out


def make_residual_attention_92_32input_preact_tiny(init_weights: bool = False) -> ResidualAttentionModel_92_32input_PreAct_Tiny:
    """
    ~10M 파라미터 수준의 경량 Residual Attention 모델을 생성 (Pre-activation 버전).
    """
    return ResidualAttentionModel_92_32input_PreAct_Tiny(init_weights=init_weights)


__all__ = [
    "ResidualAttentionModel_92_32input_PreAct",
    "ResidualAttentionModel_92_32input_PreAct_Tiny",
    "make_residual_attention_92_32input_preact_tiny",
]

