"""
Pre-activation Residual Block 모듈

Pre-activation ResNet 스타일의 Residual Block을 제공합니다.
Pre-activation 구조는 BN -> ReLU -> Conv 순서로 배치하여 그래디언트 흐름을 개선합니다.

참고:
- Pre-activation ResNet 논문: "Identity Mappings in Deep Residual Networks" (He et al., 2016)
- Pre-activation은 깊은 네트워크에서 더 나은 성능을 보임
- 그래디언트 흐름이 더 직접적이어서 학습이 더 안정적
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActivationResidualBlock(nn.Module):
    """
    Pre-activation ResNet 스타일의 Basic Residual Block

    구조:
    - 입력 x
    - Main path: BN -> ReLU -> Conv -> BN -> ReLU -> Conv
    - Shortcut: identity (채널/크기 같음) 또는 Conv (다름)
    - 출력: main_path + shortcut (ReLU 없음)

    Args:
        in_channels: 입력 채널 수
        out_channels: 출력 채널 수
        stride: 첫 번째 conv의 stride (다운샘플링용, 기본값=1)
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(PreActivationResidualBlock, self).__init__()

        # Main path: Pre-activation 구조
        # 첫 번째 BN-ReLU-Conv 블록
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)

        # 두 번째 BN-ReLU-Conv 블록
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)

        # Shortcut connection
        # 채널 수가 다르거나 stride가 1이 아닌 경우 projection 필요
        # Pre-activation ResNet에서는 shortcut은 단순히 Conv만 사용
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                      stride=stride, bias=False)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: 입력 텐서 [batch, in_channels, H, W]

        Returns:
            out: 출력 텐서 [batch, out_channels, H', W']
        """
        # Identity 또는 projection shortcut 저장
        # Pre-activation ResNet에서는 shortcut은 단순히 Conv만 사용
        identity = x
        if isinstance(self.shortcut, nn.Conv2d):
            identity = self.shortcut(x)
        # Identity shortcut: 그대로 사용 (변환 없음)

        # Main path: 첫 번째 Pre-activation (BN -> ReLU -> Conv)
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)

        # Main path: 두 번째 Pre-activation (BN -> ReLU -> Conv)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)

        # Residual connection: F(x) + x (ReLU 없음)
        out += identity

        return out

