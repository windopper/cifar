"""
Residual Block 모듈 (GELU 활성화 함수 사용)

ResNet 스타일의 Basic Residual Block과 레이어 생성 헬퍼 함수를 제공합니다.
여러 모델에서 공통으로 사용되는 기본 residual block 구현입니다.
GELU(Gaussian Error Linear Unit) 활성화 함수를 사용합니다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    ResNet 스타일의 Basic Residual Block (GELU 활성화 함수 사용)

    구조:
    - 입력 x
    - Main path: Conv -> BN -> GELU -> Conv -> BN
    - Shortcut: identity (채널/크기 같음) 또는 projection (다름)
    - 출력: GELU(main_path + shortcut)

    Args:
        in_channels: 입력 채널 수
        out_channels: 출력 채널 수
        stride: 첫 번째 conv의 stride (다운샘플링용, 기본값=1)
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Main path: 두 개의 Conv-BN 블록
        # 첫 번째 Conv-BN-GELU 블록
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 두 번째 Conv-BN 블록 (GELU는 residual connection 후에 적용)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        # 채널 수가 다르거나 stride가 1이 아닌 경우 projection 필요
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # 1x1 conv를 사용하여 채널 수와 공간 크기 조정
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: 입력 텐서 [batch, in_channels, H, W]

        Returns:
            out: 출력 텐서 [batch, out_channels, H', W']
        """
        # Identity 또는 projection shortcut 저장
        identity = self.shortcut(x)

        # Main path: 첫 번째 Conv-BN-GELU
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.gelu(out)

        # Main path: 두 번째 Conv-BN
        out = self.conv2(out)
        out = self.bn2(out)

        # Residual connection: F(x) + x
        out += identity

        # 최종 GELU 활성화
        out = F.gelu(out)

        return out


def _make_layer(in_channels, out_channels, num_blocks, stride=1):
    """
    Residual block들을 하나의 layer로 구성하는 헬퍼 함수

    Args:
        in_channels: 입력 채널 수
        out_channels: 출력 채널 수
        num_blocks: 이 layer에 포함될 residual block의 개수
        stride: 첫 번째 block의 stride (기본값=1)

    Returns:
        layers: Sequential 모듈로 구성된 residual block들
    """
    layers = []
    # 첫 번째 block: stride를 사용하여 다운샘플링 (채널 변경 또는 공간 크기 감소)
    layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
    # 나머지 block들: stride=1로 유지 (같은 채널, 같은 공간 크기)
    for _ in range(1, num_blocks):
        layers.append(ResidualBlock(out_channels, out_channels, stride=1))
    return nn.Sequential(*layers)

