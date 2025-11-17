"""
DeepBaselineNetBN3Residual15Attention
====================================

DeepBaselineNetBN3Residual15에 Residual Attention 모듈을 삽입한 변형 모델입니다.
각 stage의 residual block 구성은 그대로 유지하면서, CIFAR-10 해상도(32x32)에
맞춰 설계된 AttentionModule_stage{1,2,3}_cifar 구조를 적용했습니다.
자세한 어텐션 모듈 설계는 Residual Attention Network 논문과 공식 구현을
참고했습니다.

추가로, 전체 채널 폭(stem + stage 채널)과 block 수를 하이퍼파라미터로 노출하여
경량 버전을 쉽게 구성할 수 있으며, 10M 내외 파라미터의 Tiny 버전을 위한
빌더 함수도 제공합니다.

참고 자료:
- https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch/blob/master/Residual-Attention-Network/model/attention_module.py
- https://github.com/tengshaofeng/ResidualAttentionNetwork-pytorch/blob/master/Residual-Attention-Network/model/residual_attention_network.py
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .deep_baseline3_bn_residual_15 import ResidualBlock, _make_layer


class AttentionModuleStage1CIFAR(nn.Module):
    """32x32 해상도(또는 사용자 지정 size1) 입력을 위한 CIFAR 전용 Stage-1 Attention."""

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
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
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


class AttentionModuleStage2CIFAR(nn.Module):
    """16x16 해상도 입력을 위한 CIFAR 전용 Stage-2 Attention."""

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
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
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


class AttentionModuleStage3CIFAR(nn.Module):
    """8x8 이하 해상도 입력을 위한 CIFAR 전용 Stage-3 Attention."""

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
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
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


class DeepBaselineNetBN3Residual15Attention(nn.Module):
    """
    Residual Attention 모듈을 삽입한 DeepBaselineNetBN3Residual15 변형 버전.

    - Stage 구성(Residual block 수, 채널, stride)은 원본 모델과 동일하게 유지.
    - Stage1~4 이후에 AttentionModule_stage{1,2,3}_cifar를 배치하여
      잔차 블록과 어텐션 마스크를 곱하는 구조를 적용.
    """

    def __init__(self,
                 init_weights: bool = False,
                 stem_channels: int = 64,
                 stage_channels: Tuple[int, int, int, int] = (64, 128, 256, 512),
                 stage_blocks: Tuple[int, int, int, int] = (2, 2, 4, 2)):
        super().__init__()

        if len(stage_channels) != 4 or len(stage_blocks) != 4:
            raise ValueError("stage_channels와 stage_blocks는 길이 4의 튜플이어야 합니다.")

        c1, c2, c3, c4 = stage_channels
        b1, b2, b3, b4 = stage_blocks

        self.conv1 = nn.Conv2d(3, stem_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(stem_channels)

        # residual block 구성은 기존 DeepBaselineNetBN3Residual15 그대로 유지
        self.stage1 = _make_layer(stem_channels, c1, num_blocks=b1, stride=1)
        self.stage2 = _make_layer(c1, c2, num_blocks=b2, stride=2)
        self.stage3 = _make_layer(c2, c3, num_blocks=b3, stride=2)
        self.stage4 = _make_layer(c3, c4, num_blocks=b4, stride=2)

        self.attention1 = AttentionModuleStage1CIFAR(c1, c1, size1=(32, 32), size2=(16, 16))
        self.attention2 = AttentionModuleStage2CIFAR(c2, c2, size=(16, 16))
        self.attention3 = AttentionModuleStage3CIFAR(c3, c3)
        self.attention4 = AttentionModuleStage3CIFAR(c4, c4)

        self.classifier = nn.Linear(c4, 10)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        """기존 Residual15 모델과 동일한 Kaiming 초기화."""
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.stage1(x)
        x = self.attention1(x)

        x = self.stage2(x)
        x = self.attention2(x)

        x = self.stage3(x)
        x = self.attention3(x)

        x = self.stage4(x)
        x = self.attention4(x)

        x = F.avg_pool2d(x, kernel_size=4)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def make_deep_baseline3_bn_residual_15_attention_tiny(init_weights: bool = False) -> DeepBaselineNetBN3Residual15Attention:
    """
    채널 폭을 절반 수준으로 줄여 (~10M 파라미터) 경량 Residual Attention 모델을 생성.
    """
    return DeepBaselineNetBN3Residual15Attention(
        init_weights=init_weights,
        stem_channels=32,
        stage_channels=(32, 64, 128, 256),
        stage_blocks=(2, 2, 4, 2),
    )


__all__ = [
    "DeepBaselineNetBN3Residual15Attention",
    "make_deep_baseline3_bn_residual_15_attention_tiny",
]

