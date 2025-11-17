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

from .modules import (
    ResidualBlock,
    _make_layer,
    AttentionModuleStage1CIFAR,
    AttentionModuleStage2CIFAR,
    AttentionModuleStage3CIFAR,
)


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

