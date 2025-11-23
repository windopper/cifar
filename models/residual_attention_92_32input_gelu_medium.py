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
    def __init__(self, init_weights: bool = False):
        super(ResidualAttentionModel_92_32input_GELU_Medium, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )  # 32*32

        self.residual_block1 = ResidualBlock(64, 64)  # 32*32
        self.residual_block1_2 = ResidualBlock(64, 64)
        self.attention_module1 = AttentionModuleStage1CIFAR_GELU(64, 64, size1=(32, 32), size2=(16, 16))

        self.residual_block2 = ResidualBlock(64, 128, stride=2)
        self.residual_block2_2 = ResidualBlock(128, 128)
        self.attention_module2 = AttentionModuleStage2CIFAR_GELU(128, 128, size=(16, 16))

        self.residual_block3 = ResidualBlock(128, 256, stride=2)
        self.residual_block3_2 = ResidualBlock(256, 256)
        self.attention_module3 = AttentionModuleStage3CIFAR_GELU(256, 256)

        self.residual_block4 = ResidualBlock(256, 512)
        self.residual_block5 = ResidualBlock(512, 512)
        self.residual_block6 = ResidualBlock(512, 512)

        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.GELU(),
            nn.AvgPool2d(kernel_size=8)
        )

        self.fc = nn.Linear(512, 10)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
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
        out = self.conv1(x)

        out = self.residual_block1(out)
        out = self.residual_block1_2(out)
        out = self.attention_module1(out)

        out = self.residual_block2(out)
        out = self.residual_block2_2(out)
        out = self.attention_module2(out)

        out = self.residual_block3(out)
        out = self.residual_block3_2(out)
        out = self.attention_module3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)

        out = self.mpool2(out)

        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def make_residual_attention_92_32input_gelu_medium(init_weights: bool = False) -> ResidualAttentionModel_92_32input_GELU_Medium:
    return ResidualAttentionModel_92_32input_GELU_Medium(init_weights=init_weights)


__all__ = [
    "ResidualAttentionModel_92_32input_GELU_Medium",
    "make_residual_attention_92_32input_gelu_medium",
]

