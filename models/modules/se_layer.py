"""
Squeeze-and-Excitation Layer 모듈

Squeeze-and-Excitation Network (SE-Net)의 핵심 구성 요소인 SE Layer를 제공합니다.
채널 간 의존성을 모델링하여 중요한 채널에 집중하도록 하는 attention 메커니즘입니다.

참고 자료:
- "Squeeze-and-Excitation Networks" (Hu et al., 2018)
"""

import torch
import torch.nn as nn


class SELayer(nn.Module):
    """
    Squeeze-and-Excitation Layer
    
    구조:
    1. Squeeze: Global Average Pooling으로 공간 차원 압축
    2. Excitation: FC 레이어들로 채널별 가중치 생성
    3. Scale: 원본 feature map에 가중치를 곱하여 재조정
    
    Args:
        channel: 입력 채널 수
        reduction: 채널 축소 비율 (기본값=16)
    """
    
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: 입력 텐서 [batch, channel, H, W]
            
        Returns:
            out: SE 가중치가 적용된 출력 텐서 [batch, channel, H, W]
        """
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

