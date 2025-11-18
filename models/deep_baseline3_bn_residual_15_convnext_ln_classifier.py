"""
DeepBaselineNetBN3Residual15ConvNeXtLNClassifier: DeepBaselineNetBN3Residual15ConvNeXt의 classifier를 LayerNorm이 포함된 버전으로 변경

설계 의도:
1. ConvNeXt 블록 사용
   - Depthwise Convolution (3x3) -> LayerNorm -> Pointwise Conv (expansion) -> GELU -> Pointwise Conv (projection)
   - Residual connection을 통한 그래디언트 흐름 개선
   - Layer Normalization 사용으로 배치 크기에 덜 민감한 학습

2. 그래디언트 흐름 개선
   - Skip connection을 통해 그래디언트가 직접 전달되어 그래디언트 소실 문제 완화
   - 깊은 네트워크에서도 효과적인 학습 가능

3. 표현력 향상
   - ConvNeXt 블록은 F(x) + x 형태로, 네트워크가 identity mapping을 쉽게 학습
   - 필요시 더 복잡한 변환을 학습할 수 있어 표현력 향상

4. ConvNeXt 구조 적용
   - Depthwise convolution으로 공간 특징 추출
   - Layer Normalization으로 정규화
   - GELU 활성화 함수 사용
   - MLP 확장/축소로 채널 간 상호작용 모델링

5. Classifier 개선
   - AdaptiveAvgPool2d(1)로 공간 차원을 1x1로 축소
   - Flatten으로 1D 벡터로 변환
   - LayerNorm으로 특징 벡터 정규화
   - Linear로 최종 분류

네트워크 구조:
- conv1: 초기 feature extraction (3 -> 64)
- Stage 1: 3개의 ConvNeXt block (64 -> 64, identity shortcut)
- Stage 2: 3개의 ConvNeXt block (64 -> 128, 첫 번째는 projection shortcut + stride=2)
- Stage 3: 9개의 ConvNeXt block (128 -> 256, 첫 번째는 projection shortcut + stride=2)
- Stage 4: 3개의 ConvNeXt block (256 -> 512, 첫 번째는 projection shortcut + stride=2)
- Classifier: AdaptiveAvgPool2d(1) -> Flatten -> LayerNorm -> Linear

기존 DeepBaselineNetBN3Residual15ConvNeXt와의 차이점:
- Classifier에 LayerNorm 추가
- AdaptiveAvgPool2d(1) 사용 (고정 크기 avg_pool2d 대신)
- Flatten을 classifier 내부로 이동

참고:
- ConvNeXt 논문: "A ConvNet for the 2020s" (Liu et al., 2022)
- ConvNeXt는 Transformer의 디자인 원칙을 ConvNet에 적용한 모델
- 각 stage의 첫 번째 block에서 stride=2로 다운샘플링하여 공간 크기 감소
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.convnext_block import ConvNeXtBlock, StridedConvNeXtBlock


def _make_layer(in_channels, out_channels, num_blocks, stride=1, kernel_size=3):
    """
    ConvNeXt block들을 하나의 layer로 구성하는 헬퍼 함수
    
    Args:
        in_channels: 입력 채널 수
        out_channels: 출력 채널 수
        num_blocks: 이 layer에 포함될 ConvNeXt block의 개수
        stride: 첫 번째 block의 stride (기본값=1)
        kernel_size: Depthwise convolution의 커널 크기 (기본값=3)
    
    Returns:
        layers: Sequential 모듈로 구성된 ConvNeXt block들
    """
    layers = []
    # 첫 번째 block: stride를 사용하여 다운샘플링 (채널 변경 또는 공간 크기 감소)
    layers.append(StridedConvNeXtBlock(
        in_channels, out_channels, stride=stride, kernel_size=kernel_size
    ))
    # 나머지 block들: stride=1로 유지 (같은 채널, 같은 공간 크기)
    for _ in range(1, num_blocks):
        layers.append(ConvNeXtBlock(
            out_channels, kernel_size=kernel_size
        ))
    return nn.Sequential(*layers)


class DeepBaselineNetBN3Residual15ConvNeXtLNClassifier(nn.Module):
    """
    DeepBaselineNetBN3Residual15ConvNeXt의 classifier에 LayerNorm을 추가한 네트워크 (총 15개 ConvNeXt block)
    
    구조:
    1. 초기 Conv-BN-ReLU (3 -> 64)
    2. Stage 1: 3개의 ConvNeXt Block (64 -> 64, identity shortcut)
    3. Stage 2: 3개의 ConvNeXt Block (64 -> 128, 첫 번째는 projection shortcut + stride=2)
    4. Stage 3: 9개의 ConvNeXt Block (128 -> 256, 첫 번째는 projection shortcut + stride=2)
    5. Stage 4: 3개의 ConvNeXt Block (256 -> 512, 첫 번째는 projection shortcut + stride=2)
    6. Classifier: AdaptiveAvgPool2d(1) -> Flatten -> LayerNorm -> Linear
    """
    
    def __init__(self, init_weights=False, num_classes=10):
        super(DeepBaselineNetBN3Residual15ConvNeXtLNClassifier, self).__init__()
        
        # 초기 feature extraction
        # 입력: 3채널 (RGB), 출력: 64채널
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        # 32x32 -> 32x32
        self.bn1 = nn.BatchNorm2d(64)
        
        # ConvNeXt Blocks - 각 stage를 3, 3, 9, 3으로 구성 (총 15개)
        # Depthwise convolution 커널 크기: 3x3
        kernel_size = 7
        
        self.stage1 = _make_layer(64, 64, num_blocks=3, stride=1, kernel_size=kernel_size)
        # 32x32 -> 32x32
        self.stage2 = _make_layer(64, 128, num_blocks=3, stride=2, kernel_size=kernel_size)
        # 32x32 -> 16x16
        self.stage3 = _make_layer(128, 256, num_blocks=9, stride=2, kernel_size=kernel_size)
        # 16x16 -> 8x8
        self.stage4 = _make_layer(256, 512, num_blocks=3, stride=2, kernel_size=kernel_size)
        # 8x8 -> 4x4
        
        # Classifier: AdaptiveAvgPool2d(1) -> Flatten -> LayerNorm -> Linear
        in_channels = 512
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, num_classes),
        )
        
        if init_weights:
            self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: 입력 이미지 [batch, 3, 32, 32] (CIFAR-10)
        
        Returns:
            out: 분류 로짓 [batch, num_classes]
        """
        # 초기 Conv-BN-ReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Stage 1: 3개의 ConvNeXt block (64 -> 64)
        # 32x32 -> 32x32
        x = self.stage1(x)
        
        # Stage 2: 3개의 ConvNeXt block (64 -> 128)
        # 32x32 -> 16x16 (stride=2로 다운샘플링)
        x = self.stage2(x)
        
        # Stage 3: 9개의 ConvNeXt block (128 -> 256)
        # 16x16 -> 8x8 (stride=2로 다운샘플링)
        x = self.stage3(x)
        
        # Stage 4: 3개의 ConvNeXt block (256 -> 512)
        # 8x8 -> 4x4 (stride=2로 다운샘플링)
        x = self.stage4(x)
        
        # Classifier: AdaptiveAvgPool2d(1) -> Flatten -> LayerNorm -> Linear
        # [batch, 512, 4, 4] -> [batch, 512, 1, 1] -> [batch, 512] -> [batch, num_classes]
        x = self.classifier(x)
        
        return x

