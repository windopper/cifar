"""
DeepBaselineNetBN3Residual15ConvNeXtLNClassifierStem: DeepBaselineNetBN3Residual15ConvNeXtLNClassifier에 Stem 클래스를 추가한 버전

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

6. Stem 추가
   - Patch embedding을 위한 Stem 클래스 추가
   - Conv2d + LayerNormChannels로 구성된 패치 임베딩

네트워크 구조:
- Stem: 패치 임베딩 (3 -> 64, patch_size로 다운샘플링)
- Stage 1: 3개의 ConvNeXt block (64 -> 64, identity shortcut)
- Stage 2: 3개의 ConvNeXt block (64 -> 128, 첫 번째는 projection shortcut + stride=2)
- Stage 3: 9개의 ConvNeXt block (128 -> 256, 첫 번째는 projection shortcut + stride=2)
- Stage 4: 3개의 ConvNeXt block (256 -> 512, 첫 번째는 projection shortcut + stride=2)
- Classifier: AdaptiveAvgPool2d(1) -> Flatten -> LayerNorm -> Linear

기존 DeepBaselineNetBN3Residual15ConvNeXtLNClassifier와의 차이점:
- Stem 클래스 추가 (패치 임베딩용)

참고:
- ConvNeXt 논문: "A ConvNet for the 2020s" (Liu et al., 2022)
- ConvNeXt는 Transformer의 디자인 원칙을 ConvNet에 적용한 모델
- 각 stage의 첫 번째 block에서 stride=2로 다운샘플링하여 공간 크기 감소
- Stem은 입력 이미지를 패치로 분할하고 임베딩하는 초기 단계
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.convnext_block import ConvNeXtBlock, StridedConvNeXtBlock, LayerNormChannels


class Stem(nn.Sequential):
    """
    패치 임베딩을 위한 Stem 클래스
    
    ConvNeXt 스타일의 패치 임베딩으로, 입력 이미지를 패치 크기만큼 다운샘플링하고
    채널 수를 변경한 후 LayerNorm을 적용합니다.
    
    Args:
        in_channels: 입력 채널 수
        out_channels: 출력 채널 수
        patch_size: 패치 크기 (커널 크기 및 stride로 사용)
    """
    def __init__(self, in_channels, out_channels, patch_size):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, patch_size, stride=patch_size),
            LayerNormChannels(out_channels)
        )


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


class DeepBaselineNetBN3Residual15ConvNeXtLNClassifierStem(nn.Module):
    """
    DeepBaselineNetBN3Residual15ConvNeXtLNClassifier에 Stem을 추가한 네트워크 (총 15개 ConvNeXt block)
    
    구조:
    1. Stem: 패치 임베딩 (3 -> 64, patch_size로 다운샘플링)
    2. Stage 1: 3개의 ConvNeXt Block (64 -> 64, identity shortcut)
    3. Stage 2: 3개의 ConvNeXt Block (64 -> 128, 첫 번째는 projection shortcut + stride=2)
    4. Stage 3: 9개의 ConvNeXt Block (128 -> 256, 첫 번째는 projection shortcut + stride=2)
    5. Stage 4: 3개의 ConvNeXt Block (256 -> 512, 첫 번째는 projection shortcut + stride=2)
    6. Classifier: AdaptiveAvgPool2d(1) -> Flatten -> LayerNorm -> Linear
    """
    
    def __init__(self, init_weights=False, num_classes=10, patch_size=1):
        super(DeepBaselineNetBN3Residual15ConvNeXtLNClassifierStem, self).__init__()
        
        # Stem: 패치 임베딩
        # 입력: 3채널 (RGB), 출력: 64채널
        # patch_size=4인 경우: 32x32 -> 8x8
        self.stem = Stem(3, 64, patch_size)
        
        # ConvNeXt Blocks - 각 stage를 3, 3, 9, 3으로 구성 (총 15개)
        # Depthwise convolution 커널 크기: 7x7
        kernel_size = 7
        
        # Stem 이후 공간 크기 계산
        # patch_size=4인 경우: 32x32 -> 8x8
        spatial_size_after_stem = 32 // patch_size
        
        self.stage1 = _make_layer(64, 64, num_blocks=3, stride=1, kernel_size=kernel_size)
        # spatial_size_after_stem x spatial_size_after_stem -> spatial_size_after_stem x spatial_size_after_stem
        self.stage2 = _make_layer(64, 128, num_blocks=3, stride=2, kernel_size=kernel_size)
        # spatial_size_after_stem x spatial_size_after_stem -> (spatial_size_after_stem//2) x (spatial_size_after_stem//2)
        self.stage3 = _make_layer(128, 256, num_blocks=9, stride=2, kernel_size=kernel_size)
        # (spatial_size_after_stem//2) x (spatial_size_after_stem//2) -> (spatial_size_after_stem//4) x (spatial_size_after_stem//4)
        self.stage4 = _make_layer(256, 512, num_blocks=3, stride=2, kernel_size=kernel_size)
        # (spatial_size_after_stem//4) x (spatial_size_after_stem//4) -> (spatial_size_after_stem//8) x (spatial_size_after_stem//8)
        
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
        # Stem: 패치 임베딩
        x = self.stem(x)
        
        # Stage 1: 3개의 ConvNeXt block (64 -> 64)
        x = self.stage1(x)
        
        # Stage 2: 3개의 ConvNeXt block (64 -> 128)
        # stride=2로 다운샘플링
        x = self.stage2(x)
        
        # Stage 3: 9개의 ConvNeXt block (128 -> 256)
        # stride=2로 다운샘플링
        x = self.stage3(x)
        
        # Stage 4: 3개의 ConvNeXt block (256 -> 512)
        # stride=2로 다운샘플링
        x = self.stage4(x)
        
        # Classifier: AdaptiveAvgPool2d(1) -> Flatten -> LayerNorm -> Linear
        # [batch, 512, H, W] -> [batch, 512, 1, 1] -> [batch, 512] -> [batch, num_classes]
        x = self.classifier(x)
        
        return x

