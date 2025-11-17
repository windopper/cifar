"""
DeepBaselineNetBN3Residual15LN: DeepBaselineNetBN3Residual15의 Batch Normalization을 Layer Normalization으로 교체한 버전

설계 의도:
1. Residual Learning (잔차 학습)
   - Skip connection을 통해 그래디언트가 직접 전달되어 그래디언트 소실 문제 완화
   - 깊은 네트워크에서도 효과적인 학습 가능
   - 각 레이어가 잔차(residual)를 학습하도록 하여 학습 난이도 감소

2. 그래디언트 흐름 개선
   - Identity mapping을 통한 shortcut connection으로 그래디언트가 역전파 시
     직접 전달되어 학습 안정성 향상
   - 깊은 네트워크에서도 효과적인 학습 가능

3. 표현력 향상
   - Residual block은 F(x) + x 형태로, 네트워크가 identity mapping을 쉽게 학습
   - 필요시 더 복잡한 변환을 학습할 수 있어 표현력 향상

4. Layer Normalization 적용
   - Batch Normalization 대신 Layer Normalization 사용
   - 배치 크기에 독립적인 정규화로 작은 배치 크기에서도 안정적인 학습 가능
   - 각 샘플의 채널 차원에 대해 독립적으로 정규화

네트워크 구조:
- conv1: 초기 feature extraction (3 -> 64)
- Stage 1: 3개의 residual block (64 -> 64, identity shortcut)
- Stage 2: 3개의 residual block (64 -> 128, 첫 번째는 projection shortcut + stride=2)
- Stage 3: 6개의 residual block (128 -> 256, 첫 번째는 projection shortcut + stride=2)
- Stage 4: 3개의 residual block (256 -> 512, 첫 번째는 projection shortcut + stride=2)
- FC layers: 분류기

기존 DeepBaselineNetBN3Residual15와의 차이점:
- Batch Normalization을 Layer Normalization으로 교체
- Conv-BN-ReLU 구조가 Conv-LN-ReLU 구조로 변경

참고:
- ResNet 논문: "Deep Residual Learning for Image Recognition" (He et al., 2015)
- Layer Normalization 논문: "Layer Normalization" (Ba et al., 2016)
- Residual block은 깊은 네트워크에서 그래디언트 소실 문제를 해결하는 핵심 기술
- 각 stage의 첫 번째 block에서 stride=2로 다운샘플링하여 공간 크기 감소
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """
    2D Convolution 출력에 적용하기 위한 Layer Normalization 래퍼
    
    Layer Normalization은 마지막 차원에 대해 정규화하므로,
    2D convolution 출력 (N, C, H, W)에 적용하기 위해
    (N, H, W, C) 형태로 변환한 후 정규화하고 다시 (N, C, H, W)로 변환합니다.
    """
    def __init__(self, num_channels, eps=1e-5):
        super(LayerNorm2d, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.ln = nn.LayerNorm(num_channels, eps=eps)
    
    def forward(self, x):
        # x: (N, C, H, W)
        # (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        # Layer Normalization 적용
        x = self.ln(x)
        # (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return x


class ResidualBlock(nn.Module):
    """
    ResNet 스타일의 Basic Residual Block (Layer Normalization 사용)

    구조:
    - 입력 x
    - Main path: Conv -> LN -> ReLU -> Conv -> LN
    - Shortcut: identity (채널/크기 같음) 또는 projection (다름)
    - 출력: ReLU(main_path + shortcut)

    Args:
        in_channels: 입력 채널 수
        out_channels: 출력 채널 수
        stride: 첫 번째 conv의 stride (다운샘플링용, 기본값=1)
    """
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # Main path: 두 개의 Conv-LN 블록
        # 첫 번째 Conv-LN-ReLU 블록
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.ln1 = LayerNorm2d(out_channels)

        # 두 번째 Conv-LN 블록 (ReLU는 residual connection 후에 적용)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.ln2 = LayerNorm2d(out_channels)

        # Shortcut connection
        # 채널 수가 다르거나 stride가 1이 아닌 경우 projection 필요
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # 1x1 conv를 사용하여 채널 수와 공간 크기 조정
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                LayerNorm2d(out_channels)
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

        # Main path: 첫 번째 Conv-LN-ReLU
        out = self.conv1(x)
        out = self.ln1(out)
        out = F.relu(out)

        # Main path: 두 번째 Conv-LN
        out = self.conv2(out)
        out = self.ln2(out)

        # Residual connection: F(x) + x
        out += identity

        # 최종 ReLU 활성화
        out = F.relu(out)

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


class DeepBaselineNetBN3Residual15LN(nn.Module):
    """
    DeepBaselineNetBN3Residual15의 Batch Normalization을 Layer Normalization으로 교체한 네트워크 (총 15개 residual block)

    구조:
    1. 초기 Conv-LN-ReLU (3 -> 64)
    2. Stage 1: 3개의 Residual Block (64 -> 64, identity shortcut)
    3. Stage 2: 3개의 Residual Block (64 -> 128, 첫 번째는 projection shortcut + stride=2)
    4. Stage 3: 6개의 Residual Block (128 -> 256, 첫 번째는 projection shortcut + stride=2)
    5. Stage 4: 3개의 Residual Block (256 -> 512, 첫 번째는 projection shortcut + stride=2)
    6. Global Average Pooling
    7. Fully Connected Layers (분류기)
    """

    def __init__(self, init_weights=False):
        super(DeepBaselineNetBN3Residual15LN, self).__init__()

        # 초기 feature extraction
        # 입력: 3채널 (RGB), 출력: 64채널
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.ln1 = LayerNorm2d(64)

        # Residual Blocks - 각 stage를 3, 3, 6, 3으로 구성 (총 15개)
        # Stage 1: 3개의 block (64 -> 64, identity shortcut)
        self.stage1 = _make_layer(64, 64, num_blocks=2, stride=1)

        # Stage 2: 3개의 block (64 -> 128, 첫 번째는 projection shortcut + stride=2)
        self.stage2 = _make_layer(64, 128, num_blocks=2, stride=2)

        # Stage 3: 6개의 block (128 -> 256, 첫 번째는 projection shortcut + stride=2)
        self.stage3 = _make_layer(128, 256, num_blocks=4, stride=2)

        # Stage 4: 3개의 block (256 -> 512, 첫 번째는 projection shortcut + stride=2)
        self.stage4 = _make_layer(256, 512, num_blocks=2, stride=2)

        self.classifier = nn.Sequential(
            nn.Linear(512, 10),
        )

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        """가중치 초기화 - ReLU를 사용하므로 Kaiming initialization 사용"""
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
            elif isinstance(m, nn.LayerNorm):
                # LayerNorm2d 내부의 nn.LayerNorm도 여기서 처리됨
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: 입력 이미지 [batch, 3, 32, 32] (CIFAR-10)

        Returns:
            out: 분류 로짓 [batch, 10]
        """
        # 초기 Conv-LN-ReLU
        x = self.conv1(x)
        x = self.ln1(x)
        x = F.relu(x)

        # Stage 1: 3개의 residual block (64 -> 64)
        # 32x32 -> 32x32
        x = self.stage1(x)

        # Stage 2: 3개의 residual block (64 -> 128)
        # 32x32 -> 16x16 (stride=2로 다운샘플링)
        x = self.stage2(x)

        # Stage 3: 6개의 residual block (128 -> 256)
        # 16x16 -> 8x8 (stride=2로 다운샘플링)
        x = self.stage3(x)

        # Stage 4: 3개의 residual block (256 -> 512)
        # 8x8 -> 4x4 (stride=2로 다운샘플링)
        x = self.stage4(x)

        # Global Average Pooling: [batch, 512, 4, 4] -> [batch, 512, 1, 1]
        x = F.avg_pool2d(x, kernel_size=4)

        # Flatten: [batch, 512, 1, 1] -> [batch, 512]
        x = torch.flatten(x, 1)

        # Fully Connected Layers
        x = self.classifier(x)

        return x

