"""
DeepBaselineNetBN3ResidualBottleneck: Bottleneck 구조를 사용한 더 깊은 Residual Network

설계 의도:
1. Bottleneck 구조 (1x1 -> 3x3 -> 1x1)
   - ResNet-50/101/152에서 사용하는 Bottleneck 구조 적용
   - 1x1 conv로 채널을 확장한 후 3x3 conv로 feature extraction
   - 마지막 1x1 conv로 채널을 다시 축소하여 파라미터 효율성 향상
   - 같은 표현력을 유지하면서 파라미터 수 감소

2. 더 깊은 네트워크
   - 각 채널 레벨에서 더 많은 residual block 추가
   - 깊이를 늘려 표현력 향상
   - Bottleneck 구조로 인해 파라미터 증가를 최소화

3. 그래디언트 흐름 개선
   - Skip connection을 통한 직접적인 그래디언트 전달
   - 깊은 네트워크에서도 안정적인 학습 가능

기존 DeepBaselineNetBN3Residual와의 차이점:
1. Bottleneck Block 구조 도입
   - BasicBlock (3x3 -> 3x3) 대신 BottleneckBlock (1x1 -> 3x3 -> 1x1) 사용
   - expansion=4를 사용하여 중간 채널을 확장
   - 파라미터 효율성 향상

2. 네트워크 깊이 증가
   - 각 채널 레벨에서 더 많은 block 추가
   - 64 -> 128 -> 256 -> 512 채널 레벨에서 각각 더 많은 block 사용

3. 구조 변경
   - conv1: 초기 feature extraction (3 -> 64)
   - Layer 1: 64 채널에서 여러 bottleneck blocks
   - Layer 2: 128 채널에서 여러 bottleneck blocks
   - Layer 3: 256 채널에서 여러 bottleneck blocks
   - Layer 4: 512 채널에서 여러 bottleneck blocks
   - FC layers: 분류기

참고:
- ResNet 논문: "Deep Residual Learning for Image Recognition" (He et al., 2015)
- Bottleneck 구조는 ResNet-50 이상에서 사용되는 구조
- expansion=2를 사용하여 중간 채널을 확장 (예: 64 -> 128 -> 128) - 파라미터 최적화
- 파라미터 수를 약 10M 수준으로 제한하기 위해 구조 최적화
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckBlock(nn.Module):
    """
    ResNet 스타일의 Bottleneck Residual Block

    구조:
    - 입력 x
    - Main path: 1x1 Conv -> BN -> ReLU -> 3x3 Conv -> BN -> ReLU -> 1x1 Conv -> BN
    - Shortcut: identity (채널/크기 같음) 또는 projection (다름)
    - 출력: ReLU(main_path + shortcut)

    Args:
        in_channels: 입력 채널 수
        out_channels: 출력 채널 수 (실제로는 out_channels = in_channels * expansion)
        stride: 두 번째 conv의 stride (다운샘플링용, 기본값=1)
        expansion: 채널 확장 비율 (기본값=4)
    """
    expansion = 2  # 4에서 2로 줄여 파라미터 감소

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckBlock, self).__init__()

        # expansion을 사용하여 중간 채널 수 계산
        # out_channels는 실제 출력 채널 수 (예: 64)
        # mid_channels는 중간 확장된 채널 수 (예: 64 * 2 = 128)
        mid_channels = out_channels * self.expansion

        # Main path: 1x1 -> 3x3 -> 1x1 구조
        # 첫 번째 1x1 Conv-BN-ReLU (채널 확장)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1,
                               stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        # 두 번째 3x3 Conv-BN-ReLU (주요 feature extraction)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        # 세 번째 1x1 Conv-BN (채널 축소)
        self.conv3 = nn.Conv2d(mid_channels, out_channels * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        # Shortcut connection
        # 채널 수가 다르거나 stride가 1이 아닌 경우 projection 필요
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            # 1x1 conv를 사용하여 채널 수와 공간 크기 조정
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        """
        Forward pass

        Args:
            x: 입력 텐서 [batch, in_channels, H, W]

        Returns:
            out: 출력 텐서 [batch, out_channels*expansion, H', W']
        """
        # Identity 또는 projection shortcut 저장
        identity = self.shortcut(x)

        # Main path: 첫 번째 1x1 Conv-BN-ReLU (채널 확장)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Main path: 두 번째 3x3 Conv-BN-ReLU (주요 feature extraction)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        # Main path: 세 번째 1x1 Conv-BN (채널 축소)
        out = self.conv3(out)
        out = self.bn3(out)

        # Residual connection: F(x) + x
        out += identity

        # 최종 ReLU 활성화
        out = F.relu(out)

        return out


class DeepBaselineNetBN3ResidualBottleneck(nn.Module):
    """
    Bottleneck 구조를 사용한 더 깊은 Residual Network

    구조:
    1. 초기 Conv-BN-ReLU (3 -> 64)
    2. Layer 1: 64 채널에서 bottleneck blocks (1개)
    3. Layer 2: 128 채널에서 bottleneck blocks (2개)
    4. Layer 3: 256 채널에서 bottleneck blocks (2개)
    5. Layer 4: 256 채널에서 bottleneck blocks (1개)
    6. Fully Connected Layers (분류기)

    파라미터 최적화:
    - expansion=2로 줄여 파라미터 감소
    - 각 레이어의 block 수 감소
    - 최종 채널 수를 256으로 제한
    - FC 레이어를 단일 Linear로 축소하여 파라미터 최소화 (8192 -> 10)
    """

    def __init__(self, init_weights=False):
        super(DeepBaselineNetBN3ResidualBottleneck, self).__init__()

        # 초기 feature extraction
        # 입력: 3채널 (RGB), 출력: 64채널
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Layer 1: 64 채널에서 bottleneck blocks
        # expansion=2이므로 64 -> 64*2 = 128
        self.layer1 = nn.Sequential(
            BottleneckBlock(64, 64, stride=1),  # 64 -> 128
        )

        # Layer 2: 128 채널로 전환하고 bottleneck blocks
        # 첫 번째 block에서 채널 증가 (128 -> 128*2 = 256)
        self.layer2 = nn.Sequential(
            BottleneckBlock(128, 128, stride=1),  # 128 -> 256
            BottleneckBlock(256, 128, stride=1),  # 256 -> 256
        )

        # Layer 3: 256 채널로 전환하고 bottleneck blocks
        # 첫 번째 block에서 채널 증가 (256 -> 256*2 = 512)
        self.layer3 = nn.Sequential(
            BottleneckBlock(256, 256, stride=1),  # 256 -> 512
            BottleneckBlock(512, 256, stride=1),  # 512 -> 512
        )

        # Layer 4: 256 채널 유지 (더 이상 증가하지 않음)
        self.layer4 = nn.Sequential(
            BottleneckBlock(512, 256, stride=1),  # 512 -> 512
        )

        self.pool = nn.MaxPool2d(2, 2)

        self.classifier = nn.Linear(512 * 4 * 4, 10)

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
            elif isinstance(m, nn.BatchNorm2d):
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
        # 초기 Conv-BN-ReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        # Layer 1: 64 채널 bottleneck blocks
        x = self.layer1(x)  # [batch, 128, 32, 32]

        # Layer 2: 128 채널 bottleneck blocks
        x = self.layer2(x)  # [batch, 256, 32, 32]
        x = self.pool(x)  # 32x32 -> 16x16

        # Layer 3: 256 채널 bottleneck blocks
        x = self.layer3(x)  # [batch, 512, 16, 16]
        x = self.pool(x)  # 16x16 -> 8x8

        # Layer 4: 256 채널 bottleneck blocks
        x = self.layer4(x)  # [batch, 512, 8, 8]
        x = self.pool(x)  # 8x8 -> 4x4

        # Flatten: [batch, 512, 4, 4] -> [batch, 512*4*4]
        x = torch.flatten(x, 1)

        # Fully Connected Layers
        x = self.classifier(x)

        return x
