"""
DeepBaselineNetBN2ResidualPreAct: DeepBaselineNetBN2Residual의 Pre-activation 버전

설계 의도:
1. Pre-activation 구조
   - 기존 ResNet: Conv -> BN -> ReLU (post-activation)
   - Pre-activation ResNet: BN -> ReLU -> Conv (pre-activation)
   - 활성화 함수를 컨볼루션 레이어 앞에 배치하여 그래디언트 흐름 개선

2. 그래디언트 흐름 개선
   - Pre-activation은 각 레이어의 입력을 정규화하고 활성화한 후 컨볼루션 수행
   - 역전파 시 그래디언트가 더 직접적으로 전달되어 학습 안정성 향상
   - 깊은 네트워크에서도 효과적인 학습 가능

3. 표현력 향상
   - Pre-activation은 identity mapping을 더 쉽게 학습하도록 도와줌
   - 각 레이어가 더 독립적으로 동작하여 표현력 향상

4. Pre-activation ResNet 구조 적용
   - Residual block 내부: BN -> ReLU -> Conv -> BN -> ReLU -> Conv
   - Shortcut (projection): Conv만 사용 (BN, ReLU 없음)
   - 최종 출력: main_path + shortcut (ReLU 없음, 이미 main path에서 ReLU 적용됨)

기존 DeepBaselineNetBN2Residual와의 차이점:
1. Residual Block 구조 변경
   - 기존: Conv -> BN -> ReLU -> Conv -> BN, 최종 ReLU(main_path + shortcut)
   - Pre-activation: BN -> ReLU -> Conv -> BN -> ReLU -> Conv, 최종 main_path + shortcut
   - 각 컨볼루션 레이어 앞에 BN과 ReLU를 배치

2. Shortcut Connection 변경
   - 기존: Conv -> BN (projection인 경우)
   - Pre-activation: Conv만 사용 (projection인 경우, BN, ReLU 없음)
   - Identity shortcut은 그대로 유지

3. Forward pass 구조 변경
   - 기존: Conv -> BN -> ReLU -> Conv -> BN, ReLU(F(x) + x)
   - Pre-activation: BN -> ReLU -> Conv -> BN -> ReLU -> Conv, F(x) + x
   - 최종 ReLU가 제거되어 더 직접적인 그래디언트 흐름

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

        # 두 번째 BN-ReLU-Conv 블록 (마지막 BN 없음)
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


class DeepBaselineNetBN2ResidualPreAct(nn.Module):
    """
    DeepBaselineNetBN2Residual의 Pre-activation 버전 네트워크

    구조:
    1. 초기 Conv-BN-ReLU (3 -> 64)
    2. PreActivation Residual Block 1: 64 -> 64 (identity shortcut)
    3. PreActivation Residual Block 2: 64 -> 128 (projection shortcut)
    4. PreActivation Residual Block 3: 128 -> 256 (projection shortcut)
    5. PreActivation Residual Block 4: 256 -> 256 (identity shortcut)
    6. PreActivation Residual Block 5: 256 -> 512 (projection shortcut)
    7. Fully Connected Layers (분류기)
    """

    def __init__(self):
        super(DeepBaselineNetBN2ResidualPreAct, self).__init__()

        # 초기 feature extraction
        # 입력: 3채널 (RGB), 출력: 64채널
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Pre-activation Residual Blocks
        # Block 1: 64 -> 64 (같은 채널, identity shortcut)
        self.res_block1 = PreActivationResidualBlock(64, 64, stride=1)

        # Block 2: 64 -> 128 (채널 증가, projection shortcut 필요)
        # stride=1로 유지하고 MaxPool로 다운샘플링 (원래 구조 유지)
        self.res_block2 = PreActivationResidualBlock(64, 128, stride=1)

        # Block 3: 128 -> 256 (채널 증가, projection shortcut 필요)
        self.res_block3 = PreActivationResidualBlock(128, 256, stride=1)

        # Block 4: 256 -> 256 (같은 채널, identity shortcut)
        self.res_block4 = PreActivationResidualBlock(256, 256, stride=1)

        # Block 5: 256 -> 512 (채널 증가, projection shortcut 필요)
        self.res_block5 = PreActivationResidualBlock(256, 512, stride=1)

        # MaxPooling (원래 구조와 동일하게 유지)
        self.pool = nn.MaxPool2d(2, 2)

        # Fully Connected Layers (분류기)
        # 입력 크기: 512 * 4 * 4 = 8192 (32x32 -> 16x16 -> 8x8 -> 4x4)
        self.fc1 = nn.Linear(512 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)  # CIFAR-10: 10개 클래스

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

        # Pre-activation Residual Block 1: 64 -> 64
        x = self.res_block1(x)

        # Pre-activation Residual Block 2: 64 -> 128
        x = self.res_block2(x)
        x = self.pool(x)  # 32x32 -> 16x16

        # Pre-activation Residual Block 3: 128 -> 256
        x = self.res_block3(x)

        # Pre-activation Residual Block 4: 256 -> 256
        x = self.res_block4(x)
        x = self.pool(x)  # 16x16 -> 8x8

        # Pre-activation Residual Block 5: 256 -> 512
        x = self.res_block5(x)
        x = self.pool(x)  # 8x8 -> 4x4

        # Flatten: [batch, 512, 4, 4] -> [batch, 512*4*4]
        x = torch.flatten(x, 1)

        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # 마지막 레이어는 활성화 함수 없음 (CrossEntropyLoss 사용)

        return x
