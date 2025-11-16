"""
DeepBaselineNetBN3ResidualGroup:
DeepBaselineNetBN3Residual에 Group Convolution을 적용한 개선 버전

설계 의도:
1. 파라미터 수 및 연산량 감소
   - Conv 레이어에 group convolution을 적용하여 채널을 여러 그룹으로 나누어 연산
   - 동일 채널 수에서 일반 conv 대비 파라미터 수 감소 및 연산량 절감 효과

2. 채널 그룹별 특화 표현 학습
   - 각 그룹이 서로 다른 특징에 집중하도록 하여 표현 다양성 증가
   - ResNeXt 스타일의 "cardinality" 개념과 유사하게 채널 그룹 수를 조절 가능

3. 기존 구조 최대한 유지
   - 전체 블록 개수, 채널 진행(64 → 128 → 256 → 512) 및 풀링 위치는 동일
   - ResidualBlock 내부의 conv에 group 옵션을 추가
   - 초기 conv1에도 group을 적용할 수 있도록 옵션 제공(기본은 일반 conv)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlockGroup(nn.Module):
    """
    Group Convolution이 적용된 ResNet 스타일 Basic Residual Block

    구조:
    - 입력 x
    - Main path: (Grouped) Conv -> BN -> ReLU -> (Grouped) Conv -> BN
    - Shortcut: identity (채널/크기 같음) 또는 projection (다름)
    - 출력: ReLU(main_path + shortcut)

    Args:
        in_channels: 입력 채널 수
        out_channels: 출력 채널 수
        stride: 첫 번째 conv의 stride (다운샘플링용, 기본값=1)
        groups: group convolution에서의 그룹 수 (1이면 일반 conv와 동일)
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, groups=2):
        super(ResidualBlockGroup, self).__init__()

        # groups는 in/out 채널 수를 나눌 수 있어야 함
        assert (
            in_channels % groups == 0 and out_channels % groups == 0
        ), f"groups({groups}) must divide in_channels({in_channels}) and out_channels({out_channels})"

        # Main path: 두 개의 Grouped Conv-BN 블록
        # 첫 번째 Grouped Conv-BN-ReLU 블록
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 두 번째 Grouped Conv-BN 블록 (ReLU는 residual connection 후에 적용)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        # 채널 수가 다르거나 stride가 1이 아닌 경우 projection 필요
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Shortcut은 정보 손실을 줄이기 위해 기본적으로 일반 1x1 conv 사용(그룹 미적용)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
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

        # Main path: 첫 번째 Grouped Conv-BN-ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Main path: 두 번째 Grouped Conv-BN
        out = self.conv2(out)
        out = self.bn2(out)

        # Residual connection: F(x) + x
        out += identity

        # 최종 ReLU 활성화
        out = F.relu(out)

        return out


class DeepBaselineNetBN3ResidualGroup(nn.Module):
    """
    DeepBaselineNetBN3Residual에 Group Convolution을 적용한 네트워크

    구조:
    1. 초기 Conv-BN-ReLU (3 -> 64)
       - 옵션에 따라 group conv 사용 가능 (기본은 일반 conv)
    2. Residual Block 1: 64 -> 64 (identity shortcut, group conv)
    3. Residual Block 2: 64 -> 128 (projection shortcut, group conv)
    4. Residual Block 3: 128 -> 256 (projection shortcut, group conv)
    5. Residual Block 4: 256 -> 256 (identity shortcut, group conv)
    6. Residual Block 5: 256 -> 512 (projection shortcut, group conv)
    7. Fully Connected Layers (분류기)

    Args:
        init_weights: True이면 Kaiming initialization 수행
        groups: ResidualBlockGroup에서 사용할 그룹 수
        stem_groups: 입력 stem conv(3 -> 64)에 사용할 그룹 수 (1이면 일반 conv)
    """

    def __init__(self, init_weights: bool = False, groups: int = 2, stem_groups: int = 1):
        super(DeepBaselineNetBN3ResidualGroup, self).__init__()

        # stem conv에 group을 적용할 경우 3과 64가 stem_groups로 나누어져야 함
        assert 3 % stem_groups == 0 and 64 % stem_groups == 0, (
            f"stem_groups({stem_groups}) must divide 3 and 64"
        )

        self.groups = groups
        self.stem_groups = stem_groups

        # 초기 feature extraction
        # 입력: 3채널 (RGB), 출력: 64채널
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            padding=1,
            bias=False,
            groups=stem_groups,
        )
        self.bn1 = nn.BatchNorm2d(64)

        # Residual Blocks (모두 Group Convolution 사용)
        # Block 1: 64 -> 64 (같은 채널, identity shortcut)
        self.res_block1 = ResidualBlockGroup(64, 64, stride=1, groups=groups)

        # Block 2: 64 -> 128 (채널 증가, projection shortcut 필요)
        # stride=1로 유지하고 MaxPool로 다운샘플링 (원래 구조 유지)
        self.res_block2 = ResidualBlockGroup(64, 128, stride=1, groups=groups)

        # Block 3: 128 -> 256 (채널 증가, projection shortcut 필요)
        self.res_block3 = ResidualBlockGroup(128, 256, stride=1, groups=groups)

        # Block 4: 256 -> 256 (같은 채널, identity shortcut)
        self.res_block4 = ResidualBlockGroup(256, 256, stride=1, groups=groups)

        # Block 5: 256 -> 512 (채널 증가, projection shortcut 필요)
        self.res_block5 = ResidualBlockGroup(256, 512, stride=1, groups=groups)

        # MaxPooling (원래 구조와 동일하게 유지)
        self.pool = nn.MaxPool2d(2, 2)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 10),
        )

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        """가중치 초기화 - ReLU를 사용하므로 Kaiming initialization 사용"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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

        # Residual Block 1: 64 -> 64
        x = self.res_block1(x)

        # Residual Block 2: 64 -> 128
        x = self.res_block2(x)
        x = self.pool(x)  # 32x32 -> 16x16

        # Residual Block 3: 128 -> 256
        x = self.res_block3(x)

        # Residual Block 4: 256 -> 256
        x = self.res_block4(x)
        x = self.pool(x)  # 16x16 -> 8x8

        # Residual Block 5: 256 -> 512
        x = self.res_block5(x)
        x = self.pool(x)  # 8x8 -> 4x4

        # Flatten: [batch, 512, 4, 4] -> [batch, 512*4*4]
        x = torch.flatten(x, 1)

        # Fully Connected Layers
        x = self.classifier(x)

        return x



