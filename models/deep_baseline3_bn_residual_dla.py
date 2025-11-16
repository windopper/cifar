"""
DeepBaselineNetBN3ResidualDLA: DeepBaselineNetBN3Residual에 Deep Layer Aggregation(DLA) 아이디어를 도입한 버전

설계 의도:
1. Deep Layer Aggregation (깊은 계층 집계)
   - 네트워크 깊은 곳에서만 특징을 사용하는 것이 아니라, 여러 깊이의 특징을 계층적으로 집계
   - 얕은 레이어의 세밀한 지역 정보 + 깊은 레이어의 추상적 고수준 정보를 함께 활용
   - DLA 논문(Deep Layer Aggregation, 2017)의 아이디어를 단순화하여 적용

2. 다중 해상도 특징 집계
   - 각 Residual Block 이후의 feature map을 수집
   - Adaptive Average Pooling을 통해 모두 동일한 해상도(4x4)로 정규화
   - 1x1 Conv를 통해 채널 수를 통일한 후, 점진적으로 더해가며 집계(Iterative Aggregation)

3. 원래 구조와의 호환성 유지
   - 기본적인 stage 구성(64 -> 128 -> 256 -> 512)과 MaxPooling 위치는 DeepBaselineNetBN3Residual와 동일
   - classifier 입력 차원(512 * 4 * 4)도 동일하게 유지
   - 단, 마지막 stage의 feature뿐 아니라, 여러 stage의 feature를 집계하여 사용

구조 요약:
1. 초기 Conv-BN-ReLU (3 -> 64)
2. Residual Block 1: 64 -> 64 (identity shortcut), feature f1 (32x32)
3. Residual Block 2: 64 -> 128 (projection), MaxPool 후 feature f2 (16x16)
4. Residual Block 3: 128 -> 256, feature f3 (16x16)
5. Residual Block 4: 256 -> 256, MaxPool 후 feature f4 (8x8)
6. Residual Block 5: 256 -> 512, MaxPool 후 feature f5 (4x4)
7. Deep Layer Aggregation 모듈이 [f1, f2, f3, f4, f5]를 집계하여 f_agg (512 x 4 x 4) 생성
8. Fully Connected Layers (분류기)

참고:
- Deep Layer Aggregation 논문: "Deep Layer Aggregation" (Yu et al., 2017)
- 이 구현은 논문의 전체 구조가 아닌, "여러 깊이의 특징을 단일 feature로 집계한다"는 아이디어를
  DeepBaselineNetBN3Residual에 맞게 단순화하여 적용한 버전입니다.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    ResNet 스타일의 Basic Residual Block

    구조:
    - 입력 x
    - Main path: Conv -> BN -> ReLU -> Conv -> BN
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

        # Main path: 두 개의 Conv-BN 블록
        # 첫 번째 Conv-BN-ReLU 블록
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 두 번째 Conv-BN 블록 (ReLU는 residual connection 후에 적용)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        # 채널 수가 다르거나 stride가 1이 아닌 경우 projection 필요
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # 1x1 conv를 사용하여 채널 수와 공간 크기 조정
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

        # Main path: 첫 번째 Conv-BN-ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # Main path: 두 번째 Conv-BN
        out = self.conv2(out)
        out = self.bn2(out)

        # Residual connection: F(x) + x
        out += identity

        # 최종 ReLU 활성화
        out = F.relu(out)

        return out


class DeepLayerAggregator(nn.Module):
    """
    여러 깊이의 feature map을 모아서 하나의 깊은 표현으로 집계하는 모듈.

    설계:
    - 입력: 다양한 채널/해상도의 feature map 리스트 [f1, f2, ..., fN]
    - 각 feature를 1x1 Conv + BN + ReLU로 out_channels로 투영
    - 가장 마지막 feature의 해상도(target_size)로 AdaptiveAvgPool 수행
    - 순차적으로 더하면서(Iterative Aggregation) f_agg를 생성
    """

    def __init__(self, in_channels_list, out_channels):
        super(DeepLayerAggregator, self).__init__()
        assert len(in_channels_list) > 0, "in_channels_list는 비어 있을 수 없습니다."

        self.out_channels = out_channels
        self.proj_convs = nn.ModuleList()
        self.proj_bns = nn.ModuleList()

        for c in in_channels_list:
            self.proj_convs.append(nn.Conv2d(c, out_channels, kernel_size=1, bias=False))
            self.proj_bns.append(nn.BatchNorm2d(out_channels))

    def forward(self, features):
        """
        Args:
            features: feature map 리스트 [f1, f2, ..., fN]

        Returns:
            agg: 집계된 feature map [batch, out_channels, H_target, W_target]
        """
        assert len(features) == len(
            self.proj_convs
        ), "features 길이와 in_channels_list 길이는 동일해야 합니다."

        # 마지막 feature의 해상도를 기준 해상도로 사용
        target_h, target_w = features[-1].shape[2], features[-1].shape[3]

        agg = None
        for x, conv, bn in zip(features, self.proj_convs, self.proj_bns):
            # 1x1 Conv + BN + ReLU로 채널 투영
            x = conv(x)
            x = bn(x)
            x = F.relu(x)

            # 해상도를 target 크기로 맞춤 (보통 4x4)
            if x.shape[2] != target_h or x.shape[3] != target_w:
                x = F.adaptive_avg_pool2d(x, (target_h, target_w))

            # 순차적으로 더하면서 집계
            if agg is None:
                agg = x
            else:
                agg = agg + x

        return agg


class DeepBaselineNetBN3ResidualDLA(nn.Module):
    """
    DeepBaselineNetBN3Residual에 Deep Layer Aggregation을 도입한 네트워크

    구조:
    1. 초기 Conv-BN-ReLU (3 -> 64)
    2. Residual Block 1: 64 -> 64 (identity shortcut), feature f1 (32x32)
    3. Residual Block 2: 64 -> 128 (projection), MaxPool 후 feature f2 (16x16)
    4. Residual Block 3: 128 -> 256, feature f3 (16x16)
    5. Residual Block 4: 256 -> 256, MaxPool 후 feature f4 (8x8)
    6. Residual Block 5: 256 -> 512, MaxPool 후 feature f5 (4x4)
    7. DeepLayerAggregator가 [f1, f2, f3, f4, f5]를 집계하여 f_agg (512 x 4 x 4) 생성
    8. Fully Connected Layers (분류기)
    """

    def __init__(self, init_weights: bool = False):
        super(DeepBaselineNetBN3ResidualDLA, self).__init__()

        # 초기 feature extraction
        # 입력: 3채널 (RGB), 출력: 64채널
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual Blocks (원래 구조와 동일한 채널 구성)
        self.res_block1 = ResidualBlock(64, 64, stride=1)   # f1
        self.res_block2 = ResidualBlock(64, 128, stride=1)  # f2 (pool 후)
        self.res_block3 = ResidualBlock(128, 256, stride=1) # f3
        self.res_block4 = ResidualBlock(256, 256, stride=1) # f4 (pool 후)
        self.res_block5 = ResidualBlock(256, 512, stride=1) # f5 (pool 후)

        # MaxPooling (원래 구조와 동일하게 유지)
        self.pool = nn.MaxPool2d(2, 2)

        # Deep Layer Aggregation 모듈
        # 각 feature의 채널 수: [64, 128, 256, 256, 512]
        self.aggregator = DeepLayerAggregator(
            in_channels_list=[64, 128, 256, 256, 512],
            out_channels=512,
        )

        # 분류기: 원래 DeepBaselineNetBN3Residual과 동일
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
        f1 = x  # [B, 64, 32, 32]

        # Residual Block 2: 64 -> 128
        x = self.res_block2(x)
        x = self.pool(x)  # 32x32 -> 16x16
        f2 = x  # [B, 128, 16, 16]

        # Residual Block 3: 128 -> 256
        x = self.res_block3(x)
        f3 = x  # [B, 256, 16, 16]

        # Residual Block 4: 256 -> 256
        x = self.res_block4(x)
        x = self.pool(x)  # 16x16 -> 8x8
        f4 = x  # [B, 256, 8, 8]

        # Residual Block 5: 256 -> 512
        x = self.res_block5(x)
        x = self.pool(x)  # 8x8 -> 4x4
        f5 = x  # [B, 512, 4, 4]

        # Deep Layer Aggregation: 여러 깊이의 feature를 집계
        agg = self.aggregator([f1, f2, f3, f4, f5])  # [B, 512, 4, 4]

        # Flatten: [batch, 512, 4, 4] -> [batch, 512*4*4]
        agg = torch.flatten(agg, 1)

        # Fully Connected Layers
        out = self.classifier(agg)

        return out



