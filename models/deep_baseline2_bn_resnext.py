"""
DeepBaselineNetBN2ResNeXt: DeepBaselineNetBN2Residual에 ResNeXt 구조를 적용한 개선 버전

설계 의도:
1. ResNeXt 구조 (Aggregated Residual Transformations)
   - Cardinality 개념 도입: 병렬 경로의 개수
   - Split-Transform-Merge 전략으로 표현력 향상
   - Grouped Convolutions을 통한 효율적인 계산

2. Cardinality (병렬 경로)
   - 기존 ResNet: 단일 경로 (F(x) = conv(x))
   - ResNeXt: C개의 병렬 경로 (F(x) = sum(transform_i(x)))
   - 각 경로는 독립적으로 변환을 학습하여 더 풍부한 표현 학습

3. Grouped Convolutions
   - 3x3 conv를 grouped convolution으로 수행
   - groups=cardinality로 설정하여 각 경로가 독립적으로 작동
   - 파라미터 수를 증가시키지 않으면서 표현력 향상

4. Bottleneck 구조
   - 1x1 conv (채널 감소) -> 3x3 grouped conv -> 1x1 conv (채널 증가)
   - 각 경로는 bottleneck_width 채널을 사용
   - 전체 출력 채널은 cardinality * bottleneck_width

기존 DeepBaselineNetBN2Residual와의 차이점:
1. Residual Block 구조 변경
   - 기존: 단일 경로 (Conv -> BN -> ReLU -> Conv -> BN)
   - ResNeXt: C개의 병렬 경로 (각 경로: 1x1 -> 3x3 grouped -> 1x1)
   - 모든 경로의 출력을 합쳐서 사용

2. Cardinality 파라미터
   - cardinality: 병렬 경로의 개수 (기본값: 8)
   - bottleneck_width: 각 경로의 채널 수 (기본값: 4)
   - 출력 채널 = cardinality * bottleneck_width

3. Grouped Convolutions
   - 3x3 conv에 groups=cardinality 적용
   - 각 그룹이 독립적으로 학습하여 다양한 특징 추출

4. 네트워크 구조
   - conv1: 초기 feature extraction (3 -> 64)
   - resnext_block1: 64 -> 64 (cardinality=8, bottleneck_width=8)
   - resnext_block2: 64 -> 128 (cardinality=8, bottleneck_width=16)
   - resnext_block3: 128 -> 256 (cardinality=8, bottleneck_width=32)
   - resnext_block4: 256 -> 256 (cardinality=8, bottleneck_width=32)
   - resnext_block5: 256 -> 512 (cardinality=8, bottleneck_width=64)
   - FC layers: 분류기

참고:
- ResNeXt 논문: "Aggregated Residual Transformations for Deep Neural Networks" (Xie et al., 2017)
- Cardinality는 깊이(depth)나 너비(width)보다 더 효과적인 차원
- Grouped convolutions은 MobileNet, ShuffleNet 등에서도 사용되는 효율적인 구조
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNeXtBlock(nn.Module):
    """
    ResNeXt 스타일의 Residual Block
    
    구조:
    - 입력 x
    - C개의 병렬 경로 (cardinality)
      * 각 경로: 1x1 conv (채널 감소) -> 3x3 grouped conv -> 1x1 conv (채널 증가)
    - 모든 경로의 출력을 합침
    - Shortcut: identity (채널/크기 같음) 또는 projection (다름)
    - 출력: ReLU(sum(parallel_paths) + shortcut)
    
    Args:
        in_channels: 입력 채널 수
        out_channels: 출력 채널 수
        cardinality: 병렬 경로의 개수 (기본값: 8)
        bottleneck_width: 각 경로의 채널 수 (기본값: 4)
        stride: 첫 번째 conv의 stride (다운샘플링용, 기본값=1)
    """
    expansion = 1
    
    def __init__(self, in_channels, out_channels, cardinality=8, bottleneck_width=4, stride=1):
        super(ResNeXtBlock, self).__init__()
        
        # 각 경로의 채널 수
        # bottleneck_width는 각 경로의 중간 채널 수
        # 3x3 conv의 출력 채널 = cardinality * bottleneck_width
        bottleneck_channels = cardinality * bottleneck_width
        
        # Main path: C개의 병렬 경로
        # 각 경로는 1x1 -> 3x3 grouped -> 1x1 구조
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, 
                               stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        
        # 3x3 grouped convolution
        # groups=cardinality로 설정하여 각 경로가 독립적으로 작동
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3,
                               stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        
        # 1x1 conv로 출력 채널 수 조정
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1,
                               stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        # 채널 수가 다르거나 stride가 1이 아닌 경우 projection 필요
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # 1x1 conv를 사용하여 채널 수와 공간 크기 조정
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
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
        
        # Main path: 첫 번째 1x1 Conv-BN-ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # Main path: 3x3 Grouped Conv-BN-ReLU
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        # Main path: 두 번째 1x1 Conv-BN
        out = self.conv3(out)
        out = self.bn3(out)
        
        # Residual connection: F(x) + x
        # ResNeXt에서는 여러 경로의 출력이 이미 합쳐진 상태 (grouped conv로 자동 합쳐짐)
        out += identity
        
        # 최종 ReLU 활성화
        out = F.relu(out)
        
        return out


class DeepBaselineNetBN2ResNeXt(nn.Module):
    """
    DeepBaselineNetBN2에 ResNeXt 구조를 적용한 네트워크
    
    구조:
    1. 초기 Conv-BN-ReLU (3 -> 64)
    2. ResNeXt Block 1: 64 -> 64 (identity shortcut, cardinality=8, bottleneck_width=8)
    3. ResNeXt Block 2: 64 -> 128 (projection shortcut, cardinality=8, bottleneck_width=16)
    4. ResNeXt Block 3: 128 -> 256 (projection shortcut, cardinality=8, bottleneck_width=32)
    5. ResNeXt Block 4: 256 -> 256 (identity shortcut, cardinality=8, bottleneck_width=32)
    6. ResNeXt Block 5: 256 -> 512 (projection shortcut, cardinality=8, bottleneck_width=64)
    7. Fully Connected Layers (분류기)
    """
    
    def __init__(self, cardinality=8, bottleneck_width=4, init_weights=False):
        """
        Args:
            cardinality: 병렬 경로의 개수 (기본값: 8)
            bottleneck_width: 각 경로의 채널 수 (기본값: 4)
            init_weights: 가중치 초기화 여부
        """
        super(DeepBaselineNetBN2ResNeXt, self).__init__()
        
        # 초기 feature extraction
        # 입력: 3채널 (RGB), 출력: 64채널
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNeXt Blocks
        # Block 1: 64 -> 64 (같은 채널, identity shortcut)
        # bottleneck_width=8로 설정하여 출력 채널이 64가 되도록 함
        # (cardinality * bottleneck_width = 8 * 8 = 64)
        self.resnext_block1 = ResNeXtBlock(64, 64, cardinality=cardinality, 
                                           bottleneck_width=8, stride=1)
        
        # Block 2: 64 -> 128 (채널 증가, projection shortcut 필요)
        # bottleneck_width=16으로 설정하여 출력 채널이 128이 되도록 함
        # (cardinality * bottleneck_width = 8 * 16 = 128)
        self.resnext_block2 = ResNeXtBlock(64, 128, cardinality=cardinality,
                                           bottleneck_width=16, stride=1)
        
        # Block 3: 128 -> 256 (채널 증가, projection shortcut 필요)
        # bottleneck_width=32로 설정하여 출력 채널이 256이 되도록 함
        # (cardinality * bottleneck_width = 8 * 32 = 256)
        self.resnext_block3 = ResNeXtBlock(128, 256, cardinality=cardinality,
                                           bottleneck_width=32, stride=1)
        
        # Block 4: 256 -> 256 (같은 채널, identity shortcut)
        # bottleneck_width=32로 설정하여 출력 채널이 256이 되도록 함
        # (cardinality * bottleneck_width = 8 * 32 = 256)
        self.resnext_block4 = ResNeXtBlock(256, 256, cardinality=cardinality,
                                           bottleneck_width=32, stride=1)
        
        # Block 5: 256 -> 512 (채널 증가, projection shortcut 필요)
        # bottleneck_width=64로 설정하여 출력 채널이 512가 되도록 함
        # (cardinality * bottleneck_width = 8 * 64 = 512)
        self.resnext_block5 = ResNeXtBlock(256, 512, cardinality=cardinality,
                                           bottleneck_width=64, stride=1)
        
        # MaxPooling (원래 구조와 동일하게 유지)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully Connected Layers (분류기)
        # 입력 크기: 512 * 4 * 4 = 8192 (32x32 -> 16x16 -> 8x8 -> 4x4)
        self.fc1 = nn.Linear(512 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)  # CIFAR-10: 10개 클래스
        
        if init_weights:
            self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화 - ReLU를 사용하므로 Kaiming initialization 사용"""
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
        
        # ResNeXt Block 1: 64 -> 64
        x = self.resnext_block1(x)
        
        # ResNeXt Block 2: 64 -> 128
        x = self.resnext_block2(x)
        x = self.pool(x)  # 32x32 -> 16x16
        
        # ResNeXt Block 3: 128 -> 256
        x = self.resnext_block3(x)
        
        # ResNeXt Block 4: 256 -> 256
        x = self.resnext_block4(x)
        x = self.pool(x)  # 16x16 -> 8x8
        
        # ResNeXt Block 5: 256 -> 512
        x = self.resnext_block5(x)
        x = self.pool(x)  # 8x8 -> 4x4
        
        # Flatten: [batch, 512, 4, 4] -> [batch, 512*4*4]
        x = torch.flatten(x, 1)
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # 마지막 레이어는 활성화 함수 없음 (CrossEntropyLoss 사용)
        
        return x

