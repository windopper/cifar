"""
DeepBaselineNetBN3ResidualDeep: DeepBaselineNetBN3Residual의 깊이를 늘린 Deep 버전

설계 의도:
1. 파라미터 수 증가
   - 각 stage마다 residual block을 여러 개 쌓아 깊이를 증가시켜 파라미터 수를 대략 4배로 증가
   - 더 깊은 네트워크로 표현력 향상 기대

2. Residual Learning 유지
   - 기존 DeepBaselineNetBN3Residual의 residual connection 구조 유지
   - Skip connection을 통한 그래디언트 흐름 개선
   - 깊은 네트워크에서도 효과적인 학습 가능

3. 네트워크 구조 변경
   - 각 stage마다 residual block을 4개씩 쌓음 (기존 1개 -> 4개)
   - Stage 1: 64 -> 64 (4개 블록)
   - Stage 2: 64 -> 128 (4개 블록, 첫 번째 블록에서 채널 증가)
   - Stage 3: 128 -> 256 (4개 블록, 첫 번째 블록에서 채널 증가)
   - Stage 4: 256 -> 256 (4개 블록)
   - Stage 5: 256 -> 512 (4개 블록, 첫 번째 블록에서 채널 증가)
   - FC layers: 동일하게 유지

기존 DeepBaselineNetBN3Residual와의 차이점:
1. 각 stage마다 residual block을 4개씩 쌓아 깊이 증가
2. 총 residual block 수: 5개 -> 20개
3. 채널 수와 FC layer는 동일하게 유지
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
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 두 번째 Conv-BN 블록 (ReLU는 residual connection 후에 적용)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
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


def _make_layer(in_channels, out_channels, num_blocks, stride=1):
    """
    여러 개의 residual block을 쌓아서 layer를 만드는 헬퍼 함수
    
    Args:
        in_channels: 입력 채널 수
        out_channels: 출력 채널 수
        num_blocks: 쌓을 residual block의 개수
        stride: 첫 번째 블록의 stride (기본값=1)
        
    Returns:
        nn.Sequential: 여러 개의 residual block으로 구성된 layer
    """
    layers = []
    # 첫 번째 블록: 채널 수가 다르거나 stride가 1이 아닌 경우
    layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
    # 나머지 블록들: 같은 채널 수, stride=1
    for _ in range(1, num_blocks):
        layers.append(ResidualBlock(out_channels, out_channels, stride=1))
    return nn.Sequential(*layers)


class DeepBaselineNetBN3ResidualDeep(nn.Module):
    """
    DeepBaselineNetBN3Residual의 깊이를 늘린 Deep 버전
    
    구조:
    1. 초기 Conv-BN-ReLU (3 -> 64)
    2. Stage 1: 64 -> 64 (4개 residual block)
    3. Stage 2: 64 -> 128 (4개 residual block, 첫 번째 블록에서 채널 증가)
    4. Stage 3: 128 -> 256 (4개 residual block, 첫 번째 블록에서 채널 증가)
    5. Stage 4: 256 -> 256 (4개 residual block)
    6. Stage 5: 256 -> 512 (4개 residual block, 첫 번째 블록에서 채널 증가)
    7. Fully Connected Layers (분류기)
    """
    
    def __init__(self, init_weights=False):
        super(DeepBaselineNetBN3ResidualDeep, self).__init__()
        
        # 초기 feature extraction
        # 입력: 3채널 (RGB), 출력: 64채널
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual Layers (각 stage마다 4개씩 블록 쌓기)
        # Stage 1: 64 -> 64 (4개 블록)
        self.layer1 = _make_layer(64, 64, num_blocks=4, stride=1)
        
        # Stage 2: 64 -> 128 (4개 블록, 첫 번째 블록에서 채널 증가)
        self.layer2 = _make_layer(64, 128, num_blocks=4, stride=1)
        
        # Stage 3: 128 -> 256 (4개 블록, 첫 번째 블록에서 채널 증가)
        self.layer3 = _make_layer(128, 256, num_blocks=4, stride=1)
        
        # Stage 4: 256 -> 256 (4개 블록)
        self.layer4 = _make_layer(256, 256, num_blocks=4, stride=1)
        
        # Stage 5: 256 -> 512 (4개 블록, 첫 번째 블록에서 채널 증가)
        self.layer5 = _make_layer(256, 512, num_blocks=4, stride=1)
        
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
        
        # Stage 1: 64 -> 64 (4개 블록)
        x = self.layer1(x)
        
        # Stage 2: 64 -> 128 (4개 블록)
        x = self.layer2(x)
        x = self.pool(x)  # 32x32 -> 16x16
        
        # Stage 3: 128 -> 256 (4개 블록)
        x = self.layer3(x)
        
        # Stage 4: 256 -> 256 (4개 블록)
        x = self.layer4(x)
        x = self.pool(x)  # 16x16 -> 8x8
        
        # Stage 5: 256 -> 512 (4개 블록)
        x = self.layer5(x)
        x = self.pool(x)  # 8x8 -> 4x4
        
        # Flatten: [batch, 512, 4, 4] -> [batch, 512*4*4]
        x = torch.flatten(x, 1)
        
        # Fully Connected Layers
        x = self.classifier(x)
        
        return x

