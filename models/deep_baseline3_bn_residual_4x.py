"""
DeepBaselineNetBN3Residual4X: DeepBaselineNetBN3Residual의 파라미터 수를 대략 4배로 늘린 버전

설계 의도:
1. 파라미터 수 증가
   - 채널 수를 2배로 늘려 파라미터 수를 대략 4배로 증가
   - 더 넓은 네트워크로 표현력 향상 기대

2. Residual Learning 유지
   - 기존 DeepBaselineNetBN3Residual의 residual connection 구조 유지
   - Skip connection을 통한 그래디언트 흐름 개선

3. 네트워크 구조 변경
   - conv1: 3 -> 128 (기존 64 -> 128)
   - residual_block1: 128 -> 128 (기존 64 -> 64)
   - residual_block2: 128 -> 256 (기존 64 -> 128)
   - residual_block3: 256 -> 512 (기존 128 -> 256)
   - residual_block4: 512 -> 512 (기존 256 -> 256)
   - residual_block5: 512 -> 1024 (기존 256 -> 512)
   - FC layers: 1024*4*4 -> 1024 -> 512 -> 10 (기존 512*4*4 -> 512 -> 256 -> 10)

기존 DeepBaselineNetBN3Residual와의 차이점:
1. 모든 채널 수를 2배로 증가
2. FC layer의 크기도 비례하여 증가
3. 나머지 구조는 동일하게 유지
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


class DeepBaselineNetBN3Residual4X(nn.Module):
    """
    DeepBaselineNetBN3Residual의 파라미터 수를 대략 4배로 늘린 버전
    
    구조:
    1. 초기 Conv-BN-ReLU (3 -> 128)
    2. Residual Block 1: 128 -> 128 (identity shortcut)
    3. Residual Block 2: 128 -> 256 (projection shortcut, stride=1)
    4. Residual Block 3: 256 -> 512 (projection shortcut, stride=1)
    5. Residual Block 4: 512 -> 512 (identity shortcut)
    6. Residual Block 5: 512 -> 1024 (projection shortcut, stride=1)
    7. Fully Connected Layers (분류기)
    """
    
    def __init__(self, init_weights=False):
        super(DeepBaselineNetBN3Residual4X, self).__init__()
        
        # 초기 feature extraction
        # 입력: 3채널 (RGB), 출력: 128채널 (기존 64 -> 128)
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        
        # Residual Blocks
        # Block 1: 128 -> 128 (같은 채널, identity shortcut)
        self.res_block1 = ResidualBlock(128, 128, stride=1)
        
        # Block 2: 128 -> 256 (채널 증가, projection shortcut 필요)
        # stride=1로 유지하고 MaxPool로 다운샘플링 (원래 구조 유지)
        self.res_block2 = ResidualBlock(128, 256, stride=1)
        
        # Block 3: 256 -> 512 (채널 증가, projection shortcut 필요)
        self.res_block3 = ResidualBlock(256, 512, stride=1)
        
        # Block 4: 512 -> 512 (같은 채널, identity shortcut)
        self.res_block4 = ResidualBlock(512, 512, stride=1)
        
        # Block 5: 512 -> 1024 (채널 증가, projection shortcut 필요)
        self.res_block5 = ResidualBlock(512, 1024, stride=1)
        
        # MaxPooling (원래 구조와 동일하게 유지)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 4 * 4, 1024),  # 기존 512 -> 1024
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 512),  # 기존 256 -> 512
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10),
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
        
        # Residual Block 1: 128 -> 128
        x = self.res_block1(x)
        
        # Residual Block 2: 128 -> 256
        x = self.res_block2(x)
        x = self.pool(x)  # 32x32 -> 16x16
        
        # Residual Block 3: 256 -> 512
        x = self.res_block3(x)
        
        # Residual Block 4: 512 -> 512
        x = self.res_block4(x)
        x = self.pool(x)  # 16x16 -> 8x8
        
        # Residual Block 5: 512 -> 1024
        x = self.res_block5(x)
        x = self.pool(x)  # 8x8 -> 4x4
        
        # Flatten: [batch, 1024, 4, 4] -> [batch, 1024*4*4]
        x = torch.flatten(x, 1)
        
        # Fully Connected Layers
        x = self.classifier(x)
        
        return x

