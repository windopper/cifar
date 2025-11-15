"""
DeepBaselineNetBN4Residual: DeepBaselineNetBN3Residual의 더 깊은 버전

설계 의도:
1. 더 깊은 네트워크 구조
   - 각 채널 레벨에서 여러 개의 residual block을 사용하여 네트워크 깊이 증가
   - ResNet 스타일의 레이어 구성 (각 채널 레벨마다 2-3개의 블록)
   - Residual connection으로 인해 깊은 네트워크에서도 안정적인 학습 가능

2. 네트워크 구조:
   - 초기 Conv-BN-ReLU (3 -> 64)
   - 64 채널: 2개의 residual block
   - 128 채널: 3개의 residual block
   - 256 채널: 3개의 residual block
   - 512 채널: 2개의 residual block
   - Fully Connected Layers (분류기)

3. DeepBaselineNetBN3Residual과의 차이점:
   - 각 채널 레벨에서 여러 개의 residual block 사용
   - 총 residual block 수: 5개 -> 10개
   - 더 깊은 네트워크로 인한 표현력 향상 기대

참고:
- ResNet 논문: "Deep Residual Learning for Image Recognition" (He et al., 2015)
- 깊은 네트워크에서 residual connection이 그래디언트 소실 문제를 해결하는 핵심 기술
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


class DeepBaselineNetBN4Residual(nn.Module):
    """
    DeepBaselineNetBN3Residual의 더 깊은 버전
    
    구조:
    1. 초기 Conv-BN-ReLU (3 -> 64)
    2. 64 채널 레벨: 2개의 residual block (64 -> 64)
    3. 128 채널 레벨: 3개의 residual block (64 -> 128, 128 -> 128, 128 -> 128)
    4. 256 채널 레벨: 3개의 residual block (128 -> 256, 256 -> 256, 256 -> 256)
    5. 512 채널 레벨: 2개의 residual block (256 -> 512, 512 -> 512)
    6. Fully Connected Layers (분류기)
    
    총 10개의 residual block 사용 (기존 5개에서 증가)
    """
    
    def __init__(self, init_weights=False):
        super(DeepBaselineNetBN4Residual, self).__init__()
        
        # 초기 feature extraction
        # 입력: 3채널 (RGB), 출력: 64채널
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 64 채널 레벨: 2개의 residual block
        # 첫 번째 블록: 채널 증가 없음 (identity shortcut)
        self.res_block1_1 = ResidualBlock(64, 64, stride=1)
        self.res_block1_2 = ResidualBlock(64, 64, stride=1)
        
        # 128 채널 레벨: 3개의 residual block
        # 첫 번째 블록: 64 -> 128 (채널 증가, projection shortcut 필요)
        self.res_block2_1 = ResidualBlock(64, 128, stride=1)
        # 두 번째, 세 번째 블록: 128 -> 128 (identity shortcut)
        self.res_block2_2 = ResidualBlock(128, 128, stride=1)
        self.res_block2_3 = ResidualBlock(128, 128, stride=1)
        
        # 256 채널 레벨: 3개의 residual block
        # 첫 번째 블록: 128 -> 256 (채널 증가, projection shortcut 필요)
        self.res_block3_1 = ResidualBlock(128, 256, stride=1)
        # 두 번째, 세 번째 블록: 256 -> 256 (identity shortcut)
        self.res_block3_2 = ResidualBlock(256, 256, stride=1)
        self.res_block3_3 = ResidualBlock(256, 256, stride=1)
        
        # 512 채널 레벨: 2개의 residual block
        # 첫 번째 블록: 256 -> 512 (채널 증가, projection shortcut 필요)
        self.res_block4_1 = ResidualBlock(256, 512, stride=1)
        # 두 번째 블록: 512 -> 512 (identity shortcut)
        self.res_block4_2 = ResidualBlock(512, 512, stride=1)
        
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
        
        # 64 채널 레벨: 2개의 residual block
        x = self.res_block1_1(x)
        x = self.res_block1_2(x)
        
        # 128 채널 레벨: 3개의 residual block
        x = self.res_block2_1(x)  # 64 -> 128
        x = self.res_block2_2(x)   # 128 -> 128
        x = self.res_block2_3(x)   # 128 -> 128
        x = self.pool(x)  # 32x32 -> 16x16
        
        # 256 채널 레벨: 3개의 residual block
        x = self.res_block3_1(x)  # 128 -> 256
        x = self.res_block3_2(x)  # 256 -> 256
        x = self.res_block3_3(x)  # 256 -> 256
        x = self.pool(x)  # 16x16 -> 8x8
        
        # 512 채널 레벨: 2개의 residual block
        x = self.res_block4_1(x)  # 256 -> 512
        x = self.res_block4_2(x)   # 512 -> 512
        x = self.pool(x)  # 8x8 -> 4x4
        
        # Flatten: [batch, 512, 4, 4] -> [batch, 512*4*4]
        x = torch.flatten(x, 1)
        
        # Fully Connected Layers
        x = self.classifier(x)
        
        return x

