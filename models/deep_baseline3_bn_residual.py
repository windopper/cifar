"""
DeepBaselineNetBN2Residual: DeepBaselineNetBN2에 Residual Connection을 추가한 개선 버전

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

4. 표준 ResNet 구조 적용
   - BasicBlock 스타일의 residual block 사용
   - Conv-BN-ReLU 구조를 유지하면서 residual connection 추가
   - 채널 수가 다른 경우 1x1 conv를 사용한 shortcut connection

기존 DeepBaselineNetBN2와의 차이점:
1. Residual Block 구조 도입
   - 각 컨볼루션 레이어를 residual block으로 변경
   - 각 block은 두 개의 conv 레이어로 구성 (conv-bn-relu-conv-bn)
   - Skip connection을 통해 입력과 출력을 더함 (F(x) + x)

2. Shortcut Connection
   - 채널 수가 같은 경우: identity mapping (x 그대로 사용)
   - 채널 수가 다른 경우: 1x1 conv + BN을 사용한 projection shortcut
   - 공간 크기가 다른 경우: stride를 조정하여 다운샘플링

3. 네트워크 구조 변경
   - conv1: 초기 feature extraction (3 -> 64)
   - residual_block1: 64 -> 64 (같은 채널, identity shortcut)
   - residual_block2: 64 -> 128 (채널 증가, projection shortcut + stride=2)
   - residual_block3: 128 -> 256 (채널 증가, projection shortcut + stride=2)
   - residual_block4: 256 -> 256 (같은 채널, identity shortcut)
   - residual_block5: 256 -> 512 (채널 증가, projection shortcut + stride=2)
   - FC layers: 분류기

4. Forward pass 구조 변경
   - DeepBaselineNetBN2: Conv -> BN -> ReLU (순차적)
   - DeepBaselineNetBN2Residual: Residual Block (F(x) + x 형태)
     * 각 block 내부: Conv -> BN -> ReLU -> Conv -> BN
     * Skip connection: identity 또는 projection
     * 최종: ReLU(F(x) + x)

참고:
- ResNet 논문: "Deep Residual Learning for Image Recognition" (He et al., 2015)
- Residual block은 깊은 네트워크에서 그래디언트 소실 문제를 해결하는 핵심 기술
- Pre-activation 버전: deep_baseline2_bn_preact.py 참고
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


class DeepBaselineNetBN3Residual(nn.Module):
    """
    DeepBaselineNetBN2에 Residual Connection을 추가한 네트워크
    
    구조:
    1. 초기 Conv-BN-ReLU (3 -> 64)
    2. Residual Block 1: 64 -> 64 (identity shortcut)
    3. Residual Block 2: 64 -> 128 (projection shortcut, stride=2)
    4. Residual Block 3: 128 -> 256 (projection shortcut, stride=2)
    5. Residual Block 4: 256 -> 256 (identity shortcut)
    6. Residual Block 5: 256 -> 512 (projection shortcut, stride=2)
    7. Fully Connected Layers (분류기)
    """
    
    def __init__(self, init_weights=False):
        super(DeepBaselineNetBN3Residual, self).__init__()
        
        # 초기 feature extraction
        # 입력: 3채널 (RGB), 출력: 64채널
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual Blocks
        # Block 1: 64 -> 64 (같은 채널, identity shortcut)
        self.res_block1 = ResidualBlock(64, 64, stride=1)
        
        # Block 2: 64 -> 128 (채널 증가, projection shortcut 필요)
        # stride=1로 유지하고 MaxPool로 다운샘플링 (원래 구조 유지)
        self.res_block2 = ResidualBlock(64, 128, stride=1)
        
        # Block 3: 128 -> 256 (채널 증가, projection shortcut 필요)
        self.res_block3 = ResidualBlock(128, 256, stride=1)
        
        # Block 4: 256 -> 256 (같은 채널, identity shortcut)
        self.res_block4 = ResidualBlock(256, 256, stride=1)
        
        # Block 5: 256 -> 512 (채널 증가, projection shortcut 필요)
        self.res_block5 = ResidualBlock(256, 512, stride=1)
        
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

