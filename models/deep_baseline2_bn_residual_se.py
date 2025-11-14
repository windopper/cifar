"""
DeepBaselineNetBN2ResidualSE: DeepBaselineNetBN2Residual에 Squeeze-and-Excitation 블록을 추가한 개선 버전

설계 의도:
1. Residual Learning (잔차 학습)
   - Skip connection을 통해 그래디언트가 직접 전달되어 그래디언트 소실 문제 완화
   - 깊은 네트워크에서도 효과적인 학습 가능
   - 각 레이어가 잔차(residual)를 학습하도록 하여 학습 난이도 감소

2. Squeeze-and-Excitation (SE) 블록
   - 채널 간 의존성을 모델링하여 중요한 채널에 집중
   - Global Average Pooling으로 공간 정보를 압축 (Squeeze)
   - FC 레이어로 채널별 가중치 생성 (Excitation)
   - 원본 feature map에 채널별 가중치를 곱하여 재조정 (Scale)
   - 적은 파라미터 추가로 성능 향상

3. SE-ResNet 구조
   - 각 Residual Block에 SE 블록을 통합
   - Residual connection 후에 SE 블록을 적용하여 채널별 중요도 재조정
   - 구조: Conv -> BN -> ReLU -> Conv -> BN -> SE -> (F(x) + x) -> ReLU

기존 DeepBaselineNetBN2Residual와의 차이점:
1. SEBlock 클래스 추가
   - Squeeze: AdaptiveAvgPool2d로 공간 차원 압축
   - Excitation: 두 개의 FC 레이어로 채널별 가중치 생성
   - Scale: 원본 feature map에 가중치 적용
   - reduction ratio를 사용하여 파라미터 수 제어 (기본값: 16)

2. ResidualBlock에 SE 블록 통합
   - 각 Residual Block에 SE 블록 추가
   - Residual connection 후 SE 블록 적용
   - 채널별 중요도를 학습하여 표현력 향상

3. 네트워크 구조
   - 기본 구조는 DeepBaselineNetBN2Residual와 동일
   - 각 Residual Block 내부에 SE 블록이 추가됨
   - SE 블록은 residual connection 후에 적용

참고:
- ResNet 논문: "Deep Residual Learning for Image Recognition" (He et al., 2015)
- SE-Net 논문: "Squeeze-and-Excitation Networks" (Hu et al., 2018)
- SE 블록은 채널 간 의존성을 모델링하여 네트워크의 표현력을 향상시킴
- 적은 계산 비용으로 성능을 크게 향상시킬 수 있는 효율적인 방법
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation 블록
    
    구조:
    1. Squeeze: Global Average Pooling으로 공간 차원 압축
    2. Excitation: FC 레이어들로 채널별 가중치 생성
    3. Scale: 원본 feature map에 가중치를 곱하여 재조정
    
    Args:
        channels: 입력 채널 수
        reduction: 채널 축소 비율 (기본값=16)
    """
    
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        
        # 작은 채널 수를 고려한 reduction ratio 조정
        reduced_channels = max(1, channels // reduction)
        
        # Squeeze: Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Excitation: 두 개의 FC 레이어
        self.fc1 = nn.Linear(channels, reduced_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduced_channels, channels, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        # FC 레이어 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화 - 마지막 레이어는 0으로 초기화하여 학습 초기에 identity mapping"""
        # 첫 번째 FC 레이어: Kaiming 초기화
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        # 두 번째 FC 레이어: 0으로 초기화하여 학습 초기에 SE 블록이 identity mapping처럼 동작
        nn.init.zeros_(self.fc2.weight)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: 입력 텐서 [batch, channels, H, W]
            
        Returns:
            out: SE 블록을 통과한 텐서 [batch, channels, H, W]
        """
        b, c, _, _ = x.size()
        
        # Squeeze: Global Average Pooling
        y = self.avg_pool(x).view(b, c)
        
        # Excitation: 채널별 가중치 생성
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        
        # Scale: 원본 특징맵에 가중치 적용
        return x * y.expand_as(x)


class ResidualBlockSE(nn.Module):
    """
    ResNet 스타일의 Basic Residual Block with SE
    
    구조:
    - 입력 x
    - Main path: Conv -> BN -> ReLU -> Conv -> BN -> SE
    - Shortcut: identity (채널/크기 같음) 또는 projection (다름)
    - 출력: ReLU(main_path + shortcut)
    
    Args:
        in_channels: 입력 채널 수
        out_channels: 출력 채널 수
        stride: 첫 번째 conv의 stride (다운샘플링용, 기본값=1)
        reduction: SE 블록의 reduction ratio (기본값=16)
    """
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super(ResidualBlockSE, self).__init__()
        
        # Main path: 두 개의 Conv-BN 블록
        # 첫 번째 Conv-BN-ReLU 블록
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 두 번째 Conv-BN 블록 (ReLU는 residual connection 후에 적용)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # SE 블록 추가
        self.se = SEBlock(out_channels, reduction=reduction)
        
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
        
        # SE 블록 적용 (채널별 중요도 재조정)
        out = self.se(out)
        
        # Residual connection: F(x) + x
        out += identity
        
        # 최종 ReLU 활성화
        out = F.relu(out)
        
        return out


class DeepBaselineNetBN2ResidualSE(nn.Module):
    """
    DeepBaselineNetBN2Residual에 Squeeze-and-Excitation 블록을 추가한 네트워크
    
    구조:
    1. 초기 Conv-BN-ReLU (3 -> 64)
    2. Residual Block 1 with SE: 64 -> 64 (identity shortcut)
    3. Residual Block 2 with SE: 64 -> 128 (projection shortcut, stride=1)
    4. Residual Block 3 with SE: 128 -> 256 (projection shortcut, stride=1)
    5. Residual Block 4 with SE: 256 -> 256 (identity shortcut)
    6. Residual Block 5 with SE: 256 -> 512 (projection shortcut, stride=1)
    7. Fully Connected Layers (분류기)
    """
    
    def __init__(self, init_weights=False, se_reduction=16):
        super(DeepBaselineNetBN2ResidualSE, self).__init__()
        
        # 초기 feature extraction
        # 입력: 3채널 (RGB), 출력: 64채널
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual Blocks with SE
        # Block 1: 64 -> 64 (같은 채널, identity shortcut)
        self.res_block1 = ResidualBlockSE(64, 64, stride=1, reduction=se_reduction)
        
        # Block 2: 64 -> 128 (채널 증가, projection shortcut 필요)
        # stride=1로 유지하고 MaxPool로 다운샘플링 (원래 구조 유지)
        self.res_block2 = ResidualBlockSE(64, 128, stride=1, reduction=se_reduction)
        
        # Block 3: 128 -> 256 (채널 증가, projection shortcut 필요)
        self.res_block3 = ResidualBlockSE(128, 256, stride=1, reduction=se_reduction)
        
        # Block 4: 256 -> 256 (같은 채널, identity shortcut)
        self.res_block4 = ResidualBlockSE(256, 256, stride=1, reduction=se_reduction)
        
        # Block 5: 256 -> 512 (채널 증가, projection shortcut 필요)
        self.res_block5 = ResidualBlockSE(256, 512, stride=1, reduction=se_reduction)
        
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
        
        # Residual Block 1 with SE: 64 -> 64
        x = self.res_block1(x)
        
        # Residual Block 2 with SE: 64 -> 128
        x = self.res_block2(x)
        x = self.pool(x)  # 32x32 -> 16x16
        
        # Residual Block 3 with SE: 128 -> 256
        x = self.res_block3(x)
        
        # Residual Block 4 with SE: 256 -> 256
        x = self.res_block4(x)
        x = self.pool(x)  # 16x16 -> 8x8
        
        # Residual Block 5 with SE: 256 -> 512
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

