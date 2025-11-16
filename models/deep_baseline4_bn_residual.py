"""
ResNet-18: 표준 ResNet-18 아키텍처 구현

설계 의도:
1. 표준 ResNet-18 구조
   - ResNet 논문의 정확한 구조를 따름
   - BasicBlock을 사용한 18층 네트워크
   - 각 레이어에 2개의 residual block 사용

2. 네트워크 구조:
   - 초기 Conv-BN-ReLU (3 -> 64, stride=1, padding=1)
   - Layer 1: 64 채널, 2개의 residual block (stride=1)
   - Layer 2: 128 채널, 2개의 residual block (stride=2)
   - Layer 3: 256 채널, 2개의 residual block (stride=2)
   - Layer 4: 512 채널, 2개의 residual block (stride=2)
   - Average Pooling (4x4)
   - Fully Connected Layer (512 -> 10)

3. CIFAR-10에 최적화:
   - 작은 이미지 크기에 맞춰 초기 conv를 3x3, stride=1로 설정
   - 마지막에 4x4 average pooling 사용

참고:
- ResNet 논문: "Deep Residual Learning for Image Recognition" (He et al., 2015)
- ResNet-18: BasicBlock을 사용하여 [2, 2, 2, 2] 구조
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    ResNet의 Basic Residual Block
    
    구조:
    - 입력 x
    - Main path: Conv -> BN -> ReLU -> Conv -> BN
    - Shortcut: identity (채널/크기 같음) 또는 projection (다름)
    - 출력: ReLU(main_path + shortcut)
    
    Args:
        in_planes: 입력 채널 수
        planes: 출력 채널 수
        stride: 첫 번째 conv의 stride (다운샘플링용, 기본값=1)
    """
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        # Main path: 두 개의 Conv-BN 블록
        # 첫 번째 Conv-BN-ReLU 블록
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # 두 번째 Conv-BN 블록 (ReLU는 residual connection 후에 적용)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # Shortcut connection
        # 채널 수가 다르거나 stride가 1이 아닌 경우 projection 필요
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            # 1x1 conv를 사용하여 채널 수와 공간 크기 조정
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: 입력 텐서 [batch, in_planes, H, W]
            
        Returns:
            out: 출력 텐서 [batch, planes, H', W']
        """
        # Identity 또는 projection shortcut 저장
        identity = self.shortcut(x)
        
        # Main path: 첫 번째 Conv-BN-ReLU
        out = F.relu(self.bn1(self.conv1(x)))
        
        # Main path: 두 번째 Conv-BN
        out = self.bn2(self.conv2(out))
        
        # Residual connection: F(x) + x
        out += identity
        
        # 최종 ReLU 활성화
        out = F.relu(out)
        
        return out


class ResNet18(nn.Module):
    """
    표준 ResNet-18 아키텍처
    
    구조:
    1. 초기 Conv-BN-ReLU (3 -> 64, stride=1, padding=1)
    2. Layer 1: 64 채널, 2개의 BasicBlock (stride=1)
    3. Layer 2: 128 채널, 2개의 BasicBlock (stride=2)
    4. Layer 3: 256 채널, 2개의 BasicBlock (stride=2)
    5. Layer 4: 512 채널, 2개의 BasicBlock (stride=2)
    6. Average Pooling (4x4)
    7. Fully Connected Layer (512 -> 10)
    
    총 8개의 BasicBlock 사용 (각 레이어마다 2개씩)
    """
    
    def __init__(self, num_classes=10, init_weights=False):
        super(ResNet18, self).__init__()
        self.in_planes = 64
        
        # 초기 feature extraction
        # 입력: 3채널 (RGB), 출력: 64채널
        # CIFAR-10에 맞춰 3x3 conv, stride=1, padding=1 사용
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet-18 레이어 구성: [2, 2, 2, 2]
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        # 분류기
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)
        
        if init_weights:
            self._initialize_weights()
    
    def _make_layer(self, block, planes, num_blocks, stride):
        """
        레이어 생성 헬퍼 함수
        
        Args:
            block: 사용할 블록 타입 (BasicBlock)
            planes: 출력 채널 수
            num_blocks: 블록 개수
            stride: 첫 번째 블록의 stride
            
        Returns:
            레이어를 구성하는 Sequential 모듈
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
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
        out = F.relu(self.bn1(self.conv1(x)))
        
        # ResNet 레이어들
        out = self.layer1(out)  # 64 채널, 32x32
        out = self.layer2(out)  # 128 채널, 16x16 (stride=2)
        out = self.layer3(out)  # 256 채널, 8x8 (stride=2)
        out = self.layer4(out)  # 512 채널, 4x4 (stride=2)
        
        # Average Pooling: [batch, 512, 4, 4] -> [batch, 512, 1, 1]
        out = F.avg_pool2d(out, 4)
        
        # Flatten: [batch, 512, 1, 1] -> [batch, 512]
        out = out.view(out.size(0), -1)
        
        # Fully Connected Layer
        out = self.linear(out)
        
        return out

