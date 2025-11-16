"""
BaselineNetBN: DeepBaselineNetBN의 채널 수를 줄인 베이스라인 테스트 버전

설계 의도:
1. 내부 공변량 이동(Internal Covariate Shift) 감소
   - 각 레이어의 입력 분포가 학습 중에 변화하는 것을 BatchNorm으로 안정화
   - 이로 인해 학습이 더 안정적이고 빠르게 진행됨

2. 학습 안정성 향상
   - BatchNorm이 활성화 값을 정규화하여 그래디언트 폭발/소실 문제 완화
   - 더 큰 학습률 사용 가능하여 수렴 속도 향상

3. 정규화 효과
   - BatchNorm 자체가 약간의 정규화 효과를 제공하여 과적합 방지에 도움
   - 일반화 성능 향상 기대

4. 표준적인 Conv-BN-ReLU 구조 적용
   - 각 컨볼루션 레이어 뒤에 BatchNorm을 배치
   - ReLU 활성화 함수는 BatchNorm 이후에 적용
   - 이는 딥러닝에서 널리 사용되는 표준 패턴

DeepBaselineNetBN과의 차이점:
- 모든 컨볼루션 레이어의 채널 수를 절반으로 감소 (베이스라인 테스트 용도)
  - conv1: 3 -> 16 (원본: 3 -> 32)
  - conv2: 16 -> 32 (원본: 32 -> 64)
  - conv3: 32 -> 64 (원본: 64 -> 128)
  - conv4: 64 -> 64 (원본: 128 -> 128)
  - conv5: 64 -> 128 (원본: 128 -> 256)
- Fully-connected 레이어의 크기도 비례하여 감소
  - fc1: 128*4*4 -> 256 (원본: 256*4*4 -> 512)
  - fc2: 256 -> 128 (원본: 512 -> 256)
  - fc3: 128 -> 64 (원본: 256 -> 128)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineNetBN(nn.Module):
    def __init__(self, init_weights=False):
        super(BaselineNetBN, self).__init__()
        # Conv-BN-ReLU 블록들 (채널 수 절반으로 감소)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)
        
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
        # Conv-BN-ReLU 블록 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Conv-BN-ReLU 블록 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv-BN-ReLU 블록 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Conv-BN-ReLU 블록 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv-BN-ReLU 블록 5
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x

