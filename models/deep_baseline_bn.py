"""
DeepBaselineNetBN: DeepBaselineNet에 Batch Normalization을 추가한 개선 버전

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

기존 DeepBaselineNet과의 차이점:
- 모든 컨볼루션 레이어(conv1~conv5) 뒤에 BatchNorm2d 레이어 추가
- Forward pass에서 Conv -> BN -> ReLU 순서로 실행
- Fully-connected 레이어에는 BatchNorm을 적용하지 않음 (일반적인 관행)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepBaselineNetBN(nn.Module):
    def __init__(self):
        super(DeepBaselineNetBN, self).__init__()
        # Conv-BN-ReLU 블록들
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)

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

