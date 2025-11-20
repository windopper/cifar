"""
BaselineNetBN: BaselineNet에 BatchNorm 레이어를 추가한 버전

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

BaselineNet과의 차이점:
- 각 컨볼루션 레이어 뒤에 BatchNorm2d 레이어 추가
- Conv-BN-ReLU 순서로 구조 변경
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineNetBN(nn.Module):
    def __init__(self, init_weights=False):
        super(BaselineNetBN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
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
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool(F.relu(x))
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool(F.relu(x))
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

