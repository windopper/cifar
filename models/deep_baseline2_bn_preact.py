"""
DeepBaselineNetBN2PreAct: DeepBaselineNetBN2에 Pre-activation을 적용한 버전

Pre-activation이란:
- 일반적인 Post-activation 구조: Conv -> BN -> ReLU
- Pre-activation 구조: BN -> ReLU -> Conv
- 활성화 함수(ReLU)를 컨볼루션 연산 이전에 적용하는 방식

설계 의도:
1. Pre-activation의 장점
   - 그래디언트 흐름 개선: 활성화 함수를 먼저 적용하여 그래디언트가 더 잘 전파됨
   - 학습 안정성 향상: 정규화된 입력을 컨볼루션에 전달하여 학습이 더 안정적
   - Identity mapping 최적화: ResNet 논문에서 제시된 pre-activation이 identity mapping을 더 잘 학습함

2. 구조적 차이
   - 첫 번째 레이어(conv1): 입력이 원본 이미지(3채널)이므로 Conv -> BN -> ReLU 유지
   - 이후 레이어(conv2~conv5): BN -> ReLU -> Conv 순서로 변경
   - 각 블록의 입력을 먼저 정규화하고 활성화한 후 컨볼루션 연산 수행

3. 기대 효과
   - 더 깊은 네트워크에서도 안정적인 학습 가능
   - 그래디언트 소실 문제 완화
   - 일반화 성능 향상

DeepBaselineNetBN2와의 차이점:
1. Forward pass 구조 변경
   - DeepBaselineNetBN2: Conv -> BN -> ReLU (post-activation)
   - DeepBaselineNetBN2PreAct: BN -> ReLU -> Conv (pre-activation, conv2~conv5)
   
2. BatchNorm 위치 변경
   - conv1: Conv -> BN -> ReLU (입력이 이미지이므로 동일)
   - conv2~conv5: 이전 레이어의 출력에 BN을 먼저 적용한 후 ReLU, Conv 순서로 실행
   - 각 컨볼루션 레이어의 입력 채널 수에 맞춰 BatchNorm 채널 수 설정
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepBaselineNetBN2PreAct(nn.Module):
    def __init__(self):
        super(DeepBaselineNetBN2PreAct, self).__init__()
        
        # 첫 번째 레이어: 입력이 이미지이므로 Conv -> BN -> ReLU 구조 유지
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Pre-activation을 위한 BatchNorm 레이어들 (이전 레이어 출력에 적용)
        self.bn2_pre = nn.BatchNorm2d(64)  # conv1 출력(64채널)에 적용
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        
        self.bn3_pre = nn.BatchNorm2d(128)  # conv2 출력(128채널)에 적용
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        
        self.bn4_pre = nn.BatchNorm2d(256)  # conv3 출력(256채널)에 적용
        self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
        
        self.bn5_pre = nn.BatchNorm2d(256)  # conv4 출력(256채널)에 적용
        self.conv5 = nn.Conv2d(256, 512, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(512 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        # 첫 번째 블록: Conv -> BN -> ReLU (입력이 이미지이므로)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Pre-activation 블록 2: BN -> ReLU -> Conv
        x = self.bn2_pre(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.pool(x)
        
        # Pre-activation 블록 3: BN -> ReLU -> Conv
        x = self.bn3_pre(x)
        x = F.relu(x)
        x = self.conv3(x)
        
        # Pre-activation 블록 4: BN -> ReLU -> Conv
        x = self.bn4_pre(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.pool(x)
        
        # Pre-activation 블록 5: BN -> ReLU -> Conv
        x = self.bn5_pre(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.pool(x)
        
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x

