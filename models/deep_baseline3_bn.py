"""
DeepBaselineNetBN3: DeepBaselineNetBN2를 기반으로 한 확장 버전

설계 의도:
1. 더 깊은 네트워크 구조
   - 컨볼루션 레이어를 5개에서 8개로 확장
   - 채널 구성: 64, 64, 128, 128, 256, 256, 512, 512
   - 각 채널 크기마다 2개의 컨볼루션 레이어를 쌓아 표현력 향상

2. 단순화된 분류기
   - Fully-connected 레이어를 한 층으로 단순화
   - 과적합 위험 감소 및 모델 복잡도 감소

3. Batch Normalization 유지
   - 모든 컨볼루션 레이어에 BatchNorm 적용
   - 학습 안정성 및 수렴 속도 향상

4. 다양한 커널 사이즈 적용 (Multi-scale Feature Extraction)
   - 동일한 채널의 컨볼루션 레이어에 서로 다른 커널 사이즈 사용
   - conv1: 3x3 (공간 특징 추출)
   - conv2: 1x1 (채널 믹싱 및 비선형성 추가)
   - 이 패턴을 각 블록에 반복 적용

기존 DeepBaselineNetBN2와의 차이점:
1. 컨볼루션 레이어 확장
   - conv1: 3 → 64 (3x3 커널)
   - conv2: 64 → 64 (1x1 커널)
   - conv3: 64 → 128 (3x3 커널)
   - conv4: 128 → 128 (1x1 커널)
   - conv5: 128 → 256 (3x3 커널)
   - conv6: 256 → 256 (1x1 커널)
   - conv7: 256 → 512 (3x3 커널)
   - conv8: 512 → 512 (1x1 커널)

2. Fully-connected 레이어 단순화
   - 기존: fc1(512*4*4 → 512) → fc2(512 → 256) → fc3(256 → 128) → fc4(128 → 10)
   - 변경: fc(512*2*2 → 10) (단일 레이어)

3. Pooling 구조
   - conv2, conv4, conv6, conv8 뒤에 MaxPool2d 적용
   - 최종 feature map 크기: 2x2

다양한 커널 사이즈의 이점:
1. 다중 스케일 특징 추출 (Multi-scale Feature Extraction)
   - 3x3 커널: 공간적 지역 특징 (에지, 텍스처 등) 추출
   - 1x1 커널: 채널 간 상호작용 및 비선형성 증가

2. 효율적인 특징 표현
   - 1x1 컨볼루션은 채널 수는 유지하면서 추가 비선형성 제공
   - 계산 비용은 낮으면서 표현력 향상

3. Inception 스타일의 효과
   - 서로 다른 수용 영역(receptive field)을 가진 특징을 결합
   - 더 풍부한 특징 표현 가능

4. 정규화 효과
   - 다양한 커널 사이즈로 인한 암묵적 정규화 효과
   - 과적합 위험 감소
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepBaselineNetBN3(nn.Module):
    def __init__(self, init_weights=False):
        super(DeepBaselineNetBN3, self).__init__()
        
        # 블록 1: 64 채널 (3x3 → 1x1)
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)  # 3x3 커널: 공간 특징 추출
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 64, 1)  # 1x1 커널: 채널 믹싱
        self.bn2 = nn.BatchNorm2d(64)
        
        # 블록 2: 128 채널 (3x3 → 1x1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)  # 3x3 커널: 공간 특징 추출
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, 1)  # 1x1 커널: 채널 믹싱
        self.bn4 = nn.BatchNorm2d(128)
        
        # 블록 3: 256 채널 (3x3 → 1x1)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)  # 3x3 커널: 공간 특징 추출
        self.bn5 = nn.BatchNorm2d(256)
        
        self.conv6 = nn.Conv2d(256, 256, 1)  # 1x1 커널: 채널 믹싱
        self.bn6 = nn.BatchNorm2d(256)
        
        # 블록 4: 512 채널 (3x3 → 1x1)
        self.conv7 = nn.Conv2d(256, 512, 3, padding=1)  # 3x3 커널: 공간 특징 추출
        self.bn7 = nn.BatchNorm2d(512)
        
        self.conv8 = nn.Conv2d(512, 512, 1)  # 1x1 커널: 채널 믹싱
        self.bn8 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc = nn.Linear(512 * 2 * 2, 10)
        
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
        # Conv-BN-ReLU 블록 1-2 (64 채널): 3x3 → 1x1
        x = self.conv1(x)  # 3x3 커널: 공간 특징 추출
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)  # 1x1 커널: 채널 믹싱
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv-BN-ReLU 블록 3-4 (128 채널): 3x3 → 1x1
        x = self.conv3(x)  # 3x3 커널: 공간 특징 추출
        x = self.bn3(x)
        x = F.relu(x)
        
        x = self.conv4(x)  # 1x1 커널: 채널 믹싱
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv-BN-ReLU 블록 5-6 (256 채널): 3x3 → 1x1
        x = self.conv5(x)  # 3x3 커널: 공간 특징 추출
        x = self.bn5(x)
        x = F.relu(x)
        
        x = self.conv6(x)  # 1x1 커널: 채널 믹싱
        x = self.bn6(x)
        x = F.relu(x)
        x = self.pool(x)
        
        # Conv-BN-ReLU 블록 7-8 (512 채널): 3x3 → 1x1
        x = self.conv7(x)  # 3x3 커널: 공간 특징 추출
        x = self.bn7(x)
        x = F.relu(x)
        
        x = self.conv8(x)  # 1x1 커널: 채널 믹싱
        x = self.bn8(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = torch.flatten(x, 1)
        
        x = self.fc(x)
        
        return x

