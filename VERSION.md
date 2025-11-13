# 모델 버전 변화 내역

## v1.0: BaselineNet (`baseline.py`)

### 기본 구조
- **컨볼루션 레이어**: 2개
  - `conv1`: 3 → 6 채널, 5×5 커널
  - `conv2`: 6 → 16 채널, 5×5 커널
- **풀링**: MaxPool2d(2, 2)
- **완전연결 레이어**: 3개
  - `fc1`: 16×5×5 → 120
  - `fc2`: 120 → 84
  - `fc3`: 84 → 10 (CIFAR-10 클래스 수)
- **활성화 함수**: ReLU
- **Forward 구조**: Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → FC → ReLU → FC → ReLU → FC

---

## v2.0: DeepBaselineNet (`deep_baseline.py`)

### 주요 변경사항
- **네트워크 깊이 증가**: 2개 → 5개 컨볼루션 레이어
- **채널 수 증가**: 더 많은 특징 추출
- **컨볼루션 레이어**: 5개
  - `conv1`: 3 → 32 채널, 3×3 커널, padding=1
  - `conv2`: 32 → 64 채널, 3×3 커널, padding=1
  - `conv3`: 64 → 128 채널, 3×3 커널, padding=1
  - `conv4`: 128 → 128 채널, 3×3 커널, padding=1
  - `conv5`: 128 → 256 채널, 3×3 커널, padding=1
- **완전연결 레이어**: 4개
  - `fc1`: 256×4×4 → 512
  - `fc2`: 512 → 256
  - `fc3`: 256 → 128
  - `fc4`: 128 → 10
- **풀링 위치**: conv2, conv4, conv5 이후
- **Forward 구조**: Conv → ReLU → Conv → ReLU → Pool → Conv → ReLU → Conv → ReLU → Pool → Conv → ReLU → Pool → Flatten → FC → ReLU → FC → ReLU → FC → ReLU → FC

---

## v3.0: DeepBaselineNetBN (`deep_baseline_bn.py`)

### 주요 변경사항
- **Batch Normalization 추가**: 모든 컨볼루션 레이어에 BatchNorm2d 적용
- **학습 안정성 향상**: 내부 공변량 이동(Internal Covariate Shift) 감소
- **표준 구조 적용**: Conv-BN-ReLU 블록 패턴 사용

### 구조 변경
- 각 컨볼루션 레이어 뒤에 BatchNorm2d 레이어 추가:
  - `conv1` → `bn1` (32 채널)
  - `conv2` → `bn2` (64 채널)
  - `conv3` → `bn3` (128 채널)
  - `conv4` → `bn4` (128 채널)
  - `conv5` → `bn5` (256 채널)
- **Forward 구조 변경**: Conv → BN → ReLU 순서로 실행
- **Fully-connected 레이어**: BatchNorm 미적용 (일반적인 관행)

### 설계 의도
1. 내부 공변량 이동 감소로 학습 안정화 및 속도 향상
2. 그래디언트 폭발/소실 문제 완화
3. 더 큰 학습률 사용 가능
4. 약간의 정규화 효과로 과적합 방지

---

## v4.0: DeepBaselineNetBN2 (`deep_baseline2_bn.py`)

### 주요 변경사항
- **채널 수 증가**: 네트워크 용량 확대
- **더 깊은 특징 추출**: 각 레이어의 채널 수 2배 증가

### 구조 변경
- **컨볼루션 레이어 채널 수 변경**:
  - `conv1`: 32 → **64** 채널 (2배 증가)
  - `conv2`: 64 → **128** 채널 (2배 증가)
  - `conv3`: 128 → **256** 채널 (2배 증가)
  - `conv4`: 128 → **256** 채널 (2배 증가)
  - `conv5`: 256 → **512** 채널 (2배 증가)
- **BatchNorm 채널 수 변경**: 각 BN 레이어의 채널 수도 함께 증가
- **완전연결 레이어 입력 크기 변경**:
  - `fc1`: 256×4×4 → **512×4×4** (conv5 출력 채널 증가에 맞춤)

### 설계 의도
- 더 많은 특징 맵으로 표현력 향상
- 더 복잡한 패턴 학습 가능
- Batch Normalization과 함께 사용하여 안정적인 학습 유지

---

## v5.0: DeepBaselineNetBN2Residual (`deep_baseline2_bn_residual.py`)

### 주요 변경사항
- **Residual Connection 도입**: ResNet 스타일의 잔차 학습 구조 적용
- **ResidualBlock 클래스 추가**: 재사용 가능한 잔차 블록 구현
- **그래디언트 흐름 개선**: Skip connection을 통한 직접적인 그래디언트 전달

### 구조 변경
- **초기 Feature Extraction**:
  - `conv1`: 3 → 64 채널, 3×3 커널, padding=1
  - `bn1`: BatchNorm2d(64)
  
- **Residual Blocks** (각 블록은 2개의 Conv-BN 레이어로 구성):
  - `res_block1`: 64 → 64 (identity shortcut)
  - `res_block2`: 64 → 128 (projection shortcut)
  - `res_block3`: 128 → 256 (projection shortcut)
  - `res_block4`: 256 → 256 (identity shortcut)
  - `res_block5`: 256 → 512 (projection shortcut)

- **ResidualBlock 구조**:
  - Main path: Conv → BN → ReLU → Conv → BN
  - Shortcut: Identity (채널/크기 같음) 또는 Projection (1×1 Conv + BN)
  - 출력: ReLU(main_path + shortcut)

- **Forward 구조 변경**:
  - 초기 Conv-BN-ReLU
  - 각 Residual Block 적용
  - MaxPool은 기존 위치 유지 (res_block2, res_block4, res_block5 이후)

### 설계 의도
1. **잔차 학습 (Residual Learning)**:
   - Skip connection으로 그래디언트 직접 전달
   - 그래디언트 소실 문제 완화
   - 깊은 네트워크에서도 효과적인 학습

2. **표현력 향상**:
   - F(x) + x 형태로 identity mapping 쉽게 학습
   - 필요시 더 복잡한 변환 학습 가능

3. **표준 ResNet 구조 적용**:
   - BasicBlock 스타일의 residual block 사용
   - 채널 수가 다른 경우 1×1 conv를 사용한 projection shortcut

### 참고
- ResNet 논문: "Deep Residual Learning for Image Recognition" (He et al., 2015)
- Residual block은 깊은 네트워크에서 그래디언트 소실 문제를 해결하는 핵심 기술

---

## 변화 요약

| 버전 | 모델명 | 주요 특징 | 핵심 개선사항 |
|------|--------|----------|--------------|
| v1.0 | BaselineNet | 기본 구조 | 2 Conv + 3 FC |
| v2.0 | DeepBaselineNet | 네트워크 깊이 증가 | 5 Conv + 4 FC, 채널 수 증가 |
| v3.0 | DeepBaselineNetBN | Batch Normalization | 학습 안정성 향상 |
| v4.0 | DeepBaselineNetBN2 | 채널 수 증가 | 표현력 향상 |
| v5.0 | DeepBaselineNetBN2Residual | Residual Connection | 그래디언트 흐름 개선, 깊은 네트워크 학습 가능 |

---

