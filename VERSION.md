# 모델 버전 변화 내역

이 문서는 `baseline` → `baseline_bn` → `deep_baseline_bn` → `deep_baseline2_bn` →  
`deep_baseline2_bn_residual` → `deep_baseline3_bn_residual` → `deep_baseline3_bn_residual_15`  
→ `residual_attention_92_32input`
로 이어지는 모델 구조 변화 과정을 정리하고, 각 단계에서 어떤 설계 의도와 실험 결과를 통해
다음 버전을 선택하게 되었는지를 상세히 기록한 문서입니다.

---

## 1. `baseline`: 가장 단순한 CNN 기준선

- **구조 요약**
  - 고전적인 LeNet 스타일의 단순 CNN 구조.
  - `Conv(3→6, 5×5)` → `ReLU` → `MaxPool`  
    `Conv(6→16, 5×5)` → `ReLU` → `MaxPool`  
    `FC(16×5×5 → 120 → 84 → 10)`
- **설계 의도**
  - CIFAR-10에 대해 **가장 단순한 기준선(baseline)을 확보**하는 것이 1차 목표.
  - 복잡한 기법(배치 정규화, 잔차 연결, 깊은 네트워크 등)을 도입하기 전에,
    - 순수 CNN만으로 어느 정도까지 성능이 나오는지
    - 학습 속도, 과적합 양상, 최적화 난이도 등을 먼저 파악하기 위함.
- **관찰 & 한계**
  - 비교적 빠르게 수렴하지만, 검증 정확도는 70% 후반대 수준에 머무르며 상위 실험들과 비교했을 때 **명확한 성능 한계**가 존재.
  - 깊이가 얕고, 채널 수가 작아서 **표현력(representational power)**이 부족.
  - Batch Normalization이 없고, Fully Connected 레이어 비중이 커
    - **학습이 불안정**하고
    - **과적합**이 일찍 발생하는 경향.
  - 이 결과를 바탕으로, 보다 **깊고 정규화가 잘 된 네트워크**가 필요하다고 판단하여 다음 단계로 진행.

---

## 2. `baseline_bn`: 얕지만 BN을 도입한 기준선

- **구조 요약**
  - `BaselineNet`을 확장한 형태로, 다섯 개의 `Conv-BN-ReLU` 블록과 4개의 FC 레이어로 구성.
  - `Conv(3→16→32→64→64→128)` + 각 단계에 `BatchNorm2d` 적용, 중간에 `MaxPool`.
  - `FC(128×4×4 → 256 → 128 → 64 → 10)`.
- **설계 의도**
  1. **내부 공변량 이동(Internal Covariate Shift) 감소**
     - 학습 중 각 레이어 입력 분포가 계속 변하는 문제를 BatchNorm으로 완화.
     - 보다 **안정적이고 빠른 학습**을 기대.
  2. **학습 안정성 향상 & 더 큰 학습률 사용**
     - 활성값을 정규화함으로써 **그래디언트 폭발/소실**을 줄이고,
     - 상대적으로 **큰 학습률**을 사용해도 수렴이 잘 되는지 확인.
  3. **정규화 효과로 인한 일반화 성능 향상**
     - BatchNorm 자체가 **약한 정규화 효과**를 제공해 과적합을 완화.
  4. **표준적인 Conv-BN-ReLU 패턴 적용**
     - 이후 더 깊은 구조(`deep_baseline_*`)로 자연스럽게 확장하기 위한 기반 구조 정립.
- **실험 결과 및 인사이트**
  - `README.md`의 스케줄러/옵티마이저/증강 실험에서 대부분의 기준 모델로 `deep_baseline_bn`이 사용되지만,
    같은 설계 철학(Conv-BN-ReLU, Kaiming init)을 공유하는 `baseline_bn` 또한
    단순 `baseline`에 비해 **명확한 성능 향상과 학습 안정성 개선**을 보여줌.
  - **BatchNorm 도입만으로도 기준 성능이 크게 개선**된다는 점을 확인했고,
    이 구조를 보다 깊고 넓게 확장한 `deep_baseline_bn`을 메인 베이스라인으로 채택하기로 결정.

---

## 3. `deep_baseline_bn`: 깊이와 채널 수를 확장한 표준 베이스라인

- **구조 요약**
  - `baseline_bn`보다 채널 수를 대폭 늘린 깊은 CNN.
  - `Conv-BN-ReLU` 5개 블록:
    - `3→32→64→128→128→256` 채널,
    - 중간에 `MaxPool`을 세 번 적용하여 공간 해상도 축소.
  - `FC(256×4×4 → 512 → 256 → 128 → 10)`.
- **설계 의도**
  1. **표현력 증가**
     - 채널 수와 깊이를 늘려 CIFAR-10의 복잡한 패턴을 더 잘 모델링.
  2. **BatchNorm을 활용한 안정적인 깊은 네트워크 학습**
     - 깊어진 만큼 gradient 흐름과 학습 안정성 문제가 커질 수 있으므로,
       Conv마다 BatchNorm을 붙여 **깊이를 키워도 학습이 잘 되도록 설계**.
  3. **“합리적인 기준 모델” 확보**
     - 이후 실험(스케줄러, 옵티마이저, 증강, 레이블 스무딩, TTA 등)의
       **공통 베이스라인 네트워크**로 사용하기 위한 구조.
- **실험 결과 (요약)**
  - `Scheduler 비교`, `Optimizer/LR 비교`, `Augmentation`, `Regularization/Calibration` 등 대부분의 실험에서
    `deep_baseline_bn`이 기준 모델로 사용됨.
  - 예시:
    - Scheduler 비교(OneCycleLR 기준): 약 **87.25%** 수준의 최고 Val Acc.
    - Augmentation 실험에서 AutoAugment, CutMix 등을 적용한 조합으로 **91% 초반**까지 도달.
    - Optimizer/LR 비교에서는 Adam(1e-3) + CosineAnnealingLR 조합이 강력한 성능을 보임.
  - 이를 통해
    - **적절한 깊이 + BN + Kaiming Init** 조합이 CIFAR-10에서 매우 강력한 베이스라인이라는 것을 확인.
    - 하지만, ResNet 계열이나 더 깊은 모델과 비교하면 여전히 약간의 성능 갭이 존재.
  - 이 성능 갭을 줄이기 위해 **더 깊은 채널 구성 및 Residual 구조** 탐색을 진행하기로 함.

---

## 4. `deep_baseline2_bn`: 채널 수를 크게 늘린 딥 베이스라인

- **구조 요약**
  - `deep_baseline_bn`의 구조를 유지하면서, 채널 수를 전반적으로 2배 수준으로 확장.
  - 컨볼루션 채널:
    - `conv1: 3→64`
    - `conv2: 64→128`
    - `conv3: 128→256`
    - `conv4: 256→256`
    - `conv5: 256→512`
  - `FC(512×4×4 → 512 → 256 → 128 → 10)`.
- **설계 의도**
  1. **표현력 극대화**
     - `deep_baseline_bn`에서 이미 BN + 깊은 구조의 이점을 확인했으므로,
       **채널 수를 늘려 capacity를 극대화**하면 성능이 더 올라가는지 검증.
  2. **ResNet 계열과의 공정한 비교를 위한 채널 설정**
     - `64-128-256-512` 채널 구성은 ResNet-18/34 계열에서 흔히 쓰이는 설정이며,
       이후 Residual 구조를 붙이기에 자연스러운 형태.
- **실험 결과 (모델 비교 표 참고)**
  - `deep_baseline2_bn`는 `deep_baseline_bn`와 비슷한 수준 혹은 약간 더 나은 성능(예: 87.16% vs 87.25%)을 보이지만,
    **명확한 큰 폭의 향상은 아니었다**는 점이 중요.
  - 파라미터 수와 연산량은 상당히 늘었는데 성능 이득은 제한적이어서,
    - 단순히 채널만 늘리는 방식은 **효율성이 떨어진다**는 결론에 도달.
  - 이 결과를 계기로 **채널 수 증가 + Residual Connection**을 결합한 설계를 시도하게 됨.

---

## 5. `deep_baseline2_bn_residual`: ResNet 스타일 잔차 연결 도입

- **구조 요약**
  - `deep_baseline2_bn`의 채널 구성을 유지하면서 모든 메인 블록을 **Residual Block**으로 교체.
  - 각 Residual Block:
    - `Conv-BN-ReLU → Conv-BN` + `Skip Connection` (identity 또는 1×1 Conv projection).
    - 출력에 `ReLU(F(x) + x)` 적용.
  - 블록 구성:
    - 초기 `Conv-BN-ReLU (3→64)`
    - `ResBlock 1: 64→64`
    - `ResBlock 2: 64→128` (+ MaxPool)
    - `ResBlock 3: 128→256`
    - `ResBlock 4: 256→256` (+ MaxPool)
    - `ResBlock 5: 256→512` (+ MaxPool)
  - 이후 `FC(512×4×4 → 512 → 256 → 128 → 10)`.
- **설계 의도**
  1. **Residual Learning을 통한 깊은 네트워크 안정 학습**
     - `Deep Residual Learning for Image Recognition (He et al., 2015)`에서 제안된 방식처럼,
       깊은 네트워크에서도 **gradient가 잘 흐르도록** skip connection을 도입.
  2. **Identity mapping을 쉽게 학습**
     - 각 블록이 `F(x)` 대신 `F(x) + x`를 학습하므로,
       - 필요 시 identity에 가까운 매핑을 쉽게 학습
       - 불필요한 복잡한 변환을 강제하지 않음.
  3. **채널 증가 + Residual 구조의 결합 효과 검증**
     - 단순히 채널만 늘렸던 `deep_baseline2_bn`의 한계를 개선하고,
       같은 채널 구성에서 Residual을 추가했을 때 **성능 향상이 얼마나 되는지** 측정.
- **실험 결과 (모델 비교 표 참고)**
  - Model Comparison (60 epoch, OneCycleLR 기준):
    - `deep_baseline_bn` : 87.25%
    - `deep_baseline2_bn` : 87.16%
    - `deep_baseline2_bn_residual` : **88.73%**
  - **Residual 도입만으로 약 +1.5%p 수준의 성능 향상**을 달성.
  - ResNet 스타일 구조가 단순 채널 증가보다 훨씬 효율적임을 확인했고,
    이후의 실험과 최종 모델 설계에서도 **“Residual + BN + 충분한 채널” 조합을 기본 전제**로 사용하게 됨.

---

## 6. `deep_baseline3_bn_residual`: Residual + Dropout + 정교한 분류기

- **구조 요약**
  - `deep_baseline2_bn_residual`를 기반으로 하되, 분류기 부분에 **Dropout을 추가**하고 구조를 약간 재구성한 버전.
  - 백본(Feature Extractor):
    - 초기 `Conv-BN-ReLU (3→64)`
    - `ResBlock 1: 64→64`
    - `ResBlock 2: 64→128` (+ MaxPool)
    - `ResBlock 3: 128→256`
    - `ResBlock 4: 256→256` (+ MaxPool)
    - `ResBlock 5: 256→512` (+ MaxPool)
  - 분류기(`classifier`):
    - `Linear(512×4×4 → 512) → ReLU → Dropout(0.1)`
    - `Linear(512 → 256) → ReLU → Dropout(0.1)`
    - `Linear(256 → 10)`.
- **설계 의도**
  1. **과적합 억제**
     - 앞선 실험에서 강한 augmentation 및 CutMix, AutoAugment 등을 적용했음에도,
       깊고 폭이 넓은 네트워크에서는 여전히 **분류기 부분에서 과적합**이 관찰됨.
     - 분류기에 적절한 Dropout을 도입해 **representation은 깊게 유지하면서도 일반화 성능을 높이는 것**이 목표.
  2. **최종 실험용 주력 백본 확보**
     - TTA, 앙상블, 레이블 스무딩, AutoAugment, CutMix 등
       다양한 기법을 얹기 위한 **최종 베이스 모델**로 사용.
- **실험 결과**
  - `Final Comparison` (180 epoch, OneCycleLR, AutoAugment, Label Smoothing 등 조합):
    - `deep_baseline3_bn_residual` : **92.92%** (Label Smoothing 0)
  - `Final Comparison 2` (100 epoch, CosineAnnealingLR, AutoAugment 등):
    - `deep_baseline3_bn_residual` : **94.65%**
  - 이전 세대(`deep_baseline2_bn_residual`, ResNeXt, DLA 등)와 비교했을 때
    **가장 높은 단일 모델 성능 중 하나**를 기록.
  - 이 결과를 통해,
    - Residual + 깊은 채널 구성 + Dropout 분류기 + 강한 augmentation 조합이
      CIFAR-10에서 **상당히 강력한 성능을 내는 조합**임을 확인.
  - 다만, 여전히 다른 특화 구조들(DLA, ResNeXt 계열)과의 미세한 성능 차이를 줄이기 위해
    **백본 내부의 feature aggregation 방식**을 개선할 여지가 있다고 판단.
- **변형 실험**
  - ShakeDrop을 각 Residual Block에 적용한 `deep_baseline3_bn_residual_shakedrop`을 추가 도입.
  - 초기 블록에서는 drop 확률을 낮게, 후반 블록에서는 높게 주어 학습 안정성과 일반화를 동시에 노림.
  - 정규화 효과 검증을 위해 Model Comparison / Final Comparison 2 설정에 추후 기록 예정.

---

## 7. `deep_baseline3_bn_residual_dla`: Deep Layer Aggregation 도입

- **구조 요약**
  - `deep_baseline3_bn_residual`의 백본을 유지하되,  
    **Deep Layer Aggregation(DLA)** 아이디어를 도입해 여러 깊이의 feature를 집계하는 버전.
  - 백본:
    - 초기 `Conv-BN-ReLU (3→64)`
    - `ResBlock 1: 64→64` → feature `f1` (32×32)
    - `ResBlock 2: 64→128` → `MaxPool` → feature `f2` (16×16)
    - `ResBlock 3: 128→256` → feature `f3` (16×16)
    - `ResBlock 4: 256→256` → `MaxPool` → feature `f4` (8×8)
    - `ResBlock 5: 256→512` → `MaxPool` → feature `f5` (4×4)
  - **DeepLayerAggregator**
    - 입력: `[f1, f2, f3, f4, f5]` (채널: `[64, 128, 256, 256, 512]`).
    - 각 feature를 `1×1 Conv + BN + ReLU`로 **512채널**로 투영.
    - `AdaptiveAvgPool2d`로 모두 4×4 해상도로 맞춘 뒤,
      순차적으로 더해가며 최종 집계 feature `f_agg (512×4×4)` 생성.
  - 분류기:
    - `deep_baseline3_bn_residual`의 분류기와 동일 (`512×4×4 → 512 → 256 → 10` + Dropout).
- **설계 의도**
  1. **다양한 깊이/해상도의 정보 집약**
     - 기존 모델은 **마지막 stage의 feature만** 사용했기 때문에,
       얕은 stage의 **로컬/텍스처 정보**가 충분히 활용되지 못하는 한계가 있음.
     - DLA 아이디어를 도입해,  
       얕은 layer의 세밀한 정보 + 깊은 layer의 추상적 정보 를 **하나의 feature로 집계**.
  2. **원본 구조와의 호환성 유지**
     - 기존 `deep_baseline3_bn_residual`의 block 구성과 MaxPool 위치는 그대로 유지하여,
       - 파라미터 수와 연산량 증가를 **최소화**하면서도
       - feature aggregation 방식만 개선하도록 설계.
     - 분류기 입력 차원도 동일(512×4×4)로 유지하여, 
       실험 설정/하이퍼파라미터를 그대로 재사용 가능.
  3. **DLA 논문 아이디어의 단순화 적용**
     - 원 논문은 훨씬 복잡한 트리 구조의 aggregation을 사용하지만,
       여기서는 “여러 깊이의 feature를 단일 표현으로 집계한다”는 핵심 아이디어만
       **CIFAR-10용 커스텀 백본에 맞게 단순화**하여 적용.
- **실험 결과**
  - `Final Comparison 2` (100 epoch, CosineAnnealingLR, AutoAugment, Label Smoothing, CutMix 등):
    - `deep_baseline3_bn_residual` : 94.65%
    - `deep_baseline3_bn_residual_dla` : **94.96%**
  - 약 **+0.3%p**의 절대 성능 향상이지만,
    - 이미 94% 중반대의 높은 구간에서 얻은 추가 이득이기 때문에
    - **의미 있는 개선**으로 해석할 수 있음.
  - 또한 DLA 도입으로 인해,
    - 초기/중간 layer들이 학습하는 feature가 단순 보조 역할을 넘어
      최종 결정에 직접 기여하게 되어,
    - 시각화나 feature 분석 측면에서도 더 풍부한 정보를 제공할 가능성이 큼.

---

## 8. 요약: 왜 최종적으로 `deep_baseline3_bn_residual_dla`인가?

- **단계별 교훈 정리**
  - `baseline`  
    → 아주 단순한 CNN은 학습은 쉽지만 성능 상한이 낮음.
  - `baseline_bn`  
    → BatchNorm 도입만으로도 학습 안정성과 성능이 크게 개선.
  - `deep_baseline_bn`  
    → 충분한 깊이와 채널을 가진 Conv-BN-ReLU 구조가 강력한 베이스라인이 됨.
  - `deep_baseline2_bn`  
    → 채널 수를 더 늘리면 표현력은 증가하지만, 효율성이 떨어지고 성능 이득이 제한적.
  - `deep_baseline2_bn_residual`  
    → Residual Connection 도입으로 **효율적인 깊은 학습**이 가능해지고, 성능이 크게 상승.
  - `deep_baseline3_bn_residual`  
    → Dropout이 포함된 분류기와 튜닝된 백본으로 CIFAR-10 상위권 성능 달성.
  - `deep_baseline3_bn_residual_dla`  
    → 여러 깊이의 feature를 집계하는 DLA 아이디어를 더해,  
      **기존 백본 구조를 크게 바꾸지 않고도 최종 성능을 한 단계 더 끌어올림**.

- **최종 선택 이유**
  - `deep_baseline3_bn_residual_dla`는
    - Conv-BN-ReLU + Residual + 충분한 채널 + Dropout + Deep Layer Aggregation 을 모두 활용하면서도,
    - 학습 안정성, 구현 복잡도, 연산량 사이에서 **실용적인 균형**을 달성한 모델.
  - 실험적으로도 `Final Comparison 2`에서
    - 동일한 학습 설정 하에 `deep_baseline3_bn_residual` 대비 **더 높은 Val Accuracy(94.96%)**를 기록.
  - 따라서 본 프로젝트에서는
    - CIFAR-10 최종 단일 모델 기준으로 **`deep_baseline3_bn_residual_dla`를 최종 구조**로 채택하였고,
    - 향후 추가 실험(예: 다른 데이터셋, 더 강한 정규화/증강, 지식 증류 등)도 이 모델을 기준으로 확장하는 것을 기본 전략으로 삼는다.

