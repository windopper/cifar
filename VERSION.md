# 모델 버전 변화 내역

이 문서는 `baseline` → `baseline_bn` → `deep_baseline_bn` → `deep_baseline2_bn` →  
`deep_baseline2_bn_residual` → `deep_baseline3_bn_residual` → `deep_baseline3_bn_residual_15` →  
`wideresnet16_8` → `residual_attention_92_32input`
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

## 7. `deep_baseline3_bn_residual_15`: Residual Block 수를 늘린 딥 베이스라인

- **구조 요약**
  - `deep_baseline3_bn_residual`를 기반으로, 각 stage의 residual block 수를 **3-3-6-3** 패턴으로 늘린 더 깊은 네트워크.
  - 초기 `Conv-BN-ReLU (3→64)` 이후,
    - Stage 1: Residual Block ×3 (64→64, stride=1)
    - Stage 2: Residual Block ×3 (64→128, 첫 블록 stride=2 + projection shortcut)
    - Stage 3: Residual Block ×6 (128→256, 첫 블록 stride=2 + projection shortcut)
    - Stage 4: Residual Block ×3 (256→512, 첫 블록 stride=2 + projection shortcut)
  - 마지막에는 **Global Average Pooling** 후, 간단한 `Linear(512→10)` 분류기 사용.
- **설계 의도**
  1. **깊이 증가를 통한 표현력 극대화**
     - `deep_baseline3_bn_residual`에서 이미 Residual + 깊은 채널 구성이 효과적이라는 것을 확인했기 때문에,
       residual block 수를 늘려 **표현력과 비선형성**을 추가 확보하는 것이 목표.
  2. **ResNet 스타일 설계를 유지한 “깊은 CIFAR-10 전용 백본”**
     - 표준 ResNet의 3-4-6-3 패턴을 참고하면서도, CIFAR-10 (32×32) 입력에 맞춘 구조로
       `Global Average Pooling + 단일 Linear head` 조합을 유지해 **구현 단순성**을 보존.
  3. **파라미터 수 대비 성능의 sweet spot 탐색**
     - Parameter 수는 약 **13.5M** 수준으로 증가하지만,
       wideresnet/ConvNeXt 계열과 비교했을 때 **합리적인 규모**에서 최고 성능을 노리는 포지션.
- **실험 결과 (`README.md`의 Final Comparison 2 참고)**
  - `deep_baseline3_bn_residual`: **94.65%** (10.4M)
  - `deep_baseline3_bn_residual_15`: **94.84%** (13.5M)
  - 동일한 학습 레시피(CosineAnnealingLR, AutoAugment 등)에서
    - **약 +0.2%p 수준의 성능 향상**을 얻으면서도
    - ResNet-계열로서 비교적 단순한 구조를 유지.
  - 이후 더 깊은 변형(`deep_baseline3_bn_residual_18`, 24.5M, **95.08%**)과의 비교를 통해,
    - **성능 vs 파라미터 수** 관점에서 `deep_baseline3_bn_residual_15`가 하나의 유의미한 지점이라는 것을 확인.

---

## 8. `wideresnet16_8`: 폭을 넓힌 표준 WideResNet으로의 전환

- **구조 요약**
  - 논문 *“Wide Residual Networks”* 계열의 **WideResNet-16-8** 구조를 CIFAR-10에 맞게 사용.
  - 채널 구성: `nChannels = [16, 16×8, 32×8, 64×8] = [16, 128, 256, 512]`.
  - 깊이 `depth=16` → 각 stage 당 BasicBlock 수 `n = (16-4)/6 = 2`.
  - 각 블록은 `BN-ReLU-Conv` 구조의 **pre-activation 스타일 Residual Block**이며,
    중간에 Dropout(dropRate=0.3)을 사용해 정규화.
  - 마지막 stage 출력에 `BatchNorm + ReLU` 후, **Global Average Pooling (8×8)**,  
    `Linear(64×8 → 10)` 분류기를 거치는 전형적인 WideResNet 헤드.
- **설계 의도**
  1. **표준 강력 베이스라인과의 직접 비교**
     - 커스텀 `deep_baseline*` 계열만이 아니라,
       CIFAR-10에서 널리 사용되는 **WideResNet-16-8**과 동일/유사 설정으로 비교하여
       - 아키텍처 자체의 한계
       - 학습 레시피(optimizer, scheduler, regularization)의 영향
       을 분리해서 보고자 함.
  2. **“폭(Width)”을 늘린 residual 네트워크의 장점 검증**
     - 깊이를 크게 늘리기보다는 채널 수(widen factor=8)를 크게 키운 구조가
       CIFAR-10에서 **표현력/학습 안정성/연산 효율** 측면에서 어떤 trade-off를 보이는지 확인.
  3. **강력한 학습 레시피와의 결합**
     - SGD + Nesterov, CosineAnnealingLR, Label Smoothing,  
       AutoAugment, ASAM, EMA, CIFAR-10 Normalize 등
       다양한 최신 기법을 조합해 **“표준 구조 + 강한 레시피”의 상한선**을 측정.
- **실험 결과 (`README.md`의 Final Comparison 2 참고)**
  - Parameter Count: 약 **10.9M**.
  - 기본 설정에서는 약 **95.22%** 수준의 최고 Val Accuracy.
  - 학습 레시피를 강화하면서,
    - SGD with Nesterov, LR 0.1: **95.89%**
    - SGD + Nesterov + ASAM(rho=2.0) + EMA + Label Smoothing 0.1 등 조합에서 **96%대** 도달.
  - 최종적으로,
    - `SGD + Nesterov, LR 0.1, Label Smoothing 0.1, Epoch 200, CIFAR-10 Normalize` 설정에서  
      **96.49%**(10.9M)까지 달성하여,
      - `deep_baseline3_bn_residual_18` (95.08%, 24.5M),
      - `convnext_v2_cifar_nano_k3` (최대 96.51%, 13.3M)
      와 비교했을 때
      **매우 경쟁력 있는 성능/파라미터 비율**을 보여주는 최종 WideResNet 계열 기준선으로 자리 잡음.

---

