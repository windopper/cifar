# 모델별 최고 성능 조합 20 Epoch 기준

| 모델 | Augmentation | CutMix | Weight Init | Label Smoothing | 최고 Val Accuracy (%) |
|------|--------------|--------|-------------|-----------------|----------------------|
| deep_baseline2_bn_residual | ✅ | ✅ | ✅ | ❌ | 90.4 |
| deep_baseline2_bn_residual_preact | ✅ | ❌ | ✅ | ❌ | **90.84** |
| deep_baseline2_bn_residual_preact | ✅ | ✅ | ✅ | ❌ | 90.02 |

<details>
<summary><small>명령어 보기</small></summary>

- deep_baseline2_bn_residual (Augmentation + CutMix + Weight Init): `uv run main.py --optimizer adam --epochs 20 --lr 3e-4 --scheduler cosineannealinglr --net deep_baseline2_bn_residual --augment --cutmix --w-init`
- deep_baseline2_bn_residual_preact (Augmentation + Weight Init): `uv run main.py --optimizer adam --epochs 20 --lr 3e-4 --scheduler cosineannealinglr --net deep_baseline2_bn_residual_preact --augment --w-init`
- deep_baseline2_bn_residual_preact (Augmentation + CutMix + Weight Init): `uv run main.py --optimizer adam --epochs 20 --lr 3e-4 --scheduler cosineannealinglr --net deep_baseline2_bn_residual_preact --augment --cutmix --w-init`

</details>

**결과 요약:**
- 최고 성능: deep_baseline2_bn_residual_preact (Augmentation + Weight Init) = **90.84%**
- deep_baseline2_bn_residual_preact 모델에 Augmentation + Weight Init을 적용한 결과: **90.84%**
- 기본 deep_baseline2_bn_residual_preact (87.07%) 대비 +3.77%p 향상
- deep_baseline2_bn_residual (Augmentation + CutMix + Weight Init, 90.4%)보다 +0.44%p 향상
- deep_baseline2_bn_residual_preact (Augmentation + CutMix + Weight Init) = 90.02%
  - Augmentation + Weight Init (90.84%)보다 -0.82%p 낮음
  - CutMix 추가 시 성능이 약간 저하됨
- Pre-activation residual 모델이 Augmentation + Weight Init 조합에서 최고 성능 달성
- 모든 실험 중 최고 성능 달성


# 모델 아키텍처 비교 20 Epoch 기준

| 모델 | Batch Normalization | Residual Connection | Pre-activation | Squeeze-and-Excitation | 최고 Val Accuracy (%) |
|------|---------------------|---------------------|----------------|----------------------|
| deep_baseline | ❌ | ❌ | ❌ | ❌ | 81.35 |
| deep_baseline_bn | ✅ | ❌ | ❌ | ❌ | 87.21 |
| deep_baseline2_bn | ✅ | ❌ | ❌ | ❌ | 88.41 |
| deep_baseline2_bn_residual | ✅ | ✅ | ❌ | ❌ | **88.47** |
| deep_baseline2_bn_residual_se | ✅ | ✅ | ❌ | ✅ | 87.81 |
| deep_baseline2_bn_residual_preact | ✅ | ✅ | ✅ | ❌ | 87.07 |
| deep_baseline2_bn_resnext | ✅ | ✅ | ❌ | ❌ | 87.53 |
| deep_baseline3_bn | ✅ | ❌ | ❌ | ❌ | 87.9 |

<details>
<summary><small>명령어 보기</small></summary>
`uv run main.py --optimizer adam --epochs 20 --lr 3e-4 --scheduler cosineannealinglr --net [모델이름]`
</details>

![image](./comparison/model_comparison.png)

**결과 요약:**
- 최고 성능: deep_baseline2_bn_residual = **88.47%**
- Batch Normalization 추가 시 기본 대비 +5.86%p 향상 (deep_baseline_bn: 87.21%)
- deep_baseline2_bn이 deep_baseline_bn보다 +1.2%p 향상 (88.41% vs 87.21%)
- Residual connection 추가 시 추가 +0.06%p 향상 (deep_baseline2_bn_residual: 88.47%)
- Pre-activation residual은 이 실험에서 성능이 약간 저하됨 (87.07% vs 88.47%)
- deep_baseline2_bn_resnext (ResNeXt 구조)는 87.53%로 deep_baseline2_bn_residual_preact (87.07%)보다 약간 높은 성능
- deep_baseline3_bn은 deep_baseline2_bn보다 약간 낮은 성능 (87.9% vs 88.41%)

# 모델 비교 시에 사용하는 기본 설정
Optimizer: Adam
Epochs: 20
Learning Rate: 3e-4
Scheduler: Cosine Annealing LR
Net: deep_baseline

# Augmentation, CutMix, Weight Initialization, Label Smoothing 효과 비교 20 Epoch 기준
기본 모델: deep_baseline

| 설정 | Augmentation | CutMix | Weight Init | Label Smoothing | 최고 Val Accuracy (%) |
|------|--------------|--------|-------------|-----------------|----------------------|
| 기본 (모두 없음) | ❌ | ❌ | ❌ | ❌ | 81.35 |
| Weight Init만 | ❌ | ❌ | ✅ | ❌ | 83.29 |
| Augmentation만 | ✅ | ❌ | ❌ | ❌ | 82.84 |
| Augmentation + CutMix | ✅ | ✅ | ❌ | ❌ | 80.79 |
| Augmentation + Weight Init | ✅ | ❌ | ✅ | ❌ | 87.81 |
| Augmentation + Weight Init + Label Smoothing(0.05) | ✅ | ❌ | ✅ | ✅ (0.05) | **88.39** |
| Augmentation + CutMix + Weight Init | ✅ | ✅ | ✅ | ❌ | 86.83 |

![image](./comparison/augment_winit_comparison.png)

**결과 요약:**
- 최고 성능: Augmentation + Weight Init + Label Smoothing(0.05) = **88.39%**
- Weight Initialization만 추가해도 기본 대비 +1.94%p 향상
- Augmentation + Weight Init 조합이 가장 효과적 (87.81%)
- Label Smoothing(0.05) 추가 시 추가 +0.58%p 향상
- CutMix는 이 실험에서 오히려 성능을 약간 저하시킴 (Augmentation만: 82.84% vs Augmentation + CutMix: 80.79%)