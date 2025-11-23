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

# 모델 아키텍처 비교 20 Epoch 기준

| 모델 | Batch Normalization | Residual Connection | Pre-activation | Squeeze-and-Excitation | 최고 Val Accuracy (%) |
|------|---------------------|---------------------|----------------|----------------------|------------------|
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