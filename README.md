!git clone https://colab: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@github.com/windopper/cifar.git

# Scheduler 비교
Optimizer: Adam
Epochs: 60
Batch Size: 128
Learning Rate: 3e-4
Net: deep_baseline_bn
Weight Initialization: ✅

`uv run main.py --optimizer adam --epochs 200 --lr 3e-4 --batch-size 128 --scheduler onecyclelr --w-init --net baseline_bn`

| Scheduler | 최고 Val Accuracy (%) | 세부 사항 |
|------|------------|
| Cosine Annealing LR | 76.71 | -- |
| One Cycle LR | 87.25 | -- |
| One Cycle LR | 87.07 | Pct Start 0.2 |
| One Cycle LR | 86.57 | Pct Start 0.2, Final LR Ratio 0.07 |
| One Cycle LR | 85.57 | Pct Start 0.2, Epoch 20 |
| One Cycle LR | 85.6 | Pct Start 0.2, Final LR Ratio 0.07, Epoch 20 |
| One Cycle LR | 81.96 | Pct Start 0.2, Final LR Radio 0.07, Epoch 20, Batch Size 1024, Learning Rate: 3e-3 |
| Exponential LR | 73.43 | -- |
| ReduceLROnPlateau | 75.31 | -- |

![scheduler_comparison.png](./comparison/scheduler_comparison.png)

# Augmentation
Model: deep_baseline_bn
Optimizer: Adam
Epochs: 100
Batch Size: 128
Learning Rate: 3e-4
Scheduler: Cosine Annealing LR
Weight Initialization: ✅

Augmentation: RandomCrop(32, padding=4), RandomHorizontalFlip, RandomRotation(15)

| 설정 | Augmentation | CutMix | Mixup | AutoAugment | 최고 Val Accuracy (%) |
|------|--------------|--------|-------------|-----------------|----------------------|
| 기본 (모두 없음) | ❌ | ❌ | ❌ | ❌ | 78.43 |
| Augmentation | ✅ | ❌ | ❌ | ❌ | 90.01 |
| Augmentation + CutMix | ✅ | ✅ | ❌ | ❌ | 90.26 |
| Augmentation + CutMix (75% 에포크 시작) | ✅ | ✅ | ❌ | ❌ | 89.49 |
| Augmentation + CutMix (75% 에포크 시작) + OneCycleLR | ✅ | ✅ | ❌ | ❌ | 90.97 |
| Augmentation + Mixup | ✅ | ❌ | ✅ | ❌ | 89.85 |
| Augmentation + AutoAugment | ✅ | ❌ | ❌ | ✅ | **91.17** |
| Augmentation + CutMix + AutoAugment | ✅ | ✅ | ❌ | ✅ | 90.88 |
| Augmentation + Mixup + AutoAugment | ✅ | ❌ | ✅ | ✅ | 90.43 |

`python cifar/main.py --optimizer adam --epochs 100 --batch-size 128 --lr 3e-4 --scheduler cosineannealinglr --net deep_baseline_bn --w-init --augment --cutmix --cutmix-start-epoch-ratio 0.75`

![image](./comparison/augmentation_comparison_100epoch.png)

# Optimizer/Learning Rate Comparison
Model: deep_baseline_bn
Epochs: 40
Batch Size: 128
Scheduler: Cosine Annealing LR
Weight Initialization: ✅

| optimizer | Learning Rate | 최고 Val Accuracy (%)
|------|------------|----------------------|
| Adam | 0.01 | 84.44 |
| Adam | 0.001 | **84.76** |
| Adam | 0.0001 | 67.52 |
| AdamW | 0.01 | 82.67 |
| AdamW | 0.001 | 83.47 |
| AdamW | 0.0001 | 67.43 |
| SGD | 0.001 | 71.08 |
| SGD | 0.01 | 73.08 |
| Adagrad | 0.001 | 57.5 |
| Adagrad | 0.01 | 74.72 |
| RMSprop | 0.001 | 79.51 |
| RMSprop | 0.01 | 74.38 |

![image](./comparison/optimizer_lr_comparison.png)

<details>
<summary><small>명령어 보기</small></summary>
**기본 명령어:**
```bash
uv run main.py --optimizer [optimizer] --epochs 40 --lr [learning_rate] --batch-size 128 --scheduler cosineannealinglr --w-init --net deep_baseline_bn
```
</details>

# Regularization And Post-hoc Calibration
model: deep_baseline_bn
lr: 3e-4
batch size: 128
epoch: 60
scheduler: One Cycle LR
optimizer: Adam
Weight Initialization: ✅

| Label Smoothing | Temperature Scaling | 최고 Val Accuracy (%) | Model History |
|------|------------|----------------------|------------|
| ❌ | ❌ | 87.25 | [History](outputs/scheduler/deep_baseline_bn_adam_crossentropy_bs128_ep60_lr0.0003_mom0.9_schonecyclelr_winit_history.jsonson) |
| ✅ | ❌ | 87.12 | [History](outputs/regularization_calibration/deep_baseline_bn_adam_crossentropy_bs128_ep60_lr0.0003_mom0.9_schonecyclelr_ls0.05_winit_history.json) |
| ❌ | ✅ | 86.81 | [History](outputs/regularization_calibration/deep_baseline_bn_adam_crossentropy_bs128_ep60_lr0.0003_mom0.9_schonecyclelr_winit_calibrated_history.json) |
| ✅ | ✅ | -- | -- |

`python cifar/main.py --optimizer adam --epochs 60 --lr 3e-4 --batch-size 128 --scheduler onecyclelr --w-init --net deep_baseline_bn --calibrate


# Model Comparison
lr: 3e-4
batch size: 128
epoch: 60
scheduler: One Cycle LR
optimizer: Adam
Weight Initialization: ✅

`uv run main.py --optimizer adam --epochs 60 --lr 3e-4 --batch-size 128 --scheduler onecyclelr --w-init --net [모델이름]`

| Model | 최고 Val Accuracy (%) | Model History |
|------|------------|----------------------|
| deep_baseline_bn | 87.25 | [History](outputs/scheduler/deep_baseline_bn_adam_crossentropy_bs128_ep60_lr0.0003_mom0.9_schonecyclelr_winit_history.json)
| deep_baseline2_bn | 87.16 | [History](outputs/model_comparison/deep_baseline2_bn_adam_crossentropy_bs128_ep60_lr0.0003_mom0.9_schonecyclelr_winit_history.json)
| deep_baseline2_bn_residual | 88.73 | [History](outputs/model_comparison/deep_baseline2_bn_residual_adam_crossentropy_bs128_ep60_lr0.0003_mom0.9_schonecyclelr_winit_history.json)
| deep_baseline2_bn_residual_se | 87.88 | [History](outputs/model_comparison/deep_baseline2_bn_residual_se_adam_crossentropy_bs128_ep60_lr0.0003_mom0.9_schonecyclelr_winit_history.json)
| deep_baseline2_bn_residual_preact | 87.81 | [History](outputs/model_comparison/deep_baseline2_bn_residual_preact_adam_crossentropy_bs128_ep60_lr0.0003_mom0.9_schonecyclelr_winit_history.json)
| deep_baseline2_bn_resnext | 88.16 | [History](outputs/model_comparison/deep_baseline2_bn_resnext_adam_crossentropy_bs128_ep60_lr0.0003_mom0.9_schonecyclelr_winit_history.json)
| deep_baseline3_bn | 86.07 | [History](outputs/model_comparison/deep_baseline3_bn_adam_crossentropy_bs128_ep60_lr0.0003_mom0.9_schonecyclelr_winit_history.json)
| mxresnet56 | 87.57 | [History](outputs/model_comparison/mxresnet56_adam_crossentropy_bs128_ep60_lr0.0003_mom0.9_schonecyclelr_winit_history.json)
| dla | 87.3 | [History](outputs/model_comparison/dla_adam_crossentropy_bs128_ep60_lr0.0003_mom0.9_schonecyclelr_winit_history.json)
| resnext29_4x64d | 88.51 | [History](outputs/model_comparison/resnext29_4x64d_adam_crossentropy_bs128_ep60_lr0.0003_mom0.9_schonecyclelr_winit_history.json)
| deep_baseline2_bn_residual_grn | 81.54 | -- |
| deep_baseline3_bn_residual | **88.91** | -- |
| deep_baseline3_bn_residual_swish | 88.04 | -- |
| deep_baseline4_bn_residual | 88.67 | -- |
 
# Inference
## TTA
model: deep_baseline_bn
lr: 3e-4
batch size: 128
epoch: 60
scheduler: One Cycle LR
optimizer: Adam
Weight Initialization: ✅

| TTA | Ensemble | Temperature Scaling | Details | 최고 Val Accuracy (%) |
|------------|----------------------|----------------------|----------------------|------------|
| ❌ | ❌ | ❌ | -- | 87.25 |
| ✅ | ❌ | ❌ | -- | 88.39 |
| ❌ | ✅ | ❌ | deep_baseline_bn, deep_baseline2_bn, deep_baseline2_bn_residual | 90.27 |
| ✅ | ✅ | ❌ | deep_baseline_bn, deep_baseline2_bn, deep_baseline2_bn_residual | 90.84 |
| ❌ | ✅ | ❌ | deep_baseline_bn, deep_baseline2_bn, deep_baseline2_bn_residual, deep_baseline2_bn_residual_se, deep_baseline2_bn_resnext | 90.97 |
| ✅ | ✅ | ❌ | deep_baseline_bn, deep_baseline2_bn, deep_baseline2_bn_residual, deep_baseline2_bn_residual_se, deep_baseline2_bn_resnext | **91.06** |


# Final Comparison
lr: 3e-4
batch size: 128
epoch: 180
scheduler: One Cycle LR
optimizer: Adam
Weight Initialization: ✅
Augmentation: ✅
AutoAugment: ✅
Label Smoothing: 0.05

`python cifar/main.py --optimizer adam --epochs 180 --lr 3e-4 --batch-size 128 --scheduler onecyclelr --w-init --augment --autoaugment --net deep_baseline3_bn_residual`

| 모델 | 세부 사항 | 최고 Val Accuracy (%) |
|------|------------|----------------------|
| deep_baseline_bn | -- | 90.89 |
| deep_baseline_bn | Epoch 20, Scheduler Pct Start 0.2 | -- |
| deep_baseline_bn | Epoch 60 | 90.68 |
| deep_baseline_bn | Epoch 100 | 90.89 |
| deep_baseline_bn | Epoch 60, CutMix | 90.17 |
| deep_baseline_bn | Epoch 30 | 90.38 |
| deep_baseline_bn | Epoch 30, Batch Size 32 | 86.13 |
| deep_baseline_bn | Epoch 30, Batch Size 256 | 88.54 |
| deep_baseline2_bn_resnext | -- | 92.27 |
| deep_baseline2_bn_resnext | Epoch 60 | 91.86 |
| deep_baseline3_bn_residual | Label Smoothing 0 | 92.92 |


# Final Comparison 2
lr: 3e-4
batch size: 128
epoch: 100
scheduler: Cosine Annealing LR
optimizer: Adam
Weight Initialization: ✅

| 모델 | 세부 사항 | 최고 Val Accuracy (%) |
|------|------------|----------------------|
| deep_baseline3_bn_residual | -- | 94.65 |
| deep_baseline3_bn_residual_shakedrop | -- | 92.14 |
| deep_baseline3_bn_residual_mish | -- | 94.03 |
| deep_baseline3_bn_residual_dla_tree | -- | -- |
| deep_baseline3_bn_residual_group | -- | 93.22 |
| deep_baseline3_bn_residual_dla | -- | **94.96** |
| deep_baseline4_bn_residual | -- | 94.73 |

# BaseModel
lr: 0.001
batch size: 128
scheduler: CosineAnnealingLR
optimizer: Adam
weight init: ✅

| 모델 | Augmentation | CutMix | Label Smoothing | 최고 Val Accuracy (%) |
|------|------------|--------------|-----------------|----------------------|
| deep_baseline_bn | ✅ | ✅ | ✅ (0.05) | 91.99 |
| deep_baseline2_bn | ✅ | ✅ | ✅ (0.05) | 92.51 |
| deep_baseline2_bn_residual | ❌ | ❌ | ❌ | 대기 |
| deep_baseline2_bn_residual | ✅ | ✅ | ✅ (0.05) | 대기 |
| deep_baseline2_bn_residual_se | ❌ | ❌ | ❌ | 대기 |
| deep_baseline2_bn_residual_se | ✅ | ✅ | ✅ (0.05) | 92.9 |
| deep_baseline2_bn_residual_preact | ❌ | ❌ | ❌ | 87.36 |
| deep_baseline2_bn_residual_preact | ✅ | ✅ | ✅ (0.05) | 92.38 |
| deep_baseline2_bn_resnext | ❌ | ❌ | ❌ | 85.46 |
| deep_baseline2_bn_resnext | ✅ | ✅ | ✅ (0.05) | **92.97** |
| deep_baseline3_bn | ✅ | ✅ | ✅ (0.05) | 90.73 |
| mxresnet56 | ✅ | ✅ | ✅ (0.05) | 92.04 |
 

<details>
<summary><small>명령어 보기</small></summary>
`uv run main.py --optimizer adam --epochs 100 --lr 0.001 --batch-size 128 --scheduler cosineannealinglr --augment --cutmix --label-smoothing 0.05 --w-init --net [모델이름]`

`uv run main.py --optimizer adam --epochs 100 --lr 0.001 --batch-size 128 --scheduler cosineannealinglr --net [모델이름]`
</details>

![image](./comparison/model_comparison_100_epoch.png)

**하이퍼파라미터:**
- Epochs: 100
- Optimizer: Adam
- Learning Rate: 0.001
- Batch Size: 128
- Momentum: 0.9
- Scheduler: CosineAnnealingLR (T_max: 100, eta_min: 0.0)
- Label Smoothing: 0.05
- Data Augmentation: ✅
- CutMix: ✅
- Weight Initialization: ✅
- Seed: 42

**결과 요약:**
- 최고 성능: **91.99%** (100 에포크)
- 20 에포크 기준 최고 성능 (88.39%) 대비 +3.60%p 향상
- Augmentation + CutMix + Weight Init + Label Smoothing 조합으로 최고 성능 달성
- 100 에포크까지 학습하여 모델의 최대 성능 확인

