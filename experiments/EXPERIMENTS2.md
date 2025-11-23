# Baseline
Optimizer: Adam
Epochs: 20
Batch Size: 128
Learning Rate: 3e-4
Net: baseline

`uv run main.py --optimizer adam --epochs 20 --batch-size 128 --lr 3e-4 --net baseline`

| 세부사항 | 최고 Val Accuracy (%) |
|------|------------|
| -- | 59.48 |  
| + Weight Initialization | 61.25 |

# Deep Baseline
Optimizer: Adam
Epochs: 20
Batch Size: 128
Learning Rate: 3e-4
Net: deep_baseline
Weight Initialization: ✅

`uv run main.py --optimizer adam --epochs 20 --batch-size 32 --lr 3e-4 --net deep_baseline_bn --w-init`

| 세부사항 | 최고 Val Accuracy (%) |
|------|------------|
| -- | 76.22 |
| + Weight Initialization | 77.63 |
| + Cosine Annealing LR | 78.81 |
| + Batch Normalization | 71.93 | 


Batch Size: 32
| 세부사항 | 최고 Val Accuracy (%) |
|--|--|
| -- | 77.27 |
| + Weight Initialization | 80.52 |
| + Cosine Annealing LR | 82.05 |
| + Batch Normalization | 82.7 | 

# Scheduler 비교
Optimizer: Adam
Epochs: 60
Batch Size: 128
Learning Rate: 3e-4
Net: deep_baseline_bn
Weight Initialization: ✅

`uv run main.py --optimizer adam --epochs 200 --lr 3e-4 --batch-size 128 --scheduler onecyclelr --w-init --net baseline_bn`

| Scheduler | 최고 Val Accuracy (%) | 세부 사항 |
|------|------------|----------------------|
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
| Augmentation + Cutout | ✅ | ❌ | ❌ | ❌ | 90.11  |
| Augmentation + Cutout + AutoAugment | ✅ | ❌ | ❌ | ❌ | 90.52  |
| Augmentation + Cutout (CutLength 8) + AutoAugment  | ✅ | ❌ | ❌ | ✅ | 89.86 |
| Augmentation + CutMix + AutoAugment | ✅ | ✅ | ❌ | ✅ | 90.88 |
| Augmentation + Mixup + AutoAugment | ✅ | ❌ | ✅ | ✅ | 90.43 |

`python cifar/main.py --optimizer adam --epochs 100 --batch-size 128 --lr 3e-4 --scheduler cosineannealinglr --net deep_baseline_bn --w-init --augment --cutmix --cutmix-start-epoch-ratio 0.75`

`uv run main.py --optimizer adam --epochs 100 --batch-size 128 --lr 3e-4 --scheduler cosineannealinglr --net deep_baseline_bn --w-init --augment --cutout`

`uv run main.py --optimizer adam --epochs 100 --batch-size 128 --lr 3e-4 --scheduler cosineannealinglr --net deep_baseline_bn --w-init --augment --cutout --cutout-length 8`

`uv run main.py --optimizer adam --epochs 100 --batch-size 128 --lr 3e-4 --scheduler cosineannealinglr --net deep_baseline_bn --w-init --augment --cutout --autoaugment`

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
| SGD with Nestrov | 0.01 | -- |
| SGD with Nestrov | 0.1 | -- |
| Adagrad | 0.001 | 57.5 |
| Adagrad | 0.01 | 74.72 |
| RMSprop | 0.001 | 79.51 |
| RMSprop | 0.01 | 74.38 |

![image](./comparison/optimizer_lr_comparison.png)

<details>
<summary><small>명령어 보기</small></summary>
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
Augmentation: ✅
AutoAugment: ✅

`python cifar/main.py --optimizer adamw --epochs 100 --lr 2e-3 --batch-size 128 --weight-decay 0.05 --scheduler cosineannealingwarmuprestarts --w-init --augment --autoaugment --sam --sam-rho 2.0 --sam-adaptive --ema --scheduler-warmup-epochs 5 --net convnext_v2_cifar_nano_k3`

`python cifar/main.py --optimizer adam --epochs 100 --lr 3e-4 --batch-size 128 --scheduler cosineannealinglr --w-init --augment --autoaugment --net wideresnet16_8`

| 모델 | 세부 사항 | 최고 Val Accuracy (%) | Parameter Count |
|------|------------|----------------------|----------------------|
| baseline_bn | -- | 86.75 | 0.7 M |
| baseline_bn | Warmup 10 Epochs | -- | 0.7 M |
| baseline_bn | Normalize | 86.96 | 0.7 M |
| baseline_bn | Criterion SupCon | 81.46 | 0.7 M |
| baseline_bn | Label Smoothing 0.1 | 86.57 | 0.7 M |
| baseline_bn | CutMix | 85.39 | 0.7 M |

| 모델 | 세부 사항 | 최고 Val Accuracy (%) | Parameter Count |
|------|------------|----------------------|----------------------|
| deep_baseline3_bn_residual_gap_gmp_s3_f64_128_256_b5 | -- | 92.82 | 7.8 M |
| deep_baseline3_bn_residual | -- | **94.65** | 10.4 M |
| deep_baseline3_bn_residual_bottleneck | -- | 89.69 | 10.3 M |
| deep_baseline3_bn_residual_convnext_stride | -- | -- | 10.8 M |
| deep_baseline3_bn_residual_15 | -- | **94.84** | 13.5 M |
| deep_baseline3_bn_residual_15 | AdamW | 94.39 | 13.5 M |
| deep_baseline3_bn_residual_18 | -- | **95.08** | 24.5 M |
| deep_baseline3_bn_residual_15_convnext | -- | 83.76 | 11.0 M |
| deep_baseline3_bn_residual_15_convnext | SGD with Nestrov, Learning Rate 0.1 | -- | 11.0 M |
| deep_baseline3_bn_residual_15_convnext | AdamW, Weight Decay 5e-5 | 92.82 | 11.0 M | 
| deep_baseline3_bn_residual_15_convnext | AdamW, Weight Decay 5e-3 | 93.19 | 11.0 M |
| deep_baseline3_bn_residual_15_convnext_ln_classifier | AdamW | 91.29 | 11.0 M |
| deep_baseline3_bn_residual_15_convnext_ln_classifier_stem | AdamW | -- | 12.4 M |
| deep_baseline3_bn_residual_15_ln | -- | 92.40 | 13.5 M |
| deep_baseline3_bn_residual_15_attention_tiny | -- | 93.48 | 13.0 M |

| 모델 | 세부 사항 | 최고 Val Accuracy (%) | Parameter Count |
|------|------------|----------------------|----------------------|
| wideresnet16_8 | -- | 95.08 | 10.9 M |
| wideresnet16_8 | SGD with Nestrov, Learning Rate 0.1 | 95.89 | 10.9 M |
| + Remove First ReLU | SGD with Nestrov, Learning Rate 0.1 | 94.78 | 10.9 M |
| + Last Batch Norm | SGD with Nestrov, Learning Rate 0.1 | 95.08 | 10.9 M |
| wideresnet16_8 | SGD with Nestrov, Learning Rate 0.1, ShakeDrop 1 | -- | 10.9 M |
| wideresnet16_8 | SGD with Nestrov, ASAM (rho=2.0), Learning Rate 0.1 | 96.34 | 10.9 M |
| wideresnet16_8 | SGD with Nestrov, ASAM (rho=2.0), Learning Rate 0.1, EMA | 96.4 | 10.9 M |
| wideresnet16_8 | SGD with Nestrov, ASAM (rho=2.0), Learning Rate 0.1, EMA, Label Smoothing 0.1 | 96.86 | 10.9 M |
| wideresnet16_8 | + TTA 2 | 96.70 | 10.9 M |
| wideresnet16_8 | SGD with Nestrov, ASAM (rho=2.0), Learning Rate 0.025, Label Smoothing 0.1, Batch Size 32 | 96.43 | 10.9 M |
| wideresnet16_8 | SGD with Nestrov, ASAM (rho=2.0), Learning Rate 0.025, EMA, Label Smoothing 0.1, Batch Size 32 | 96.47 | 10.9 M |
| wideresnet16_8 | SGD with Nestrov, ASAM (rho=2.0), Learning Rate 0.1, EMA, Label Smoothing 0.1, Use CIFAR-10 Normalize | 96.61 | 10.9 M |
| wideresnet16_8 | SGD with Nestrov, Learning Rate 0.1, Label Smoothing 0.1, Epoch 200, Use CIFAR-10 Normalize | 96.49 | 10.9 M |
| wideresnet16_8 | SGD with Nestrov, ASAM (rho=2.0), Learning Rate 0.1, EMA, Label Smoothing 0.1, Epoch 200 | **97.07** | 10.9 M |
|---|---|---|---|
| wideresnet16_8 | SGD with Nestrov, ASAM (rho=2.0), Learning Rate 0.1, Focal Loss Adaptive (gamma=3.0)| 95.53 | 10.9 M |
| wideresnet16_8 | SGD with Nestrov, ASAM (rho=2.0), Learning Rate 0.1, EMA, Focal Loss Adaptive (gamma=3.0)| 95.65 | 10.9 M |
| wideresnet16_8 | SGD with Nestrov, ASAM (rho=2.0), Learning Rate 0.1, EMA, Label Smoothing 0.1, Weighted CE | 96.56 | 10.9 M |
| wideresnet16_8 | + TTA 2 | 96.74 | 10.9 M |

`python cifar/main.py --optimizer sgd --epochs 100 --lr 0.1 --batch-size 128 --scheduler cosineannealinglr --w-init --augment --autoaugment --nesterov --ema --sam --sam-rho 2.0 --sam-adaptive --label-smoothing 0.1 --weighted-ce --net wideresnet16_8`

`python cifar/main.py --optimizer sgd --epochs 100 --criterion focal_loss_adaptive --lr 0.1 --batch-size 128 --scheduler cosineannealinglr --w-init --augment --autoaugment --nesterov --ema --sam --sam-rho 2.0 --sam-adaptive --label-smoothing 0.1 --grad-norm 2.0 --net wideresnet16_8`

```bash
python cifar/main.py --optimizer sgd --epochs 100 --lr 0.025 --batch-size 32 --scheduler cosineannealinglr --w-init --augment --autoaugment --nesterov --ema --sam --sam-rho 2.0 --sam-adaptive --label-smoothing 0.1 --net wideresnet16_8
```

| 모델 | 세부 사항 | 최고 Val Accuracy (%) | Parameter Count |
|------|------------|----------------------|----------------------|
| pyramidnet110_150 | SGD with Nestrov, Learning Rate 0.1 | 96.82 | 10.9 M |
| pyramidnet110_150 | SGD with Nestrov, Learning Rate 0.1, ShakeDrop 1 | 92.61 | 10.9 M |
| pyramidnet110_150 | SGD with Nestrov, Learning Rate 0.1, ShakeDrop 0.5 | 96.90 | 10.9 M |
| pyramidnet110_150 | SGD with Nestrov, ASAM (rho=2.0), Learning Rate 0.1, ShakeDrop 0.5, Label Smoothing 0.1 | 97.0 | 10.9 M |
| pyramidnet110_150 | SGD with Nestrov, ASAM (rho=2.0), Learning Rate 0.1, EMA, Label Smoothing 0.1, Epoch 200, ShakeDrop 0.5 | 97.48 | 10.9 M |
| pyramidnet110_150 | SGD with Nestrov, ASAM (rho=2.0), Learning Rate 0.1, EMA, Label Smoothing 0.1, Epoch 400, ShakeDrop 0.5 | **97.70** | 10.9 M |
| pyramidnet110_150 | + TTA 4 | 97.43 | 10.9 M |
| pyramidnet110_150 | + TTA 2 | 97.67 | 10.9 M |

`python cifar/main.py --optimizer sgd --epochs 100 --lr 0.1 --batch-size 128 --scheduler cosineannealinglr --w-init --augment --autoaugment --nesterov --net pyramidnet110_150`

`python cifar/main.py --optimizer sgd --epochs 100 --lr 0.1 --batch-size 128 --scheduler cosineannealinglr --w-init --augment --autoaugment --nesterov --net pyramidnet110_150 --shakedrop 1`

`python cifar/main.py --optimizer sgd --epochs 200 --lr 0.1 --batch-size 128 --scheduler cosineannealinglr --w-init --augment --autoaugment --nesterov --ema --sam --sam-rho 2.0 --sam-adaptive --label-smoothing 0.1 --net pyramidnet110_150 --shakedrop 0.5`

`python cifar/main.py --optimizer sgd --epochs 100 --lr 0.1 --batch-size 128 --scheduler cosineannealinglr --w-init --augment --autoaugment --nesterov --sam --sam-rho 2.0 --sam-adaptive --label-smoothing 0.1 --net pyramidnet110_150 --shakedrop 0.5`


```bash
python cifar/main.py --optimizer sgd --epochs 400 --lr 0.1 --batch-size 128 --scheduler cosineannealinglr --w-init --augment --autoaugment --nesterov --ema --sam --sam-rho 2.0 --sam-adaptive --label-smoothing 0.1 --net pyramidnet272_200_bottleneck --shakedrop 0.5
```


| 모델 | 세부 사항 | 최고 Val Accuracy (%) | Parameter Count |
|------|------------|----------------------|----------------------|
| convnext_v2_cifar_nano | AdamW, Learning Rate 2e-3, Weight Decay 0.05, Warmup Epoch 5 | 94.51 | 13.3 M |
| convnext_v2_cifar_nano_k3 | AdamW, Learning Rate 2e-3, Weight Decay 0.05, Warmup Epoch 5 | 95.48 | 13.3 M |
| convnext_v2_cifar_nano_k3 | AdamW, ASAM (rho=2.0), Learning Rate 2e-3, Weight Decay 0.05, Warmup Epoch 5 | 95.93 | 13.3 M |
| convnext_v2_cifar_nano_k3 | AdamW, ASAM (rho=2.0), Learning Rate 2e-3, Weight Decay 0.05, Warmup Epoch 5, EMA | 95.95 | 13.3 M |
| convnext_v2_cifar_nano_k3 | AdamW, Warmup Epoch 5, CutMix, Mixup, Label Smoothing 0.1, Epoch 200 | **96.51** | 13.3 M |

`python cifar/main.py --optimizer adamw --epochs 100 --lr 2e-3 --batch-size 128 --weight-decay 0.05 --scheduler cosineannealingwarmuprestarts --w-init --augment --autoaugment --sam --sam-rho 2.0 --sam-adaptive --ema --scheduler-warmup-epochs 5 --net convnext_v2_cifar_nano_k3`

| 모델 | 세부 사항 | 최고 Val Accuracy (%) | Parameter Count |
|------|------------|----------------------|----------------------|
| residual_attention_92_32input_tiny | -- | 95.43 | 14.5 M |
| residual_attention_92_32input | Epoch 200 | -- | 160.9 M |   
| residual_attention_92_32input_gelu_tiny | -- | **95.47** | 14.5 M |
| residual_attention_92_32input_gelu_medium | -- | 95.36 | 25.5 M |
| residual_attention_92_32input_gelu_tiny | AdamW | 94.99 | 14.5 M |
| residual_attention_92_32input_se_tiny | -- | 94.96 | 14.6 M |
| residual_attention_92_32input_preact_tiny | -- | 94.82 | 14.5 M |
| residual_attention_92_32input_gelu_tiny_dla_tiny | -- | 94.05 | 12.6 M |
| residual_attention_92_32input_gelu_tiny_dla | Batch Size 32 | -- | 55.1 M |

| 모델 | 세부 사항 | 최고 Val Accuracy (%) | Parameter Count |
|------|------------|----------------------|----------------------|
| deep_baseline3_bn_residual_deep | -- | 94.85 | 32.7 M |
| deep_baseline3_bn_residual_wide | -- | -- | 41.6 M |
| deep_baseline3_bn_residual_4x | Epoch 800 TMax 200 | 95.46 | 41.6 M |
| deep_baseline3_bn_residual_shakedrop | -- | 92.14 | 10.4 M |
| deep_baseline3_bn_residual_mish | -- | 94.03 | -- |
| deep_baseline3_bn_residual_dla_tree | -- | **95.59** | 42.6 M |
| deep_baseline3_bn_residual_group | -- | 93.22 | -- |
| deep_baseline3_bn_residual_dla | -- | 94.96 | 11 M |
| deep_baseline4_bn_residual | -- | 94.73 | 11.1 M |
| DLA | -- | 94.9 | 16.2M |
| rdnet_tiny | -- | -- | 22.8 M |

![image](./comparison/final_comparison_2.png)

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

# Ensemble
| 방법 | 최고 Val Accuracy (%) |
|------|------------|
| Hard Voting | 97.71% |
| Soft Voting | 97.71% |
| 0.5 0.25 0.25 | 97.75% |

## MetaLearner
| 세부사항 | 최고 Val Accuracy (%) |
|------|------------|
| seed=250 | 97.78% |

