# CIFAR-10 97.78%

## Installation
### Installing uv

#### Windows (PowerShell)
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

#### Linux/macOS
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Alternative: Using pip
```bash
pip install uv
```

### Setting up the project

Once uv is installed, you can install the project dependencies:

```bash
uv sync
```

## Train
| Model | Best Val Accuracy (%) | Parameter Count |
|------|------------|----------------------|
| wideresnet16_8 | **97.07** | 10.9 M |
| pyramidnet110_150 (200 Epochs) | **97.48** | 10.9 M |
| pyramidnet110_150 (400 Epochs) | **97.70** | 10.9 M |

```bash
uv run train.py --optimizer sgd --epochs 200 --lr 0.1 --batch-size 128 --scheduler cosineannealinglr --w-init --augment --autoaugment --nesterov --ema --sam --sam-rho 2.0 --sam-adaptive --label-smoothing 0.1 --net wideresnet16_8
uv run train.py --optimizer sgd --epochs 200 --lr 0.1 --batch-size 128 --scheduler cosineannealinglr --w-init --augment --autoaugment --nesterov --ema --sam --sam-rho 2.0 --sam-adaptive --label-smoothing 0.1 --net pyramidnet110_150 --shakedrop 0.5
uv run train.py --optimizer sgd --epochs 400 --lr 0.1 --batch-size 128 --scheduler cosineannealinglr --w-init --augment --autoaugment --nesterov --ema --sam --sam-rho 2.0 --sam-adaptive --label-smoothing 0.1 --net pyramidnet110_150 --shakedrop 0.5
```

### Stacking
| Model | Best Val Accuracy (%) | Parameter Count |
|------|------------|----------------------|
| pyramidnet110_150 (400 Epochs) + pyramidnet110_150 (200 Epochs) + wideresnet16_8 | **97.78** | 10.9 M |

```bash
uv run stacking.py --seed 250
```

## Inference
wideresnet16_8 (97.07%)
```bash
uv run inference.py --history outputs/wideresnet16_8_sgd_crossentropy_bs128_ep200_lr0.1_mom0.9_nesterov_schcosineannealinglr_tmax200_ls0.1_aug_autoaug_winit_ema_emad0.999_sam_samrho2.0_samadaptive_history.json
```

pyramidnet110_150 (200 Epochs) (97.48%)
```bash
uv run inference.py --history outputs/pyramidnet110_150_sgd_crossentropy_bs128_ep200_lr0.1_mom0.9_nesterov_schcosineannealinglr_tmax200_ls0.1_aug_autoaug_winit_ema_emad0.999_sam_samrho2.0_samadaptive_shakedrop_history.json
```

pyramidnet110_150 (400 Epochs) (97.70%)
```bash
uv run inference.py --history outputs/pyramidnet110_150_sgd_crossentropy_bs128_ep400_lr0.1_mom0.9_nesterov_schcosineannealinglr_tmax400_ls0.1_aug_autoaug_winit_ema_emad0.999_sam_samrho2.0_samadaptive_shakedrop_history.json
```
