#!/bin/bash

# ============================================================================
# 실행 방법
# ============================================================================
# 이 스크립트는 Augmentation 비교 실험을 자동으로 실행합니다.
#
# 실행 방법:
#   1. Git Bash 또는 Linux/Mac 터미널에서 실행:
#      bash scripts/run_augmentation_comparison_100epoch.sh
#
#   2. 실행 권한 부여 후 직접 실행:
#      chmod +x scripts/run_augmentation_comparison_100epoch.sh
#      ./scripts/run_augmentation_comparison_100epoch.sh
#
#   3. Windows PowerShell에서 Git Bash를 통해 실행:
#      bash scripts/run_augmentation_comparison_100epoch.sh
#
# 주의사항:
#   - 스크립트는 프로젝트 루트 디렉토리에서 실행해야 합니다.
#   - 모든 실험은 순차적으로 실행되며, 총 7개의 실험이 진행됩니다.
#   - 각 실험은 100 에포크를 실행하므로 전체 실행 시간이 오래 걸릴 수 있습니다.
#   - 하이퍼파라미터를 변경하려면 스크립트 상단의 변수들을 수정하세요.
#   - CutMix와 Mixup은 동시에 사용할 수 없습니다.
# ============================================================================

# Augmentation 비교 실험 (100 Epoch 기준)
# 하이퍼파라미터 설정
OPTIMIZER="adam"
EPOCHS=100
BATCH_SIZE=128
LR=3e-4
SCHEDULER="cosineannealinglr"
NET="deep_baseline_bn"
WEIGHT_INIT="--w-init"  # Weight Initialization: ✅

BASE_CMD="python cifar/main.py --optimizer $OPTIMIZER --epochs $EPOCHS --batch-size $BATCH_SIZE --lr $LR --scheduler $SCHEDULER --net $NET $WEIGHT_INIT"

echo "=========================================="
echo "Augmentation 비교 실험 시작 (100 Epoch)"
echo "=========================================="
echo ""

# 1. 기본 (모두 없음)
echo "[1/7] 기본 (모두 없음) - Augmentation ❌, CutMix ❌, Mixup ❌, AutoAugment ❌"
$BASE_CMD
echo ""

# 2. Augmentation만
echo "[2/7] Augmentation만 - Augmentation ✅, CutMix ❌, Mixup ❌, AutoAugment ❌"
$BASE_CMD --augment
echo ""

# 3. Augmentation + CutMix
echo "[3/7] Augmentation + CutMix - Augmentation ✅, CutMix ✅, Mixup ❌, AutoAugment ❌"
$BASE_CMD --augment --cutmix
echo ""

# 4. Augmentation + Mixup
echo "[4/7] Augmentation + Mixup - Augmentation ✅, CutMix ❌, Mixup ✅, AutoAugment ❌"
$BASE_CMD --augment --mixup
echo ""

# 5. Augmentation + AutoAugment
echo "[5/7] Augmentation + AutoAugment - Augmentation ✅, CutMix ❌, Mixup ❌, AutoAugment ✅"
$BASE_CMD --augment --autoaugment
echo ""

# 6. Augmentation + CutMix + AutoAugment
echo "[6/7] Augmentation + CutMix + AutoAugment - Augmentation ✅, CutMix ✅, Mixup ❌, AutoAugment ✅"
$BASE_CMD --augment --cutmix --autoaugment
echo ""

# 7. Augmentation + Mixup + AutoAugment
echo "[7/7] Augmentation + Mixup + AutoAugment - Augmentation ✅, CutMix ❌, Mixup ✅, AutoAugment ✅"
$BASE_CMD --augment --mixup --autoaugment
echo ""

echo "=========================================="
echo "Completed Augmentation Comparison Experiment"
echo "=========================================="

