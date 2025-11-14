#!/bin/bash

# ============================================================================
# 실행 방법
# ============================================================================
# 이 스크립트는 Scheduler 비교 실험을 자동으로 실행합니다.
#
# 실행 방법:
#   1. Git Bash 또는 Linux/Mac 터미널에서 실행:
#      bash scripts/run_scheduler_comparison.sh
#
#   2. 실행 권한 부여 후 직접 실행:
#      chmod +x scripts/run_scheduler_comparison.sh
#      ./scripts/run_scheduler_comparison.sh
#
#   3. Windows PowerShell에서 Git Bash를 통해 실행:
#      bash scripts/run_scheduler_comparison.sh
#
# 주의사항:
#   - 스크립트는 프로젝트 루트 디렉토리에서 실행해야 합니다.
#   - 모든 실험은 순차적으로 실행되며, 총 4개의 실험이 진행됩니다.
#   - 각 실험은 60 에포크를 실행하므로 전체 실행 시간이 오래 걸릴 수 있습니다.
#   - 하이퍼파라미터를 변경하려면 스크립트 상단의 변수들을 수정하세요.
# ============================================================================

# Scheduler 비교 실험
# 하이퍼파라미터 설정 (README.md 기준)
OPTIMIZER="adam"
EPOCHS=60
BATCH_SIZE=128
LR=3e-4
NET="deep_baseline_bn"
WEIGHT_INIT="--w-init"  # Weight Initialization: ✅

echo "=========================================="
echo "Scheduler 비교 실험 시작"
echo "=========================================="
echo "공통 설정:"
echo "  - Optimizer: $OPTIMIZER"
echo "  - Epochs: $EPOCHS"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Learning Rate: $LR"
echo "  - Net: $NET"
echo "  - Weight Initialization: ✅"
echo "=========================================="
echo ""

COUNTER=1
TOTAL=4

# 1. Cosine Annealing LR
echo "[$COUNTER/$TOTAL] Cosine Annealing LR"
echo "  - T_max: $EPOCHS (default)"
echo "  - eta_min: 0.0 (default)"
uv run main.py --optimizer $OPTIMIZER --epochs $EPOCHS --lr $LR --batch-size $BATCH_SIZE \
  --scheduler cosineannealinglr --scheduler-t-max $EPOCHS --net $NET $WEIGHT_INIT
echo ""
COUNTER=$((COUNTER+1))

# 2. One Cycle LR
echo "[$COUNTER/$TOTAL] One Cycle LR"
echo "  - max_lr: $(echo "$LR * 10" | bc) (default: lr * 10)"
uv run main.py --optimizer $OPTIMIZER --epochs $EPOCHS --lr $LR --batch-size $BATCH_SIZE \
  --scheduler onecyclelr --net $NET $WEIGHT_INIT
echo ""
COUNTER=$((COUNTER+1))

# 3. Exponential LR
echo "[$COUNTER/$TOTAL] Exponential LR"
echo "  - gamma: 0.95 (default)"
uv run main.py --optimizer $OPTIMIZER --epochs $EPOCHS --lr $LR --batch-size $BATCH_SIZE \
  --scheduler exponentiallr --scheduler-gamma 0.95 --net $NET $WEIGHT_INIT
echo ""
COUNTER=$((COUNTER+1))

# 4. ReduceLROnPlateau
echo "[$COUNTER/$TOTAL] ReduceLROnPlateau"
echo "  - factor: 0.1 (default)"
echo "  - patience: 5 (조정됨, 60 epoch 기준)"
echo "  - mode: min (default)"
uv run main.py --optimizer $OPTIMIZER --epochs $EPOCHS --lr $LR --batch-size $BATCH_SIZE \
  --scheduler reducelronplateau --scheduler-factor 0.1 --scheduler-patience 5 --scheduler-mode min \
  --net $NET $WEIGHT_INIT
echo ""
COUNTER=$((COUNTER+1))

echo "=========================================="
echo "Scheduler 비교 실험 완료"
echo "총 $TOTAL개의 실험이 완료되었습니다."
echo "=========================================="
echo ""
echo "결과 파일 위치:"
echo "  - outputs/"


