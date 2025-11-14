#!/bin/bash

# ============================================================================
# 실행 방법
# ============================================================================
# 이 스크립트는 Optimizer별 최적 Learning Rate 비교 실험을 자동으로 실행합니다.
#
# 실행 방법:
#   1. Git Bash 또는 Linux/Mac 터미널에서 실행:
#      bash scripts/run_optimizer_lr_comparison.sh
#
#   2. 실행 권한 부여 후 직접 실행:
#      chmod +x scripts/run_optimizer_lr_comparison.sh
#      ./scripts/run_optimizer_lr_comparison.sh
#
#   3. Windows PowerShell에서 Git Bash를 통해 실행:
#      bash scripts/run_optimizer_lr_comparison.sh
#
# 주의사항:
#   - 스크립트는 프로젝트 루트 디렉토리에서 실행해야 합니다.
#   - 모든 실험은 순차적으로 실행되며, 총 13개의 실험이 진행됩니다.
#   - 각 실험은 40 에포크를 실행하므로 전체 실행 시간이 오래 걸릴 수 있습니다.
#   - 하이퍼파라미터를 변경하려면 스크립트 상단의 변수들을 수정하세요.
# ============================================================================

# Optimizer별 최적 Learning Rate 비교 실험
# 하이퍼파라미터 설정
EPOCHS=40
BATCH_SIZE=128
SCHEDULER="cosineannealinglr"
NET="deep_baseline_bn"
WEIGHT_INIT="--w-init"  # Weight Initialization: ✅

echo "=========================================="
echo "Optimizer별 최적 Learning Rate 비교 실험 시작"
echo "=========================================="
echo ""

COUNTER=1
TOTAL=13

# Adam 실험
echo "[$COUNTER/$TOTAL] Adam - Learning Rate: 0.01"
uv run main.py --optimizer adam --epochs $EPOCHS --lr 0.01 --batch-size $BATCH_SIZE --scheduler $SCHEDULER --net $NET $WEIGHT_INIT
echo ""
COUNTER=$((COUNTER+1))

echo "[$COUNTER/$TOTAL] Adam - Learning Rate: 0.001"
uv run main.py --optimizer adam --epochs $EPOCHS --lr 0.001 --batch-size $BATCH_SIZE --scheduler $SCHEDULER --net $NET $WEIGHT_INIT
echo ""
COUNTER=$((COUNTER+1))

echo "[$COUNTER/$TOTAL] Adam - Learning Rate: 0.0001"
uv run main.py --optimizer adam --epochs $EPOCHS --lr 0.0001 --batch-size $BATCH_SIZE --scheduler $SCHEDULER --net $NET $WEIGHT_INIT
echo ""
COUNTER=$((COUNTER+1))

# AdamW 실험
echo "[$COUNTER/$TOTAL] AdamW - Learning Rate: 0.01"
uv run main.py --optimizer adamw --epochs $EPOCHS --lr 0.01 --batch-size $BATCH_SIZE --scheduler $SCHEDULER --net $NET $WEIGHT_INIT
echo ""
COUNTER=$((COUNTER+1))

echo "[$COUNTER/$TOTAL] AdamW - Learning Rate: 0.001"
uv run main.py --optimizer adamw --epochs $EPOCHS --lr 0.001 --batch-size $BATCH_SIZE --scheduler $SCHEDULER --net $NET $WEIGHT_INIT
echo ""
COUNTER=$((COUNTER+1))

echo "[$COUNTER/$TOTAL] AdamW - Learning Rate: 0.0001"
uv run main.py --optimizer adamw --epochs $EPOCHS --lr 0.0001 --batch-size $BATCH_SIZE --scheduler $SCHEDULER --net $NET $WEIGHT_INIT
echo ""
COUNTER=$((COUNTER+1))

# SGD 실험
echo "[$COUNTER/$TOTAL] SGD - Learning Rate: 0.001"
uv run main.py --optimizer sgd --epochs $EPOCHS --lr 0.001 --batch-size $BATCH_SIZE --scheduler $SCHEDULER --net $NET $WEIGHT_INIT
echo ""
COUNTER=$((COUNTER+1))

echo "[$COUNTER/$TOTAL] SGD - Learning Rate: 0.01"
uv run main.py --optimizer sgd --epochs $EPOCHS --lr 0.01 --batch-size $BATCH_SIZE --scheduler $SCHEDULER --net $NET $WEIGHT_INIT
echo ""
COUNTER=$((COUNTER+1))

# Adagrad 실험
echo "[$COUNTER/$TOTAL] Adagrad - Learning Rate: 0.001"
uv run main.py --optimizer adagrad --epochs $EPOCHS --lr 0.001 --batch-size $BATCH_SIZE --scheduler $SCHEDULER --net $NET $WEIGHT_INIT
echo ""
COUNTER=$((COUNTER+1))

echo "[$COUNTER/$TOTAL] Adagrad - Learning Rate: 0.01"
uv run main.py --optimizer adagrad --epochs $EPOCHS --lr 0.01 --batch-size $BATCH_SIZE --scheduler $SCHEDULER --net $NET $WEIGHT_INIT
echo ""
COUNTER=$((COUNTER+1))

# RMSprop 실험
echo "[$COUNTER/$TOTAL] RMSprop - Learning Rate: 0.001"
uv run main.py --optimizer rmsprop --epochs $EPOCHS --lr 0.001 --batch-size $BATCH_SIZE --scheduler $SCHEDULER --net $NET $WEIGHT_INIT
echo ""
COUNTER=$((COUNTER+1))

echo "[$COUNTER/$TOTAL] RMSprop - Learning Rate: 0.01"
uv run main.py --optimizer rmsprop --epochs $EPOCHS --lr 0.01 --batch-size $BATCH_SIZE --scheduler $SCHEDULER --net $NET $WEIGHT_INIT
echo ""
COUNTER=$((COUNTER+1))

echo "=========================================="
echo "Optimizer별 최적 Learning Rate 비교 실험 완료"
echo "총 $TOTAL개의 실험이 완료되었습니다."
echo "=========================================="

