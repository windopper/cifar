#!/bin/bash

# ============================================================================
# 실행 방법
# ============================================================================
# 이 스크립트는 모델별 성능 비교 실험을 자동으로 실행합니다.
#
# 실행 방법:
#   1. 모든 모델 실행:
#      bash scripts/run_model_comparison.sh
#
#   2. 특정 모델만 실행:
#      bash scripts/run_model_comparison.sh --net [모델이름]
#      예: bash scripts/run_model_comparison.sh --net deep_baseline_bn
#
#   3. 실행 권한 부여 후 직접 실행:
#      chmod +x scripts/run_model_comparison.sh
#      ./scripts/run_model_comparison.sh [--net 모델이름]
#
# 주의사항:
#   - 스크립트는 프로젝트 루트 디렉토리에서 실행해야 합니다.
#   - --net 인수가 없으면 모든 모델(총 8개)이 순차적으로 실행됩니다.
#   - 각 실험은 60 에포크를 실행하므로 전체 실행 시간이 오래 걸릴 수 있습니다.
#   - 하이퍼파라미터를 변경하려면 스크립트 상단의 변수들을 수정하세요.
# ============================================================================

# 모델별 성능 비교 실험
# 하이퍼파라미터 설정
EPOCHS=60
BATCH_SIZE=128
LEARNING_RATE=0.0003  # 3e-4
SCHEDULER="onecyclelr"
OPTIMIZER="adam"
WEIGHT_INIT="--w-init"  # Weight Initialization: ✅

# 명령줄 인수 파싱
SELECTED_NET=""
if [ "$1" == "--net" ] && [ -n "$2" ]; then
    SELECTED_NET="$2"
fi

# 사용 가능한 모델 목록
declare -a MODELS=(
    "deep_baseline_bn"
    "deep_baseline2_bn"
    "deep_baseline2_bn_residual"
    "deep_baseline2_bn_residual_se"
    "deep_baseline2_bn_residual_preact"
    "deep_baseline2_bn_resnext"
    "deep_baseline3_bn"
    "mxresnet56"
)

# 특정 모델이 지정된 경우 해당 모델만 실행
if [ -n "$SELECTED_NET" ]; then
    # 모델이 목록에 있는지 확인
    FOUND=0
    for model in "${MODELS[@]}"; do
        if [ "$model" == "$SELECTED_NET" ]; then
            FOUND=1
            break
        fi
    done
    
    if [ $FOUND -eq 0 ]; then
        echo "오류: '$SELECTED_NET' 모델을 찾을 수 없습니다."
        echo "사용 가능한 모델:"
        for model in "${MODELS[@]}"; do
            echo "  - $model"
        done
        exit 1
    fi
    
    echo "=========================================="
    echo "모델별 성능 비교 실험 시작"
    echo "선택된 모델: $SELECTED_NET"
    echo "=========================================="
    echo ""
    
    COUNTER=1
    TOTAL=1
else
    echo "=========================================="
    echo "모델별 성능 비교 실험 시작"
    echo "모든 모델 실행 (총 ${#MODELS[@]}개)"
    echo "=========================================="
    echo ""
    
    COUNTER=1
    TOTAL=${#MODELS[@]}
fi

# 모델별 실험 실행
for NET in "${MODELS[@]}"; do
    # 특정 모델이 지정된 경우 해당 모델만 실행
    if [ -n "$SELECTED_NET" ] && [ "$NET" != "$SELECTED_NET" ]; then
        continue
    fi
    
    echo "[$COUNTER/$TOTAL] $NET"
    python cifar/main.py --optimizer $OPTIMIZER --epochs $EPOCHS --lr $LEARNING_RATE --batch-size $BATCH_SIZE --scheduler $SCHEDULER --net $NET $WEIGHT_INIT
    echo ""
    COUNTER=$((COUNTER+1))
done

echo "=========================================="
echo "모델별 성능 비교 실험 완료"
echo "총 $TOTAL개의 실험이 완료되었습니다."
echo "=========================================="

