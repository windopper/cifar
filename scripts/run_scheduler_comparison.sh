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
# 특정 단계만 실행하기:
#   - 단일 단계: bash scripts/run_scheduler_comparison.sh --step 1
#   - 여러 단계: bash scripts/run_scheduler_comparison.sh --step 1,3,4
#   - 단계 목록:
#       1: Cosine Annealing LR
#       2: One Cycle LR
#       3: Exponential LR
#       4: ReduceLROnPlateau
#
# 주의사항:
#   - 스크립트는 프로젝트 루트 디렉토리에서 실행해야 합니다.
#   - 인수 없이 실행하면 모든 실험이 순차적으로 실행됩니다.
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

BASE_CMD="python cifar/main.py --optimizer $OPTIMIZER --epochs $EPOCHS --batch-size $BATCH_SIZE --lr $LR --net $NET $WEIGHT_INIT"

# 명령줄 인수 파싱
SELECTED_STEPS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --step|-s)
            SELECTED_STEPS="$2"
            shift 2
            ;;
        --help|-h)
            echo "사용법: $0 [옵션]"
            echo ""
            echo "옵션:"
            echo "  --step, -s STEPS    실행할 단계 번호 (쉼표로 구분, 예: 1,3,4)"
            echo "  --help, -h          이 도움말 표시"
            echo ""
            echo "단계 목록:"
            echo "  1: Cosine Annealing LR"
            echo "  2: One Cycle LR"
            echo "  3: Exponential LR"
            echo "  4: ReduceLROnPlateau"
            echo ""
            echo "예제:"
            echo "  $0                  # 모든 단계 실행"
            echo "  $0 --step 1        # 단계 1만 실행"
            echo "  $0 --step 1,3,4    # 단계 1, 3, 4 실행"
            exit 0
            ;;
        *)
            echo "알 수 없는 옵션: $1"
            echo "도움말을 보려면 --help 또는 -h를 사용하세요."
            exit 1
            ;;
    esac
done

# 단계 실행 여부 확인 함수
should_run_step() {
    local step=$1
    if [[ -z "$SELECTED_STEPS" ]]; then
        return 0  # 인수가 없으면 모든 단계 실행
    fi
    # 쉼표로 구분된 단계 목록에서 현재 단계가 포함되어 있는지 확인
    IFS=',' read -ra STEPS <<< "$SELECTED_STEPS"
    for s in "${STEPS[@]}"; do
        if [[ "$s" == "$step" ]]; then
            return 0
        fi
    done
    return 1
}

# 각 실험을 함수로 정의
run_step1() {
    echo "Cosine Annealing LR"
    echo "  - T_max: $EPOCHS (default)"
    echo "  - eta_min: 0.0 (default)"
    $BASE_CMD --scheduler cosineannealinglr --scheduler-t-max $EPOCHS
    echo ""
}

run_step2() {
    echo "One Cycle LR"
    echo "  - max_lr: $(echo "$LR * 10" | bc) (default: lr * 10)"
    $BASE_CMD --scheduler onecyclelr
    echo ""
}

run_step3() {
    echo "Exponential LR"
    echo "  - gamma: 0.95 (default)"
    $BASE_CMD --scheduler exponentiallr --scheduler-gamma 0.95
    echo ""
}

run_step4() {
    echo "ReduceLROnPlateau"
    echo "  - factor: 0.1 (default)"
    echo "  - patience: 5 (조정됨, 60 epoch 기준)"
    echo "  - mode: min (default)"
    $BASE_CMD --scheduler reducelronplateau --scheduler-factor 0.1 --scheduler-patience 5 --scheduler-mode min
    echo ""
}

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
if [[ -n "$SELECTED_STEPS" ]]; then
    echo "  - 선택된 단계: $SELECTED_STEPS"
fi
echo "=========================================="
echo ""

# 선택된 단계 실행
TOTAL=0
if should_run_step 1; then TOTAL=$((TOTAL+1)); fi
if should_run_step 2; then TOTAL=$((TOTAL+1)); fi
if should_run_step 3; then TOTAL=$((TOTAL+1)); fi
if should_run_step 4; then TOTAL=$((TOTAL+1)); fi

COUNTER=0

if should_run_step 1; then
    COUNTER=$((COUNTER+1))
    echo "[$COUNTER/$TOTAL]"
    run_step1
fi

if should_run_step 2; then
    COUNTER=$((COUNTER+1))
    echo "[$COUNTER/$TOTAL]"
    run_step2
fi

if should_run_step 3; then
    COUNTER=$((COUNTER+1))
    echo "[$COUNTER/$TOTAL]"
    run_step3
fi

if should_run_step 4; then
    COUNTER=$((COUNTER+1))
    echo "[$COUNTER/$TOTAL]"
    run_step4
fi

echo "=========================================="
echo "Scheduler 비교 실험 완료"
echo "총 $TOTAL개의 실험이 완료되었습니다."
echo "=========================================="
echo ""
echo "결과 파일 위치:"
echo "  - outputs/"


