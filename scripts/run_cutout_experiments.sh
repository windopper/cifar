#!/bin/bash

# ============================================================================
# 실행 방법
# ============================================================================
# 이 스크립트는 README.md 54-58 라인의 Cutout 관련 실험을 자동으로 실행합니다.
#
# 실행 방법:
#   1. Git Bash 또는 Linux/Mac 터미널에서 실행:
#      bash scripts/run_cutout_experiments.sh
#
#   2. 실행 권한 부여 후 직접 실행:
#      chmod +x scripts/run_cutout_experiments.sh
#      ./scripts/run_cutout_experiments.sh
#
#   3. Windows PowerShell에서 Git Bash를 통해 실행:
#      bash scripts/run_cutout_experiments.sh
#
# 주의사항:
#   - 스크립트는 프로젝트 루트 디렉토리에서 실행해야 합니다.
#   - 모든 실험은 순차적으로 실행되며, 총 3개의 실험이 진행됩니다.
#   - 각 실험은 100 에포크를 실행하므로 전체 실행 시간이 오래 걸릴 수 있습니다.
# ============================================================================

echo "=========================================="
echo "Cutout 실험 시작 (README.md 54-58 라인)"
echo "=========================================="
echo ""

# 1. Cutout 기본 (기본 길이)
echo "[1/3] Cutout 기본 - Augmentation ✅, Cutout ✅"
uv run main.py --optimizer adam --epochs 100 --batch-size 128 --lr 3e-4 --scheduler cosineannealinglr --net deep_baseline_bn --w-init --augment --cutout
echo ""

# 2. Cutout (길이 8)
echo "[2/3] Cutout (길이 8) - Augmentation ✅, Cutout ✅, Cutout Length 8"
uv run main.py --optimizer adam --epochs 100 --batch-size 128 --lr 3e-4 --scheduler cosineannealinglr --net deep_baseline_bn --w-init --augment --cutout --cutout-length 8
echo ""

# 3. Cutout + AutoAugment
echo "[3/3] Cutout + AutoAugment - Augmentation ✅, Cutout ✅, AutoAugment ✅"
uv run main.py --optimizer adam --epochs 100 --batch-size 128 --lr 3e-4 --scheduler cosineannealinglr --net deep_baseline_bn --w-init --augment --cutout --autoaugment
echo ""

echo "=========================================="
echo "Completed Cutout Experiments"
echo "=========================================="

