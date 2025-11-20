"""
pyramidnet164_118의 깊이를 줄였을 때 동일한 파라미터 수를 유지하기 위한 alpha 값 계산
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
from models.wideresnet_pyramid import WideResNetPyramid
from utils.parameter_count import count_parameters

def find_alpha_for_same_params(target_depth, target_params, initial_alpha=118, tolerance=10000):
    """
    목표 깊이에서 목표 파라미터 수를 달성하기 위한 alpha 값을 찾습니다.
    
    Args:
        target_depth: 목표 깊이
        target_params: 목표 파라미터 수
        initial_alpha: 초기 alpha 값
        tolerance: 허용 오차 (파라미터 수)
    
    Returns:
        alpha 값과 실제 파라미터 수
    """
    # 이진 탐색으로 alpha 값 찾기
    low_alpha = 1
    high_alpha = initial_alpha * 3  # 충분히 큰 범위
    best_alpha = initial_alpha
    best_diff = float('inf')
    
    for _ in range(50):  # 최대 50번 반복
        mid_alpha = (low_alpha + high_alpha) // 2
        
        try:
            model = WideResNetPyramid(
                depth=target_depth,
                num_classes=10,
                widen_factor=1,
                dropRate=0.0,
                shakedrop_prob=0.5,
                use_pyramid=True,
                alpha=mid_alpha,
                use_original_depth=True
            )
            params = count_parameters(model)
            diff = abs(params - target_params)
            
            if diff < best_diff:
                best_diff = diff
                best_alpha = mid_alpha
            
            if params < target_params:
                low_alpha = mid_alpha + 1
            else:
                high_alpha = mid_alpha - 1
            
            if diff <= tolerance:
                break
        except Exception as e:
            print(f"Error with alpha={mid_alpha}: {e}")
            break
    
    # 최종 확인
    model = WideResNetPyramid(
        depth=target_depth,
        num_classes=10,
        widen_factor=1,
        dropRate=0.0,
        shakedrop_prob=0.5,
        use_pyramid=True,
        alpha=best_alpha,
        use_original_depth=True
    )
    final_params = count_parameters(model)
    
    return best_alpha, final_params

# 현재 모델의 파라미터 수 확인
print("현재 모델 파라미터 수 계산 중...")
current_model = WideResNetPyramid(
    depth=164,
    num_classes=10,
    widen_factor=1,
    dropRate=0.0,
    shakedrop_prob=0.5,
    use_pyramid=True,
    alpha=118,
    use_original_depth=True
)
current_params = count_parameters(current_model)
print(f"pyramidnet164_118 파라미터 수: {current_params:,}")
print(f"목표 파라미터 수: {current_params:,}")
print()

# 다양한 깊이에 대해 alpha 값 계산
target_depths = [110, 98, 86, 74, 62, 50]
print("=" * 70)
print("깊이별 동일 파라미터 수를 위한 alpha 값:")
print("=" * 70)
print(f"{'깊이':<10} {'n_units':<10} {'alpha':<15} {'파라미터 수':<20} {'차이':<15}")
print("-" * 70)

for depth in target_depths:
    n_units = (depth - 2) // 6
    if (depth - 2) % 6 != 0:
        print(f"{depth:<10} {'N/A':<10} {'N/A':<15} {'Invalid depth':<20} {'N/A':<15}")
        continue
    
    alpha, params = find_alpha_for_same_params(depth, current_params, initial_alpha=118)
    diff = abs(params - current_params)
    print(f"{depth:<10} {n_units:<10} {alpha:<15} {params:,} {diff:,}")

print("=" * 70)

