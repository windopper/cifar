"""
CIFAR-10 전체 데이터셋의 클래스 비율을 탐색하는 스크립트
"""
import torch
import torchvision
import torchvision.transforms as transforms
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

CLASS_NAMES = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')


def explore_class_distribution():
    """
    CIFAR-10 전체 데이터셋(train + test)의 클래스 비율을 탐색합니다.
    """
    print("=" * 60)
    print("CIFAR-10 전체 데이터셋 클래스 비율 탐색")
    print("=" * 60)
    
    # Transform: ToTensor만 적용 (라벨 정보만 필요)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Train 데이터셋 로드
    print("\n[1/2] Train 데이터셋 로드 중...")
    train_set = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # Test 데이터셋 로드
    print("[2/2] Test 데이터셋 로드 중...")
    test_set = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    print(f"\nTrain set 크기: {len(train_set):,}")
    print(f"Test set 크기: {len(test_set):,}")
    print(f"전체 데이터셋 크기: {len(train_set) + len(test_set):,}")
    
    # Train set 클래스 분포 계산
    print("\nTrain set 클래스 분포 계산 중...")
    train_labels = [train_set[i][1] for i in range(len(train_set))]
    train_class_counts = Counter(train_labels)
    
    # Test set 클래스 분포 계산
    print("Test set 클래스 분포 계산 중...")
    test_labels = [test_set[i][1] for i in range(len(test_set))]
    test_class_counts = Counter(test_labels)
    
    # 전체 데이터셋 클래스 분포 계산
    all_labels = train_labels + test_labels
    all_class_counts = Counter(all_labels)
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("클래스별 분포 결과")
    print("=" * 60)
    
    print(f"\n{'클래스':<12} {'Train':<10} {'Test':<10} {'전체':<10} {'Train 비율':<12} {'Test 비율':<12} {'전체 비율':<12}")
    print("-" * 80)
    
    total_train = len(train_set)
    total_test = len(test_set)
    total_all = len(train_set) + len(test_set)
    
    for class_idx in range(10):
        class_name = CLASS_NAMES[class_idx]
        train_count = train_class_counts[class_idx]
        test_count = test_class_counts[class_idx]
        all_count = all_class_counts[class_idx]
        
        train_ratio = (train_count / total_train) * 100
        test_ratio = (test_count / total_test) * 100
        all_ratio = (all_count / total_all) * 100
        
        print(f"{class_name:<12} {train_count:<10} {test_count:<10} {all_count:<10} "
              f"{train_ratio:>10.2f}% {test_ratio:>10.2f}% {all_ratio:>10.2f}%")
    
    print("-" * 80)
    print(f"{'합계':<12} {total_train:<10} {total_test:<10} {total_all:<10} "
          f"{100.0:>10.2f}% {100.0:>10.2f}% {100.0:>10.2f}%")
    
    # 요약 통계
    print("\n" + "=" * 60)
    print("요약 통계")
    print("=" * 60)
    
    train_ratios = [train_class_counts[i] / total_train * 100 for i in range(10)]
    test_ratios = [test_class_counts[i] / total_test * 100 for i in range(10)]
    all_ratios = [all_class_counts[i] / total_all * 100 for i in range(10)]
    
    print(f"\nTrain set:")
    print(f"  평균 비율: {np.mean(train_ratios):.2f}%")
    print(f"  표준편차: {np.std(train_ratios):.2f}%")
    print(f"  최소 비율: {np.min(train_ratios):.2f}% ({CLASS_NAMES[np.argmin(train_ratios)]})")
    print(f"  최대 비율: {np.max(train_ratios):.2f}% ({CLASS_NAMES[np.argmax(train_ratios)]})")
    
    print(f"\nTest set:")
    print(f"  평균 비율: {np.mean(test_ratios):.2f}%")
    print(f"  표준편차: {np.std(test_ratios):.2f}%")
    print(f"  최소 비율: {np.min(test_ratios):.2f}% ({CLASS_NAMES[np.argmin(test_ratios)]})")
    print(f"  최대 비율: {np.max(test_ratios):.2f}% ({CLASS_NAMES[np.argmax(test_ratios)]})")
    
    print(f"\n전체 데이터셋:")
    print(f"  평균 비율: {np.mean(all_ratios):.2f}%")
    print(f"  표준편차: {np.std(all_ratios):.2f}%")
    print(f"  최소 비율: {np.min(all_ratios):.2f}% ({CLASS_NAMES[np.argmin(all_ratios)]})")
    print(f"  최대 비율: {np.max(all_ratios):.2f}% ({CLASS_NAMES[np.argmax(all_ratios)]})")
    
    # 균형성 확인
    print("\n" + "=" * 60)
    print("데이터셋 균형성 분석")
    print("=" * 60)
    
    expected_ratio = 100.0 / 10  # 각 클래스당 10%
    train_imbalance = max([abs(r - expected_ratio) for r in train_ratios])
    test_imbalance = max([abs(r - expected_ratio) for r in test_ratios])
    all_imbalance = max([abs(r - expected_ratio) for r in all_ratios])
    
    print(f"\n이상적인 클래스 비율: {expected_ratio:.2f}% (균형 데이터셋)")
    print(f"\nTrain set 최대 편차: {train_imbalance:.2f}%")
    print(f"Test set 최대 편차: {test_imbalance:.2f}%")
    print(f"전체 데이터셋 최대 편차: {all_imbalance:.2f}%")
    
    if all_imbalance < 0.1:
        print("\n✓ 데이터셋이 매우 균형잡혀 있습니다.")
    elif all_imbalance < 1.0:
        print("\n✓ 데이터셋이 대체로 균형잡혀 있습니다.")
    else:
        print("\n⚠ 데이터셋에 클래스 불균형이 있습니다.")
    
    print("\n" + "=" * 60)
    
    # 시각화
    try:
        print("\n클래스 분포 시각화 생성 중...")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Train set
        axes[0].bar(range(10), [train_class_counts[i] for i in range(10)], 
                    color='skyblue', edgecolor='black')
        axes[0].set_title('Train Set 클래스 분포', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('클래스', fontsize=12)
        axes[0].set_ylabel('샘플 수', fontsize=12)
        axes[0].set_xticks(range(10))
        axes[0].set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Test set
        axes[1].bar(range(10), [test_class_counts[i] for i in range(10)], 
                    color='lightcoral', edgecolor='black')
        axes[1].set_title('Test Set 클래스 분포', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('클래스', fontsize=12)
        axes[1].set_ylabel('샘플 수', fontsize=12)
        axes[1].set_xticks(range(10))
        axes[1].set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
        axes[1].grid(axis='y', alpha=0.3)
        
        # 전체 데이터셋
        axes[2].bar(range(10), [all_class_counts[i] for i in range(10)], 
                    color='lightgreen', edgecolor='black')
        axes[2].set_title('전체 데이터셋 클래스 분포', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('클래스', fontsize=12)
        axes[2].set_ylabel('샘플 수', fontsize=12)
        axes[2].set_xticks(range(10))
        axes[2].set_xticklabels(CLASS_NAMES, rotation=45, ha='right')
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # 저장
        output_path = 'class_distribution.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"시각화 결과가 '{output_path}'에 저장되었습니다.")
        
        # 표시 (선택사항)
        # plt.show()
        
    except Exception as e:
        print(f"\n시각화 생성 중 오류 발생: {e}")
        print("텍스트 결과만 출력합니다.")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    explore_class_distribution()

