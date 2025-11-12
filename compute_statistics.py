"""
CIFAR-10 train set의 mean과 std 값을 계산하는 스크립트
"""
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np


def compute_mean_std(dataset, batch_size=128, num_workers=2):
    """
    데이터셋의 채널별 mean과 std를 계산합니다.
    
    Args:
        dataset: PyTorch 데이터셋
        batch_size: 배치 크기
        num_workers: 데이터 로더 워커 수
    
    Returns:
        mean: (R, G, B) 채널별 평균값
        std: (R, G, B) 채널별 표준편차
    """
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0
    
    print("Mean과 std 계산 중...")
    for images, _ in tqdm(dataloader, desc="Processing"):
        # 배치 차원과 공간 차원을 평탄화: [B, C, H, W] -> [B*H*W, C]
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        
        # 각 채널별로 평균 계산
        mean += images.mean(2).sum(0)
        total_samples += batch_samples
    
    # 전체 평균 계산
    mean /= total_samples
    
    # 표준편차 계산
    total_samples = 0
    for images, _ in tqdm(dataloader, desc="Computing std"):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        
        # 각 채널별로 분산 계산
        std += ((images - mean.view(1, 3, 1)) ** 2).mean(2).sum(0)
        total_samples += batch_samples
    
    # 전체 표준편차 계산
    std = torch.sqrt(std / total_samples)
    
    return mean, std


def main():
    print("=" * 60)
    print("CIFAR-10 Train Set 통계 계산")
    print("=" * 60)
    
    # ToTensor만 적용 (Normalize는 하지 않음)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Train 데이터셋 로드
    print("\n데이터셋 로드 중...")
    train_set = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    print(f"Train set 크기: {len(train_set)}")
    print(f"이미지 크기: {train_set[0][0].shape}")
    print()
    
    # Mean과 std 계산
    mean, std = compute_mean_std(train_set, batch_size=128, num_workers=2)
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("계산 결과")
    print("=" * 60)
    print(f"\nMean (R, G, B):")
    print(f"  {mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}")
    print(f"\nStd (R, G, B):")
    print(f"  {std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}")
    
    # PyTorch Normalize에서 사용할 수 있는 형식으로 출력
    print("\n" + "=" * 60)
    print("PyTorch transforms.Normalize에서 사용할 수 있는 형식:")
    print("=" * 60)
    print(f"\nmean = ({mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f})")
    print(f"std = ({std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f})")
    
    # 코드 예시 출력
    print("\n" + "=" * 60)
    print("사용 예시:")
    print("=" * 60)
    print(f"""
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=({mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}),
        std=({std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f})
    )
])
    """)
    
    print("=" * 60)


if __name__ == '__main__':
    main()

