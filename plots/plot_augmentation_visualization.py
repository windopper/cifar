"""
각 augmentation 기법을 적용한 이미지를 시각화하는 스크립트
- Standard Augmentation
- CutMix
- Mixup
- AutoAugment
- Cutout
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision
import torchvision.transforms as transforms

# 프로젝트 루트를 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from utils.cutout import Cutout
from utils.cutmix import cutmix
from utils.mixup import mixup


# Seaborn 스타일 설정
sns.set_style("whitegrid")
sns.set_palette("husl")
sns.set_context("paper", font_scale=1.2)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
plt.rcParams['figure.facecolor'] = 'white'


def denormalize(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """정규화된 텐서를 원본 이미지로 변환"""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


def tensor_to_numpy(tensor):
    """텐서를 numpy 배열로 변환 (시각화용)"""
    # (C, H, W) -> (H, W, C)
    img = tensor.permute(1, 2, 0).numpy()
    # [0, 1] 범위로 클리핑
    img = np.clip(img, 0, 1)
    return img


def get_standard_augmentation_transform():
    """Standard augmentation transform 생성"""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def get_autoaugment_transform():
    """AutoAugment transform 생성"""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def get_cutout_transform():
    """Cutout augmentation transform 생성"""
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        Cutout(n_holes=1, length=16, prob=1.0),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def apply_cutmix_visualization(img1, img2):
    """CutMix을 시각화용으로 적용 (단일 이미지 쌍)"""
    # 배치 차원 추가: (C, H, W) -> (1, C, H, W)
    batch1 = img1.unsqueeze(0)
    batch2 = img2.unsqueeze(0)
    
    # 배치 결합: (2, C, H, W)
    batch = torch.cat([batch1, batch2], dim=0)
    targets = torch.tensor([0, 1])  # 더미 타겟
    
    # CutMix 적용
    mixed_batch, _ = cutmix((batch, targets), alpha=1.0)
    
    # 첫 번째 이미지 반환
    return mixed_batch[0]


def apply_mixup_visualization(img1, img2):
    """Mixup을 시각화용으로 적용 (단일 이미지 쌍)"""
    # 배치 차원 추가: (C, H, W) -> (1, C, H, W)
    batch1 = img1.unsqueeze(0)
    batch2 = img2.unsqueeze(0)
    
    # 배치 결합: (2, C, H, W)
    batch = torch.cat([batch1, batch2], dim=0)
    targets = torch.tensor([0, 1])  # 더미 타겟
    
    # Mixup 적용
    mixed_batch, _ = mixup((batch, targets), alpha=1.0)
    
    # 첫 번째 이미지 반환
    return mixed_batch[0]


def plot_augmentation_visualization(save_path="comparison/augmentation_visualization.png", num_samples=5):
    """각 augmentation을 적용한 이미지 시각화"""
    
    # CIFAR-10 데이터셋 로드 (원본 이미지 표시용 - ToTensor만)
    cifar10_original = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transforms.ToTensor()
    )
    
    # CIFAR-10 데이터셋 로드 (Augmentation 적용용 - PIL Image)
    cifar10_pil = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=None
    )
    
    # 랜덤하게 샘플 선택
    indices = np.random.choice(len(cifar10_original), num_samples, replace=False)
    
    # 각 augmentation transform 생성
    standard_transform = get_standard_augmentation_transform()
    autoaugment_transform = get_autoaugment_transform()
    cutout_transform = get_cutout_transform()
    
    # 시각화 준비
    fig, axes = plt.subplots(num_samples, 6, figsize=(18, 3 * num_samples))
    
    # 컬럼 제목
    column_titles = ['원본', 'Standard Aug', 'CutMix', 'Mixup', 'AutoAugment', 'Cutout']
    for col_idx, title in enumerate(column_titles):
        axes[0, col_idx].set_title(title, fontsize=14, fontweight='bold', pad=10)
    
    for row_idx, idx in enumerate(indices):
        # 원본 이미지 로드 (텐서)
        original_img, label = cifar10_original[idx]
        
        # 원본 이미지 표시 (정규화 없음)
        axes[row_idx, 0].imshow(tensor_to_numpy(original_img))
        axes[row_idx, 0].axis('off')
        
        # PIL Image로 원본 이미지 가져오기
        pil_img, _ = cifar10_pil[idx]
        
        # Standard Augmentation 적용
        # 여러 번 시도해서 좋은 결과 선택
        best_std_img = None
        for _ in range(3):
            std_img = standard_transform(pil_img)
            if best_std_img is None:
                best_std_img = std_img
            # 시각적으로 더 변화가 큰 것을 선택
            if torch.std(std_img) > torch.std(best_std_img):
                best_std_img = std_img
        std_img_denorm = denormalize(best_std_img)
        axes[row_idx, 1].imshow(tensor_to_numpy(std_img_denorm))
        axes[row_idx, 1].axis('off')
        
        # CutMix 적용 (다른 이미지와 섞기)
        # 랜덤하게 다른 이미지 선택
        other_idx = np.random.choice([i for i in range(len(cifar10_pil)) if i != idx])
        other_pil_img, _ = cifar10_pil[other_idx]
        
        # Standard augmentation을 먼저 적용한 후 CutMix (실제 사용 시나리오와 유사하게)
        std_img1 = standard_transform(pil_img)
        std_img2 = standard_transform(other_pil_img)
        
        # CutMix 적용 (정규화된 이미지에 적용)
        cutmix_img = apply_cutmix_visualization(std_img1, std_img2)
        cutmix_img_denorm = denormalize(cutmix_img)
        axes[row_idx, 2].imshow(tensor_to_numpy(cutmix_img_denorm))
        axes[row_idx, 2].axis('off')
        
        # Mixup 적용 (정규화된 이미지에 적용)
        mixup_img = apply_mixup_visualization(std_img1, std_img2)
        mixup_img_denorm = denormalize(mixup_img)
        axes[row_idx, 3].imshow(tensor_to_numpy(mixup_img_denorm))
        axes[row_idx, 3].axis('off')
        
        # AutoAugment 적용
        # 여러 번 시도해서 좋은 결과 선택
        best_aa_img = None
        for _ in range(3):
            aa_img = autoaugment_transform(pil_img)
            if best_aa_img is None:
                best_aa_img = aa_img
            if torch.std(aa_img) > torch.std(best_aa_img):
                best_aa_img = aa_img
        aa_img_denorm = denormalize(best_aa_img)
        axes[row_idx, 4].imshow(tensor_to_numpy(aa_img_denorm))
        axes[row_idx, 4].axis('off')
        
        # Cutout 적용
        # 여러 번 시도해서 좋은 결과 선택
        best_cutout_img = None
        for _ in range(3):
            cutout_img = cutout_transform(pil_img)
            if best_cutout_img is None:
                best_cutout_img = cutout_img
            if torch.std(cutout_img) > torch.std(best_cutout_img):
                best_cutout_img = cutout_img
        cutout_img_denorm = denormalize(best_cutout_img)
        axes[row_idx, 5].imshow(tensor_to_numpy(cutout_img_denorm))
        axes[row_idx, 5].axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"그래프가 저장되었습니다: {save_path}")
    plt.close()


def main():
    """메인 함수"""
    print("Augmentation 시각화를 생성합니다...")
    plot_augmentation_visualization(num_samples=3)
    print("완료되었습니다!")


if __name__ == "__main__":
    main()

