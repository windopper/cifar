"""
CIFAR-10 데이터셋의 랜덤 이미지 시각화 유틸리티

CIFAR-10 데이터셋에서 랜덤으로 선택한 이미지들을 그리드 형태로 시각화합니다.
"""
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def denormalize(tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """
    정규화된 텐서를 원본 이미지 형태로 변환합니다.
    
    Args:
        tensor: 정규화된 이미지 텐서 (C, H, W)
        mean: 정규화에 사용된 평균값
        std: 정규화에 사용된 표준편차
    
    Returns:
        denormalized: 역정규화된 텐서
    """
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


def show_random_images(
    num_images: int = 20,
    rows: int = 10,
    cols: int = 2,
    data_root: str = './data',
    save_path: str = None,
    use_train: bool = True,
    seed: int = None
):
    """
    CIFAR-10 데이터셋에서 랜덤 이미지를 선택하여 그리드 형태로 시각화합니다.
    
    Args:
        num_images: 표시할 이미지 개수 (기본값: 20)
        rows: 그리드 행 개수 (기본값: 10)
        cols: 그리드 열 개수 (기본값: 2)
        data_root: 데이터셋 저장 경로 (기본값: './data')
        save_path: 이미지 저장 경로 (None이면 './images/random_images.png'로 저장, 기본값: None)
        use_train: 학습 데이터셋 사용 여부 (False면 테스트 데이터셋 사용, 기본값: True)
        seed: 랜덤 시드 (기본값: None)
    """
    # 행과 열의 곱이 이미지 개수와 일치하는지 확인
    if rows * cols != num_images:
        raise ValueError(f"rows * cols ({rows} * {cols} = {rows * cols})가 num_images ({num_images})와 일치하지 않습니다.")
    
    # 랜덤 시드 설정
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # CIFAR-10 클래스 이름
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # 데이터셋 로드 (정규화 없이 원본 이미지 사용)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=use_train,
        download=True,
        transform=transform
    )
    
    # 랜덤 인덱스 선택
    indices = np.random.choice(len(dataset), num_images, replace=False)
    
    # 이미지와 레이블 수집
    images = []
    labels = []
    for idx in indices:
        image, label = dataset[idx]
        images.append(image)
        labels.append(label)
    
    # 이미지를 numpy 배열로 변환
    images = torch.stack(images)
    labels = torch.tensor(labels)
    
    # 역정규화 (ToTensor만 사용했으므로 0-1 범위)
    # matplotlib는 0-1 범위의 float 또는 0-255 범위의 int를 기대
    images = images.permute(0, 2, 3, 1).numpy()  # (N, H, W, C) 형태로 변환
    images = np.clip(images, 0, 1)  # 0-1 범위로 클리핑
    
    # 그리드 생성
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    
    # 각 이미지 표시
    for i in range(num_images):
        row = i // cols
        col = i % cols
        
        if rows == 1:
            ax = axes[col] if cols > 1 else axes
        elif cols == 1:
            ax = axes[row] if rows > 1 else axes
        else:
            ax = axes[row, col]
        
        ax.imshow(images[i])
        ax.set_title(f'{classes[labels[i]]}', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    
    # 저장 경로 설정 (기본값: ./images/random_images.png)
    if save_path is None:
        images_dir = Path('./images')
        images_dir.mkdir(exist_ok=True)
        save_path = images_dir / 'random_images.png'
    
    # 이미지 저장
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"이미지가 저장되었습니다: {save_path}")
    
    plt.close()


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='CIFAR-10 랜덤 이미지 시각화')
    parser.add_argument('--num-images', type=int, default=20,
                        help='표시할 이미지 개수 (기본값: 20)')
    parser.add_argument('--rows', type=int, default=2,
                        help='그리드 행 개수 (기본값: 2)')
    parser.add_argument('--cols', type=int, default=10,
                        help='그리드 열 개수 (기본값: 10)')
    parser.add_argument('--data-root', type=str, default='./data',
                        help='데이터셋 저장 경로 (기본값: ./data)')
    parser.add_argument('--save-path', type=str, default=None,
                        help='이미지 저장 경로 (지정하지 않으면 ./images/random_images.png로 저장)')
    parser.add_argument('--test', action='store_true',
                        help='테스트 데이터셋 사용 (기본값: 학습 데이터셋 사용)')
    parser.add_argument('--seed', type=int, default=None,
                        help='랜덤 시드 (기본값: None)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    show_random_images(
        num_images=args.num_images,
        rows=args.rows,
        cols=args.cols,
        data_root=args.data_root,
        save_path=args.save_path,
        use_train=not args.test,
        seed=args.seed
    )

