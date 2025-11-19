"""CIFAR-10 데이터셋 로딩 유틸리티"""
import torch
import torchvision
import torchvision.transforms as transforms
from utils.cutout import Cutout


def get_normalize_values(use_cifar_normalize: bool = False):
    """
    Normalize 값 반환
    
    Args:
        use_cifar_normalize: CIFAR-10 표준 Normalize 값 사용 여부
    
    Returns:
        normalize_mean: 평균 튜플
        normalize_std: 표준편차 튜플
    """
    if use_cifar_normalize:
        # CIFAR-10 표준 Normalize 값
        normalize_mean = (0.4914, 0.4822, 0.4465)
        normalize_std = (0.2470, 0.2434, 0.2615)
    else:
        # 기본값
        normalize_mean = (0.5, 0.5, 0.5)
        normalize_std = (0.5, 0.5, 0.5)
    
    return normalize_mean, normalize_std


def get_train_transform(
    augment: bool = False,
    autoaugment: bool = False,
    cutout: bool = False,
    cutout_n_holes: int = 1,
    cutout_length: int = 16,
    cutout_prob: float = 0.5,
    use_cifar_normalize: bool = False
):
    """
    학습용 데이터 변환 생성
    
    Args:
        augment: 데이터 증강 사용 여부
        autoaugment: AutoAugment 사용 여부
        cutout: Cutout 사용 여부
        cutout_n_holes: Cutout 마스킹할 영역의 개수
        cutout_length: Cutout 마스킹 영역의 크기
        cutout_prob: Cutout 적용 확률
        use_cifar_normalize: CIFAR-10 표준 Normalize 값 사용 여부
    
    Returns:
        train_transform: 학습용 변환
    """
    normalize_mean, normalize_std = get_normalize_values(use_cifar_normalize)
    
    train_transform_list = []
    
    if augment:
        train_transform_list.append(transforms.RandomCrop(32, padding=4))
        train_transform_list.append(transforms.RandomHorizontalFlip())
        
        if autoaugment:
            # AutoAugment 사용: CIFAR-10 정책 적용
            train_transform_list.append(transforms.AutoAugment(
                policy=transforms.AutoAugmentPolicy.CIFAR10))
        else:
            # 기본 데이터 증강: RandomRotation
            train_transform_list.append(transforms.RandomRotation(15))
    
    # 공통 변환: ToTensor와 Normalize는 항상 적용
    train_transform_list.append(transforms.ToTensor())
    
    # Cutout 적용 (--augment가 활성화되어 있을 때만)
    if cutout and augment:
        train_transform_list.append(Cutout(
            n_holes=cutout_n_holes,
            length=cutout_length,
            prob=cutout_prob
        ))
    
    train_transform_list.append(
        transforms.Normalize(normalize_mean, normalize_std))
    
    return transforms.Compose(train_transform_list)


def get_val_transform(use_cifar_normalize: bool = False):
    """
    검증용 데이터 변환 생성
    
    Args:
        use_cifar_normalize: CIFAR-10 표준 Normalize 값 사용 여부
    
    Returns:
        val_transform: 검증용 변환
    """
    normalize_mean, normalize_std = get_normalize_values(use_cifar_normalize)
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])
    
    return val_transform


def get_cifar10_loaders(
    batch_size: int = 16,
    augment: bool = False,
    autoaugment: bool = False,
    cutout: bool = False,
    cutout_n_holes: int = 1,
    cutout_length: int = 16,
    cutout_prob: float = 0.5,
    use_cifar_normalize: bool = False,
    num_workers: int = None,
    collate_fn=None,
    data_root: str = './data'
):
    """
    CIFAR-10 데이터셋과 DataLoader 생성
    
    Args:
        batch_size: 배치 크기
        augment: 데이터 증강 사용 여부
        autoaugment: AutoAugment 사용 여부
        cutout: Cutout 사용 여부
        cutout_n_holes: Cutout 마스킹할 영역의 개수
        cutout_length: Cutout 마스킹 영역의 크기
        cutout_prob: Cutout 적용 확률
        use_cifar_normalize: CIFAR-10 표준 Normalize 값 사용 여부
        num_workers: DataLoader의 워커 수 (None이면 자동 설정)
        collate_fn: collate 함수 (CutMix/Mixup용)
        data_root: 데이터셋 저장 경로
    
    Returns:
        train_loader: 학습용 DataLoader
        val_loader: 검증용 DataLoader
        train_set: 학습용 데이터셋
        val_set: 검증용 데이터셋
    """
    # Transform 생성
    train_transform = get_train_transform(
        augment=augment,
        autoaugment=autoaugment,
        cutout=cutout,
        cutout_n_holes=cutout_n_holes,
        cutout_length=cutout_length,
        cutout_prob=cutout_prob,
        use_cifar_normalize=use_cifar_normalize
    )
    val_transform = get_val_transform(use_cifar_normalize=use_cifar_normalize)
    
    # 데이터셋 로딩
    train_set = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform)
    val_set = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=val_transform)
    
    # num_workers 자동 설정: AutoAugment 사용 시 더 많은 워커 필요
    if num_workers is None:
        # AutoAugment는 CPU에서 무거운 작업이므로 더 많은 워커 사용
        num_workers = 4 if autoaugment and augment else 2
    
    # GPU 사용 시 pin_memory 활성화로 전송 속도 향상
    pin_memory = torch.cuda.is_available()
    # persistent_workers로 워커 재생성 오버헤드 감소 (num_workers > 0일 때만)
    persistent_workers = num_workers > 0
    
    # DataLoader 생성
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers, collate_fn=collate_fn)
    
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers)
    
    return train_loader, val_loader, train_set, val_set

