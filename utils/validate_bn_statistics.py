"""
BN 통계량 불일치 가설 검증 유틸리티

가설: 만약 BN 통계량 불일치가 원인이라면, 학습 때 저장된 running_mean, running_var를 버리고,
현재 들어오는 TTA 데이터들의 평균과 분산을 사용하여 추론하면 성능이 올라가야 합니다.

검증 방법:
1. 모델의 가중치(Weight)는 고정(Freeze)합니다.
2. TTA가 적용된 테스트 데이터셋을 모델에 통과시키며 BN 층의 running_mean과 running_var만 업데이트합니다.
3. 업데이트된 통계량을 가진 모델로 다시 TTA 추론을 수행합니다.
"""
import os
import json
import sys
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.optim.swa_utils import update_bn
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from tqdm import tqdm
from typing import Tuple, Optional
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.net import get_net
from train import CLASS_NAMES


def load_model_from_history(history_path: str) -> Tuple[str, str, tuple, tuple, float]:
    """
    History 파일에서 모델 정보를 추출하고 모델 경로를 반환합니다.
    
    Args:
        history_path: History 파일 경로
        
    Returns:
        (model_name, model_path, normalize_mean, normalize_std, shakedrop_prob) 튜플
    """
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"History 파일을 찾을 수 없습니다: {history_path}")
    
    with open(history_path, 'r') as f:
        history_data = json.load(f)
    
    # 모델 이름 추출
    if 'hyperparameters' in history_data and 'net' in history_data['hyperparameters']:
        model_name = history_data['hyperparameters']['net']
    else:
        raise ValueError(f"History 파일에 'hyperparameters.net' 정보가 없습니다: {history_path}")
    
    # 모델 파일 경로 자동 생성
    if history_path.endswith('_history.json'):
        model_path = history_path.replace('_history.json', '.pth')
    else:
        base_path = history_path.rsplit('.json', 1)[0]
        model_path = f"{base_path.rsplit('_history', 1)[0]}.pth"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    # Normalize 값 추출
    normalize_mean = (0.5, 0.5, 0.5)
    normalize_std = (0.5, 0.5, 0.5)
    shakedrop_prob = 0.0
    
    if 'hyperparameters' in history_data:
        hp = history_data['hyperparameters']
        if 'normalize_mean' in hp and 'normalize_std' in hp:
            normalize_mean = tuple(hp['normalize_mean'])
            normalize_std = tuple(hp['normalize_std'])
        if 'shakedrop_prob' in hp and hp['shakedrop_prob'] is not None:
            shakedrop_prob = hp['shakedrop_prob']
    
    return model_name, model_path, normalize_mean, normalize_std, shakedrop_prob


def create_tta_dataloader(
    normalize_mean: Tuple[float, float, float],
    normalize_std: Tuple[float, float, float],
    tta_mode: int = 5,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 2
):
    """
    TTA용 데이터로더를 생성합니다.
    
    Args:
        normalize_mean: 정규화 평균값
        normalize_std: 정규화 표준편차
        tta_mode: TTA 모드 (2: 원본+Flip, 3: 원본+Flip+확대, 5: AutoAugment 5회)
        batch_size: 배치 크기
        shuffle: 셔플 여부
        num_workers: 워커 수
        
    Returns:
        DataLoader: TTA용 데이터로더
    """
    if tta_mode == 5:
        # AutoAugment를 포함한 변환 (학습 시와 유사한 augmentation)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
            # AutoAugment는 ToTensor 이후에 적용할 수 없으므로,
            # 여기서는 기본 augmentation만 적용하고
            # 실제 TTA는 추론 시 각 배치마다 적용합니다.
        ])
    elif tta_mode == 2:
        # 원본 + Horizontal Flip
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std)
        ])
    elif tta_mode == 3:
        # 원본 + Horizontal Flip + 확대
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std)
        ])
    else:
        # 기본 변환
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std)
        ])
    
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    
    return test_loader


def apply_tta(net, images, tta_mode, normalize_mean, normalize_std, device):
    """
    TTA를 적용하여 추론합니다.
    
    Args:
        net: 모델
        images: 입력 이미지
        tta_mode: TTA 모드
        normalize_mean: 정규화 평균값
        normalize_std: 정규화 표준편차
        device: 디바이스
        
    Returns:
        outputs: 모델 출력
    """
    if tta_mode == 2:
        outputs_original = net(images)
        outputs_flipped = net(torch.flip(images, [3]))
        return (outputs_original + outputs_flipped) / 2.0
    elif tta_mode == 3:
        outputs_original = net(images)
        outputs_flipped = net(torch.flip(images, [3]))
        
        scale_factor = 1.1
        h, w = images.shape[2], images.shape[3]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        images_upscaled = F.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        images_cropped = images_upscaled[:, :, start_h:start_h+h, start_w:start_w+w]
        outputs_upscaled = net(images_cropped)
        
        return (outputs_original + outputs_flipped + outputs_upscaled) / 3.0
    elif tta_mode == 5:
        outputs_list = [net(images)]
        
        mean_tensor = torch.tensor(normalize_mean, device=device).view(1, 3, 1, 1)
        std_tensor = torch.tensor(normalize_std, device=device).view(1, 3, 1, 1)
        images_denorm = images * std_tensor + mean_tensor
        images_denorm = torch.clamp(images_denorm, 0.0, 1.0)
        
        to_pil = transforms.ToPILImage()
        autoaugment = AutoAugment(policy=AutoAugmentPolicy.CIFAR10)
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(normalize_mean, normalize_std)
        
        for _ in range(4):
            augmented_images = []
            for img in images_denorm:
                img_pil = to_pil(img.cpu())
                img_aug = autoaugment(img_pil)
                img_tensor = to_tensor(img_aug).to(device)
                img_normalized = normalize(img_tensor)
                augmented_images.append(img_normalized)
            augmented_batch = torch.stack(augmented_images)
            outputs_list.append(net(augmented_batch))
        
        return sum(outputs_list) / len(outputs_list)
    else:
        return net(images)


def evaluate_model(
    net,
    test_loader,
    tta_mode: int,
    normalize_mean: Tuple[float, float, float],
    normalize_std: Tuple[float, float, float],
    device: torch.device,
    description: str = "평가"
):
    """
    모델을 평가합니다.
    
    Args:
        net: 모델
        test_loader: 테스트 데이터로더
        tta_mode: TTA 모드
        normalize_mean: 정규화 평균값
        normalize_std: 정규화 표준편차
        device: 디바이스
        description: 평가 설명
        
    Returns:
        accuracy: 정확도
    """
    net.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc=description)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = apply_tta(net, images, tta_mode, normalize_mean, normalize_std, device)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            current_acc = 100 * correct / total
            pbar.set_postfix({'accuracy': f'{current_acc:.2f}%'})
    
    accuracy = 100 * correct / total
    return accuracy


def update_bn_with_tta(
    net,
    tta_loader,
    tta_mode: int,
    normalize_mean: Tuple[float, float, float],
    normalize_std: Tuple[float, float, float],
    device: torch.device
):
    """
    TTA를 적용하면서 BN 통계량을 업데이트합니다.
    
    Args:
        net: 모델 (train 모드여야 함)
        tta_loader: TTA 데이터로더
        tta_mode: TTA 모드
        normalize_mean: 정규화 평균값
        normalize_std: 정규화 표준편차
        device: 디바이스
    """
    net.train()
    
    if tta_mode == 5:
        # AutoAugment를 사용하는 경우, 각 배치마다 다른 augmentation 적용
        to_pil = transforms.ToPILImage()
        autoaugment = AutoAugment(policy=AutoAugmentPolicy.CIFAR10)
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(normalize_mean, normalize_std)
        
        mean_tensor = torch.tensor(normalize_mean, device=device).view(1, 3, 1, 1)
        std_tensor = torch.tensor(normalize_std, device=device).view(1, 3, 1, 1)
        
        pbar = tqdm(tta_loader, desc="BN 통계량 업데이트 (TTA 모드 5)")
        for images, _ in pbar:
            images = images.to(device)
            
            # 원본 이미지로 BN 통계량 업데이트
            _ = net(images)
            
            # AutoAugment 적용된 이미지들로도 BN 통계량 업데이트
            images_denorm = images * std_tensor + mean_tensor
            images_denorm = torch.clamp(images_denorm, 0.0, 1.0)
            
            # 여러 번 augmentation 적용 (TTA와 동일하게)
            for _ in range(4):
                augmented_images = []
                for img in images_denorm:
                    img_pil = to_pil(img.cpu())
                    img_aug = autoaugment(img_pil)
                    img_tensor = to_tensor(img_aug).to(device)
                    img_normalized = normalize(img_tensor)
                    augmented_images.append(img_normalized)
                augmented_batch = torch.stack(augmented_images)
                _ = net(augmented_batch)
    elif tta_mode == 2:
        # 원본 + Horizontal Flip
        pbar = tqdm(tta_loader, desc="BN 통계량 업데이트 (TTA 모드 2)")
        for images, _ in pbar:
            images = images.to(device)
            _ = net(images)
            _ = net(torch.flip(images, [3]))
    elif tta_mode == 3:
        # 원본 + Horizontal Flip + 확대
        pbar = tqdm(tta_loader, desc="BN 통계량 업데이트 (TTA 모드 3)")
        for images, _ in pbar:
            images = images.to(device)
            _ = net(images)
            _ = net(torch.flip(images, [3]))
            
            scale_factor = 1.1
            h, w = images.shape[2], images.shape[3]
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            images_upscaled = F.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            images_cropped = images_upscaled[:, :, start_h:start_h+h, start_w:start_w+w]
            _ = net(images_cropped)
    else:
        # 기본 모드 (TTA 없음)
        update_bn(tta_loader, net, device=device)


def validate_bn_statistics(
    history_path: str,
    tta_mode: int = 5,
    bn_update_batch_size: int = 256,
    eval_batch_size: int = 128,
    device: Optional[torch.device] = None
):
    """
    BN 통계량 불일치 가설을 검증합니다.
    
    Args:
        history_path: History 파일 경로
        tta_mode: TTA 모드 (2: 원본+Flip, 3: 원본+Flip+확대, 5: AutoAugment 5회)
        bn_update_batch_size: BN 통계량 업데이트용 배치 크기
        eval_batch_size: 평가용 배치 크기
        device: 디바이스 (None이면 자동 선택)
        
    Returns:
        dict: 검증 결과 (업데이트 전 정확도, 업데이트 후 정확도, 개선율)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("=" * 80)
    print("BN 통계량 불일치 가설 검증")
    print("=" * 80)
    print(f"History 파일: {history_path}")
    print(f"TTA 모드: {tta_mode}")
    print(f"BN 업데이트 배치 크기: {bn_update_batch_size}")
    print(f"평가 배치 크기: {eval_batch_size}")
    print(f"Device: {device}")
    print("=" * 80)
    
    # 1. 모델 정보 로드
    print("\n[1/5] 모델 정보 로드 중...")
    model_name, model_path, normalize_mean, normalize_std, shakedrop_prob = load_model_from_history(history_path)
    print(f"  모델 이름: {model_name}")
    print(f"  모델 경로: {model_path}")
    print(f"  Normalize: mean={normalize_mean}, std={normalize_std}")
    print(f"  ShakeDrop 확률: {shakedrop_prob}")
    
    # 2. 모델 로드
    print("\n[2/5] 모델 로드 중...")
    net = get_net(model_name, shakedrop_prob=shakedrop_prob)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()
    print("  모델 로드 완료")
    
    # 3. 평가용 데이터로더 생성 (업데이트 전 성능 측정용)
    print("\n[3/5] 평가용 데이터로더 생성 중...")
    eval_loader = create_tta_dataloader(
        normalize_mean, normalize_std, tta_mode=tta_mode,
        batch_size=eval_batch_size, shuffle=False, num_workers=2
    )
    
    # 4. 업데이트 전 성능 측정
    print("\n[4/5] BN 통계량 업데이트 전 성능 측정 중...")
    accuracy_before = evaluate_model(
        net, eval_loader, tta_mode, normalize_mean, normalize_std, device,
        description="업데이트 전 평가"
    )
    print(f"\n  업데이트 전 정확도: {accuracy_before:.2f}%")
    
    # 5. BN 통계량 업데이트
    print("\n[5/5] BN 통계량 업데이트 중...")
    print("  주의: 모델 가중치는 고정하고 BN 통계량만 업데이트합니다.")
    
    # 모델을 train 모드로 변경 (BN 통계량 업데이트를 위해)
    net.train()
    
    # 모든 파라미터의 gradient를 비활성화 (가중치 고정)
    for param in net.parameters():
        param.requires_grad = False
    
    # TTA 데이터로더 생성 (BN 통계량 업데이트용, 배치 크기는 클수록 좋음)
    tta_loader = create_tta_dataloader(
        normalize_mean, normalize_std, tta_mode=tta_mode,
        batch_size=bn_update_batch_size, shuffle=True, num_workers=2
    )
    
    # TTA를 적용하면서 BN 통계량 업데이트
    print("  TTA 데이터로 BN 통계량 재계산 중...")
    update_bn_with_tta(net, tta_loader, tta_mode, normalize_mean, normalize_std, device)
    print("  BN 통계량 업데이트 완료")
    
    # 모델을 eval 모드로 변경
    net.eval()
    
    # 6. 업데이트 후 성능 측정
    print("\n[6/6] BN 통계량 업데이트 후 성능 측정 중...")
    accuracy_after = evaluate_model(
        net, eval_loader, tta_mode, normalize_mean, normalize_std, device,
        description="업데이트 후 평가"
    )
    print(f"\n  업데이트 후 정확도: {accuracy_after:.2f}%")
    
    # 결과 요약
    improvement = accuracy_after - accuracy_before
    improvement_rate = (improvement / accuracy_before) * 100 if accuracy_before > 0 else 0
    
    print("\n" + "=" * 80)
    print("검증 결과 요약")
    print("=" * 80)
    print(f"업데이트 전 정확도: {accuracy_before:.2f}%")
    print(f"업데이트 후 정확도: {accuracy_after:.2f}%")
    print(f"개선율: {improvement:+.2f}%p ({improvement_rate:+.2f}%)")
    print("=" * 80)
    
    if improvement > 0:
        print("\n✓ 가설이 지지됩니다: BN 통계량 업데이트로 성능이 향상되었습니다.")
    elif improvement < 0:
        print("\n✗ 가설이 기각됩니다: BN 통계량 업데이트로 성능이 하락했습니다.")
    else:
        print("\n- 가설이 기각됩니다: BN 통계량 업데이트로 성능 변화가 없습니다.")
    
    return {
        'accuracy_before': accuracy_before,
        'accuracy_after': accuracy_after,
        'improvement': improvement,
        'improvement_rate': improvement_rate
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='BN 통계량 불일치 가설 검증')
    parser.add_argument('--history', type=str, required=True,
                        help='History 파일 경로')
    parser.add_argument('--tta', type=int, default=5, choices=[2, 3, 5],
                        help='TTA 모드 (2: 원본+Flip, 3: 원본+Flip+확대, 5: AutoAugment 5회, default: 5)')
    parser.add_argument('--bn-update-batch-size', type=int, default=256,
                        help='BN 통계량 업데이트용 배치 크기 (default: 256)')
    parser.add_argument('--eval-batch-size', type=int, default=128,
                        help='평가용 배치 크기 (default: 128)')
    parser.add_argument('--device', type=str, default=None,
                        help='디바이스 (cuda/cpu, default: 자동 선택)')
    
    args = parser.parse_args()
    
    device = None
    if args.device:
        device = torch.device(args.device)
    
    validate_bn_statistics(
        history_path=args.history,
        tta_mode=args.tta,
        bn_update_batch_size=args.bn_update_batch_size,
        eval_batch_size=args.eval_batch_size,
        device=device
    )

