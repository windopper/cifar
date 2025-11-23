"""
Top Loss Samples 시각화 유틸리티

모델이 강하게 확신했지만 틀린 예측(top loss samples)을 시각화합니다.
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import platform
from train import CLASS_NAMES

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows: 맑은 고딕
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'  # macOS: AppleGothic
else:
    plt.rcParams['font.family'] = 'NanumGothic'  # Linux: 나눔고딕
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


def denormalize(tensor: torch.Tensor, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> torch.Tensor:
    """
    정규화된 텐서를 원본 이미지 형태로 변환합니다.
    
    Args:
        tensor: 정규화된 이미지 텐서 (C, H, W) 또는 (N, C, H, W)
        mean: 정규화에 사용된 평균값
        std: 정규화에 사용된 표준편차
    
    Returns:
        denormalized: 역정규화된 텐서
    """
    if tensor.dim() == 3:
        mean_tensor = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
        std_tensor = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    else:  # dim == 4
        mean_tensor = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
        std_tensor = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
    
    return tensor * std_tensor + mean_tensor


def visualize_top_loss_samples(
    images: List[torch.Tensor],
    true_labels: List[int],
    predicted_labels: List[int],
    probabilities: List[float],
    losses: List[float],
    normalize_mean: Tuple[float, float, float],
    normalize_std: Tuple[float, float, float],
    num_samples: int,
    save_path: Optional[str] = None,
    class_names: Tuple[str, ...] = CLASS_NAMES
) -> None:
    """
    Top loss samples를 시각화합니다.
    
    Args:
        images: 이미지 텐서 리스트 (각각 (C, H, W) 형태)
        true_labels: 실제 레이블 리스트
        predicted_labels: 예측 레이블 리스트
        probabilities: 예측 확률 리스트 (예측 클래스에 대한 확률)
        losses: loss 값 리스트
        normalize_mean: 정규화에 사용된 평균값
        normalize_std: 정규화에 사용된 표준편차
        num_samples: 표시할 샘플 개수
        save_path: 저장 경로 (None이면 './images/top_loss_samples.png'로 저장)
        class_names: 클래스 이름 튜플
    """
    if len(images) == 0:
        print("경고: 시각화할 샘플이 없습니다.")
        return
    
    # loss가 높은 순서대로 정렬 (상위 num_samples개만 선택)
    num_samples = min(num_samples, len(images))
    sorted_indices = sorted(range(len(losses)), key=lambda i: losses[i], reverse=True)[:num_samples]
    
    # 선택된 샘플들
    selected_images = [images[i] for i in sorted_indices]
    selected_true_labels = [true_labels[i] for i in sorted_indices]
    selected_predicted_labels = [predicted_labels[i] for i in sorted_indices]
    selected_probabilities = [probabilities[i] for i in sorted_indices]
    selected_losses = [losses[i] for i in sorted_indices]
    
    # 이미지 역정규화 및 numpy 변환
    denormalized_images = []
    for img in selected_images:
        img_denorm = denormalize(img, normalize_mean, normalize_std)
        img_denorm = torch.clamp(img_denorm, 0.0, 1.0)
        img_np = img_denorm.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
        denormalized_images.append(img_np)
    
    # 그리드 크기 계산
    cols = min(5, num_samples)
    rows = (num_samples + cols - 1) // cols
    
    # 플롯 생성
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    if num_samples == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = axes.flatten()
    
    # 각 샘플 표시
    for i in range(num_samples):
        ax = axes[i]
        
        # 이미지 표시
        ax.imshow(denormalized_images[i])
        
        # 제목 설정
        true_class = class_names[selected_true_labels[i]]
        pred_class = class_names[selected_predicted_labels[i]]
        prob = selected_probabilities[i] * 100
        loss = selected_losses[i]
        
        title = f"True: {true_class}\nPred: {pred_class} ({prob:.1f}%)\nLoss: {loss:.4f}"
        ax.set_title(title, fontsize=9, pad=5)
        ax.axis('off')
    
    # 빈 subplot 숨기기
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # 저장 경로 설정
    if save_path is None:
        images_dir = Path('./images')
        images_dir.mkdir(exist_ok=True)
        save_path = images_dir / 'top_loss_samples.png'
    
    # 이미지 저장
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Top loss samples 이미지가 저장되었습니다: {save_path}")
    
    plt.close()

