import json
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Seaborn 스타일 설정
sns.set_style("whitegrid")
sns.set_palette("husl")
sns.set_context("paper", font_scale=1.2)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
plt.rcParams['figure.facecolor'] = 'white'


def load_histories(file_names, output_dir="outputs/augmentation"):
    """특정 파일명들의 history를 로드"""
    histories = []

    for file_name in file_names:
        # _history.json이 없으면 추가
        if not file_name.endswith(".json"):
            if file_name.endswith("_history"):
                file_name = file_name + ".json"
            else:
                file_name = file_name + "_history.json"

        file_path = os.path.join(output_dir, file_name)

        if not os.path.exists(file_path):
            print(f"Warning: {file_path} 파일을 찾을 수 없습니다.")
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
                history['file_path'] = file_path
                history['file_name'] = file_name
                histories.append(history)
        except Exception as e:
            print(f"Warning: {file_path} 로드 실패: {e}")

    return histories


def get_config_label(file_name):
    """파일명에서 augmentation, cutmix, mixup, autoaugment 설정을 파악하여 레이블 생성"""
    has_aug = '_aug' in file_name
    has_cutmix = '_cutmix' in file_name
    has_mixup = '_mixup' in file_name
    has_autoaug = '_autoaug' in file_name
    
    # 레이블 구성 요소
    components = []
    
    if has_aug:
        components.append('Augmentation')
    if has_cutmix:
        components.append('CutMix')
    if has_mixup:
        components.append('Mixup')
    if has_autoaug:
        components.append('AutoAugment')
    
    if not components:
        return '기본 (모두 없음)'
    
    # 레이블 생성
    label = ' + '.join(components)
    
    return label


def plot_augmentation_comparison(histories, save_path="comparison/augmentation_comparison_100epoch.png"):
    """Augmentation 비교 그래프 작성 (100 Epoch 기준)"""

    if not histories:
        print("로드된 history 파일이 없습니다.")
        return

    # 각 파일의 epoch 길이 확인 및 가장 짧은 epoch 찾기
    epoch_lengths = []
    for history in histories:
        file_name = history.get('file_name', 'unknown')
        epoch_len = len(history.get('train_loss', []))
        epoch_lengths.append({
            'file_name': file_name,
            'epoch_len': epoch_len
        })

    if not epoch_lengths or all(e['epoch_len'] == 0 for e in epoch_lengths):
        print("유효한 데이터가 없습니다.")
        return

    # 가장 짧은 epoch 찾기
    min_epoch_info = min(epoch_lengths, key=lambda x: x['epoch_len'])
    min_epochs = min_epoch_info['epoch_len']

    print(f"\n각 설정의 epoch 길이:")
    for info in epoch_lengths:
        marker = " ← 가장 짧음" if info == min_epoch_info else ""
        print(f"  - {info['file_name']}: {info['epoch_len']} epochs{marker}")
    print(f"\n가장 짧은 epoch 길이: {min_epochs}")

    # 그래프 설정
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Augmentation 비교 (100 Epoch 기준)', fontsize=18, fontweight='bold', y=1.02)

    # Seaborn 색상 팔레트 사용
    n_configs = len(histories)
    palette = sns.color_palette("husl", n_configs)
    
    # 더 예쁜 색상 팔레트 (명확한 구분을 위해)
    base_colors = sns.color_palette("Set2", n_configs)
    if n_configs > len(base_colors):
        base_colors = sns.color_palette("husl", n_configs)
    
    base_styles = ['-', '--', '-.', ':', '-', '--', '-.']
    
    # 모든 고유 레이블 수집
    all_labels = set()
    for history in histories:
        file_name = history.get('file_name', 'unknown')
        label = get_config_label(file_name)
        all_labels.add(label)
    
    # 레이블 순서 정의 (README의 표 순서와 동일)
    label_order = [
        '기본 (모두 없음)',
        'Augmentation',
        'Augmentation + CutMix',
        'Augmentation + Mixup',
        'Augmentation + AutoAugment',
        'Augmentation + CutMix + AutoAugment',
        'Augmentation + Mixup + AutoAugment',
    ]
    
    # 순서에 없는 레이블들을 뒤에 추가
    for label in sorted(all_labels):
        if label not in label_order:
            label_order.append(label)
    
    # 색상 및 스타일 매핑 생성
    config_colors = {}
    config_styles = {}
    color_idx = 0
    for i, label in enumerate(label_order):
        if label in all_labels:
            config_colors[label] = base_colors[color_idx % len(base_colors)]
            config_styles[label] = base_styles[color_idx % len(base_styles)]
            color_idx += 1

    # 각 설정에 대한 데이터 준비
    config_data = []

    for history in histories:
        file_name = history.get('file_name', 'unknown')
        label = get_config_label(file_name)

        train_loss = history.get('train_loss', [])[:min_epochs]
        val_loss = history.get('val_loss', [])[:min_epochs]
        val_acc = history.get('val_accuracy', [])[:min_epochs]

        epochs = list(range(1, min_epochs + 1))

        config_data.append({
            'label': label,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'epochs': epochs,
            'color': config_colors.get(label, '#000000'),
            'linestyle': config_styles.get(label, '-')
        })
    
    # 레이블 순서대로 정렬
    config_data.sort(key=lambda x: label_order.index(x['label']) if x['label'] in label_order else 999)

    # 1. Train Loss
    ax1 = axes[0]
    for data in config_data:
        ax1.plot(data['epochs'], data['train_loss'],
                 label=data['label'], color=data['color'],
                 linestyle=data['linestyle'], linewidth=2.5, 
                 marker='o', markersize=5, markevery=10, alpha=0.8)

    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Train Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Train Loss', fontsize=14, fontweight='bold', pad=10)
    ax1.legend(fontsize=9, loc='best', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    sns.despine(ax=ax1, top=True, right=True)

    # 2. Validation Loss
    ax2 = axes[1]
    for data in config_data:
        ax2.plot(data['epochs'], data['val_loss'],
                 label=data['label'], color=data['color'],
                 linestyle=data['linestyle'], linewidth=2.5,
                 marker='s', markersize=5, markevery=10, alpha=0.8)

    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Validation Loss', fontsize=14, fontweight='bold', pad=10)
    ax2.legend(fontsize=9, loc='best', frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    sns.despine(ax=ax2, top=True, right=True)

    # 3. Validation Accuracy
    ax3 = axes[2]
    for data in config_data:
        ax3.plot(data['epochs'], data['val_accuracy'],
                 label=data['label'], color=data['color'],
                 linestyle=data['linestyle'], linewidth=2.5,
                 marker='^', markersize=5, markevery=10, alpha=0.8)

    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Validation Accuracy', fontsize=14, fontweight='bold', pad=10)
    ax3.legend(fontsize=9, loc='best', frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3, linestyle='--')
    sns.despine(ax=ax3, top=True, right=True)

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # 디렉토리 생성
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(
        save_path) else '.', exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"그래프가 저장되었습니다: {save_path}")
    plt.close()


def main():
    """메인 함수"""
    # 100 Epoch 기준 Augmentation 비교
    # 기본 모델명 (augmentation 관련 옵션 제외)
    base_model_name = "deep_baseline_bn_adam_crossentropy_bs128_ep100_lr0.0003_mom0.9_schcosineannealinglr_tmax100"
    
    # 여러 조합: 기본, aug, aug+cutmix, aug+mixup, aug+autoaug, aug+cutmix+autoaug, aug+mixup+autoaug
    file_names = [
        f"{base_model_name}_winit_history.json",                    # 기본 (모두 없음)
        f"{base_model_name}_aug_winit_history.json",                # Augmentation
        f"{base_model_name}_aug_cutmix_winit_history.json",         # Augmentation + CutMix
        f"{base_model_name}_aug_mixup_winit_history.json",         # Augmentation + Mixup
        f"{base_model_name}_aug_autoaug_winit_history.json",        # Augmentation + AutoAugment
        f"{base_model_name}_aug_autoaug_cutmix_winit_history.json", # Augmentation + CutMix + AutoAugment
        f"{base_model_name}_aug_autoaug_mixup_winit_history.json",  # Augmentation + Mixup + AutoAugment
    ]

    print("비교할 파일 목록:")
    for i, file_name in enumerate(file_names, 1):
        print(f"  {i}. {file_name}")

    # History 파일 로드
    histories = load_histories(file_names)

    if not histories:
        print("\n로드된 history 파일이 없습니다.")
        print("파일명을 확인하거나 base_model_name을 수정하세요.")
        return

    print(f"\n총 {len(histories)}개의 파일이 로드되었습니다.")

    # 그래프 작성
    plot_augmentation_comparison(histories)


if __name__ == "__main__":
    main()

