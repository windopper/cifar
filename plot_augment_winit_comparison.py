import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


def load_histories(file_names, output_dir="outputs"):
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
    """파일명에서 augmentation, cutmix, weight initialization, label smoothing 설정을 파악하여 레이블 생성"""
    has_aug = '_aug' in file_name
    has_cutmix = '_cutmix' in file_name
    has_winit = '_winit' in file_name
    
    # 라벨 스무딩 값 추출 (ls0.05 형식)
    ls_match = re.search(r'ls([\d.]+)', file_name)
    ls_value = ls_match.group(1) if ls_match else None
    has_ls = ls_value is not None
    
    # 레이블 구성 요소
    components = []
    
    if has_aug:
        components.append('Augmentation')
    if has_cutmix:
        components.append('CutMix')
    if has_winit:
        components.append('Weight Init')
    if has_ls:
        components.append(f'Label Smoothing({ls_value})')
    
    if not components:
        return '기본 (모두 없음)'
    
    # 레이블 생성
    label = ' + '.join(components)
    
    # 특별한 경우: Augmentation만, Weight Init만
    if len(components) == 1:
        if components[0] == 'Augmentation':
            return 'Augmentation만'
        elif components[0] == 'Weight Init':
            return 'Weight Init만'
    
    return label


def plot_model_comparison(histories, save_path="comparison/augment_winit_comparison.png"):
    """Augmentation, CutMix, Weight Initialization, Label Smoothing 비교 그래프 작성"""

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
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Augmentation, CutMix, Weight Initialization, Label Smoothing 효과 비교', fontsize=16, fontweight='bold')

    # 색상 및 스타일 매핑 (일관성 있게)
    # 기본 색상 팔레트
    base_colors = [
        '#1f77b4',  # 파란색
        '#ff7f0e',  # 주황색
        '#2ca02c',  # 초록색
        '#9467bd',  # 보라색
        '#d62728',  # 빨간색
        '#8c564b',  # 갈색
        '#e377c2',  # 분홍색
        '#7f7f7f',  # 회색
        '#bcbd22',  # 올리브색
        '#17becf',  # 청록색
    ]
    
    base_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--']
    
    # 모든 고유 레이블 수집
    all_labels = set()
    for history in histories:
        file_name = history.get('file_name', 'unknown')
        label = get_config_label(file_name)
        all_labels.add(label)
    
    # 레이블 순서 정의
    label_order = [
        '기본 (모두 없음)',
        'Weight Init만',
        'Augmentation만',
        'Augmentation + CutMix',
        'Augmentation + Weight Init',
        'Augmentation + CutMix + Weight Init',
        'Augmentation + Weight Init + Label Smoothing(0.05)',
        'Augmentation + CutMix + Weight Init + Label Smoothing(0.05)',
    ]
    
    # 순서에 없는 레이블들을 뒤에 추가
    for label in sorted(all_labels):
        if label not in label_order:
            label_order.append(label)
    
    # 색상 및 스타일 매핑 생성
    config_colors = {}
    config_styles = {}
    for i, label in enumerate(label_order):
        if label in all_labels:
            config_colors[label] = base_colors[i % len(base_colors)]
            config_styles[label] = base_styles[i % len(base_styles)]

    # 각 모델에 대한 데이터 준비
    model_data = []

    for history in histories:
        file_name = history.get('file_name', 'unknown')
        label = get_config_label(file_name)

        train_loss = history.get('train_loss', [])[:min_epochs]
        val_loss = history.get('val_loss', [])[:min_epochs]
        val_acc = history.get('val_accuracy', [])[:min_epochs]

        epochs = list(range(1, min_epochs + 1))

        model_data.append({
            'label': label,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'epochs': epochs,
            'color': config_colors.get(label, '#000000'),
            'linestyle': config_styles.get(label, '-')
        })
    
    # 레이블 순서대로 정렬
    label_order_for_sort = [
        '기본 (모두 없음)',
        'Weight Init만',
        'Augmentation만',
        'Augmentation + CutMix',
        'Augmentation + Weight Init',
        'Augmentation + CutMix + Weight Init',
        'Augmentation + Weight Init + Label Smoothing(0.05)',
        'Augmentation + CutMix + Weight Init + Label Smoothing(0.05)',
    ]
    # 순서에 없는 레이블들을 뒤에 추가
    all_labels_for_sort = set(x['label'] for x in model_data)
    for label in sorted(all_labels_for_sort):
        if label not in label_order_for_sort:
            label_order_for_sort.append(label)
    
    model_data.sort(key=lambda x: label_order_for_sort.index(x['label']) if x['label'] in label_order_for_sort else 999)

    # 1. Train Loss
    ax1 = axes[0]
    for data in model_data:
        ax1.plot(data['epochs'], data['train_loss'],
                 label=data['label'], color=data['color'],
                 linestyle=data['linestyle'], linewidth=2.5, marker='o', markersize=4)

    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Train Loss', fontsize=11)
    ax1.set_title('Train Loss', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)

    # 2. Validation Loss
    ax2 = axes[1]
    for data in model_data:
        ax2.plot(data['epochs'], data['val_loss'],
                 label=data['label'], color=data['color'],
                 linestyle=data['linestyle'], linewidth=2.5, marker='s', markersize=4)

    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Validation Loss', fontsize=11)
    ax2.set_title('Validation Loss', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)

    # 3. Validation Accuracy
    ax3 = axes[2]
    for data in model_data:
        ax3.plot(data['epochs'], data['val_accuracy'],
                 label=data['label'], color=data['color'],
                 linestyle=data['linestyle'], linewidth=2.5, marker='^', markersize=4)

    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Validation Accuracy (%)', fontsize=11)
    ax3.set_title('Validation Accuracy', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9, loc='best')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # 디렉토리 생성
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(
        save_path) else '.', exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"그래프가 저장되었습니다: {save_path}")
    plt.close()


def main():
    """메인 함수"""
    # 동일한 모델에 대한 augmentation과 weight initialization 조합 비교
    # 기본 모델명 (aug, winit 제외)
    base_model_name = "deep_baseline_adam_crossentropy_bs16_ep20_lr0.0003_mom0.9_schcosineannealinglr_tmax20"
    
    # 여러 조합: 기본, winit만, aug만, aug+cutmix, aug+winit, aug+winit+ls, aug+cutmix+winit
    file_names = [
        f"{base_model_name}_history.json",                    # 기본 (둘 다 없음)
        f"{base_model_name}_winit_history.json",              # Weight Init만
        f"{base_model_name}_aug_history.json",                # Augmentation만
        f"{base_model_name}_aug_cutmix_history.json",         # Augmentation + CutMix
        f"{base_model_name}_aug_winit_history.json",          # Augmentation + Weight Init
        f"{base_model_name}_ls0.05_aug_winit_history.json",   # Augmentation + Weight Init + Label Smoothing=0.05
        f"{base_model_name}_aug_cutmix_winit_history.json",   # Augmentation + CutMix + Weight Init
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
    plot_model_comparison(histories)


if __name__ == "__main__":
    main()
