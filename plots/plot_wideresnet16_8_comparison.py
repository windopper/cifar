import json
import os
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


def load_histories(file_names, output_dir="outputs/final2"):
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
    """파일명에서 설정을 파악하여 레이블 생성"""
    # README.md의 표 순서에 맞춰 레이블 생성
    if 'sam_samrho2.0_samadaptive' in file_name and 'ema_emad0.999' in file_name and 'ls0.1' in file_name:
        return 'SGD + Nesterov + ASAM + EMA + Label Smoothing'
    elif 'sam_samrho2.0_samadaptive' in file_name and 'ema_emad0.999' in file_name:
        return 'SGD + Nesterov + ASAM + EMA'
    elif 'sam_samrho2.0_samadaptive' in file_name:
        return 'SGD + Nesterov + ASAM (rho=2.0)'
    elif 'ls0.1' in file_name and 'cifar_normalize' in file_name and 'ep200' in file_name:
        return 'SGD + Nesterov + Label Smoothing 0.1 + Epoch 200 + CIFAR-10 Normalize'
    elif 'nesterov' in file_name and 'lr0.1' in file_name and 'sam' not in file_name:
        return 'SGD + Nesterov, Learning Rate 0.1'
    else:
        return '기본 설정'


def plot_wideresnet_comparison(histories, save_path="comparison/wideresnet16_8_comparison.png"):
    """WideResNet16_8 모델들의 설정별 비교 그래프 작성"""
    
    if not histories:
        print("로드된 history 파일이 없습니다.")
        return
    
    # 각 파일의 epoch 길이 확인
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
    
    print(f"\n각 설정의 epoch 길이:")
    for info in epoch_lengths:
        print(f"  - {info['file_name']}: {info['epoch_len']} epochs")
    
    # 그래프 설정
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('WideResNet16_8: 설정별 성능 비교', fontsize=18, fontweight='bold', y=0.995)
    
    # Seaborn 색상 팔레트 사용
    n_configs = len(histories)
    palette = sns.color_palette("Set2", n_configs)
    if n_configs > len(palette):
        palette = sns.color_palette("husl", n_configs)
    
    # 레이블 순서 정의 (README의 표 순서와 동일)
    label_order = [
        '기본 설정',
        'SGD + Nesterov, Learning Rate 0.1',
        'SGD + Nesterov + ASAM (rho=2.0)',
        'SGD + Nesterov + ASAM + EMA',
        'SGD + Nesterov + ASAM + EMA + Label Smoothing',
        'SGD + Nesterov + ASAM + EMA + Label Smoothing + CIFAR-10 Normalize',
        'SGD + Nesterov + Label Smoothing 0.1 + Epoch 200 + CIFAR-10 Normalize',
    ]
    
    # 각 설정에 대한 데이터 준비
    config_data = []
    
    for i, history in enumerate(histories):
        file_name = history.get('file_name', 'unknown')
        label = get_config_label(file_name)
        
        train_loss = history.get('train_loss', [])
        val_loss = history.get('val_loss', [])
        val_acc = history.get('val_accuracy', [])
        
        epochs = list(range(1, len(train_loss) + 1))
        
        # 최고 정확도 찾기
        max_val_acc = max(val_acc) if val_acc else 0
        
        config_data.append({
            'label': label,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'epochs': epochs,
            'color': palette[i % len(palette)],
            'max_val_acc': max_val_acc,
            'file_name': file_name
        })
    
    # 레이블 순서대로 정렬
    def get_label_order(label):
        try:
            return label_order.index(label)
        except ValueError:
            return 999
    
    config_data.sort(key=lambda x: get_label_order(x['label']))
    
    # 1. Train Loss
    ax1 = axes[0, 0]
    for data in config_data:
        ax1.plot(data['epochs'], data['train_loss'],
                 label=data['label'], color=data['color'],
                 linewidth=2.5, marker='o', markersize=4, 
                 markevery=max(1, len(data['epochs'])//20), alpha=0.8)
    
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Train Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Train Loss', fontsize=14, fontweight='bold', pad=10)
    ax1.legend(fontsize=9, loc='best', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    sns.despine(ax=ax1, top=True, right=True)
    
    # 2. Validation Loss
    ax2 = axes[0, 1]
    for data in config_data:
        ax2.plot(data['epochs'], data['val_loss'],
                 label=data['label'], color=data['color'],
                 linewidth=2.5, marker='s', markersize=4,
                 markevery=max(1, len(data['epochs'])//20), alpha=0.8)
    
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Validation Loss', fontsize=14, fontweight='bold', pad=10)
    ax2.legend(fontsize=9, loc='best', frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='--')
    sns.despine(ax=ax2, top=True, right=True)
    
    # 3. Validation Accuracy
    ax3 = axes[1, 0]
    for data in config_data:
        ax3.plot(data['epochs'], data['val_accuracy'],
                 label=f"{data['label']} (최고: {data['max_val_acc']:.2f}%)", 
                 color=data['color'],
                 linewidth=2.5, marker='^', markersize=4,
                 markevery=max(1, len(data['epochs'])//20), alpha=0.8)
    
    ax3.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Validation Accuracy', fontsize=14, fontweight='bold', pad=10)
    ax3.legend(fontsize=9, loc='best', frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3, linestyle='--')
    sns.despine(ax=ax3, top=True, right=True)
    
    # 4. 최종 성능 요약 (Bar Chart)
    ax4 = axes[1, 1]
    final_val_accs = []
    config_labels = []
    bar_colors = []
    
    for data in config_data:
        if data['val_accuracy']:
            final_val_accs.append(data['max_val_acc'])
            config_labels.append(data['label'])
            bar_colors.append(data['color'])
    
    if final_val_accs:
        x_pos = np.arange(len(config_labels))
        
        bars = ax4.bar(x_pos, final_val_accs, color=bar_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # 각 막대 위에 값 표시
        for i, (bar, acc) in enumerate(zip(bars, final_val_accs)):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.2f}%',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax4.set_xlabel('설정', fontsize=12, fontweight='bold')
        ax4.set_ylabel('최고 Validation Accuracy (%)', fontsize=12, fontweight='bold')
        ax4.set_title('최고 성능 비교', fontsize=14, fontweight='bold', pad=10)
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(config_labels, rotation=15, ha='right', fontsize=9)
        ax4.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax4.set_ylim([min(final_val_accs) - 0.5, max(final_val_accs) + 0.5])
        sns.despine(ax=ax4, top=True, right=True)
    
    plt.tight_layout()
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n그래프가 저장되었습니다: {save_path}")
    plt.close()


def print_summary(histories):
    """모델별 요약 정보 출력"""
    print("\n" + "="*80)
    print("WideResNet16_8 설정별 요약")
    print("="*80)
    
    for i, history in enumerate(histories, 1):
        hyperparams = history.get('hyperparameters', {})
        file_name = history.get('file_name', 'unknown')
        label = get_config_label(file_name)
        
        val_loss = history.get('val_loss', [])
        val_acc = history.get('val_accuracy', [])
        
        final_val_loss = val_loss[-1] if val_loss else None
        max_val_acc = max(val_acc) if val_acc else None
        
        print(f"\n[{i}] {label}")
        print(f"    파일명: {file_name}")
        print(f"    Optimizer: {hyperparams.get('optimizer', 'N/A').upper()}")
        print(f"    Learning Rate: {hyperparams.get('learning_rate', 'N/A')}")
        print(f"    Epochs: {len(val_loss)}")
        if final_val_loss is not None:
            print(f"    최종 Val Loss: {final_val_loss:.4f}")
        if max_val_acc is not None:
            print(f"    최고 Val Accuracy: {max_val_acc:.2f}%")


def main():
    """메인 함수"""
    # README.md의 표에 해당하는 파일명들 (확장자 없이)
    # 주의: 1번 "기본 설정"은 히스토리 파일이 없을 수 있으므로 제외
    file_names = [
        # 2. SGD with Nestrov, Learning Rate 0.1
        "wideresnet16_8_sgd_crossentropy_bs128_ep100_lr0.1_mom0.9_nesterov_schcosineannealinglr_tmax100_aug_autoaug_winit",
        # 3. SGD with Nestrov, ASAM (rho=2.0), Learning Rate 0.1
        "wideresnet16_8_sgd_crossentropy_bs128_ep100_lr0.1_mom0.9_nesterov_schcosineannealinglr_tmax100_aug_autoaug_winit_sam_samrho2.0_samadaptive",
        # 4. SGD with Nestrov, ASAM (rho=2.0), Learning Rate 0.1, EMA
        "wideresnet16_8_sgd_crossentropy_bs128_ep100_lr0.1_mom0.9_nesterov_schcosineannealinglr_tmax100_aug_autoaug_winit_ema_emad0.999_sam_samrho2.0_samadaptive",
        # 6. SGD with Nestrov, Learning Rate 0.1, Label Smoothing 0.1, Epoch 200, Use CIFAR-10 Normalize
        "wideresnet16_8_sgd_crossentropy_bs128_ep200_lr0.1_mom0.9_nesterov_schcosineannealinglr_tmax200_ls0.1_aug_autoaug_winit_cifar_normalize",
        
        "wideresnet16_8_sgd_crossentropy_bs128_ep100_lr0.1_mom0.9_nesterov_schcosineannealinglr_tmax100_ls0.1_aug_autoaug_winit_ema_emad0.999_sam_samrho2.0_samadaptive",
        "wideresnet16_8_sgd_crossentropy_bs128_ep100_lr0.1_mom0.9_nesterov_schcosineannealinglr_tmax100_ls0.1_aug_autoaug_winit_cifar_normalize_ema_emad0.999_sam_samrho2.0_samadaptive_history"
    ]
    
    print("\n" + "="*80)
    print("WideResNet16_8 비교 시작")
    print("="*80)
    
    histories = load_histories(file_names, output_dir="outputs/final2/wideresnet")
    
    if histories:
        # 요약 정보 출력
        print_summary(histories)
        
        # 그래프 작성
        plot_wideresnet_comparison(histories)
    else:
        print("WideResNet16_8 비교용 history 파일을 찾을 수 없습니다.")


if __name__ == "__main__":
    main()

