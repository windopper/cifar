import json
import os
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


def plot_model_comparison(histories, save_path="comparison/model_comparison.png"):
    """모델 비교 그래프 작성"""

    if not histories:
        print("로드된 history 파일이 없습니다.")
        return

    # 각 파일의 epoch 길이 확인 및 가장 짧은 epoch 찾기
    epoch_lengths = []
    for history in histories:
        hyperparams = history.get('hyperparameters', {})
        net_name = hyperparams.get('net', 'unknown')
        file_name = history.get('file_name', 'unknown')
        epoch_len = len(history.get('train_loss', []))
        epoch_lengths.append({
            'net_name': net_name,
            'file_name': file_name,
            'epoch_len': epoch_len
        })

    if not epoch_lengths or all(e['epoch_len'] == 0 for e in epoch_lengths):
        print("유효한 데이터가 없습니다.")
        return

    # 가장 짧은 epoch 찾기
    min_epoch_info = min(epoch_lengths, key=lambda x: x['epoch_len'])
    min_epochs = min_epoch_info['epoch_len']

    print(f"\n각 모델의 epoch 길이:")
    for info in epoch_lengths:
        marker = " ← 가장 짧음" if info == min_epoch_info else ""
        print(
            f"  - {info['net_name']} ({info['file_name']}): {info['epoch_len']} epochs{marker}")
    print(f"\n가장 짧은 epoch 길이: {min_epochs} (모델: {min_epoch_info['net_name']})")

    # 그래프 설정
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('모델 성능 비교', fontsize=16, fontweight='bold')

    # 색상 팔레트
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

    # 각 모델에 대한 데이터 준비
    model_data = []

    for i, history in enumerate(histories):
        hyperparams = history.get('hyperparameters', {})
        net_name = hyperparams.get('net', 'unknown')

        train_loss = history.get('train_loss', [])[:min_epochs]
        val_loss = history.get('val_loss', [])[:min_epochs]
        val_acc = history.get('val_accuracy', [])[:min_epochs]

        epochs = list(range(1, min_epochs + 1))

        model_data.append({
            'label': net_name,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'epochs': epochs,
            'color': colors[i]
        })

    # 1. Train Loss
    ax1 = axes[0]
    for data in model_data:
        ax1.plot(data['epochs'], data['train_loss'],
                 label=data['label'], color=data['color'],
                 linewidth=2, marker='o', markersize=3)

    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Train Loss', fontsize=11)
    ax1.set_title('Train Loss', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # 2. Validation Loss
    ax2 = axes[1]
    for data in model_data:
        ax2.plot(data['epochs'], data['val_loss'],
                 label=data['label'], color=data['color'],
                 linewidth=2, marker='s', markersize=3)

    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Validation Loss', fontsize=11)
    ax2.set_title('Validation Loss', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. Validation Accuracy
    ax3 = axes[2]
    for data in model_data:
        ax3.plot(data['epochs'], data['val_accuracy'],
                 label=data['label'], color=data['color'],
                 linewidth=2, marker='^', markersize=3)

    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Validation Accuracy (%)', fontsize=11)
    ax3.set_title('Validation Accuracy', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # 디렉토리 생성
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(
        save_path) else '.', exist_ok=True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"그래프가 저장되었습니다: {save_path}")
    plt.close()


def plot_model_comparison_20_epoch(save_path="comparison/model_comparison_20_epoch.png"):
    """20 Epoch 기준 모델 성능 비교 그래프 작성"""
    file_names = [
        "deep_baseline2_bn_residual_preact_adam_crossentropy_bs16_ep20_lr0.0003_mom0.9_schcosineannealinglr_tmax20_history.json",
        "deep_baseline_adam_crossentropy_bs16_ep20_lr0.0003_mom0.9_schcosineannealinglr_tmax20_history.json",
        "deep_baseline_bn_adam_crossentropy_bs16_ep20_lr0.0003_mom0.9_schcosineannealinglr_tmax20_history.json",
        "deep_baseline2_bn_adam_crossentropy_bs16_ep20_lr0.0003_mom0.9_schcosineannealinglr_tmax20_history.json",
        "deep_baseline2_bn_residual_adam_crossentropy_bs16_ep20_lr0.0003_mom0.9_schcosineannealinglr_tmax20_history.json",
        "deep_baseline2_bn_resnext_adam_crossentropy_bs16_ep20_lr0.0003_mom0.9_schcosineannealinglr_tmax20_history.json",
        "deep_baseline3_bn_adam_crossentropy_bs16_ep20_lr0.0003_mom0.9_schcosineannealinglr_tmax20_history.json",
    ]

    if not file_names:
        print("비교할 파일 리스트가 비어있습니다. file_names 리스트에 파일명을 추가하세요.")
        return

    # History 파일 로드
    histories = load_histories(file_names)

    if not histories:
        print("로드된 history 파일이 없습니다.")
        return

    # 그래프 작성
    plot_model_comparison(histories, save_path=save_path)
    
def plot_model_comparison_100_epoch(save_path="comparison/model_comparison_100_epoch.png"):
    file_names = [
        "deep_baseline_bn_adam_crossentropy_bs128_ep100_lr0.001_mom0.9_schcosineannealinglr_tmax100_ls0.05_aug_cutmix_winit_history.json",
        "deep_baseline2_bn_adam_crossentropy_bs128_ep100_lr0.001_mom0.9_schcosineannealinglr_tmax100_ls0.05_aug_cutmix_winit_history.json",
        "deep_baseline2_bn_resnext_adam_crossentropy_bs128_ep100_lr0.001_mom0.9_schcosineannealinglr_tmax100_winit_history.json",
        "deep_baseline2_bn_residual_adam_crossentropy_bs128_ep100_lr0.001_mom0.9_schcosineannealinglr_tmax100_ls0.05_aug_cutmix_winit_history.json",
        "deep_baseline2_bn_residual_preact_adam_crossentropy_bs128_ep100_lr0.001_mom0.9_schcosineannealinglr_tmax100_ls0.05_aug_cutmix_winit_history.json",
    ]
    
    if not file_names:
        print("로드된 history 파일이 없습니다.")
        return
    
    histories = load_histories(file_names)
    
    # 100 Epoch 기준 모델 성능 비교 그래프 작성
    plot_model_comparison(histories, save_path=save_path)


if __name__ == "__main__":
    plot_model_comparison_20_epoch()
    plot_model_comparison_100_epoch()
