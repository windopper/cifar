import json
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Seaborn 스타일 설정
sns.set_style("whitegrid")
sns.set_palette("husl")
sns.set_context("paper", font_scale=1.2)

# 기본 설정
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'


def load_histories_from_dir(output_dir):
    """디렉토리에서 모든 history 파일을 로드 (adam, last_bn_remove, remove_first_relu, cifar_normalize 제외)"""
    histories = []
    
    if not os.path.exists(output_dir):
        print(f"Warning: {output_dir} 디렉토리를 찾을 수 없습니다.")
        return histories
    
    # 디렉토리 내의 모든 _history.json 파일 찾기
    pattern = os.path.join(output_dir, "*_history.json")
    file_paths = glob.glob(pattern)
    
    excluded_count = 0
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        
        # adam, last_bn_remove, remove_first_relu, cifar_normalize 파일 제외
        if 'adam' in file_name.lower():
            print(f"Excluded: {file_name} (adam)")
            excluded_count += 1
            continue
        if 'last_bn_remove' in file_name:
            print(f"Excluded: {file_name} (last_bn_remove)")
            excluded_count += 1
            continue
        if 'remove_first_relu' in file_name:
            print(f"Excluded: {file_name} (remove_first_relu)")
            excluded_count += 1
            continue
        if 'cifar_normalize' in file_name:
            print(f"Excluded: {file_name} (cifar_normalize)")
            excluded_count += 1
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
                history['file_path'] = file_path
                history['file_name'] = file_name
                histories.append(history)
                print(f"Loaded: {file_name}")
        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
    
    print(f"\nTotal {excluded_count} files excluded")
    return histories


def extract_config_info(file_name, hyperparams):
    """파일명과 하이퍼파라미터에서 설정 정보 추출"""
    info = {
        'has_sam': False,
        'sam_rho': 0,
        'sam_adaptive': False,
        'has_ema': False,
        'ema_decay': 0,
        'label_smoothing': 0.0,
        'epochs': 0,
    }
    
    # SAM 여부
    info['has_sam'] = hyperparams.get('sam', False)
    info['sam_rho'] = hyperparams.get('sam_rho', 0)
    info['sam_adaptive'] = hyperparams.get('sam_adaptive', False)
    
    # EMA 여부
    info['has_ema'] = hyperparams.get('ema', False)
    info['ema_decay'] = hyperparams.get('ema_decay', 0)
    
    # Label Smoothing 여부
    info['label_smoothing'] = hyperparams.get('label_smoothing', 0.0)
    
    # Epoch 수
    info['epochs'] = hyperparams.get('epochs', 0)
    
    return info


def create_label(info):
    """설정 정보로부터 레이블 생성"""
    label_parts = ['SGD + Nesterov']
    
    if info['has_sam']:
        if info['sam_adaptive']:
            label_parts.append(f"ASAM(ρ={info['sam_rho']})")
        else:
            label_parts.append(f"SAM(ρ={info['sam_rho']})")
    
    if info['has_ema']:
        label_parts.append(f"EMA({info['ema_decay']})")
    
    if info['label_smoothing'] > 0:
        label_parts.append(f"LS={info['label_smoothing']}")
    
    if info['epochs'] != 100:
        label_parts.append(f"Ep={info['epochs']}")
    
    return ' + '.join(label_parts)


def plot_wideresnet_comparison(input_dir="outputs/final2/wideresnet", save_path="comparison/wideresnet_sgd_only_comparison.png"):
    """WideResNet16_8 SGD 모델들의 비교 그래프 작성"""
    
    # 히스토리 파일 로드
    histories = load_histories_from_dir(input_dir)
    
    if not histories:
        print(f"No history files found in {input_dir}")
        return
    
    print(f"\nLoaded files: {len(histories)}")
    
    # 각 파일의 epoch 길이 확인
    epoch_lengths = []
    for history in histories:
        epoch_len = len(history.get('train_loss', []))
        epoch_lengths.append(epoch_len)
        print(f"  - {history.get('file_name', 'unknown')}: {epoch_len} epochs")
    
    if not epoch_lengths:
        print("No valid data found.")
        return
    
    # 그래프 설정 (가로 방향)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.patch.set_facecolor('white')
    
    # 각 모델에 대한 데이터 준비
    plot_data = []
    
    for history in histories:
        file_name = history.get('file_name', 'unknown')
        hyperparams = history.get('hyperparameters', {})
        info = extract_config_info(file_name, hyperparams)
        
        train_loss = history.get('train_loss', [])
        val_loss = history.get('val_loss', [])
        val_acc = history.get('val_accuracy', [])
        
        epochs = list(range(1, len(train_loss) + 1))
        
        # 최고 validation accuracy 찾기
        max_val_acc = max(val_acc) if val_acc else 0
        max_val_acc_epoch = val_acc.index(max_val_acc) + 1 if val_acc else 0
        
        label = create_label(info)
        
        plot_data.append({
            'label': label,
            'file_name': file_name,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'epochs': epochs,
            'max_val_acc': max_val_acc,
            'max_val_acc_epoch': max_val_acc_epoch,
            'info': info
        })
    
    if not plot_data:
        print("No plot data available.")
        plt.close()
        return
    
    # 정렬: 기본 -> SAM -> SAM+EMA -> SAM+EMA+LS 순서
    def sort_key(x):
        info = x['info']
        return (
            info['has_sam'],
            info['has_ema'],
            info['label_smoothing'] > 0,
            -x['max_val_acc']  # 같은 설정이면 높은 정확도 순
        )
    
    plot_data.sort(key=sort_key)
    
    # 색상 팔레트 - seaborn 컬러 팔레트 사용
    colors = sns.color_palette("husl", len(plot_data))
    
    # 선 스타일과 마커
    linestyles = ['-', '--', '-.', ':', '-', '--', '-.']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']
    
    for i, data in enumerate(plot_data):
        data['color'] = colors[i % len(colors)]
        data['linestyle'] = linestyles[i % len(linestyles)]
        data['marker'] = markers[i % len(markers)]
    
    # 공통 스타일 함수
    def style_axis(ax, ylabel):
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        ax.tick_params(colors='#666666', labelsize=11)
        ax.set_xlabel('Epoch', fontsize=12, color='#333333')
        ax.set_ylabel(ylabel, fontsize=12, color='#333333')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, color='#CCCCCC')
    
    # 1. Train Loss
    ax1 = axes[0]
    for data in plot_data:
        epoch_len = len(data['epochs'])
        ax1.plot(data['epochs'], data['train_loss'],
                label=data['label'], 
                color=data['color'],
                linewidth=2.5,
                linestyle=data['linestyle'],
                marker=data['marker'], 
                markersize=6,
                markevery=max(1, epoch_len // 10),
                alpha=0.85,
                markerfacecolor='white',
                markeredgewidth=2,
                markeredgecolor=data['color'])
    
    style_axis(ax1, 'Train Loss')
    ax1.set_title('Train Loss', fontsize=14, fontweight='bold', pad=12, color='#333333')
    
    # 2. Validation Loss
    ax2 = axes[1]
    for data in plot_data:
        epoch_len = len(data['epochs'])
        ax2.plot(data['epochs'], data['val_loss'],
                label=data['label'], 
                color=data['color'],
                linewidth=2.5,
                linestyle=data['linestyle'],
                marker=data['marker'],
                markersize=6,
                markevery=max(1, epoch_len // 10),
                alpha=0.85,
                markerfacecolor='white',
                markeredgewidth=2,
                markeredgecolor=data['color'])
    
    style_axis(ax2, 'Validation Loss')
    ax2.set_title('Validation Loss', fontsize=14, fontweight='bold', pad=12, color='#333333')
    
    # 3. Validation Accuracy
    ax3 = axes[2]
    for data in plot_data:
        epoch_len = len(data['epochs'])
        ax3.plot(data['epochs'], data['val_accuracy'],
                label=f"{data['label']} (Max: {data['max_val_acc']:.2f}%)", 
                color=data['color'],
                linewidth=2.5,
                linestyle=data['linestyle'],
                marker=data['marker'],
                markersize=6,
                markevery=max(1, epoch_len // 10),
                alpha=0.85,
                markerfacecolor='white',
                markeredgewidth=2,
                markeredgecolor=data['color'])
    
    style_axis(ax3, 'Validation Accuracy (%)')
    ax3.set_title('Validation Accuracy', fontsize=14, fontweight='bold', pad=12, color='#333333')
    
    # 공통 legend를 그래프 아래에 배치
    handles, labels = ax3.get_legend_handles_labels()
    fig.legend(handles, labels, 
               loc='lower center',
               bbox_to_anchor=(0.5, -0.12),
               ncol=2,
               fontsize=9,
               frameon=True,
               fancybox=True,
               shadow=True,
               framealpha=0.95,
               edgecolor='#E0E0E0',
               facecolor='white')
    
    # 전체 레이아웃 조정
    plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.18, wspace=0.22)
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nGraph saved to: {save_path}")
    plt.close()
    
    # 결과 요약 출력
    print(f"\n{'='*100}")
    print(f"Best Validation Accuracy by Model (Sorted by Accuracy)")
    print(f"{'='*100}")
    
    # 정확도 순으로 정렬
    sorted_data = sorted(plot_data, key=lambda x: x['max_val_acc'], reverse=True)
    
    for i, data in enumerate(sorted_data, 1):
        print(f"\n[Rank {i}] {data['label']}")
        print(f"    File: {data['file_name']}")
        print(f"    Best Val Accuracy: {data['max_val_acc']:.2f}% (Epoch {data['max_val_acc_epoch']})")
        print(f"    Final Train Loss: {data['train_loss'][-1]:.4f}")
        print(f"    Final Val Loss: {data['val_loss'][-1]:.4f}")
        print(f"    Total Epochs: {len(data['epochs'])}")


def main():
    """메인 함수"""
    print("\n" + "="*100)
    print("WideResNet16_8 SGD Comparison (Excluding Adam, Last_BN_Remove, Remove_First_ReLU, CIFAR-Normalize)")
    print("="*100 + "\n")
    
    # 그래프 작성
    plot_wideresnet_comparison(
        input_dir="outputs/final2/wideresnet",
        save_path="comparison/wideresnet_sgd_only_comparison.png"
    )


if __name__ == "__main__":
    main()
