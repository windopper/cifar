import json
import os
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


def load_history(file_path):
    """히스토리 파일 로드"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            history = json.load(f)
            history['file_path'] = file_path
            history['file_name'] = os.path.basename(file_path)
            return history
    except Exception as e:
        print(f"Warning: {file_path} 로드 실패: {e}")
        return None


def extract_model_info(file_name):
    """파일명에서 모델 정보 추출"""
    info = {
        'model_type': 'unknown',
        'display_name': 'Unknown Model'
    }
    
    file_name_lower = file_name.lower()
    
    # 모델 타입 추출
    if 'wideresnet16_8' in file_name_lower:
        info['model_type'] = 'wideresnet'
        info['display_name'] = 'WideResNet16-8'
    elif 'deep_baseline3_bn_residual_15' in file_name_lower:
        info['model_type'] = 'deep_baseline3_bn_residual_15'
        info['display_name'] = 'Deep Baseline3 BN Residual 15'
    elif 'deep_baseline3_bn_residual' in file_name_lower:
        info['model_type'] = 'deep_baseline3_bn_residual'
        info['display_name'] = 'Deep Baseline3 BN Residual'
    elif 'deep_baseline_bn' in file_name_lower:
        info['model_type'] = 'deep_baseline_bn'
        info['display_name'] = 'Deep Baseline BN'
    
    return info


def plot_final_comparison(history_files, save_path="comparison/final_comparison.png"):
    """최종 모델들의 비교 그래프 작성"""
    
    # 히스토리 파일 로드
    histories = []
    for file_path in history_files:
        history = load_history(file_path)
        if history:
            histories.append(history)
    
    if not histories:
        print("유효한 히스토리 파일이 없습니다.")
        return
    
    print(f"\n로드된 파일: {len(histories)}개")
    
    # 각 파일의 epoch 길이 확인 및 가장 짧은 epoch 찾기
    epoch_lengths = []
    for history in histories:
        epoch_len = len(history.get('train_loss', []))
        epoch_lengths.append(epoch_len)
    
    if not epoch_lengths:
        print("유효한 데이터가 없습니다.")
        return
    
    min_epochs = min(epoch_lengths)
    print(f"가장 짧은 epoch 길이: {min_epochs}")
    
    # 그래프 설정 (가로 방향)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor('white')
    
    # 각 모델에 대한 데이터 준비
    plot_data = []
    
    for history in histories:
        file_name = history.get('file_name', 'unknown')
        info = extract_model_info(file_name)
        
        train_loss = history.get('train_loss', [])[:min_epochs]
        val_loss = history.get('val_loss', [])[:min_epochs]
        val_acc = history.get('val_accuracy', [])[:min_epochs]
        
        epochs = list(range(1, min_epochs + 1))
        
        # 최고 validation accuracy 찾기
        max_val_acc = max(val_acc) if val_acc else 0
        
        label = info['display_name']
        
        plot_data.append({
            'label': label,
            'file_name': file_name,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'epochs': epochs,
            'max_val_acc': max_val_acc,
            'model_type': info['model_type']
        })
    
    if not plot_data:
        print("플롯 데이터가 없습니다.")
        plt.close()
        return
    
    # 정렬: 최고 validation accuracy 순서
    plot_data.sort(key=lambda x: x['max_val_acc'], reverse=True)
    
    # 색상 팔레트 - seaborn 컬러 팔레트 사용
    colors = sns.color_palette("husl", len(plot_data))
    
    # 선 스타일과 마커
    linestyles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D']
    
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
        ax1.plot(data['epochs'], data['train_loss'],
                label=data['label'], 
                color=data['color'],
                linewidth=2.5,
                linestyle=data['linestyle'],
                marker=data['marker'], 
                markersize=6,
                markevery=max(1, min_epochs // 10),
                alpha=0.85,
                markerfacecolor='white',
                markeredgewidth=2,
                markeredgecolor=data['color'])
    
    style_axis(ax1, 'Train Loss')
    
    # 2. Validation Loss
    ax2 = axes[1]
    for data in plot_data:
        ax2.plot(data['epochs'], data['val_loss'],
                label=data['label'], 
                color=data['color'],
                linewidth=2.5,
                linestyle=data['linestyle'],
                marker=data['marker'],
                markersize=6,
                markevery=max(1, min_epochs // 10),
                alpha=0.85,
                markerfacecolor='white',
                markeredgewidth=2,
                markeredgecolor=data['color'])
    
    style_axis(ax2, 'Validation Loss')
    
    # 3. Validation Accuracy
    ax3 = axes[2]
    for data in plot_data:
        ax3.plot(data['epochs'], data['val_accuracy'],
                label=f"{data['label']} (Max: {data['max_val_acc']:.2f}%)", 
                color=data['color'],
                linewidth=2.5,
                linestyle=data['linestyle'],
                marker=data['marker'],
                markersize=6,
                markevery=max(1, min_epochs // 10),
                alpha=0.85,
                markerfacecolor='white',
                markeredgewidth=2,
                markeredgecolor=data['color'])
    
    style_axis(ax3, 'Validation Accuracy (%)')
    
    # 공통 legend를 그래프 아래에 배치 (글씨 크기 증가)
    handles, labels = ax3.get_legend_handles_labels()
    fig.legend(handles, labels, 
               loc='lower center',
               bbox_to_anchor=(0.5, -0.15),
               ncol=2,
               fontsize=11,  # 9.5에서 11로 증가
               frameon=True,
               fancybox=True,
               shadow=True,
               framealpha=0.95,
               edgecolor='#E0E0E0',
               facecolor='white')
    
    # 전체 레이아웃 조정
    plt.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.15, wspace=0.25)
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n그래프가 저장되었습니다: {save_path}")
    plt.close()
    
    # 결과 요약 출력
    print(f"\n=== 모델별 최고 Validation Accuracy ===")
    for data in plot_data:
        print(f"{data['label']}: {data['max_val_acc']:.2f}%")


def main():
    """메인 함수"""
    print("최종 모델 비교 그래프를 생성합니다...")
    
    # 비교할 히스토리 파일 목록
    history_files = [
        "outputs/final2/wideresnet/wideresnet16_8_adam_crossentropy_bs128_ep100_lr0.0003_mom0.9_schcosineannealinglr_tmax100_aug_autoaug_winit_history.json",
        "outputs/final2/deep_baseline3_bn_residual_15_adam_crossentropy_bs128_ep100_lr0.0003_mom0.9_schcosineannealinglr_tmax100_aug_autoaug_winit_history.json",
        "outputs/final2/deep_baseline3_bn_residual_adam_crossentropy_bs128_ep100_lr0.0003_mom0.9_schcosineannealinglr_tmax100_aug_autoaug_winit_history.json",
        "outputs/augmentation/deep_baseline_bn_adam_crossentropy_bs128_ep100_lr0.0003_mom0.9_schcosineannealinglr_tmax100_aug_autoaug_winit_history.json"
    ]
    
    # 그래프 작성
    plot_final_comparison(
        history_files=history_files,
        save_path="comparison/final_comparison.png"
    )


if __name__ == "__main__":
    main()

