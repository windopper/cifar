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
        'model_type': 'pyramidnet',
        'display_name': 'PyramidNet110-150'
    }
    
    file_name_lower = file_name.lower()
    
    # Epoch 정보 추출
    if 'ep200' in file_name_lower:
        info['display_name'] = 'PyramidNet110-150 (Epoch 200)'
        info['epoch'] = 200
    elif 'ep400' in file_name_lower:
        info['display_name'] = 'PyramidNet110-150 (Epoch 400)'
        info['epoch'] = 400
    
    return info


def plot_pyramidnet_comparison(history_files, save_path="comparison/pyramidnet_comparison.png"):
    """PyramidNet 모델들의 비교 그래프 작성"""
    
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
    
    # 각 파일의 epoch 길이 확인
    epoch_lengths = []
    for history in histories:
        epoch_len = len(history.get('train_loss', []))
        epoch_lengths.append(epoch_len)
    
    if not epoch_lengths:
        print("유효한 데이터가 없습니다.")
        return
    
    print(f"각 모델의 epoch 길이: {epoch_lengths}")
    
    # 그래프 설정 (가로 방향)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.patch.set_facecolor('white')
    
    # 각 모델에 대한 데이터 준비
    plot_data = []
    
    for history in histories:
        file_name = history.get('file_name', 'unknown')
        info = extract_model_info(file_name)
        
        # 전체 데이터 사용 (자르지 않음)
        train_loss = history.get('train_loss', [])
        val_loss = history.get('val_loss', [])
        val_acc = history.get('val_accuracy', [])
        
        # 각 모델의 전체 epoch 길이에 맞춰 epochs 생성
        num_epochs = len(train_loss)
        epochs = list(range(1, num_epochs + 1))
        
        # 최고 validation accuracy 찾기 및 해당 epoch 찾기
        if val_acc:
            max_val_acc = max(val_acc)
            max_val_acc_epoch = val_acc.index(max_val_acc) + 1  # epoch는 1부터 시작
        else:
            max_val_acc = 0
            max_val_acc_epoch = 0
        
        label = info['display_name']
        
        plot_data.append({
            'label': label,
            'file_name': file_name,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'epochs': epochs,
            'num_epochs': num_epochs,
            'max_val_acc': max_val_acc,
            'max_val_acc_epoch': max_val_acc_epoch,
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
                markevery=max(1, data['num_epochs'] // 10),
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
                markevery=max(1, data['num_epochs'] // 10),
                alpha=0.85,
                markerfacecolor='white',
                markeredgewidth=2,
                markeredgecolor=data['color'])
    
    style_axis(ax2, 'Validation Loss')
    
    # 3. Validation Accuracy (최고점 표시 포함)
    ax3 = axes[2]
    for data in plot_data:
        ax3.plot(data['epochs'], data['val_accuracy'],
                label=f"{data['label']} (Max: {data['max_val_acc']:.2f}%)", 
                color=data['color'],
                linewidth=2.5,
                linestyle=data['linestyle'],
                marker=data['marker'],
                markersize=6,
                markevery=max(1, data['num_epochs'] // 10),
                alpha=0.85,
                markerfacecolor='white',
                markeredgewidth=2,
                markeredgecolor=data['color'])
        
        # 최고점 표시
        if data['max_val_acc_epoch'] > 0:
            max_epoch_idx = data['max_val_acc_epoch'] - 1  # 인덱스는 0부터 시작
            if max_epoch_idx < len(data['val_accuracy']):
                ax3.scatter(data['max_val_acc_epoch'], data['max_val_acc'],
                           color=data['color'],
                           s=150,  # 마커 크기
                           zorder=5,  # 다른 요소 위에 표시
                           edgecolors='black',
                           linewidths=2,
                           marker='*')  # 별 모양 마커
                
                # 최고점에 텍스트 주석 추가
                ax3.annotate(f'{data["max_val_acc"]:.2f}%',
                            xy=(data['max_val_acc_epoch'], data['max_val_acc']),
                            xytext=(10, 10),
                            textcoords='offset points',
                            fontsize=10,
                            fontweight='bold',
                            color=data['color'],
                            bbox=dict(boxstyle='round,pad=0.3', 
                                    facecolor='white', 
                                    edgecolor=data['color'],
                                    alpha=0.8),
                            arrowprops=dict(arrowstyle='->',
                                         connectionstyle='arc3,rad=0',
                                         color=data['color'],
                                         lw=1.5))
    
    style_axis(ax3, 'Validation Accuracy (%)')
    
    # 공통 legend를 그래프 아래에 배치 (글씨 크기 증가)
    handles, labels = ax3.get_legend_handles_labels()
    fig.legend(handles, labels, 
               loc='lower center',
               bbox_to_anchor=(0.5, -0.15),
               ncol=2,
               fontsize=11,
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
        print(f"{data['label']}: {data['max_val_acc']:.2f}% (Epoch {data['max_val_acc_epoch']})")


def main():
    """메인 함수"""
    print("PyramidNet 모델 비교 그래프를 생성합니다...")
    
    # 비교할 히스토리 파일 목록 (README.md 308-309 라인 참조)
    history_files = [
        "outputs/final2/pyramidnet/pyramidnet110_150_sgd_crossentropy_bs128_ep200_lr0.1_mom0.9_nesterov_schcosineannealinglr_tmax200_ls0.1_aug_autoaug_winit_ema_emad0.999_sam_samrho2.0_samadaptive_shakedrop_history.json",
        "outputs/final2/pyramidnet/pyramidnet110_150_sgd_crossentropy_bs128_ep400_lr0.1_mom0.9_nesterov_schcosineannealinglr_tmax400_ls0.1_aug_autoaug_winit_ema_emad0.999_sam_samrho2.0_samadaptive_shakedrop_history.json"
    ]
    
    # 그래프 작성
    plot_pyramidnet_comparison(
        history_files=history_files,
        save_path="comparison/pyramidnet_comparison.png"
    )


if __name__ == "__main__":
    main()

