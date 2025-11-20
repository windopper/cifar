import json
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import numpy as np
import pandas as pd

# Seaborn 스타일 설정
sns.set_style("whitegrid")
sns.set_palette("husl")
sns.set_context("paper", font_scale=1.2)

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
plt.rcParams['figure.facecolor'] = 'white'


def load_histories_from_dir(output_dir):
    """디렉토리에서 모든 history 파일을 로드"""
    histories = []
    
    if not os.path.exists(output_dir):
        print(f"Warning: {output_dir} 디렉토리를 찾을 수 없습니다.")
        return histories
    
    # 디렉토리 내의 모든 _history.json 파일 찾기
    pattern = os.path.join(output_dir, "*_history.json")
    file_paths = glob.glob(pattern)
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
                history['file_path'] = file_path
                history['file_name'] = os.path.basename(file_path)
                histories.append(history)
        except Exception as e:
            print(f"Warning: {file_path} 로드 실패: {e}")
    
    return histories


def extract_model_info(file_name):
    """파일명에서 모델 정보 추출"""
    info = {
        'model_type': 'unknown',
        'model_variant': None,  # bn, se 등 모델 변형
        'has_winit': False,
        'has_scheduler': False,
        'scheduler': None
    }
    
    file_name_lower = file_name.lower()
    
    # 모델 타입 추출
    if file_name_lower.startswith('baseline_') and not file_name_lower.startswith('baseline_deep'):
        info['model_type'] = 'baseline'
    elif file_name_lower.startswith('deep_baseline_'):
        info['model_type'] = 'deep_baseline'
        # 모델 변형 추출 (bn, se 등)
        if '_bn_' in file_name_lower:
            info['model_variant'] = 'bn'
        elif '_se_' in file_name_lower:
            info['model_variant'] = 'se'
    
    # weight_init 확인
    if 'winit' in file_name_lower:
        info['has_winit'] = True
    
    # scheduler 확인
    if 'schcosineannealinglr' in file_name_lower:
        info['has_scheduler'] = True
        info['scheduler'] = 'cosineannealinglr'
    elif 'schonecyclelr' in file_name_lower:
        info['has_scheduler'] = True
        info['scheduler'] = 'onecyclelr'
    elif 'schexponentiallr' in file_name_lower:
        info['has_scheduler'] = True
        info['scheduler'] = 'exponentiallr'
    elif 'schreducelronplateau' in file_name_lower:
        info['has_scheduler'] = True
        info['scheduler'] = 'reducelronplateau'
    
    return info


def create_comparison_label(info):
    """비교를 위한 레이블 생성"""
    model_name = info['model_type'].replace('_', ' ').title()
    if info['model_variant']:
        model_name += f" ({info['model_variant'].upper()})"
    label_parts = [model_name]
    
    if info['has_winit']:
        label_parts.append('w/ Weight Init')
    
    if info['has_scheduler']:
        scheduler_display = {
            'cosineannealinglr': 'CosineAnnealingLR',
            'onecyclelr': 'OneCycleLR',
            'exponentiallr': 'ExponentialLR',
            'reducelronplateau': 'ReduceLROnPlateau'
        }.get(info['scheduler'], info['scheduler'])
        label_parts.append(f'w/ {scheduler_display}')
    
    return ' + '.join(label_parts)


def plot_model_comparison(histories, model_type, save_path):
    """특정 모델 타입의 비교 그래프 작성"""
    
    if not histories:
        print(f"{model_type} 모델의 history 파일이 없습니다.")
        return
    
    # 각 파일의 epoch 길이 확인 및 가장 짧은 epoch 찾기
    epoch_lengths = []
    for history in histories:
        epoch_len = len(history.get('train_loss', []))
        epoch_lengths.append(epoch_len)
    
    if not epoch_lengths:
        print(f"{model_type} 모델에 유효한 데이터가 없습니다.")
        return
    
    min_epochs = min(epoch_lengths)
    print(f"\n{model_type} 모델의 가장 짧은 epoch 길이: {min_epochs}")
    
    # A4 용지 크기에 맞게 그래프 설정 (세로 방향: 가로 8.27 인치, 높이 축소)
    fig, axes = plt.subplots(3, 1, figsize=(8.27, 7))
    fig.patch.set_facecolor('white')
    
    # 각 모델에 대한 데이터 준비
    plot_data = []
    
    for history in histories:
        file_name = history.get('file_name', 'unknown')
        info = extract_model_info(file_name)
        
        if info['model_type'] != model_type:
            continue
        
        train_loss = history.get('train_loss', [])[:min_epochs]
        val_loss = history.get('val_loss', [])[:min_epochs]
        val_acc = history.get('val_accuracy', [])[:min_epochs]
        
        epochs = list(range(1, min_epochs + 1))
        
        # 최고 validation accuracy 찾기
        max_val_acc = max(val_acc) if val_acc else 0
        
        label = create_comparison_label(info)
        
        plot_data.append({
            'label': label,
            'model_type': info['model_type'],
            'model_variant': info.get('model_variant'),
            'file_name': file_name,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'epochs': epochs,
            'max_val_acc': max_val_acc,
            'has_winit': info['has_winit'],
            'has_scheduler': info['has_scheduler']
        })
    
    if not plot_data:
        print(f"{model_type} 모델에 대한 플롯 데이터가 없습니다.")
        plt.close()
        return
    
    # 모델 변형별로 정렬
    plot_data.sort(key=lambda x: (
        x['model_variant'] or '',
        x['has_winit'],
        x['has_scheduler']
    ))
    
    # 각 모델 조합에 대해 고유한 스타일 할당
    # 색상 팔레트: 명확하게 구분되는 색상들
    colors = [
        '#2E86AB',  # 진한 파란색
        '#06A77D',  # 청록색
        '#F24236',  # 산호색
        '#9B59B6',  # 보라색
        '#E67E22',  # 주황색
        '#1ABC9C',  # 터키석색
    ]
    
    # 선 스타일: 실선, 점선, 점선-점 등
    linestyles = ['-', '--', '-.', ':', '-', '--']
    
    # 마커 스타일
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    for i, data in enumerate(plot_data):
        data['color'] = colors[i % len(colors)]
        data['linestyle'] = linestyles[i % len(linestyles)]
        data['marker'] = markers[i % len(markers)]
    
    # 공통 스타일 함수 (제목 제거)
    def style_axis(ax, ylabel, show_xlabel=False):
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        ax.tick_params(colors='#666666', labelsize=11)
        if show_xlabel:
            ax.set_xlabel('Epoch', fontsize=13, fontweight='bold', color='#333333')
        else:
            ax.set_xlabel('', fontsize=13, fontweight='bold', color='#333333')
            ax.tick_params(labelbottom=False)  # x축 틱 레이블 숨기기
        ax.set_ylabel(ylabel, fontsize=13, fontweight='bold', color='#333333')
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
                markersize=7,
                markevery=max(1, min_epochs // 10),
                alpha=0.9,
                markerfacecolor='white',
                markeredgewidth=2,
                markeredgecolor=data['color'])
    
    style_axis(ax1, 'Train Loss', show_xlabel=False)
    ax1.legend(fontsize=10, loc='best', frameon=True, fancybox=True, shadow=True, 
               framealpha=0.95, edgecolor='#E0E0E0', facecolor='white',
               columnspacing=0.8, handlelength=2.5, handletextpad=0.5)
    
    # 2. Validation Loss
    ax2 = axes[1]
    for data in plot_data:
        ax2.plot(data['epochs'], data['val_loss'],
                label=data['label'], 
                color=data['color'],
                linewidth=2.5,
                linestyle=data['linestyle'],
                marker=data['marker'],
                markersize=7,
                markevery=max(1, min_epochs // 10),
                alpha=0.9,
                markerfacecolor='white',
                markeredgewidth=2,
                markeredgecolor=data['color'])
    
    style_axis(ax2, 'Validation Loss', show_xlabel=False)
    ax2.legend(fontsize=10, loc='best', frameon=True, fancybox=True, shadow=True,
               framealpha=0.95, edgecolor='#E0E0E0', facecolor='white',
               columnspacing=0.8, handlelength=2.5, handletextpad=0.5)
    
    # 3. Validation Accuracy
    ax3 = axes[2]
    for data in plot_data:
        ax3.plot(data['epochs'], data['val_accuracy'],
                label=f"{data['label']} (최고: {data['max_val_acc']:.2f}%)", 
                color=data['color'],
                linewidth=2.5,
                linestyle=data['linestyle'],
                marker=data['marker'],
                markersize=7,
                markevery=max(1, min_epochs // 10),
                alpha=0.9,
                markerfacecolor='white',
                markeredgewidth=2,
                markeredgecolor=data['color'])
    
    style_axis(ax3, 'Validation Accuracy (%)', show_xlabel=True)
    ax3.legend(fontsize=10, loc='best', frameon=True, fancybox=True, shadow=True,
               framealpha=0.95, edgecolor='#E0E0E0', facecolor='white',
               columnspacing=0.8, handlelength=2.5, handletextpad=0.5)
    
    # 전체 레이아웃 조정 - 세로 배치에 맞게 여백 조정
    plt.subplots_adjust(left=0.12, right=0.95, top=0.97, bottom=0.08, hspace=0.25)
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n그래프가 저장되었습니다: {save_path}")
    plt.close()
    
    # 결과 요약 출력
    print(f"\n=== {model_type} 모델별 최고 Validation Accuracy ===")
    for data in plot_data:
        print(f"{data['label']}: {data['max_val_acc']:.2f}%")


def plot_baseline_deep_baseline_comparison(
    baseline_dir="outputs/baseline",
    deep_baseline_dir="outputs/deep_baseline",
    baseline_save_path="comparison/baseline_comparison.png",
    deep_baseline_save_path="comparison/deep_baseline_comparison.png"
):
    """Baseline과 Deep Baseline을 각각 별도 그래프로 작성"""
    
    # 히스토리 파일 로드
    baseline_histories = load_histories_from_dir(baseline_dir)
    deep_baseline_histories = load_histories_from_dir(deep_baseline_dir)
    
    print(f"\n로드된 파일:")
    print(f"  - Baseline: {len(baseline_histories)}개")
    print(f"  - Deep Baseline: {len(deep_baseline_histories)}개")
    
    # Baseline 그래프 생성
    if baseline_histories:
        plot_model_comparison(baseline_histories, 'baseline', baseline_save_path)
    
    # Deep Baseline 그래프 생성
    if deep_baseline_histories:
        plot_model_comparison(deep_baseline_histories, 'deep_baseline', deep_baseline_save_path)


def main():
    """메인 함수"""
    print("Baseline과 Deep Baseline 비교 그래프를 생성합니다...")
    
    # 그래프 작성
    plot_baseline_deep_baseline_comparison(
        baseline_dir="outputs/baseline",
        deep_baseline_dir="outputs/deep_baseline",
        baseline_save_path="comparison/baseline_comparison.png",
        deep_baseline_save_path="comparison/deep_baseline_comparison.png"
    )


if __name__ == "__main__":
    main()

