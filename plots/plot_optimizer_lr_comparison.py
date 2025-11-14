import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# matplotlib 스타일 설정 (한글 폰트 설정 전에)
plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'seaborn-darkgrid' if 'seaborn-darkgrid' in plt.style.available else 'default')

# 한글 폰트 설정 (스타일 적용 후에 다시 설정)
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
plt.rcParams['font.size'] = 10

# 추가 스타일 설정
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = '#E0E0E0'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linewidth'] = 0.8


def load_histories_from_dir(output_dir="outputs/optimizer"):
    """outputs/optimizer 디렉토리에서 모든 history 파일을 로드"""
    histories = []
    
    if not os.path.exists(output_dir):
        print(f"Warning: {output_dir} 디렉토리를 찾을 수 없습니다.")
        return histories
    
    # 디렉토리 내의 모든 _history.json 파일 찾기
    for file_name in os.listdir(output_dir):
        if file_name.endswith("_history.json"):
            file_path = os.path.join(output_dir, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                    history['file_path'] = file_path
                    history['file_name'] = file_name
                    histories.append(history)
            except Exception as e:
                print(f"Warning: {file_path} 로드 실패: {e}")
    
    return histories


def parse_optimizer_and_lr(file_name):
    """파일명에서 optimizer와 learning rate 추출"""
    # optimizer 추출 (adam, adamw, sgd, adagrad, rmsprop)
    optimizer_match = re.search(r'_(adam|adamw|sgd|adagrad|rmsprop)_', file_name.lower())
    optimizer = optimizer_match.group(1).lower() if optimizer_match else 'unknown'
    
    # learning rate 추출 (lr0.001, lr0.01 등)
    lr_match = re.search(r'lr([\d.]+)', file_name.lower())
    lr = float(lr_match.group(1)) if lr_match else None
    
    return optimizer, lr


def get_optimizer_display_name(optimizer):
    """Optimizer 표시 이름 반환"""
    display_names = {
        'adam': 'Adam',
        'adamw': 'AdamW',
        'sgd': 'SGD',
        'adagrad': 'Adagrad',
        'rmsprop': 'RMSprop'
    }
    return display_names.get(optimizer.lower(), optimizer.capitalize())


def plot_optimizer_lr_comparison(histories, save_path="comparison/optimizer_lr_comparison.png"):
    """Optimizer별 Learning Rate 비교 그래프 작성"""
    
    if not histories:
        print("로드된 history 파일이 없습니다.")
        return
    
    # optimizer별로 그룹화
    optimizer_groups = defaultdict(list)
    
    for history in histories:
        file_name = history.get('file_name', 'unknown')
        optimizer, lr = parse_optimizer_and_lr(file_name)
        
        if optimizer == 'unknown' or lr is None:
            print(f"Warning: {file_name}에서 optimizer 또는 learning rate를 추출할 수 없습니다.")
            continue
        
        optimizer_groups[optimizer].append({
            'history': history,
            'lr': lr,
            'file_name': file_name
        })
    
    if not optimizer_groups:
        print("유효한 데이터가 없습니다.")
        return
    
    # 각 optimizer별로 learning rate 순으로 정렬
    for optimizer in optimizer_groups:
        optimizer_groups[optimizer].sort(key=lambda x: x['lr'])
    
    # 각 파일의 epoch 길이 확인 및 가장 짧은 epoch 찾기
    all_epoch_lengths = []
    for optimizer, group in optimizer_groups.items():
        for item in group:
            epoch_len = len(item['history'].get('train_loss', []))
            all_epoch_lengths.append(epoch_len)
    
    if not all_epoch_lengths:
        print("유효한 데이터가 없습니다.")
        return
    
    min_epochs = min(all_epoch_lengths)
    print(f"\n가장 짧은 epoch 길이: {min_epochs}")
    
    # 그래프 설정 (더 큰 크기와 여백)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Optimizer별 Learning Rate 비교', fontsize=18, fontweight='bold', y=1.02)
    fig.patch.set_facecolor('white')
    
    # 더 세련된 색상 팔레트 (optimizer별로 다른 색상)
    optimizer_colors = {
        'adam': '#2E86AB',      # 진한 파란색
        'adamw': '#F24236',     # 산호색
        'sgd': '#06A77D',       # 청록색
        'adagrad': '#9B59B6',   # 보라색
        'rmsprop': '#E67E22',   # 주황색
    }
    
    # Learning rate별 선 스타일과 마커
    lr_styles = {
        0.0001: {'linestyle': '-', 'marker': 'o', 'markevery': 5, 'linewidth': 2.5},
        0.001: {'linestyle': '--', 'marker': 's', 'markevery': 5, 'linewidth': 2.5},
        0.01: {'linestyle': '-.', 'marker': '^', 'markevery': 5, 'linewidth': 2.5},
        0.1: {'linestyle': ':', 'marker': 'D', 'markevery': 5, 'linewidth': 2.5},
    }
    
    # 각 optimizer에 대한 데이터 준비
    plot_data = []
    
    for optimizer, group in optimizer_groups.items():
        optimizer_color = optimizer_colors.get(optimizer, '#000000')
        optimizer_display = get_optimizer_display_name(optimizer)
        
        for item in group:
            history = item['history']
            lr = item['lr']
            
            train_loss = history.get('train_loss', [])[:min_epochs]
            val_loss = history.get('val_loss', [])[:min_epochs]
            val_acc = history.get('val_accuracy', [])[:min_epochs]
            
            epochs = list(range(1, min_epochs + 1))
            
            # 최고 validation accuracy 찾기
            max_val_acc = max(val_acc) if val_acc else 0
            
            label = f"{optimizer_display} (lr={lr})"
            
            lr_style = lr_styles.get(lr, {'linestyle': '-', 'marker': 'o', 'markevery': 5, 'linewidth': 2.5})
            
            plot_data.append({
                'label': label,
                'optimizer': optimizer,
                'lr': lr,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'epochs': epochs,
                'color': optimizer_color,
                'linestyle': lr_style['linestyle'],
                'marker': lr_style['marker'],
                'markevery': lr_style['markevery'],
                'linewidth': lr_style['linewidth'],
                'max_val_acc': max_val_acc
            })
    
    # Optimizer별, Learning Rate 순으로 정렬
    plot_data.sort(key=lambda x: (x['optimizer'], x['lr']))
    
    # 공통 스타일 함수
    def style_axis(ax, title, ylabel):
        ax.set_facecolor('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['bottom'].set_color('#CCCCCC')
        ax.tick_params(colors='#666666', labelsize=10)
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold', color='#333333')
        ax.set_ylabel(ylabel, fontsize=12, fontweight='bold', color='#333333')
        ax.set_title(title, fontsize=14, fontweight='bold', color='#333333', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, color='#CCCCCC')
    
    # 1. Train Loss
    ax1 = axes[0]
    for data in plot_data:
        ax1.plot(data['epochs'], data['train_loss'],
                label=data['label'], color=data['color'],
                linestyle=data['linestyle'],
                linewidth=data['linewidth'],
                marker=data['marker'], 
                markersize=6,
                markevery=data['markevery'],
                alpha=0.85,
                markerfacecolor='white',
                markeredgewidth=1.5,
                markeredgecolor=data['color'])
    
    style_axis(ax1, 'Train Loss', 'Train Loss')
    ax1.legend(fontsize=9, loc='best', frameon=True, fancybox=True, shadow=True, 
               framealpha=0.9, edgecolor='#E0E0E0', facecolor='white')
    
    # 2. Validation Loss
    ax2 = axes[1]
    for data in plot_data:
        ax2.plot(data['epochs'], data['val_loss'],
                label=data['label'], color=data['color'],
                linestyle=data['linestyle'],
                linewidth=data['linewidth'],
                marker=data['marker'],
                markersize=6,
                markevery=data['markevery'],
                alpha=0.85,
                markerfacecolor='white',
                markeredgewidth=1.5,
                markeredgecolor=data['color'])
    
    style_axis(ax2, 'Validation Loss', 'Validation Loss')
    ax2.legend(fontsize=9, loc='best', frameon=True, fancybox=True, shadow=True,
               framealpha=0.9, edgecolor='#E0E0E0', facecolor='white')
    
    # 3. Validation Accuracy
    ax3 = axes[2]
    for data in plot_data:
        ax3.plot(data['epochs'], data['val_accuracy'],
                label=f"{data['label']} (최고: {data['max_val_acc']:.2f}%)", 
                color=data['color'],
                linestyle=data['linestyle'],
                linewidth=data['linewidth'],
                marker=data['marker'],
                markersize=6,
                markevery=data['markevery'],
                alpha=0.85,
                markerfacecolor='white',
                markeredgewidth=1.5,
                markeredgecolor=data['color'])
    
    style_axis(ax3, 'Validation Accuracy', 'Validation Accuracy (%)')
    ax3.legend(fontsize=9, loc='best', frameon=True, fancybox=True, shadow=True,
               framealpha=0.9, edgecolor='#E0E0E0', facecolor='white')
    
    # 전체 레이아웃 조정
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n그래프가 저장되었습니다: {save_path}")
    plt.close()
    
    # 결과 요약 출력
    print("\n=== Optimizer별 최고 Validation Accuracy ===")
    optimizer_best = defaultdict(lambda: {'lr': None, 'acc': 0})
    
    for data in plot_data:
        optimizer = data['optimizer']
        lr = data['lr']
        acc = data['max_val_acc']
        
        if acc > optimizer_best[optimizer]['acc']:
            optimizer_best[optimizer] = {'lr': lr, 'acc': acc}
    
    for optimizer in sorted(optimizer_best.keys()):
        best = optimizer_best[optimizer]
        optimizer_display = get_optimizer_display_name(optimizer)
        print(f"{optimizer_display}: lr={best['lr']}, 최고 Accuracy={best['acc']:.2f}%")


def main():
    """메인 함수"""
    print("outputs/optimizer 디렉토리에서 history 파일을 로드합니다...")
    
    # outputs/optimizer 디렉토리에서 모든 history 파일 로드
    histories = load_histories_from_dir("outputs/optimizer")
    
    if not histories:
        print("\n로드된 history 파일이 없습니다.")
        print("outputs/optimizer 디렉토리에 _history.json 파일이 있는지 확인하세요.")
        return
    
    print(f"\n총 {len(histories)}개의 파일이 로드되었습니다.")
    
    # 각 파일의 optimizer와 learning rate 출력
    print("\n로드된 파일 목록:")
    for i, history in enumerate(histories, 1):
        file_name = history.get('file_name', 'unknown')
        optimizer, lr = parse_optimizer_and_lr(file_name)
        optimizer_display = get_optimizer_display_name(optimizer)
        print(f"  {i}. {file_name}")
        print(f"     → Optimizer: {optimizer_display}, Learning Rate: {lr}")
    
    # 그래프 작성
    plot_optimizer_lr_comparison(histories)


if __name__ == "__main__":
    main()

