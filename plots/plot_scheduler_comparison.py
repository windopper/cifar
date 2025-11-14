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


def load_histories_from_dir(output_dir="outputs/scheduler"):
    """outputs/scheduler 디렉토리에서 모든 history 파일을 로드"""
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


def parse_scheduler(file_name):
    """파일명에서 scheduler 추출"""
    # scheduler 추출 (schcosineannealinglr, schonecyclelr, schexponentiallr, schreducelronplateau)
    scheduler_match = re.search(r'sch(cosineannealinglr|onecyclelr|exponentiallr|reducelronplateau)', file_name.lower())
    scheduler = scheduler_match.group(1) if scheduler_match else 'unknown'
    
    return scheduler


def get_scheduler_display_name(scheduler):
    """Scheduler 표시 이름 반환"""
    display_names = {
        'cosineannealinglr': 'Cosine Annealing LR',
        'onecyclelr': 'One Cycle LR',
        'exponentiallr': 'Exponential LR',
        'reducelronplateau': 'ReduceLROnPlateau'
    }
    return display_names.get(scheduler.lower(), scheduler.capitalize())


def plot_scheduler_comparison(histories, save_path="comparison/scheduler_comparison.png"):
    """Scheduler 비교 그래프 작성"""
    
    if not histories:
        print("로드된 history 파일이 없습니다.")
        return
    
    # scheduler별로 그룹화
    scheduler_data = []
    
    for history in histories:
        file_name = history.get('file_name', 'unknown')
        scheduler = parse_scheduler(file_name)
        
        if scheduler == 'unknown':
            print(f"Warning: {file_name}에서 scheduler를 추출할 수 없습니다.")
            continue
        
        scheduler_data.append({
            'history': history,
            'scheduler': scheduler,
            'file_name': file_name
        })
    
    if not scheduler_data:
        print("유효한 데이터가 없습니다.")
        return
    
    # 각 파일의 epoch 길이 확인 및 가장 짧은 epoch 찾기
    all_epoch_lengths = []
    for item in scheduler_data:
        epoch_len = len(item['history'].get('train_loss', []))
        all_epoch_lengths.append(epoch_len)
    
    if not all_epoch_lengths:
        print("유효한 데이터가 없습니다.")
        return
    
    min_epochs = min(all_epoch_lengths)
    print(f"\n가장 짧은 epoch 길이: {min_epochs}")
    
    # 그래프 설정
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Scheduler 비교', fontsize=18, fontweight='bold', y=1.02)
    fig.patch.set_facecolor('white')
    
    # Scheduler별 색상 팔레트
    scheduler_colors = {
        'cosineannealinglr': '#2E86AB',      # 진한 파란색
        'onecyclelr': '#F24236',             # 산호색
        'exponentiallr': '#06A77D',          # 청록색
        'reducelronplateau': '#9B59B6',      # 보라색
    }
    
    # 각 scheduler에 대한 데이터 준비
    plot_data = []
    
    for item in scheduler_data:
        history = item['history']
        scheduler = item['scheduler']
        
        train_loss = history.get('train_loss', [])[:min_epochs]
        val_loss = history.get('val_loss', [])[:min_epochs]
        val_acc = history.get('val_accuracy', [])[:min_epochs]
        
        epochs = list(range(1, min_epochs + 1))
        
        # 최고 validation accuracy 찾기
        max_val_acc = max(val_acc) if val_acc else 0
        
        scheduler_display = get_scheduler_display_name(scheduler)
        color = scheduler_colors.get(scheduler, '#000000')
        
        plot_data.append({
            'label': scheduler_display,
            'scheduler': scheduler,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'epochs': epochs,
            'color': color,
            'max_val_acc': max_val_acc
        })
    
    # Scheduler 이름 순으로 정렬 (README 순서와 동일하게)
    scheduler_order = ['cosineannealinglr', 'onecyclelr', 'exponentiallr', 'reducelronplateau']
    plot_data.sort(key=lambda x: scheduler_order.index(x['scheduler']) if x['scheduler'] in scheduler_order else 999)
    
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
                linewidth=2.5,
                marker='o', 
                markersize=6,
                markevery=5,
                alpha=0.85,
                markerfacecolor='white',
                markeredgewidth=1.5,
                markeredgecolor=data['color'])
    
    style_axis(ax1, 'Train Loss', 'Train Loss')
    ax1.legend(fontsize=10, loc='best', frameon=True, fancybox=True, shadow=True, 
               framealpha=0.9, edgecolor='#E0E0E0', facecolor='white')
    
    # 2. Validation Loss
    ax2 = axes[1]
    for data in plot_data:
        ax2.plot(data['epochs'], data['val_loss'],
                label=data['label'], color=data['color'],
                linewidth=2.5,
                marker='s',
                markersize=6,
                markevery=5,
                alpha=0.85,
                markerfacecolor='white',
                markeredgewidth=1.5,
                markeredgecolor=data['color'])
    
    style_axis(ax2, 'Validation Loss', 'Validation Loss')
    ax2.legend(fontsize=10, loc='best', frameon=True, fancybox=True, shadow=True,
               framealpha=0.9, edgecolor='#E0E0E0', facecolor='white')
    
    # 3. Validation Accuracy
    ax3 = axes[2]
    for data in plot_data:
        ax3.plot(data['epochs'], data['val_accuracy'],
                label=f"{data['label']} (최고: {data['max_val_acc']:.2f}%)", 
                color=data['color'],
                linewidth=2.5,
                marker='^',
                markersize=6,
                markevery=5,
                alpha=0.85,
                markerfacecolor='white',
                markeredgewidth=1.5,
                markeredgecolor=data['color'])
    
    style_axis(ax3, 'Validation Accuracy', 'Validation Accuracy (%)')
    ax3.legend(fontsize=10, loc='best', frameon=True, fancybox=True, shadow=True,
               framealpha=0.9, edgecolor='#E0E0E0', facecolor='white')
    
    # 전체 레이아웃 조정
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n그래프가 저장되었습니다: {save_path}")
    plt.close()
    
    # 결과 요약 출력
    print("\n=== Scheduler별 최고 Validation Accuracy ===")
    for data in plot_data:
        print(f"{data['label']}: {data['max_val_acc']:.2f}%")


def main():
    """메인 함수"""
    print("outputs/scheduler 디렉토리에서 history 파일을 로드합니다...")
    
    # outputs/scheduler 디렉토리에서 모든 history 파일 로드
    histories = load_histories_from_dir("outputs/scheduler")
    
    if not histories:
        print("\n로드된 history 파일이 없습니다.")
        print("outputs/scheduler 디렉토리에 _history.json 파일이 있는지 확인하세요.")
        return
    
    print(f"\n총 {len(histories)}개의 파일이 로드되었습니다.")
    
    # 각 파일의 scheduler 출력
    print("\n로드된 파일 목록:")
    for i, history in enumerate(histories, 1):
        file_name = history.get('file_name', 'unknown')
        scheduler = parse_scheduler(file_name)
        scheduler_display = get_scheduler_display_name(scheduler)
        print(f"  {i}. {file_name}")
        print(f"     → Scheduler: {scheduler_display}")
    
    # 그래프 작성
    plot_scheduler_comparison(histories)


if __name__ == "__main__":
    main()

