import json
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from collections import defaultdict

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

def load_history_files(output_dir="outputs"):
    """outputs 디렉토리에서 baseline 모델의 history 파일들을 로드"""
    history_files = glob.glob(os.path.join(output_dir, "*_history.json"))
    
    baseline_histories = []
    for file_path in history_files:
        # baseline 모델만 필터링 (deep_baseline 제외)
        filename = os.path.basename(file_path)
        if filename.startswith("baseline_") and not filename.startswith("baseline_deep"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    history = json.load(f)
                    # 파일 경로도 함께 저장
                    history['file_path'] = file_path
                    baseline_histories.append(history)
            except Exception as e:
                print(f"Warning: {file_path} 로드 실패: {e}")
    
    return baseline_histories

def group_by_criterion(histories):
    """criterion별로 히스토리 그룹화"""
    grouped = defaultdict(list)
    
    for history in histories:
        criterion = history.get('hyperparameters', {}).get('criterion', 'unknown')
        grouped[criterion].append(history)
    
    return grouped

def plot_criterion_comparison(grouped_histories, save_path="comparison/baseline_criteria_comparison.png"):
    """다양한 criterion에 대한 loss 및 accuracy 그래프 작성"""
    
    # criterion 개수 확인
    criteria = sorted(grouped_histories.keys())
    
    if not criteria:
        print("baseline 모델의 history 파일을 찾을 수 없습니다.")
        return
    
    # 그래프 설정
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Baseline 모델: Criterion별 성능 비교', fontsize=16, fontweight='bold')
    
    # 색상 팔레트
    colors = plt.cm.tab10(np.linspace(0, 1, len(criteria)))
    color_map = {criterion: colors[i] for i, criterion in enumerate(criteria)}
    
    # 각 criterion에 대해 평균 계산
    criterion_data = {}
    
    for criterion in criteria:
        histories = grouped_histories[criterion]
        
        # 가장 긴 epoch 길이 찾기
        max_epochs = max(len(h.get('train_loss', [])) for h in histories)
        
        # 각 메트릭별로 평균 계산
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(max_epochs):
            epoch_train_losses = []
            epoch_val_losses = []
            epoch_val_accs = []
            
            for history in histories:
                train_loss = history.get('train_loss', [])
                val_loss = history.get('val_loss', [])
                val_acc = history.get('val_accuracy', [])
                
                if epoch < len(train_loss):
                    epoch_train_losses.append(train_loss[epoch])
                if epoch < len(val_loss):
                    epoch_val_losses.append(val_loss[epoch])
                if epoch < len(val_acc):
                    epoch_val_accs.append(val_acc[epoch])
            
            train_losses.append(np.mean(epoch_train_losses) if epoch_train_losses else None)
            val_losses.append(np.mean(epoch_val_losses) if epoch_val_losses else None)
            val_accuracies.append(np.mean(epoch_val_accs) if epoch_val_accs else None)
        
        criterion_data[criterion] = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'val_accuracy': val_accuracies,
            'epochs': list(range(1, max_epochs + 1)),
            'count': len(histories)
        }
    
    # 1. Train Loss
    ax1 = axes[0, 0]
    for criterion in criteria:
        data = criterion_data[criterion]
        epochs = data['epochs']
        train_loss = [x for x in data['train_loss'] if x is not None]
        epochs_filtered = epochs[:len(train_loss)]
        
        label = f"{criterion.upper()}"
        if data['count'] > 1:
            label += f" (n={data['count']})"
        
        ax1.plot(epochs_filtered, train_loss, 
                label=label, color=color_map[criterion], linewidth=2, marker='o', markersize=4)
    
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Train Loss', fontsize=11)
    ax1.set_title('Train Loss 비교', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Validation Loss
    ax2 = axes[0, 1]
    for criterion in criteria:
        data = criterion_data[criterion]
        epochs = data['epochs']
        val_loss = [x for x in data['val_loss'] if x is not None]
        epochs_filtered = epochs[:len(val_loss)]
        
        label = f"{criterion.upper()}"
        if data['count'] > 1:
            label += f" (n={data['count']})"
        
        ax2.plot(epochs_filtered, val_loss, 
                label=label, color=color_map[criterion], linewidth=2, marker='s', markersize=4)
    
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Validation Loss', fontsize=11)
    ax2.set_title('Validation Loss 비교', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Validation Accuracy
    ax3 = axes[1, 0]
    for criterion in criteria:
        data = criterion_data[criterion]
        epochs = data['epochs']
        val_acc = [x for x in data['val_accuracy'] if x is not None]
        epochs_filtered = epochs[:len(val_acc)]
        
        label = f"{criterion.upper()}"
        if data['count'] > 1:
            label += f" (n={data['count']})"
        
        ax3.plot(epochs_filtered, val_acc, 
                label=label, color=color_map[criterion], linewidth=2, marker='^', markersize=4)
    
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Validation Accuracy (%)', fontsize=11)
    ax3.set_title('Validation Accuracy 비교', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. 최종 성능 요약 (Bar Chart)
    ax4 = axes[1, 1]
    final_val_losses = []
    final_val_accs = []
    criterion_labels = []
    
    for criterion in criteria:
        data = criterion_data[criterion]
        val_loss = [x for x in data['val_loss'] if x is not None]
        val_acc = [x for x in data['val_accuracy'] if x is not None]
        
        if val_loss and val_acc:
            final_val_losses.append(val_loss[-1])
            final_val_accs.append(val_acc[-1])
            criterion_labels.append(criterion.upper())
    
    if final_val_losses and final_val_accs:
        x_pos = np.arange(len(criterion_labels))
        width = 0.35
        
        # 정규화된 값으로 표시 (loss는 작을수록 좋고, accuracy는 클수록 좋음)
        normalized_loss = [1 - (loss / max(final_val_losses)) for loss in final_val_losses]
        normalized_acc = [acc / 100 for acc in final_val_accs]
        
        ax4.bar(x_pos - width/2, normalized_loss, width, label='Loss (정규화, 역)', 
               color=[color_map[c.lower()] for c in criterion_labels], alpha=0.7)
        ax4.bar(x_pos + width/2, normalized_acc, width, label='Accuracy (정규화)', 
               color=[color_map[c.lower()] for c in criterion_labels], alpha=0.7)
        
        ax4.set_xlabel('Criterion', fontsize=11)
        ax4.set_ylabel('정규화된 값', fontsize=11)
        ax4.set_title('최종 성능 요약 (정규화)', fontsize=12, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(criterion_labels)
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"그래프가 저장되었습니다: {save_path}")
    plt.show()

def print_summary(grouped_histories):
    """criterion별 요약 정보 출력"""
    print("\n" + "="*60)
    print("Baseline 모델 Criterion별 요약")
    print("="*60)
    
    for criterion in sorted(grouped_histories.keys()):
        histories = grouped_histories[criterion]
        print(f"\n[{criterion.upper()}]")
        print(f"  파일 개수: {len(histories)}")
        
        for i, history in enumerate(histories, 1):
            hyperparams = history.get('hyperparameters', {})
            filename = os.path.basename(history['file_path'])
            
            val_loss = history.get('val_loss', [])
            val_acc = history.get('val_accuracy', [])
            
            final_val_loss = val_loss[-1] if val_loss else None
            final_val_acc = val_acc[-1] if val_acc else None
            
            print(f"  [{i}] {filename}")
            print(f"      Optimizer: {hyperparams.get('optimizer', 'N/A')}")
            print(f"      Learning Rate: {hyperparams.get('learning_rate', 'N/A')}")
            print(f"      Epochs: {len(val_loss)}")
            if final_val_loss is not None:
                print(f"      최종 Val Loss: {final_val_loss:.4f}")
            if final_val_acc is not None:
                print(f"      최종 Val Accuracy: {final_val_acc:.2f}%")

def main():
    """메인 함수"""
    # History 파일 로드
    histories = load_history_files()
    
    if not histories:
        print("baseline 모델의 history 파일을 찾을 수 없습니다.")
        return
    
    # Criterion별로 그룹화
    grouped_histories = group_by_criterion(histories)
    
    # 요약 정보 출력
    print_summary(grouped_histories)
    
    # 그래프 작성
    plot_criterion_comparison(grouped_histories)

if __name__ == "__main__":
    main()

