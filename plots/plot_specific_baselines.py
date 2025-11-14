import json
import os
import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

def load_specific_histories(file_names, output_dir="outputs"):
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

def plot_optimizer_comparison(histories, save_path="comparison/specific_baselines_optimizer_comparison.png"):
    """특정 baseline 모델들의 optimizer별 비교 그래프 작성"""
    
    if not histories:
        print("로드된 history 파일이 없습니다.")
        return
    
    # 그래프 설정
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Baseline 모델: Optimizer별 성능 비교', fontsize=16, fontweight='bold')
    
    # 색상 팔레트
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    
    # 각 모델에 대한 데이터 준비
    model_data = []
    
    for i, history in enumerate(histories):
        hyperparams = history.get('hyperparameters', {})
        optimizer = hyperparams.get('optimizer', 'unknown')
        lr = hyperparams.get('learning_rate', 'N/A')
        epochs = hyperparams.get('epochs', len(history.get('train_loss', [])))
        
        # 레이블 생성
        label = optimizer.upper()
        if optimizer == 'adamw':
            wd = hyperparams.get('weight_decay', 'N/A')
            label += f" (wd={wd})"
        if epochs != 24:
            label += f" (ep={epochs})"
        
        train_loss = history.get('train_loss', [])
        val_loss = history.get('val_loss', [])
        val_acc = history.get('val_accuracy', [])
        
        model_data.append({
            'label': label,
            'optimizer': optimizer,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'epochs': list(range(1, len(train_loss) + 1)),
            'color': colors[i],
            'file_name': history.get('file_name', 'unknown')
        })
    
    # 1. Train Loss
    ax1 = axes[0, 0]
    for data in model_data:
        ax1.plot(data['epochs'], data['train_loss'], 
                label=data['label'], color=data['color'], 
                linewidth=2, marker='o', markersize=4)
    
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Train Loss', fontsize=11)
    ax1.set_title('Train Loss 비교', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Validation Loss
    ax2 = axes[0, 1]
    for data in model_data:
        ax2.plot(data['epochs'], data['val_loss'], 
                label=data['label'], color=data['color'], 
                linewidth=2, marker='s', markersize=4)
    
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Validation Loss', fontsize=11)
    ax2.set_title('Validation Loss 비교', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Validation Accuracy
    ax3 = axes[1, 0]
    for data in model_data:
        ax3.plot(data['epochs'], data['val_accuracy'], 
                label=data['label'], color=data['color'], 
                linewidth=2, marker='^', markersize=4)
    
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Validation Accuracy (%)', fontsize=11)
    ax3.set_title('Validation Accuracy 비교', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. 최종 성능 요약 (Bar Chart)
    ax4 = axes[1, 1]
    final_val_losses = []
    final_val_accs = []
    optimizer_labels = []
    bar_colors = []
    
    for data in model_data:
        if data['val_loss'] and data['val_accuracy']:
            final_val_losses.append(data['val_loss'][-1])
            final_val_accs.append(data['val_accuracy'][-1])
            optimizer_labels.append(data['label'])
            bar_colors.append(data['color'])
    
    if final_val_losses and final_val_accs:
        x_pos = np.arange(len(optimizer_labels))
        width = 0.35
        
        # Loss와 Accuracy를 같은 스케일로 비교하기 위해 정규화
        max_loss = max(final_val_losses)
        min_loss = min(final_val_losses)
        normalized_loss = [(max_loss - loss) / (max_loss - min_loss) if max_loss != min_loss else 0.5 
                          for loss in final_val_losses]
        normalized_acc = [acc / 100 for acc in final_val_accs]
        
        ax4.bar(x_pos - width/2, normalized_loss, width, label='Loss (정규화, 역)', 
               color=bar_colors, alpha=0.7)
        ax4.bar(x_pos + width/2, normalized_acc, width, label='Accuracy (정규화)', 
               color=bar_colors, alpha=0.7)
        
        ax4.set_xlabel('Optimizer', fontsize=11)
        ax4.set_ylabel('정규화된 값', fontsize=11)
        ax4.set_title('최종 성능 요약 (정규화)', fontsize=12, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(optimizer_labels, rotation=15, ha='right')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"그래프가 저장되었습니다: {save_path}")
    plt.close()

def plot_scheduler_comparison(histories, save_path="comparison/specific_baselines_scheduler_comparison.png"):
    """특정 baseline 모델들의 scheduler별 비교 그래프 작성"""
    
    if not histories:
        print("로드된 history 파일이 없습니다.")
        return
    
    # 그래프 설정
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Baseline 모델: Scheduler별 성능 비교', fontsize=16, fontweight='bold')
    
    # 색상 팔레트
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    
    # 각 모델에 대한 데이터 준비
    model_data = []
    
    for i, history in enumerate(histories):
        hyperparams = history.get('hyperparameters', {})
        filename = history.get('file_name', 'unknown')
        
        # 파일명에서 스케줄러 정보 추출
        scheduler_name = 'None'
        scheduler_params = ''
        
        # 베이스라인 파일 확인 (스케줄러가 없는 경우)
        is_baseline = 'baseline_sgd_crossentropy_bs16_ep24_lr0.001_mom0.9_history' in filename and \
                     'schexponentiallr' not in filename and \
                     'schonecyclelr' not in filename and \
                     'schreducelronplateau' not in filename
        
        if is_baseline:
            scheduler_name = 'Baseline'
        elif 'schexponentiallr' in filename:
            scheduler_name = 'ExponentialLR'
            if 'gamma0.95' in filename:
                scheduler_params = ' (γ=0.95)'
        elif 'schonecyclelr' in filename:
            scheduler_name = 'OneCycleLR'
        elif 'schreducelronplateau' in filename:
            scheduler_name = 'ReduceLROnPlateau'
            if 'factor0.1' in filename:
                scheduler_params = ' (factor=0.1'
            if 'patience3' in filename:
                scheduler_params += ', patience=3)' if scheduler_params else ' (patience=3)'
        
        # 레이블 생성
        label = scheduler_name + scheduler_params
        
        train_loss = history.get('train_loss', [])
        val_loss = history.get('val_loss', [])
        val_acc = history.get('val_accuracy', [])
        
        model_data.append({
            'label': label,
            'scheduler': scheduler_name,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'epochs': list(range(1, len(train_loss) + 1)),
            'color': colors[i],
            'file_name': history.get('file_name', 'unknown')
        })
    
    # 1. Train Loss
    ax1 = axes[0, 0]
    for data in model_data:
        ax1.plot(data['epochs'], data['train_loss'], 
                label=data['label'], color=data['color'], 
                linewidth=2, marker='o', markersize=4)
    
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Train Loss', fontsize=11)
    ax1.set_title('Train Loss 비교', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Validation Loss
    ax2 = axes[0, 1]
    for data in model_data:
        ax2.plot(data['epochs'], data['val_loss'], 
                label=data['label'], color=data['color'], 
                linewidth=2, marker='s', markersize=4)
    
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Validation Loss', fontsize=11)
    ax2.set_title('Validation Loss 비교', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Validation Accuracy
    ax3 = axes[1, 0]
    for data in model_data:
        ax3.plot(data['epochs'], data['val_accuracy'], 
                label=data['label'], color=data['color'], 
                linewidth=2, marker='^', markersize=4)
    
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Validation Accuracy (%)', fontsize=11)
    ax3.set_title('Validation Accuracy 비교', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. 최종 성능 요약 (Bar Chart)
    ax4 = axes[1, 1]
    final_val_losses = []
    final_val_accs = []
    scheduler_labels = []
    bar_colors = []
    
    for data in model_data:
        if data['val_loss'] and data['val_accuracy']:
            final_val_losses.append(data['val_loss'][-1])
            final_val_accs.append(data['val_accuracy'][-1])
            scheduler_labels.append(data['label'])
            bar_colors.append(data['color'])
    
    if final_val_losses and final_val_accs:
        x_pos = np.arange(len(scheduler_labels))
        width = 0.35
        
        # Loss와 Accuracy를 같은 스케일로 비교하기 위해 정규화
        max_loss = max(final_val_losses)
        min_loss = min(final_val_losses)
        normalized_loss = [(max_loss - loss) / (max_loss - min_loss) if max_loss != min_loss else 0.5 
                          for loss in final_val_losses]
        normalized_acc = [acc / 100 for acc in final_val_accs]
        
        ax4.bar(x_pos - width/2, normalized_loss, width, label='Loss (정규화, 역)', 
               color=bar_colors, alpha=0.7)
        ax4.bar(x_pos + width/2, normalized_acc, width, label='Accuracy (정규화)', 
               color=bar_colors, alpha=0.7)
        
        ax4.set_xlabel('Scheduler', fontsize=11)
        ax4.set_ylabel('정규화된 값', fontsize=11)
        ax4.set_title('최종 성능 요약 (정규화)', fontsize=12, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(scheduler_labels, rotation=15, ha='right')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"그래프가 저장되었습니다: {save_path}")
    plt.close()

def plot_technique_comparison(histories, save_path="comparison/specific_baselines_technique_comparison.png"):
    """특정 baseline 모델들의 기법별 비교 그래프 작성"""
    
    if not histories:
        print("로드된 history 파일이 없습니다.")
        return
    
    # 그래프 설정
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Baseline 모델: 기법별 성능 비교', fontsize=16, fontweight='bold')
    
    # 색상 팔레트
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))
    
    # 각 모델에 대한 데이터 준비
    model_data = []
    
    for i, history in enumerate(histories):
        hyperparams = history.get('hyperparameters', {})
        filename = history.get('file_name', 'unknown')
        
        # 파일명에서 기법 정보 추출
        technique_name = 'Baseline'
        
        if 'calibrated' in filename:
            technique_name = 'Calibrated'
        elif 'aug' in filename:
            technique_name = 'Data Augmentation'
        elif 'ls0.1' in filename:
            technique_name = 'Label Smoothing (0.1)'
        
        # 레이블 생성
        label = technique_name
        
        train_loss = history.get('train_loss', [])
        val_loss = history.get('val_loss', [])
        val_acc = history.get('val_accuracy', [])
        
        model_data.append({
            'label': label,
            'technique': technique_name,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'epochs': list(range(1, len(train_loss) + 1)),
            'color': colors[i],
            'file_name': history.get('file_name', 'unknown')
        })
    
    # 1. Train Loss
    ax1 = axes[0, 0]
    for data in model_data:
        ax1.plot(data['epochs'], data['train_loss'], 
                label=data['label'], color=data['color'], 
                linewidth=2, marker='o', markersize=4)
    
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Train Loss', fontsize=11)
    ax1.set_title('Train Loss 비교', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # 2. Validation Loss
    ax2 = axes[0, 1]
    for data in model_data:
        ax2.plot(data['epochs'], data['val_loss'], 
                label=data['label'], color=data['color'], 
                linewidth=2, marker='s', markersize=4)
    
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Validation Loss', fontsize=11)
    ax2.set_title('Validation Loss 비교', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # 3. Validation Accuracy
    ax3 = axes[1, 0]
    for data in model_data:
        ax3.plot(data['epochs'], data['val_accuracy'], 
                label=data['label'], color=data['color'], 
                linewidth=2, marker='^', markersize=4)
    
    ax3.set_xlabel('Epoch', fontsize=11)
    ax3.set_ylabel('Validation Accuracy (%)', fontsize=11)
    ax3.set_title('Validation Accuracy 비교', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. 최종 성능 요약 (Bar Chart)
    ax4 = axes[1, 1]
    final_val_losses = []
    final_val_accs = []
    technique_labels = []
    bar_colors = []
    
    for data in model_data:
        if data['val_loss'] and data['val_accuracy']:
            final_val_losses.append(data['val_loss'][-1])
            final_val_accs.append(data['val_accuracy'][-1])
            technique_labels.append(data['label'])
            bar_colors.append(data['color'])
    
    if final_val_losses and final_val_accs:
        x_pos = np.arange(len(technique_labels))
        width = 0.35
        
        # Loss와 Accuracy를 같은 스케일로 비교하기 위해 정규화
        max_loss = max(final_val_losses)
        min_loss = min(final_val_losses)
        normalized_loss = [(max_loss - loss) / (max_loss - min_loss) if max_loss != min_loss else 0.5 
                          for loss in final_val_losses]
        normalized_acc = [acc / 100 for acc in final_val_accs]
        
        ax4.bar(x_pos - width/2, normalized_loss, width, label='Loss (정규화, 역)', 
               color=bar_colors, alpha=0.7)
        ax4.bar(x_pos + width/2, normalized_acc, width, label='Accuracy (정규화)', 
               color=bar_colors, alpha=0.7)
        
        ax4.set_xlabel('기법', fontsize=11)
        ax4.set_ylabel('정규화된 값', fontsize=11)
        ax4.set_title('최종 성능 요약 (정규화)', fontsize=12, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(technique_labels, rotation=15, ha='right')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"그래프가 저장되었습니다: {save_path}")
    plt.close()

def print_summary(histories, comparison_type="Optimizer"):
    """모델별 요약 정보 출력"""
    print("\n" + "="*60)
    print(f"Baseline 모델 {comparison_type}별 요약")
    print("="*60)
    
    for i, history in enumerate(histories, 1):
        hyperparams = history.get('hyperparameters', {})
        filename = history.get('file_name', 'unknown')
        
        optimizer = hyperparams.get('optimizer', 'N/A')
        lr = hyperparams.get('learning_rate', 'N/A')
        epochs = hyperparams.get('epochs', len(history.get('train_loss', [])))
        
        val_loss = history.get('val_loss', [])
        val_acc = history.get('val_accuracy', [])
        
        final_val_loss = val_loss[-1] if val_loss else None
        final_val_acc = val_acc[-1] if val_acc else None
        
        print(f"\n[{i}] {filename}")
        print(f"    Optimizer: {optimizer.upper()}")
        print(f"    Learning Rate: {lr}")
        if hyperparams.get('weight_decay'):
            print(f"    Weight Decay: {hyperparams.get('weight_decay')}")
        
        # 스케줄러 정보 출력
        if comparison_type == "Scheduler":
            # 베이스라인 파일 확인
            is_baseline = 'baseline_sgd_crossentropy_bs16_ep24_lr0.001_mom0.9_history' in filename and \
                         'schexponentiallr' not in filename and \
                         'schonecyclelr' not in filename and \
                         'schreducelronplateau' not in filename
            
            if is_baseline:
                print(f"    Scheduler: Baseline (None)")
            elif 'schexponentiallr' in filename:
                scheduler_name = 'ExponentialLR'
                if 'gamma0.95' in filename:
                    print(f"    Scheduler: {scheduler_name} (γ=0.95)")
                else:
                    print(f"    Scheduler: {scheduler_name}")
            elif 'schonecyclelr' in filename:
                scheduler_name = 'OneCycleLR'
                print(f"    Scheduler: {scheduler_name}")
            elif 'schreducelronplateau' in filename:
                scheduler_name = 'ReduceLROnPlateau'
                print(f"    Scheduler: {scheduler_name} (factor=0.1, patience=3)")
            else:
                print(f"    Scheduler: None")
        
        # 기법 정보 출력
        if comparison_type == "Technique":
            technique_name = 'Baseline'
            if 'calibrated' in filename:
                technique_name = 'Calibrated'
                print(f"    기법: {technique_name}")
            elif 'aug' in filename:
                technique_name = 'Data Augmentation'
                print(f"    기법: {technique_name}")
            elif 'ls0.1' in filename:
                technique_name = 'Label Smoothing (0.1)'
                print(f"    기법: {technique_name}")
            else:
                print(f"    기법: {technique_name}")
        
        print(f"    Epochs: {epochs}")
        if final_val_loss is not None:
            print(f"    최종 Val Loss: {final_val_loss:.4f}")
        if final_val_acc is not None:
            print(f"    최종 Val Accuracy: {final_val_acc:.2f}%")

def main():
    """메인 함수"""
    # 베이스라인 파일명
    baseline_file_name = "baseline_sgd_crossentropy_bs16_ep24_lr0.001_mom0.9_history"
    
    # Optimizer 비교용 파일명들 (확장자 없이)
    optimizer_file_names = [
        baseline_file_name,  # 베이스라인을 첫 번째로
        "baseline_adam_crossentropy_bs16_ep24_lr0.001_mom0.9_history",
        "baseline_adamw_crossentropy_bs16_ep24_lr0.001_mom0.9_wd0.0005_history",
        "baseline_adagrad_crossentropy_bs16_ep24_lr0.001_mom0.9_history"
    ]
    
    # Scheduler 비교용 파일명들 (확장자 없이)
    scheduler_file_names = [
        baseline_file_name,  # 베이스라인을 첫 번째로
        "baseline_adam_crossentropy_bs16_ep24_lr0.001_mom0.9_schexponentiallr_gamma0.95_history",
        "baseline_adam_crossentropy_bs16_ep24_lr0.001_mom0.9_schonecyclelr_history",
        "baseline_adam_crossentropy_bs16_ep24_lr0.001_mom0.9_schreducelronplateau_factor0.1_patience3_history"
    ]
    
    # 기법 비교용 파일명들 (확장자 없이)
    technique_file_names = [
        "baseline_sgd_crossentropy_bs16_ep24_lr0.001_mom0.9_history",
        "baseline_sgd_crossentropy_bs16_ep24_lr0.001_mom0.9_calibrated_history",
        "baseline_sgd_crossentropy_bs16_ep24_lr0.001_mom0.9_aug_history",
        "baseline_sgd_crossentropy_bs16_ep24_lr0.001_mom0.9_ls0.1_history"
    ]
    
    # Optimizer 비교
    print("\n" + "="*60)
    print("Optimizer 비교 시작")
    print("="*60)
    optimizer_histories = load_specific_histories(optimizer_file_names)
    
    if optimizer_histories:
        # 요약 정보 출력
        print_summary(optimizer_histories, comparison_type="Optimizer")
        
        # 그래프 작성
        plot_optimizer_comparison(optimizer_histories)
    else:
        print("Optimizer 비교용 history 파일을 찾을 수 없습니다.")
    
    # Scheduler 비교
    print("\n" + "="*60)
    print("Scheduler 비교 시작")
    print("="*60)
    scheduler_histories = load_specific_histories(scheduler_file_names)
    
    if scheduler_histories:
        # 요약 정보 출력
        print_summary(scheduler_histories, comparison_type="Scheduler")
        
        # 그래프 작성
        plot_scheduler_comparison(scheduler_histories)
    else:
        print("Scheduler 비교용 history 파일을 찾을 수 없습니다.")
    
    # 기법 비교
    print("\n" + "="*60)
    print("기법 비교 시작")
    print("="*60)
    technique_histories = load_specific_histories(technique_file_names)
    
    if technique_histories:
        # 요약 정보 출력
        print_summary(technique_histories, comparison_type="Technique")
        
        # 그래프 작성
        plot_technique_comparison(technique_histories)
    else:
        print("기법 비교용 history 파일을 찾을 수 없습니다.")

if __name__ == "__main__":
    main()

