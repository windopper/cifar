import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import json
import os
from tqdm import tqdm
from main import get_net
import hashlib

# -----------------------------------------------------------
# 1. Configuration & Setup
# -----------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS_L1 = 30
STACKING_DIR = "stacking"
STACKING_META_FILE = os.path.join(STACKING_DIR, "meta.json")
STACKING_TRAIN_FILE = os.path.join(STACKING_DIR, "S_train.npy")
STACKING_TEST_FILE = os.path.join(STACKING_DIR, "S_test.npy")

def load_model_from_history(history_path):
    """History 파일에서 모델 정보를 추출"""
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"History 파일을 찾을 수 없습니다: {history_path}")
    
    with open(history_path, 'r') as f:
        history_data = json.load(f)
    
    # 모델 이름 추출
    if 'hyperparameters' in history_data and 'net' in history_data['hyperparameters']:
        model_name = history_data['hyperparameters']['net']
    else:
        raise ValueError(f"History 파일에 'hyperparameters.net' 정보가 없습니다: {history_path}")
    
    # 모델 파일 경로 자동 생성
    if history_path.endswith('_history.json'):
        model_path = history_path.replace('_history.json', '.pth')
    else:
        base_path = history_path.rsplit('.json', 1)[0]
        model_path = f"{base_path.rsplit('_history', 1)[0]}.pth"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    # Normalize 값 추출 (있는 경우)
    normalize_mean = (0.4914, 0.4822, 0.4465)  # CIFAR-10 기본값
    normalize_std = (0.2023, 0.1994, 0.2010)  # CIFAR-10 기본값
    if 'hyperparameters' in history_data:
        hp = history_data['hyperparameters']
        if 'normalize_mean' in hp and 'normalize_std' in hp:
            normalize_mean = tuple(hp['normalize_mean'])
            normalize_std = tuple(hp['normalize_std'])
    
    # ShakeDrop 확률 추출
    shakedrop_prob = 0.0
    if 'hyperparameters' in history_data:
        hp = history_data['hyperparameters']
        if 'shakedrop_prob' in hp and hp['shakedrop_prob'] is not None:
            shakedrop_prob = hp['shakedrop_prob']
    
    return model_name, model_path, normalize_mean, normalize_std, shakedrop_prob

def get_history_paths_hash(history_paths):
    """히스토리 경로들의 해시값을 생성"""
    paths_str = json.dumps(sorted(history_paths), sort_keys=True)
    return hashlib.md5(paths_str.encode()).hexdigest()

def load_cached_meta_features():
    """저장된 meta-features를 로드"""
    if not os.path.exists(STACKING_META_FILE):
        return None, None
    
    try:
        with open(STACKING_META_FILE, 'r') as f:
            meta_data = json.load(f)
        
        if not os.path.exists(STACKING_TRAIN_FILE) or not os.path.exists(STACKING_TEST_FILE):
            return None, None
        
        S_train = np.load(STACKING_TRAIN_FILE)
        S_test = np.load(STACKING_TEST_FILE)
        
        return meta_data, (S_train, S_test)
    except Exception as e:
        print(f"캐시 로드 중 오류 발생: {e}")
        return None, None

def save_meta_features(history_paths, S_train, S_test):
    """생성된 meta-features를 저장"""
    os.makedirs(STACKING_DIR, exist_ok=True)
    
    # 히스토리 경로 해시 저장
    meta_data = {
        'history_paths': history_paths,
        'history_paths_hash': get_history_paths_hash(history_paths),
        'num_models': len(history_paths),
        'num_classes': NUM_CLASSES
    }
    
    with open(STACKING_META_FILE, 'w') as f:
        json.dump(meta_data, f, indent=2)
    
    # NumPy 배열 저장
    np.save(STACKING_TRAIN_FILE, S_train)
    np.save(STACKING_TEST_FILE, S_test)
    
    print(f"\n>>> Meta-features 저장 완료: {STACKING_DIR}/ <<<")

# -----------------------------------------------------------
# 4. Helper Functions (Predict)
# -----------------------------------------------------------
def predict(model, loader):
    """모델 추론 함수"""
    model.eval()
    preds_list = []
    with torch.no_grad():
        for inputs, _ in tqdm(loader, desc="  예측 중", leave=False):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            # Softmax 대신 Logits를 내보내는 것이 Meta-Learner에게 더 풍부한 정보를 줌
            preds_list.append(outputs.cpu().numpy())
    return np.vstack(preds_list)

# -----------------------------------------------------------
# 6. Level 1 Model (Meta Learner) Construction
# -----------------------------------------------------------
class MetaLearner(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MetaLearner, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3), # Overfitting 방지 중요
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3), # Overfitting 방지 중요
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.net(x)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none', weight=alpha)

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss) # 예측 확률
        # 맞춘 샘플(pt가 큼)은 Loss가 0에 수렴, 틀린 샘플에 가중치 부여
        loss = (1 - pt) ** self.gamma * ce_loss 
        return loss.mean()

# -----------------------------------------------------------
# Main Execution
# -----------------------------------------------------------
if __name__ == '__main__':
    print(f"Running on {DEVICE}")
    
    # -----------------------------------------------------------
    # 2. 히스토리 파일에서 모델 정보 로드
    # -----------------------------------------------------------
    HISTORY_PATHS = [
        "outputs/final2/pyramidnet/pyramidnet110_150_sgd_crossentropy_bs128_ep400_lr0.1_mom0.9_nesterov_schcosineannealinglr_tmax400_ls0.1_aug_autoaug_winit_ema_emad0.999_sam_samrho2.0_samadaptive_shakedrop_history.json",
        "outputs/final2/pyramidnet/pyramidnet110_150_sgd_crossentropy_bs128_ep200_lr0.1_mom0.9_nesterov_schcosineannealinglr_tmax200_ls0.1_aug_autoaug_winit_ema_emad0.999_sam_samrho2.0_samadaptive_shakedrop_history.json",
        "outputs/final2/wideresnet/wideresnet16_8_sgd_crossentropy_bs128_ep200_lr0.1_mom0.9_nesterov_schcosineannealinglr_tmax200_ls0.1_aug_autoaug_winit_ema_emad0.999_sam_samrho2.0_samadaptive_history.json"
    ]
    
    # 히스토리 파일에서 모델 정보 로드
    model_infos = []
    for history_path in HISTORY_PATHS:
        model_name, model_path, normalize_mean, normalize_std, shakedrop_prob = load_model_from_history(history_path)
        model_infos.append({
            'name': model_name,
            'path': model_path,
            'normalize_mean': normalize_mean,
            'normalize_std': normalize_std,
            'shakedrop_prob': shakedrop_prob
        })
        print(f"모델 로드: {model_name}")
        print(f"  경로: {model_path}")
        print(f"  Normalize: mean={normalize_mean}, std={normalize_std}")
        print(f"  ShakeDrop: {shakedrop_prob}")
        print()
    
    # 첫 번째 모델의 normalize 값을 사용 (모든 모델이 동일한 데이터셋을 사용한다고 가정)
    normalize_mean = model_infos[0]['normalize_mean']
    normalize_std = model_infos[0]['normalize_std']
    
    # -----------------------------------------------------------
    # 3. Data Preparation (CIFAR-10)
    # -----------------------------------------------------------
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
    ])
    
    # Train 데이터는 원본 데이터 사용 (augmentation 없음)
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    # Stacking을 위해 원본 Train Label을 확보
    train_targets = np.array(train_dataset.targets)
    
    # -----------------------------------------------------------
    # 5. Stacking Pipeline: Generate Meta-Features
    # -----------------------------------------------------------
    # 저장된 meta-features 확인
    current_hash = get_history_paths_hash(HISTORY_PATHS)
    cached_meta, cached_data = load_cached_meta_features()
    
    if cached_meta is not None and cached_data is not None:
        cached_hash = cached_meta.get('history_paths_hash')
        if cached_hash == current_hash:
            print(">>> 저장된 Meta-features 발견! Level 0 건너뛰기 <<<")
            S_train, S_test = cached_data
            print(f"S_train shape: {S_train.shape}")
            print(f"S_test shape: {S_test.shape}")
        else:
            print(">>> 히스토리 경로가 변경됨. Level 0 재실행 <<<")
            cached_meta = None
            cached_data = None
    
    if cached_meta is None or cached_data is None:
        # 저장 공간: [데이터수, 모델수 * 클래스수]
        num_train_samples = len(train_dataset)
        num_test_samples = len(test_dataset)
        
        # Train 데이터에 대한 예측값 (Level 1의 Training Data가 됨)
        S_train = np.zeros((num_train_samples, len(model_infos) * NUM_CLASSES))
        # Test 데이터에 대한 예측값 (Level 1의 Test Data가 됨)
        S_test = np.zeros((num_test_samples, len(model_infos) * NUM_CLASSES))
        
        # 데이터 로더 생성
        # Windows 호환성을 위해 num_workers=0 사용
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        print(">>> Start Level 0 Inference (Base Models) <<<")
        
        for model_idx, model_info in enumerate(tqdm(model_infos, desc="모델 처리", unit="model")):
            model_name = model_info['name']
            model_path = model_info['path']
            shakedrop_prob = model_info['shakedrop_prob']
            
            print(f"\n모델 {model_idx+1}/{len(model_infos)}: {model_name}")
            print(f"  모델 경로: {model_path}")
            
            # 모델 생성 및 가중치 로드
            model = get_net(model_name, init_weights=False, shakedrop_prob=shakedrop_prob)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model = model.to(DEVICE)
            model.eval()
            
            # Train set에 대한 예측 (전체 학습된 모델 사용)
            print(f"  Train set 예측 중...")
            train_preds = predict(model, train_loader)
            start_col = model_idx * NUM_CLASSES
            end_col = (model_idx + 1) * NUM_CLASSES
            S_train[:, start_col:end_col] = train_preds
            
            # Test set에 대한 예측 (전체 학습된 모델 사용)
            print(f"  Test set 예측 중...")
            test_preds = predict(model, test_loader)
            S_test[:, start_col:end_col] = test_preds
        
        print("\n>>> Level 0 Inference Complete. Meta-features generated.")
        print(f"S_train shape: {S_train.shape}") # (50000, 모델수 * 10)
        print(f"S_test shape: {S_test.shape}")   # (10000, 모델수 * 10)
        
        # Meta-features 저장
        save_meta_features(HISTORY_PATHS, S_train, S_test)
    
    # -----------------------------------------------------------
    # 6. Level 1 Model (Meta Learner) Construction & Training
    # -----------------------------------------------------------
    # Meta-Features(Logits)를 입력으로 받아 최종 Class를 예측하는 딥러닝 모델
    
    # Meta Dataset 준비 (Tensor 변환)
    X_meta_train = torch.FloatTensor(S_train)
    y_meta_train = torch.LongTensor(train_targets)
    X_meta_test = torch.FloatTensor(S_test)
    y_meta_test = torch.LongTensor(np.array(test_dataset.targets))
    
    # Meta DataLoader
    meta_train_dataset = torch.utils.data.TensorDataset(X_meta_train, y_meta_train)
    meta_train_loader = DataLoader(meta_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    meta_test_dataset = torch.utils.data.TensorDataset(X_meta_test, y_meta_test)
    meta_test_loader = DataLoader(meta_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Meta Model 학습
    meta_model = MetaLearner(input_dim=len(model_infos)*NUM_CLASSES, num_classes=NUM_CLASSES).to(DEVICE)
    # meta_optimizer = optim.SGD(meta_model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
    meta_optimizer = optim.Adam(meta_model.parameters(), lr=0.001, weight_decay=1e-2)
    meta_scheduler = optim.lr_scheduler.CosineAnnealingLR(meta_optimizer, T_max=EPOCHS_L1)
    # meta_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    meta_criterion = FocalLoss(gamma=2)
    
    print("\n>>> Start Level 1 Training (Stacking) <<<")
    
    best_acc = 0.0
    for epoch in tqdm(range(EPOCHS_L1), desc="Meta Learner 학습", unit="epoch"):
        meta_model.train()
        total_loss = 0
        pbar_train = tqdm(meta_train_loader, desc=f"  Epoch {epoch+1}/{EPOCHS_L1} [Train]", leave=False)
        for inputs, targets in pbar_train:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            meta_optimizer.zero_grad()
            outputs = meta_model(inputs)
            loss = meta_criterion(outputs, targets)
            loss.backward()
            meta_optimizer.step()
            total_loss += loss.item()
            pbar_train.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Evaluation
        meta_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            pbar_test = tqdm(meta_test_loader, desc=f"  Epoch {epoch+1}/{EPOCHS_L1} [Eval]", leave=False)
            for inputs, targets in pbar_test:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = meta_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                pbar_test.set_postfix({'acc': f'{100 * correct / total:.2f}%'})
                
        acc = 100 * correct / total
        current_lr = meta_scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1}/{EPOCHS_L1} | Loss: {total_loss/len(meta_train_loader):.4f} | Test Acc: {acc:.2f}% | LR: {current_lr:.6f}")
        meta_scheduler.step()
    
    print("\n>>> Stacking Complete. <<<")