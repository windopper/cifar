import argparse
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import json
import os
from models.baseline import BaselineNet
from models.deep_baseline import DeepBaselineNet
from models.deep_baseline_silu import DeepBaselineNetSilu
from models.deep_baseline_bn import DeepBaselineNetBN
from models.deep_baseline2_bn import DeepBaselineNetBN2
from models.deep_baseline2_bn_residual import DeepBaselineNetBN2Residual
from models.deep_baseline2_bn_residual_preact import DeepBaselineNetBN2ResidualPreAct
from models.deep_baseline_bn_dropout import DeepBaselineNetBNDropout
from models.deep_baseline_bn_dropout_resnet import DeepBaselineNetBNDropoutResNet
from models.deep_baseline_gap import DeepBaselineNetGAP
from models.deep_baseline_se import DeepBaselineNetSE
from calibration import calibrate_temperature
from models.resnet import ResNet18
from models.vgg import VGG
from models.mobilenetv2 import MobileNetV2
from models.densenet import DenseNet121

CLASS_NAMES = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')


def set_seed(seed: int = 42):
    """시드 고정 함수 - 재현성을 위한 모든 랜덤 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed fixed to: {seed}")


def get_criterion(name: str, label_smoothing: float = 0.0):
    """Criterion 팩토리 함수"""
    criterions = {
        'crossentropy': nn.CrossEntropyLoss(label_smoothing=label_smoothing),
        'mse': nn.MSELoss(),
        'nll': nn.NLLLoss(),
    }
    if name.lower() not in criterions:
        raise ValueError(
            f"Unknown criterion: {name}. Available: {list(criterions.keys())}")
    return criterions[name.lower()]


def get_net(name: str):
    """Network 팩토리 함수"""
    nets = {
        'baseline': BaselineNet(),
        'deep_baseline': DeepBaselineNet(),
        'deep_baseline_silu': DeepBaselineNetSilu(),
        'deep_baseline_bn': DeepBaselineNetBN(),
        'deep_baseline2_bn': DeepBaselineNetBN2(),
        'deep_baseline2_bn_residual': DeepBaselineNetBN2Residual(),
        'deep_baseline_gap': DeepBaselineNetGAP(),
        'deep_baseline_bn_dropout': DeepBaselineNetBNDropout(),
        'deep_baseline_bn_dropout_resnet': DeepBaselineNetBNDropoutResNet(),
        'deep_baseline2_bn_residual_preact': DeepBaselineNetBN2ResidualPreAct(),
        'deep_baseline_se': DeepBaselineNetSE(),
        'resnet18': ResNet18(),
        'vgg16': VGG('VGG16'),
        'mobilenetv2': MobileNetV2(),
        'densenet121': DenseNet121(),
    }
    if name.lower() not in nets:
        raise ValueError(
            f"Unknown net: {name}. Available: {list(nets.keys())}")
    return nets[name.lower()]


def get_optimizer(name: str, net: nn.Module, lr: float = 0.001, momentum: float = 0.9, weight_decay: float = 5e-4):
    """Optimizer 팩토리 함수"""
    optimizers = {
        'sgd': optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay),
        'adam': optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay),
        'adamw': optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay),
        'adagrad': optim.Adagrad(net.parameters(), lr=lr, weight_decay=weight_decay),
        'rmsprop': optim.RMSprop(net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay),
    }
    if name.lower() not in optimizers:
        raise ValueError(
            f"Unknown optimizer: {name}. Available: {list(optimizers.keys())}")
    return optimizers[name.lower()]


def get_scheduler(name: str, optimizer, epochs: int = 24, steps_per_epoch: int = 1,
                  lr: float = 0.001, gamma: float = 0.95, max_lr: float = None,
                  factor: float = 0.1, patience: int = 10, mode: str = 'min',
                  t_max: int = None, eta_min: float = 0.0):
    """Learning Rate Scheduler 팩토리 함수"""
    if name is None or (isinstance(name, str) and name.lower() == 'none'):
        return None

    schedulers = {}

    if name.lower() == 'exponentiallr':
        schedulers['exponentiallr'] = lr_scheduler.ExponentialLR(
            optimizer, gamma=gamma)
    elif name.lower() == 'onecyclelr':
        if max_lr is None:
            max_lr = lr * 10  # 기본값: 초기 lr의 10배
        total_steps = epochs * steps_per_epoch
        schedulers['onecyclelr'] = lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps
        )
    elif name.lower() == 'reducelronplateau':
        schedulers['reducelronplateau'] = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, verbose=True
        )
    elif name.lower() == 'cosineannealinglr':
        if t_max is None:
            t_max = epochs  # 기본값: 전체 epochs 수
        schedulers['cosineannealinglr'] = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min
        )

    if name.lower() not in schedulers:
        raise ValueError(
            f"Unknown scheduler: {name}. Available: ['exponentiallr', 'onecyclelr', 'reducelronplateau', 'cosineannealinglr', 'none']")

    return schedulers[name.lower()]


def validate(net, criterion, val_loader, device):
    """Validation 함수 - loss와 accuracy 계산"""
    net.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    net.train()
    avg_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='CIFAR-10 Training')
    parser.add_argument('--batch-size', type=int,
                        default=16, help='배치 크기 (default: 16)')
    parser.add_argument('--criterion', type=str, default='crossentropy',
                        choices=['crossentropy', 'mse', 'nll'],
                        help='손실 함수 (default: crossentropy)')
    parser.add_argument('--net', type=str, default='baseline',
                        choices=['baseline', 'deep_baseline', 'deep_baseline_silu',
                                 'deep_baseline_bn', 'deep_baseline_gap', 'deep_baseline_bn_dropout',
                                 'deep_baseline_bn_dropout_resnet', 'deep_baseline_se', 'resnet18',
                                 'vgg16', 'mobilenetv2', 'densenet121', 'deep_baseline2_bn', 'deep_baseline2_bn_residual',
                                 'deep_baseline2_bn_residual_preact'],

                        help='네트워크 모델 (default: baseline)')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam', 'adamw',
                                 'adagrad', 'rmsprop', 'muon'],
                        help='옵티마이저 (default: sgd)')
    parser.add_argument('--epochs', type=int, default=24,
                        help='에포크 수 (default: 24)')
    parser.add_argument('--save-prefix', type=str, default='',
                        help='모델 저장 파일명 접두사 (default: baseline)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='학습률 (default: 0.001)')
    parser.add_argument('--momentum', type=float,
                        default=0.9, help='모멘텀 (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay (default: 5e-4)')
    parser.add_argument('--scheduler', type=str, default='none',
                        choices=['none', 'exponentiallr', 'onecyclelr',
                                 'reducelronplateau', 'cosineannealinglr'],
                        help='Learning rate scheduler (default: none)')
    parser.add_argument('--scheduler-gamma', type=float, default=0.95,
                        help='ExponentialLR의 gamma 값 (default: 0.95)')
    parser.add_argument('--scheduler-max-lr', type=float, default=None,
                        help='OneCycleLR의 max_lr 값 (default: lr * 10)')
    parser.add_argument('--scheduler-factor', type=float, default=0.1,
                        help='ReduceLROnPlateau의 factor 값 (default: 0.1)')
    parser.add_argument('--scheduler-patience', type=int, default=3,
                        help='ReduceLROnPlateau의 patience 값 (default: 3)')
    parser.add_argument('--scheduler-mode', type=str, default='min',
                        choices=['min', 'max'],
                        help='ReduceLROnPlateau의 mode 값 (default: min)')
    parser.add_argument('--scheduler-t-max', type=int, default=None,
                        help='CosineAnnealingLR의 T_max 값 (default: epochs)')
    parser.add_argument('--scheduler-eta-min', type=float, default=0.0,
                        help='CosineAnnealingLR의 eta_min 값 (default: 0.0)')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='Label smoothing 값 (0.0~1.0, 권장: 0.05~0.1, default: 0.0)')
    parser.add_argument('--augment', action='store_true',
                        help='데이터 증강 사용 (default: False)')
    parser.add_argument('--calibrate', action='store_true',
                        help='Temperature Scaling 캘리브레이션 수행 (default: False)')
    parser.add_argument('--use-cifar-normalize', action='store_true',
                        help='CIFAR-10 표준 Normalize 값 사용 (mean: 0.4914 0.4822 0.4465, std: 0.2023 0.1994 0.2010, default: False)')
    parser.add_argument('--seed', type=int, default=42,
                        help='랜덤 시드 값 (default: 42)')
    return parser.parse_args()


def main():
    args = parse_args()

    # 시드 고정
    set_seed(args.seed)

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Save path에 모델 이름과 설정 정보 자동 추가
    model_name_parts = [
        args.save_prefix,
        args.net,
        args.optimizer,
        args.criterion,
        f"bs{args.batch_size}",
        f"ep{args.epochs}",
        f"lr{args.lr}",
        f"mom{args.momentum}"
    ]
    if args.optimizer.lower() == 'adamw':
        model_name_parts.append(f"wd{args.weight_decay}")
    if args.scheduler and args.scheduler.lower() != 'none':
        model_name_parts.append(f"sch{args.scheduler}")
        if args.scheduler.lower() == 'exponentiallr':
            model_name_parts.append(f"gamma{args.scheduler_gamma}")
        elif args.scheduler.lower() == 'onecyclelr' and args.scheduler_max_lr:
            model_name_parts.append(f"maxlr{args.scheduler_max_lr}")
        elif args.scheduler.lower() == 'reducelronplateau':
            model_name_parts.append(f"factor{args.scheduler_factor}")
            model_name_parts.append(f"patience{args.scheduler_patience}")
        elif args.scheduler.lower() == 'cosineannealinglr':
            t_max = args.scheduler_t_max if args.scheduler_t_max else args.epochs
            model_name_parts.append(f"tmax{t_max}")
            if args.scheduler_eta_min > 0.0:
                model_name_parts.append(f"etamin{args.scheduler_eta_min}")
    if args.label_smoothing > 0.0:
        model_name_parts.append(f"ls{args.label_smoothing}")
    if args.augment:
        model_name_parts.append("aug")
    if args.calibrate:
        model_name_parts.append("calibrated")
    if args.use_cifar_normalize:
        model_name_parts.append("cifar_normalize")

    model_name = "_".join(filter(None, model_name_parts))  # 빈 문자열 제거
    SAVE_PATH = f"outputs/{model_name}.pth"
    HISTORY_PATH = f"outputs/{model_name}_history.json"

    # outputs 디렉토리 생성
    os.makedirs("outputs", exist_ok=True)

    # Normalize 값 설정
    if args.use_cifar_normalize:
        # CIFAR-10 표준 Normalize 값
        normalize_mean = (0.4914, 0.4822, 0.4465)
        normalize_std = (0.2470, 0.2434, 0.2615)
    else:
        # 기본값
        normalize_mean = (0.5, 0.5, 0.5)
        normalize_std = (0.5, 0.5, 0.5)

    # Train 데이터 변환: 증강 on/off에 따라 다르게 설정
    if args.augment:
        # 데이터 증강: RandomCrop(32, padding=4), RandomHorizontalFlip(), 약한 ColorJitter
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std)
        ])
    else:
        # 데이터 증강 없이 기본 변환만
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std)
        ])

    # Validation: 데이터 증강 없이 기본 변환만
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)

    val_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    net = get_net(args.net)
    net = net.to(device)
    criterion = get_criterion(
        args.criterion, label_smoothing=args.label_smoothing)
    optimizer = get_optimizer(args.optimizer, net, lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)

    # Learning rate scheduler 생성
    steps_per_epoch = len(train_loader)
    scheduler = get_scheduler(
        args.scheduler, optimizer, epochs=args.epochs,
        steps_per_epoch=steps_per_epoch, lr=args.lr,
        gamma=args.scheduler_gamma, max_lr=args.scheduler_max_lr,
        factor=args.scheduler_factor, patience=args.scheduler_patience,
        mode=args.scheduler_mode, t_max=args.scheduler_t_max,
        eta_min=args.scheduler_eta_min
    )

    # 학습 히스토리 저장용 리스트
    history = {
        'hyperparameters': {
            'batch_size': args.batch_size,
            'criterion': args.criterion,
            'net': args.net,
            'optimizer': args.optimizer,
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'momentum': args.momentum,
            'scheduler': args.scheduler,
            'label_smoothing': args.label_smoothing,
            'data_augment': args.augment,
            'normalize_mean': list(normalize_mean),
            'normalize_std': list(normalize_std),
            'seed': args.seed,
            'device': str(device)
        },
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'best_val_accuracy': None
    }

    # Scheduler 관련 하이퍼파라미터 추가
    if args.scheduler and args.scheduler.lower() != 'none':
        if args.scheduler.lower() == 'exponentiallr':
            history['hyperparameters']['scheduler_gamma'] = args.scheduler_gamma
        elif args.scheduler.lower() == 'onecyclelr':
            history['hyperparameters']['scheduler_max_lr'] = args.scheduler_max_lr if args.scheduler_max_lr else args.lr * 10
        elif args.scheduler.lower() == 'reducelronplateau':
            history['hyperparameters']['scheduler_factor'] = args.scheduler_factor
            history['hyperparameters']['scheduler_patience'] = args.scheduler_patience
            history['hyperparameters']['scheduler_mode'] = args.scheduler_mode
            history['hyperparameters']['scheduler_metric'] = 'val_loss'
        elif args.scheduler.lower() == 'cosineannealinglr':
            history['hyperparameters']['scheduler_t_max'] = args.scheduler_t_max if args.scheduler_t_max else args.epochs
            history['hyperparameters']['scheduler_eta_min'] = args.scheduler_eta_min

    # Optimizer 관련 하이퍼파라미터 추가
    if args.optimizer.lower() == 'adamw':
        history['hyperparameters']['weight_decay'] = args.weight_decay

    print(f"Training configuration:")
    print(f"  Seed: {args.seed}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Criterion: {args.criterion}")
    print(f"  Net: {args.net}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    if args.optimizer.lower() == 'adamw':
        print(f"  Weight decay: {args.weight_decay}")
    print(f"  Label smoothing: {args.label_smoothing}")
    print(f"  Data augmentation: {args.augment}")
    print(f"  Normalize mean: {normalize_mean}")
    print(f"  Normalize std: {normalize_std}")
    print(f"  Scheduler: {args.scheduler}")
    if args.scheduler and args.scheduler.lower() != 'none':
        if args.scheduler.lower() == 'exponentiallr':
            print(f"  Scheduler gamma: {args.scheduler_gamma}")
        elif args.scheduler.lower() == 'onecyclelr':
            max_lr = args.scheduler_max_lr if args.scheduler_max_lr else args.lr * 10
            print(f"  Scheduler max_lr: {max_lr}")
        elif args.scheduler.lower() == 'reducelronplateau':
            print(f"  Scheduler factor: {args.scheduler_factor}")
            print(f"  Scheduler patience: {args.scheduler_patience}")
            print(f"  Scheduler mode: {args.scheduler_mode}")
            print(f"  Scheduler metric: val_loss")
        elif args.scheduler.lower() == 'cosineannealinglr':
            t_max = args.scheduler_t_max if args.scheduler_t_max else args.epochs
            print(f"  Scheduler T_max: {t_max}")
            print(f"  Scheduler eta_min: {args.scheduler_eta_min}")
    print(f"  Save path: {SAVE_PATH}")
    print(f"  History path: {HISTORY_PATH}")
    print()

    # 최고 검증 정확도 추적
    best_val_acc = -1.0

    for epoch in range(args.epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}')
        for i, data in enumerate(pbar, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # OneCycleLR은 각 step마다 업데이트
            if scheduler is not None and isinstance(scheduler, lr_scheduler.OneCycleLR):
                scheduler.step()

            running_loss += loss.item()

            # tqdm 진행률 표시줄에 현재 loss 업데이트
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(
                {'loss': f'{running_loss / (i + 1):.3f}', 'lr': f'{current_lr:.6f}'})

        # Epoch 종료 후 평균 train loss 계산
        avg_train_loss = running_loss / len(train_loader)

        # Validation 수행
        val_loss, val_acc = validate(net, criterion, val_loader, device)

        # ExponentialLR과 CosineAnnealingLR은 각 epoch마다 업데이트
        if scheduler is not None and isinstance(scheduler, lr_scheduler.ExponentialLR):
            scheduler.step()
        if scheduler is not None and isinstance(scheduler, lr_scheduler.CosineAnnealingLR):
            scheduler.step()

        # ReduceLROnPlateau는 validation loss를 기반으로 각 epoch마다 업데이트
        if scheduler is not None and isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)

        # 히스토리에 저장
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_acc)

        # 최고 검증 정확도일 때만 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            history['best_val_accuracy'] = best_val_acc
            torch.save(net.state_dict(), SAVE_PATH)
            print(f"  [Best Model Saved] Val Accuracy: {val_acc:.2f}%")

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{args.epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        print()

        # 매 epoch마다 히스토리 저장 (중간에 중단되어도 데이터 보존)
        with open(HISTORY_PATH, 'w') as f:
            json.dump(history, f, indent=2)

    print('Finished Training')

    # Temperature Scaling 캘리브레이션 수행
    optimal_temperature = None
    if args.calibrate:
        print("\n" + "="*50)
        print("Temperature Scaling 캘리브레이션 시작...")
        print("="*50)
        optimal_temperature = calibrate_temperature(net, val_loader, device)
        print(f"최적의 Temperature: {optimal_temperature:.4f}")

        # 캘리브레이션 후 검증 정확도 확인
        net.eval()
        calibrated_correct = 0
        calibrated_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                logits = net(inputs)
                # Temperature로 스케일링
                scaled_logits = logits / optimal_temperature
                _, predicted = torch.max(scaled_logits.data, 1)
                calibrated_total += labels.size(0)
                calibrated_correct += (predicted == labels).sum().item()

        calibrated_acc = 100 * calibrated_correct / calibrated_total
        print(f"캘리브레이션 후 검증 정확도: {calibrated_acc:.2f}%")
        print("="*50 + "\n")

        # 히스토리에 temperature 저장
        history['hyperparameters']['temperature'] = optimal_temperature
        history['calibrated_val_accuracy'] = calibrated_acc

        # 히스토리 파일 업데이트
        with open(HISTORY_PATH, 'w') as f:
            json.dump(history, f, indent=2)

    # 최고 모델은 이미 저장되어 있음
    print(
        f"Best model (Val Accuracy: {best_val_acc:.2f}%) saved to: {SAVE_PATH}")
    print(f"Training history saved to: {HISTORY_PATH}")

    # Temperature가 있으면 별도 파일로도 저장
    if optimal_temperature is not None:
        TEMP_PATH = f"outputs/{model_name}_temperature.json"
        with open(TEMP_PATH, 'w') as f:
            json.dump({'temperature': optimal_temperature}, f, indent=2)
        print(f"Temperature saved to: {TEMP_PATH}")


if __name__ == '__main__':
    main()
