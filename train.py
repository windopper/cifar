import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import json
import os
from models.wideresnet import (
    WideResNet
)
from utils.augmentation.cutmix import CutMixCollator, CutMixCriterion
from utils.augmentation.mixup import MixupCollator, MixupCriterion
from utils.augmentation.switch import SwitchCollator
from utils.dataset import get_cifar10_loaders, get_normalize_values
from utils.history import (
    create_history, update_history_scheduler_params, update_history_optimizer_params,
    update_history_epoch, update_history_best, update_history_early_stop,
    save_history
)
from utils.model_name import get_model_name_parts
from utils.net import get_available_nets
from utils.net import get_net
from utils.training_config import print_training_configuration


from utils.loss.supcon import SupConLoss
from utils.scheduler.cosine_annealing_warmup_restarts import CosineAnnealingWarmupRestarts
from utils.ema import ModelEMA
from utils.sam import SAM
from utils.loss.focal_loss_adaptive import FocalLossAdaptive

CLASS_NAMES = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')


def set_seed(seed: int = 42):
    """시드 고정 함수 - 재현성을 위한 모든 랜덤 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = True
    print(f"Seed fixed to: {seed}")


def get_criterion(name: str, label_smoothing: float = 0.0, weight: torch.Tensor = None, 
                  gamma: float = 3.0, device: torch.device = None):
    criterions = {
        'crossentropy': nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=weight),
        'mse': nn.MSELoss(),
        'nll': nn.NLLLoss(),
        'focal_loss_adaptive': FocalLossAdaptive(gamma=gamma, size_average=True, device=device),
    }
    if name.lower() not in criterions:
        raise ValueError(
            f"Unknown criterion: {name}. Available: {list(criterions.keys())}")
    return criterions[name.lower()]

def get_optimizer(name: str, net: nn.Module, lr: float = 0.001, momentum: float = 0.9, weight_decay: float = 5e-4, nesterov: bool = False,
                  use_sam: bool = False, sam_rho: float = 0.05, sam_adaptive: bool = False):
    """Optimizer 팩토리 함수"""
    name_lower = name.lower()
    base_optimizer_cls = None
    base_optimizer_kwargs = {}

    if name_lower == 'sgd':
        base_optimizer_cls = optim.SGD
        base_optimizer_kwargs = {'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay, 'nesterov': nesterov}
    elif name_lower == 'adam':
        base_optimizer_cls = optim.Adam
        base_optimizer_kwargs = {'lr': lr, 'weight_decay': weight_decay}
    elif name_lower == 'adamw':
        base_optimizer_cls = optim.AdamW
        base_optimizer_kwargs = {'lr': lr, 'weight_decay': weight_decay}
    elif name_lower == 'adagrad':
        base_optimizer_cls = optim.Adagrad
        base_optimizer_kwargs = {'lr': lr, 'weight_decay': weight_decay}
    elif name_lower == 'rmsprop':
        base_optimizer_cls = optim.RMSprop
        base_optimizer_kwargs = {'lr': lr, 'momentum': momentum, 'weight_decay': weight_decay}
    else:
        raise ValueError(
            f"Unknown optimizer: {name}. Available: ['sgd', 'adam', 'adamw', 'adagrad', 'rmsprop']")
            
    if use_sam:
        return SAM(net.parameters(), base_optimizer_cls, rho=sam_rho, adaptive=sam_adaptive, **base_optimizer_kwargs)
    else:
        return base_optimizer_cls(net.parameters(), **base_optimizer_kwargs)


def get_scheduler(name: str, optimizer, epochs: int = 24, steps_per_epoch: int = 1,
                  lr: float = 0.001, gamma: float = 0.95, max_lr: float = None,
                  factor: float = 0.1, patience: int = 10, mode: str = 'min',
                  t_max: int = None, eta_min: float = 0.0, pct_start: float = 0.3,
                  final_lr_ratio: float = 0.0001, warmup_epochs: int = 0,
                  first_cycle_steps: int = None, cycle_mult: float = 1.0,
                  scheduler_min_lr: float = None, scheduler_gamma: float = 1.0):
    if name is None or (isinstance(name, str) and name.lower() == 'none'):
        return None

    schedulers = {}

    if name.lower() == 'exponentiallr':
        schedulers['exponentiallr'] = lr_scheduler.ExponentialLR(
            optimizer, gamma=gamma)
    elif name.lower() == 'onecyclelr':
        if max_lr is None:
            max_lr = lr * 10

        div_factor = 25.0 
        final_div_factor = max_lr / \
            (div_factor * lr * final_lr_ratio) if final_lr_ratio > 0 else 10000.0

        total_steps = epochs * steps_per_epoch
        schedulers['onecyclelr'] = lr_scheduler.OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=pct_start,
            div_factor=div_factor, final_div_factor=final_div_factor
        )
    elif name.lower() == 'reducelronplateau':
        schedulers['reducelronplateau'] = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience
        )
    elif name.lower() == 'cosineannealinglr':
        if t_max is None:
            t_max = epochs
        
        schedulers['cosineannealinglr'] = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=t_max, eta_min=eta_min
        )
    elif name.lower() == 'cosineannealingwarmuprestarts':
        if first_cycle_steps is None:
            first_cycle_steps = epochs * steps_per_epoch
        if max_lr is None:
            max_lr = lr
        if scheduler_min_lr is None:
            scheduler_min_lr = eta_min if eta_min > 0 else lr * 0.001
        
        warmup_steps = warmup_epochs * steps_per_epoch
        
        if warmup_steps >= first_cycle_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps})는 first_cycle_steps ({first_cycle_steps})보다 작아야 합니다.")
        
        schedulers['cosineannealingwarmuprestarts'] = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=first_cycle_steps,
            cycle_mult=cycle_mult,
            max_lr=max_lr,
            min_lr=scheduler_min_lr,
            warmup_steps=warmup_steps,
            gamma=scheduler_gamma
        )

    if name.lower() not in schedulers:
        raise ValueError(
            f"Unknown scheduler: {name}. Available: ['exponentiallr', 'onecyclelr', 'reducelronplateau', 'cosineannealinglr', 'cosineannealingwarmuprestarts', 'none']")

    return schedulers[name.lower()]


def validate(net, criterion, val_loader, device):
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

    # 학습 모드로 복원
    net.train()
    avg_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 Training')
    parser.add_argument('--batch-size', type=int,
                        default=16, help='배치 크기 (default: 16)')
    parser.add_argument('--criterion', type=str, default='crossentropy',
                        choices=['crossentropy', 'mse', 'nll', 'supcon_ce', 'focal_loss_adaptive'],
                        help='손실 함수 (default: crossentropy)')
    parser.add_argument('--flsd-gamma', type=float, default=3.0,
                        help='Focal Loss의 gamma 값 (default: 3.0)')
    parser.add_argument('--net', type=str, default='baseline',
                        choices=get_available_nets(),
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
    parser.add_argument('--nesterov', action='store_true',
                        help='SGD에서 Nesterov momentum 사용 (default: False)')
    parser.add_argument('--scheduler', type=str, default='none',
                        choices=['none', 'exponentiallr', 'onecyclelr',
                                 'reducelronplateau', 'cosineannealinglr', 'cosineannealingwarmuprestarts'],
                        help='Learning rate scheduler (default: none)')
    parser.add_argument('--scheduler-gamma', type=float, default=0.95,
                        help='ExponentialLR의 gamma 값 (default: 0.95)')
    parser.add_argument('--scheduler-max-lr', type=float, default=None,
                        help='OneCycleLR의 max_lr 값 (default: lr * 10)')
    parser.add_argument('--scheduler-pct-start', type=float, default=0.3,
                        help='OneCycleLR의 pct_start 값 (0.0~1.0, 최고 학습률 도달 시점 비율, default: 0.3)')
    parser.add_argument('--scheduler-final-lr-ratio', type=float, default=0.0001,
                        help='OneCycleLR의 마지막 학습률 비율 (원래 lr 대비, PyTorch 기본값: 0.0001, final_div_factor=10000)')
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
    parser.add_argument('--scheduler-warmup-epochs', type=int, default=0,
                        help='CosineAnnealingWarmupRestarts의 warmup 에포크 수 (default: 0, warmup 비활성화)')
    parser.add_argument('--scheduler-first-cycle-steps', type=int, default=None,
                        help='CosineAnnealingWarmupRestarts의 first_cycle_steps 값 (default: epochs * steps_per_epoch)')
    parser.add_argument('--scheduler-cycle-mult', type=float, default=1.0,
                        help='CosineAnnealingWarmupRestarts의 cycle_mult 값 (default: 1.0)')
    parser.add_argument('--scheduler-min-lr', type=float, default=None,
                        help='CosineAnnealingWarmupRestarts의 min_lr 값 (default: scheduler-eta-min 또는 lr * 0.001)')
    parser.add_argument('--scheduler-gamma-restarts', type=float, default=1.0,
                        help='CosineAnnealingWarmupRestarts의 gamma 값 (default: 1.0)')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='Label smoothing 값 (0.0~1.0, 권장: 0.05~0.1, default: 0.0)')
    parser.add_argument('--augment', action='store_true',
                        help='데이터 증강 사용 (default: False)')
    parser.add_argument('--autoaugment', action='store_true',
                        help='AutoAugment 사용 (--augment가 활성화되어 있을 때만 동작, default: False)')
    parser.add_argument('--cutmix', action='store_true',
                        help='CutMix 증강 사용 (--augment가 활성화되어 있을 때만 동작, default: False)')
    parser.add_argument('--cutmix-alpha', type=float, default=1.0,
                        help='CutMix의 alpha 값 (Beta 분포 파라미터, default: 1.0)')
    parser.add_argument('--cutmix-prob', type=float, default=1,
                        help='CutMix 적용 확률 (0.0~1.0, default: 1)')
    parser.add_argument('--mixup', action='store_true',
                        help='Mixup 증강 사용 (--augment가 활성화되어 있을 때만 동작, default: False)')
    parser.add_argument('--mixup-alpha', type=float, default=1.0,
                        help='Mixup의 alpha 값 (Beta 분포 파라미터, default: 1.0)')
    parser.add_argument('--mixup-prob', type=float, default=1,
                        help='Mixup 적용 확률 (0.0~1.0, default: 1)')
    parser.add_argument('--switch-prob', type=float, default=0.5,
                        help='CutMix와 Mixup이 동시에 활성화되었을 때 Mixup을 선택할 확률 (0.0~1.0, default: 0.5)')
    parser.add_argument('--cutout', action='store_true',
                        help='Cutout 증강 사용 (--augment가 활성화되어 있을 때만 동작, default: False)')
    parser.add_argument('--cutout-length', type=int, default=16,
                        help='Cutout 마스킹 영역의 크기 (픽셀 단위, default: 16)')
    parser.add_argument('--cutout-n-holes', type=int, default=1,
                        help='Cutout 마스킹할 영역의 개수 (default: 1)')
    parser.add_argument('--cutout-prob', type=float, default=0.5,
                        help='Cutout을 적용할 확률 (0.0~1.0, default: 1.0)')
    parser.add_argument('--use-cifar-normalize', action='store_true',
                        help='CIFAR-10 표준 Normalize 값 사용 (mean: 0.4914 0.4822 0.4465, std: 0.2023 0.1994 0.2010, default: False)')
    parser.add_argument('--seed', type=int, default=42,
                        help='랜덤 시드 값 (default: 42)')
    parser.add_argument('--w-init', action='store_true',
                        help='Weight initialization 사용 (deep_baseline 모델에만 적용, default: False)')
    parser.add_argument('--early-stopping', action='store_true',
                        help='Early stopping 사용 (default: False)')
    parser.add_argument('--early-stopping-patience', type=int, default=5,
                        help='Early stopping patience 값 (default: 5)')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='모델 및 히스토리 파일 저장 디렉토리 (default: outputs)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='DataLoader의 워커 수 (None이면 자동 설정: AutoAugment 사용 시 4, 그 외 2, default: None)')
    parser.add_argument('--ema', action='store_true',
                        help='EMA (Exponential Moving Average) 사용 (default: False)')
    parser.add_argument('--ema-decay', type=float, default=0.999,
                        help='EMA decay 값 (default: 0.999)')
    parser.add_argument('--sam', action='store_true',
                        help='SAM optimizer 사용 (default: False)')
    parser.add_argument('--sam-rho', type=float, default=0.05,
                        help='SAM rho 값 (default: 0.05)')
    parser.add_argument('--sam-adaptive', action='store_true',
                        help='SAM adaptive 모드 사용 (default: False)')
    parser.add_argument('--shakedrop', type=float, default=0.0,
                        help='ShakeDrop 확률 (0.0~1.0, WideResNet 모델에만 적용, default: 0.0)')
    parser.add_argument('--weighted-ce', action='store_true',
                        help='Weighted Cross Entropy 사용 (cat, dog 클래스에 1.5배 가중치 부여, default: False)')
    parser.add_argument('--grad-norm', type=float, default=None,
                        help='Gradient clipping의 최대 norm 값 (None이면 비활성화, default: None)')
    parser.add_argument('--amp', action='store_true',
                        help='AMP (Automatic Mixed Precision) 사용 (default: False)')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='torch.compile을 사용하여 모델 최적화 (default: True)')
    parser.add_argument('--compile-mode', type=str, default='default',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile 모드 (default: default)')
    return parser.parse_args()


def main():
    args = parse_args()

    # 시드 고정
    set_seed(args.seed)

    # torch.set_float32_matmul_precision('high')

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Save path에 모델 이름과 설정 정보 추가
    model_name_parts = get_model_name_parts(args)
    model_name = "_".join(filter(None, model_name_parts))

    # 출력 디렉토리 설정
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    SAVE_PATH = os.path.join(output_dir, f"{model_name}.pth")
    EMA_SAVE_PATH = os.path.join(output_dir, f"{model_name}_ema.pth")
    HISTORY_PATH = os.path.join(output_dir, f"{model_name}_history.json")

    # Normalize 값 설정
    normalize_mean, normalize_std = get_normalize_values(args.use_cifar_normalize)

    # CutMix/Mixup 설정
    cutmix_collator = None
    cutmix_criterion = None
    mixup_collator = None
    mixup_criterion = None
    switch_collator = None
    use_both = False

    if args.cutmix and args.augment and args.mixup and args.augment:
        use_both = True
        cutmix_collator = CutMixCollator(alpha=args.cutmix_alpha, prob=args.cutmix_prob)
        cutmix_criterion = CutMixCriterion(
            reduction='mean', label_smoothing=args.label_smoothing)
        mixup_collator = MixupCollator(alpha=args.mixup_alpha, prob=args.mixup_prob)
        mixup_criterion = MixupCriterion(
            reduction='mean', label_smoothing=args.label_smoothing)
        switch_collator = SwitchCollator(
            cutmix_collator=cutmix_collator,
            mixup_collator=mixup_collator,
            switch_prob=args.switch_prob
        )
    elif args.cutmix and args.augment:
        cutmix_collator = CutMixCollator(alpha=args.cutmix_alpha, prob=args.cutmix_prob)
        cutmix_criterion = CutMixCriterion(
            reduction='mean', label_smoothing=args.label_smoothing)
    elif args.mixup and args.augment:
        mixup_collator = MixupCollator(alpha=args.mixup_alpha, prob=args.mixup_prob)
        mixup_criterion = MixupCriterion(
            reduction='mean', label_smoothing=args.label_smoothing)

    # 초기 collate_fn 설정
    collate_fn = None
    if use_both:
        collate_fn = switch_collator
    elif args.cutmix and args.augment:
        collate_fn = cutmix_collator
    elif args.mixup and args.augment:
        collate_fn = mixup_collator

    # 데이터셋 및 DataLoader 생성
    train_loader, val_loader, train_set, val_set = get_cifar10_loaders(
        batch_size=args.batch_size,
        augment=args.augment,
        autoaugment=args.autoaugment,
        cutout=args.cutout,
        cutout_n_holes=args.cutout_n_holes,
        cutout_length=args.cutout_length,
        cutout_prob=args.cutout_prob,
        use_cifar_normalize=args.use_cifar_normalize,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        data_root='./data'
    )
    
    # num_workers 및 pin_memory 값 가져오기
    num_workers = train_loader.num_workers
    pin_memory = torch.cuda.is_available()

    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # ShakeDrop은 WideResNet 및 PyramidNet 모델에만 적용
    wideresnet_models = ['wideresnet28_10', 'wideresnet16_8', 'wideresnet28_10_pyramid', 
                         'wideresnet16_8_pyramid', 'wideresnet28_10_fullpyramid', 
                         'wideresnet16_8_fullpyramid', 'pyramidnet110_270', 'pyramidnet110_150', 'pyramidnet164_118',
                         'pyramidnet272_200_bottleneck',
                         'wideresnet28_10_remove_first_relu', 'wideresnet28_10_last_bn_remove_first_relu',
                         'wideresnet16_8_remove_first_relu', 'wideresnet16_8_last_bn_remove_first_relu']
    shakedrop_prob = args.shakedrop if args.net in wideresnet_models else 0.0
    if args.shakedrop > 0.0 and args.net not in wideresnet_models:
        print(f"[경고] --shakedrop은 WideResNet/PyramidNet 모델에만 적용됩니다. 현재 모델: {args.net}")
    
    net = get_net(args.net, init_weights=args.w_init, shakedrop_prob=shakedrop_prob)
    net = net.to(device)

    # EMA 초기화
    ema_model = None
    if args.ema:
        ema_model = ModelEMA(net, decay=args.ema_decay, device=device)
        print(f"EMA 활성화됨 (decay: {args.ema_decay})")
    
    # torch.compile 적용
    if args.compile:
        if hasattr(torch, 'compile'):
            try:
                net = torch.compile(net, mode=args.compile_mode)
                print(f"torch.compile 활성화됨 (mode: {args.compile_mode})")
            except Exception as e:
                print(f"[경고] torch.compile 적용 실패: {e}")
                print("  모델을 컴파일하지 않고 계속 진행합니다.")
        else:
            print("[경고] 현재 PyTorch 버전에서 torch.compile을 지원하지 않습니다. (PyTorch 2.0+ 필요)")
            args.compile = False

    # AMP GradScaler 초기화
    scaler = None
    if args.amp:
        if device.type == 'cuda':
            scaler = GradScaler()
            print(f"AMP (Automatic Mixed Precision) 활성화됨")
        else:
            print("[경고] AMP는 CUDA 디바이스에서만 지원됩니다. CPU에서는 비활성화됩니다.")
            args.amp = False

    # Weighted Cross Entropy 가중치 설정
    class_weights = None
    if args.weighted_ce:
        class_weights = torch.ones(10, dtype=torch.float32)
        class_weights[3] = 1.5
        class_weights[5] = 1.5
        class_weights = class_weights.to(device)
        print(f"Weighted Cross Entropy 활성화됨:")
        print(f"  - cat (인덱스 3): 가중치 1.5")
        print(f"  - dog (인덱스 5): 가중치 1.5")
        print(f"  - 기타 클래스: 가중치 1.0")
    
    # Criterion 설정
    supcon_criterion = None
    supcon_weight = 0.1
    if args.criterion.lower() == 'supcon_ce':
        # SupCon + CrossEntropy 조합
        criterion = get_criterion(
            'crossentropy', label_smoothing=args.label_smoothing, weight=class_weights)
        supcon_criterion = SupConLoss(temperature=0.07)
        if args.net != 'baseline_bn':
            print("[경고] --criterion supcon_ce는 현재 baseline_bn 모델에 최적화되어 있습니다.")
    else:
        criterion = get_criterion(
            args.criterion, label_smoothing=args.label_smoothing, weight=class_weights,
            gamma=args.flsd_gamma, device=device)
    optimizer = get_optimizer(args.optimizer, net, lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov,
                              use_sam=args.sam, sam_rho=args.sam_rho, sam_adaptive=args.sam_adaptive)

    # Learning rate scheduler 생성
    steps_per_epoch = len(train_loader)
    scheduler = get_scheduler(
        args.scheduler, optimizer, epochs=args.epochs,
        steps_per_epoch=steps_per_epoch, lr=args.lr,
        gamma=args.scheduler_gamma, max_lr=args.scheduler_max_lr,
        factor=args.scheduler_factor, patience=args.scheduler_patience,
        mode=args.scheduler_mode, t_max=args.scheduler_t_max,
        eta_min=args.scheduler_eta_min, pct_start=args.scheduler_pct_start,
        final_lr_ratio=args.scheduler_final_lr_ratio,
        warmup_epochs=args.scheduler_warmup_epochs,
        first_cycle_steps=args.scheduler_first_cycle_steps,
        cycle_mult=args.scheduler_cycle_mult,
        scheduler_min_lr=args.scheduler_min_lr,
        scheduler_gamma=args.scheduler_gamma_restarts
    )

    # 학습 히스토리 초기화
    history = create_history(
        args, normalize_mean, normalize_std, device, num_workers, pin_memory, steps_per_epoch)
    
    # Scheduler 관련 하이퍼파라미터 추가
    update_history_scheduler_params(history, args, steps_per_epoch)
    
    # Optimizer 관련 하이퍼파라미터 추가
    update_history_optimizer_params(history, args)

    # Training configuration 출력
    print_training_configuration(
        args, normalize_mean, normalize_std, SAVE_PATH, HISTORY_PATH)

    # 최고 검증 정확도 추적
    best_val_acc = -1.0

    # Early stopping 설정
    early_stopping_enabled = args.early_stopping
    early_stopping_patience = args.early_stopping_patience if early_stopping_enabled else None
    patience_counter = 0

    for epoch in range(args.epochs):
        use_cutmix = args.cutmix and args.augment
        use_mixup = args.mixup and args.augment
        use_both_current = use_both

        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{args.epochs}')
        for i, data in enumerate(pbar, 0):
            inputs, labels = data
            inputs = inputs.to(device)

            # CutMix/Mixup 사용 시 labels는 (targets1, targets2, lam) 튜플 형태
            if isinstance(labels, (tuple, list)):
                targets1, targets2, lam = labels
                labels = (targets1.to(device), targets2.to(device), lam)
            else:
                labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            def compute_loss(inputs, labels):
                # AMP autocast 적용
                with autocast('cuda', enabled=args.amp):
                    outputs = net(inputs)

                    # 기본 CrossEntropy / CutMix / Mixup 손실
                    # 둘 다 사용 중일 때는 배치마다 선택되므로, labels가 tuple인지만 확인
                    if isinstance(labels, tuple):
                        if use_both_current:
                            loss_ce = cutmix_criterion(outputs, labels)
                        elif use_cutmix:
                            loss_ce = cutmix_criterion(outputs, labels)
                        elif use_mixup:
                            loss_ce = mixup_criterion(outputs, labels)
                        else:
                            loss_ce = criterion(outputs, labels)
                    else:
                        loss_ce = criterion(outputs, labels)

                    # SupCon 손실 추가 (--criterion supcon_ce, CutMix/Mixup 미사용 시)
                    if supcon_criterion is not None:
                        if use_cutmix or use_mixup or use_both_current:
                            raise ValueError(
                                "SupCon CE(--criterion supcon_ce)는 CutMix/Mixup과 함께 사용할 수 없습니다.")
                        if hasattr(net, "forward_features"):
                            features = net.forward_features(inputs)
                        else:
                            features = outputs
                        # SupCon은 hard label만 사용
                        supcon_loss = supcon_criterion(features, labels)
                        loss = loss_ce + supcon_weight * supcon_loss
                    else:
                        loss = loss_ce
                
                return loss, outputs

            loss, outputs = compute_loss(inputs, labels)
            
            # AMP를 사용할 경우 scaler를 통해 backward
            if args.amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient clipping
            if args.grad_norm is not None and not args.sam:
                if args.amp:
                    # AMP 사용 시 scaler를 통해 gradient unscale 후 clipping
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_norm)
            
            if args.sam:
                # SAM optimizer는 closure를 통해 두 번의 forward-backward를 수행
                # AMP 사용 시 첫 번째 backward 후 unscale 필요
                if args.amp:
                    scaler.unscale_(optimizer)
                
                # SAM의 first step (gradient 계산)
                optimizer.first_step(zero_grad=True)
                
                # SAM의 second step을 위한 closure
                optimizer.zero_grad(set_to_none=True)
                loss, _ = compute_loss(inputs, labels)
                
                # 두 번째 backward는 AMP 없이 수행
                loss.backward()
                optimizer.second_step()
                
                # AMP 사용 시 scaler update
                if args.amp:
                    scaler.update()
            else:
                if args.amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

            # EMA 업데이트
            if ema_model is not None:
                ema_model.update(net)

            # OneCycleLR과 CosineAnnealingWarmupRestarts는 각 step마다 업데이트
            if scheduler is not None and isinstance(scheduler, lr_scheduler.OneCycleLR):
                scheduler.step()
            if scheduler is not None and isinstance(scheduler, CosineAnnealingWarmupRestarts):
                scheduler.step()

            running_loss += loss.item()

            # tqdm 진행률 표시줄에 현재 loss 업데이트
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix(
                {'loss': f'{running_loss / (i + 1):.3f}', 'lr': f'{current_lr:.6f}'})

        # Epoch 종료 후 평균 train loss 계산
        avg_train_loss = running_loss / len(train_loader)
        
        val_loss, val_acc = validate(net, criterion, val_loader, device)

        # ExponentialLR, CosineAnnealingLR은 각 epoch마다 업데이트
        if scheduler is not None and isinstance(scheduler, lr_scheduler.ExponentialLR):
            scheduler.step()
        if scheduler is not None and isinstance(scheduler, lr_scheduler.CosineAnnealingLR):
            scheduler.step()

        # ReduceLROnPlateau는 validation loss를 기반으로 각 epoch마다 업데이트
        if scheduler is not None and isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)

        # 히스토리에 저장
        update_history_epoch(history, avg_train_loss, val_loss, val_acc)

        # EMA 모델은 매 에포크마다 항상 저장
        if ema_model is not None:
            torch.save(ema_model.get_model().state_dict(), EMA_SAVE_PATH)

        # 최고 검증 정확도일 때만 원본 모델 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            update_history_best(history, best_val_acc)
            
            torch.save(net.state_dict(), SAVE_PATH)
            print(f"  [Best Model Saved] Val Accuracy: {val_acc:.2f}%")
            if ema_model is not None:
                print(f"    - Original model: {SAVE_PATH}")
                print(f"    - EMA model: {EMA_SAVE_PATH} (updated every epoch)")
            # Early stopping 카운터 리셋 (활성화된 경우에만)
            if early_stopping_enabled:
                patience_counter = 0
        else:
            # 개선이 없으면 patience 카운터 증가 (활성화된 경우에만)
            if early_stopping_enabled:
                patience_counter += 1

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1}/{args.epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Val Accuracy: {val_acc:.2f}%")
        print(f"  Learning Rate: {current_lr:.6f}")
        if early_stopping_enabled and patience_counter > 0:
            print(
                f"  Early Stopping: {patience_counter}/{early_stopping_patience} (no improvement)")
        print()

        # 매 epoch마다 히스토리 저장 (중간에 중단되어도 데이터 보존)
        save_history(history, HISTORY_PATH)

        # Early stopping 체크 (활성화된 경우에만)
        if early_stopping_enabled and patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(
                f"No improvement in validation accuracy for {early_stopping_patience} consecutive epochs")
            update_history_early_stop(history, True, epoch + 1, early_stopping_patience)
            # 히스토리 파일 업데이트
            save_history(history, HISTORY_PATH)
            break

    # Early stopping이 발생하지 않은 경우 히스토리 업데이트
    if not early_stopping_enabled or (early_stopping_enabled and patience_counter < early_stopping_patience):
        update_history_early_stop(history, False, early_stopping_patience=early_stopping_patience if early_stopping_enabled else None)
        save_history(history, HISTORY_PATH)

    print('Finished Training')
    print(f"Training history saved to: {HISTORY_PATH}")

if __name__ == '__main__':
    main()
