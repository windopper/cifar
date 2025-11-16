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
from models.baseline_bn import BaselineNetBN
from models.deep_baseline import DeepBaselineNet
from models.deep_baseline_silu import DeepBaselineNetSilu
from models.deep_baseline_bn import DeepBaselineNetBN
from models.deep_baseline2_bn import DeepBaselineNetBN2
from models.deep_baseline3_bn import DeepBaselineNetBN3
from models.deep_baseline2_bn_residual import DeepBaselineNetBN2Residual
from models.deep_baseline2_bn_residual_preact import DeepBaselineNetBN2ResidualPreAct
from models.deep_baseline2_bn_resnext import DeepBaselineNetBN2ResNeXt
from models.deep_baseline2_bn_residual_se import DeepBaselineNetBN2ResidualSE
from models.deep_baseline2_bn_residual_grn import DeepBaselineNetBN2ResidualGRN
from models.deep_baseline_bn_dropout import DeepBaselineNetBNDropout
from models.deep_baseline_bn_dropout_resnet import DeepBaselineNetBNDropoutResNet
from models.deep_baseline_gap import DeepBaselineNetGAP
from models.deep_baseline_se import DeepBaselineNetSE
from models.convnext_step1_patchify import ConvNeXtPatchifyClassifier
from models.convnext_step2_local_block import ConvNeXtLocalBlockClassifier
from models.convnext_step3_full import ConvNeXtCIFAR, convnext_tiny
from calibration import calibrate_temperature
from models.resnet import ResNet18
from models.vgg import VGG
from models.mobilenetv2 import MobileNetV2
from models.densenet import DenseNet121
from models.mxresnet import MXResNet20, MXResNet32, MXResNet44, MXResNet56
from models.dla import DLA
from models.resnext import ResNeXt29_4x64d
from utils.cutmix import CutMixCollator, CutMixCriterion
from utils.mixup import MixupCollator, MixupCriterion
from utils.model_name import get_model_name_parts
from utils.training_config import print_training_configuration
from utils.supcon import SupConLoss
from models.deep_baseline3_bn_residual import DeepBaselineNetBN3Residual
from models.deep_baseline3_bn_residual_wide import DeepBaselineNetBN3ResidualWide
from models.deep_baseline3_bn_residual_4x import DeepBaselineNetBN3Residual4X
from models.deep_baseline3_bn_residual_deep import DeepBaselineNetBN3ResidualDeep
from models.deep_baseline3_bn_residual_preact import DeepBaselineNetBN3ResidualPreAct
from models.deep_baseline3_bn_residual_swish import DeepBaselineNetBN3ResidualSwish
from models.deep_baseline3_bn_residual_swiglu import DeepBaselineNetBN3ResidualSwiGLU
from models.deep_baseline3_bn_residual_dla import DeepBaselineNetBN3ResidualDLA
from models.deep_baseline3_bn_residual_dla_tree import DeepBaselineNetBN3ResidualDLATree
from models.deep_baseline3_bn_residual_group import DeepBaselineNetBN3ResidualGroup
from models.deep_baseline3_bn_residual_shakedrop import DeepBaselineNetBN3ResidualShakeDrop
from models.deep_baseline3_bn_residual_mish import DeepBaselineNetBN3ResidualMish
from models.deep_baseline3_bn_residual_gap_gmp import (
    DeepBaselineNetBN3ResidualGAPGMP, 
    make_deep_baseline3_bn_residual_gap_gmp,
    DeepBaselineNetBN3ResidualGAPGMP_S3_F8_16_32_B2,
    DeepBaselineNetBN3ResidualGAPGMP_S3_F16_32_64_B3,
    DeepBaselineNetBN3ResidualGAPGMP_S3_F32_64_128_B5,
    DeepBaselineNetBN3ResidualGAPGMP_S3_F64_128_256_B5,
    DeepBaselineNetBN3ResidualGAPGMP_S4_F64_128_256_512_B5
)
from models.deep_baseline4_bn_residual import ResNet18 as DeepBaselineNetBN4Residual
from models.deep_baseline4_bn_residual_shakedrop import ResNet18ShakeDrop as DeepBaselineNetBN4ResidualShakeDrop

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


def get_net(name: str, init_weights: bool = False):
    """Network 팩토리 함수"""
    nets = {
        'baseline': BaselineNet(),
        'baseline_bn': BaselineNetBN(init_weights=init_weights),
        'deep_baseline': DeepBaselineNet(init_weights=init_weights),
        'deep_baseline_silu': DeepBaselineNetSilu(),
        'deep_baseline_bn': DeepBaselineNetBN(init_weights=init_weights),
        'deep_baseline_gap': DeepBaselineNetGAP(),
        'deep_baseline_bn_dropout': DeepBaselineNetBNDropout(),
        'deep_baseline_bn_dropout_resnet': DeepBaselineNetBNDropoutResNet(),
        'deep_baseline2_bn': DeepBaselineNetBN2(init_weights=init_weights),
        'deep_baseline2_bn_residual': DeepBaselineNetBN2Residual(init_weights=init_weights),
        'deep_baseline2_bn_residual_se': DeepBaselineNetBN2ResidualSE(init_weights=init_weights),   
        'deep_baseline2_bn_resnext': DeepBaselineNetBN2ResNeXt(init_weights=init_weights),
        'deep_baseline2_bn_residual_preact': DeepBaselineNetBN2ResidualPreAct(),
        'deep_baseline2_bn_residual_grn': DeepBaselineNetBN2ResidualGRN(init_weights=init_weights),
        'deep_baseline3_bn': DeepBaselineNetBN3(init_weights=init_weights),
        'deep_baseline3_bn_residual': DeepBaselineNetBN3Residual(init_weights=init_weights),
        'deep_baseline3_bn_residual_wide': DeepBaselineNetBN3ResidualWide(init_weights=init_weights),
        'deep_baseline3_bn_residual_4x': DeepBaselineNetBN3Residual4X(init_weights=init_weights),
        'deep_baseline3_bn_residual_deep': DeepBaselineNetBN3ResidualDeep(init_weights=init_weights),
        'deep_baseline3_bn_residual_preact': DeepBaselineNetBN3ResidualPreAct(init_weights=init_weights),
        'deep_baseline3_bn_residual_swish': DeepBaselineNetBN3ResidualSwish(init_weights=init_weights),
        'deep_baseline3_bn_residual_swiglu': DeepBaselineNetBN3ResidualSwiGLU(init_weights=init_weights),
        'deep_baseline3_bn_residual_dla': DeepBaselineNetBN3ResidualDLA(init_weights=init_weights),
        'deep_baseline3_bn_residual_dla_tree': DeepBaselineNetBN3ResidualDLATree(init_weights=init_weights),
        'deep_baseline3_bn_residual_group': DeepBaselineNetBN3ResidualGroup(init_weights=init_weights),
        'deep_baseline3_bn_residual_shakedrop': DeepBaselineNetBN3ResidualShakeDrop(init_weights=init_weights),
        'deep_baseline3_bn_residual_mish': DeepBaselineNetBN3ResidualMish(init_weights=init_weights),
        'deep_baseline3_bn_residual_gap_gmp': DeepBaselineNetBN3ResidualGAPGMP(init_weights=init_weights),
        'deep_baseline3_bn_residual_gap_gmp_s3_f8_16_32_b2': DeepBaselineNetBN3ResidualGAPGMP_S3_F8_16_32_B2(init_weights=init_weights),
        'deep_baseline3_bn_residual_gap_gmp_s3_f16_32_64_b3': DeepBaselineNetBN3ResidualGAPGMP_S3_F16_32_64_B3(init_weights=init_weights),
        'deep_baseline3_bn_residual_gap_gmp_s3_f32_64_128_b5': DeepBaselineNetBN3ResidualGAPGMP_S3_F32_64_128_B5(init_weights=init_weights),
        'deep_baseline3_bn_residual_gap_gmp_s3_f64_128_256_b5': DeepBaselineNetBN3ResidualGAPGMP_S3_F64_128_256_B5(init_weights=init_weights),
        'deep_baseline3_bn_residual_gap_gmp_s4_f64_128_256_512_b5': DeepBaselineNetBN3ResidualGAPGMP_S4_F64_128_256_512_B5(init_weights=init_weights),
        'deep_baseline4_bn_residual': DeepBaselineNetBN4Residual(init_weights=init_weights),
        'deep_baseline4_bn_residual_shakedrop': DeepBaselineNetBN4ResidualShakeDrop(init_weights=init_weights),
        'deep_baseline_se': DeepBaselineNetSE(),
        'convnext_patchify': ConvNeXtPatchifyClassifier(init_weights=init_weights),
        'convnext_local': ConvNeXtLocalBlockClassifier(init_weights=init_weights),
        'convnext_cifar': ConvNeXtCIFAR(init_weights=init_weights),
        'convnext_tiny': convnext_tiny(init_weights=init_weights),
        'resnet18': ResNet18(),
        'vgg16': VGG('VGG16'),
        'mobilenetv2': MobileNetV2(),
        'densenet121': DenseNet121(),
        'mxresnet20': MXResNet20(init_weights=init_weights),
        'mxresnet32': MXResNet32(init_weights=init_weights),
        'mxresnet44': MXResNet44(init_weights=init_weights),
        'mxresnet56': MXResNet56(init_weights=init_weights),
        'dla': DLA(),
        'resnext29_4x64d': ResNeXt29_4x64d(),
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
                  t_max: int = None, eta_min: float = 0.0, pct_start: float = 0.3,
                  final_lr_ratio: float = 0.0001):
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
        
        # OneCycleLR의 동작 방식:
        # 초기 학습률 = max_lr / div_factor (div_factor 기본값: 25.0)
        # 마지막 학습률 = (max_lr / div_factor) / final_div_factor = max_lr / (div_factor * final_div_factor)
        # 사용자가 원하는 마지막 학습률 = lr * final_lr_ratio
        # 따라서: max_lr / (div_factor * final_div_factor) = lr * final_lr_ratio
        # final_div_factor = max_lr / (div_factor * lr * final_lr_ratio)
        
        div_factor = 25.0  # PyTorch 기본값
        final_div_factor = max_lr / (div_factor * lr * final_lr_ratio) if final_lr_ratio > 0 else 10000.0
        
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
                        choices=['crossentropy', 'mse', 'nll', 'supcon_ce'],
                        help='손실 함수 (default: crossentropy)')
    parser.add_argument('--net', type=str, default='baseline',
                        choices=['baseline', 'baseline_bn', 'deep_baseline', 'deep_baseline_silu',
                                 'deep_baseline_bn', 'deep_baseline_gap', 'deep_baseline_bn_dropout',
                                 'deep_baseline_bn_dropout_resnet', 'deep_baseline_se', 'resnet18',
                                 'vgg16', 'mobilenetv2', 'densenet121', 'deep_baseline2_bn', 'deep_baseline2_bn_residual',
                                 'deep_baseline2_bn_residual_preact', 'deep_baseline3_bn', 'deep_baseline2_bn_resnext', 'deep_baseline2_bn_residual_se',
                                 'deep_baseline2_bn_residual_grn', 'deep_baseline3_bn_residual', 
                                 'deep_baseline3_bn_residual_preact', 'deep_baseline3_bn_residual_wide', 'deep_baseline3_bn_residual_4x', 'deep_baseline3_bn_residual_deep', 
                                 'deep_baseline3_bn_residual_swish', 'deep_baseline3_bn_residual_swiglu', 
                                 'deep_baseline3_bn_residual_dla', 'deep_baseline3_bn_residual_dla_tree', 'deep_baseline3_bn_residual_group', 
                                 'deep_baseline3_bn_residual_mish', 'deep_baseline3_bn_residual_gap_gmp', 
                                 'deep_baseline3_bn_residual_gap_gmp_s3_f8_16_32_b2', 
                                 'deep_baseline3_bn_residual_gap_gmp_s3_f16_32_64_b3', 
                                 'deep_baseline3_bn_residual_gap_gmp_s3_f32_64_128_b5', 
                                 'deep_baseline3_bn_residual_gap_gmp_s3_f64_128_256_b5', 
                                 'deep_baseline3_bn_residual_gap_gmp_s4_f64_128_256_512_b5', 
                                 'deep_baseline4_bn_residual',
                                 'deep_baseline3_bn_residual_shakedrop',
                                 'deep_baseline4_bn_residual_shakedrop',
                                 'convnext_patchify', 'convnext_local', 'convnext_cifar', 'convnext_tiny',
                                 'mxresnet20', 'mxresnet32', 'mxresnet44', 'mxresnet56', 'dla', 'resnext29_4x64d'],

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
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='Label smoothing 값 (0.0~1.0, 권장: 0.05~0.1, default: 0.0)')
    parser.add_argument('--augment', action='store_true',
                        help='데이터 증강 사용 (default: False)')
    parser.add_argument('--autoaugment', action='store_true',
                        help='AutoAugment 사용 (--augment가 활성화되어 있을 때만 동작, default: False)')
    parser.add_argument('--cutmix', action='store_true',
                        help='CutMix 증강 사용 (--augment가 활성화되어 있을 때만 동작, default: False)')
    parser.add_argument('--cutmix-start-epoch-ratio', type=float, default=0.0,
                        help='CutMix 시작 에포크 비율 (0.0~1.0, 예: 0.3이면 전체 에포크의 30%% 이후부터 적용, default: 0.0)')
    parser.add_argument('--mixup', action='store_true',
                        help='Mixup 증강 사용 (--augment가 활성화되어 있을 때만 동작, default: False)')
    parser.add_argument('--mixup-start-epoch-ratio', type=float, default=0.0,
                        help='Mixup 시작 에포크 비율 (0.0~1.0, 예: 0.3이면 전체 에포크의 30%% 이후부터 적용, default: 0.0)')
    parser.add_argument('--calibrate', action='store_true',
                        help='Temperature Scaling 캘리브레이션 수행 (default: False)')
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
    return parser.parse_args()


def main():
    args = parse_args()

    # 시드 고정
    set_seed(args.seed)

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Save path에 모델 이름과 설정 정보 자동 추가
    model_name_parts = get_model_name_parts(args)
    model_name = "_".join(filter(None, model_name_parts))  # 빈 문자열 제거
    
    # 출력 디렉토리 설정
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    SAVE_PATH = os.path.join(output_dir, f"{model_name}.pth")
    HISTORY_PATH = os.path.join(output_dir, f"{model_name}_history.json")

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
    train_transform_list = []
    
    if args.augment:
        train_transform_list.append(transforms.RandomCrop(32, padding=4))
        train_transform_list.append(transforms.RandomHorizontalFlip())
        
        if args.autoaugment:
            # AutoAugment 사용: CIFAR-10 정책 적용
            train_transform_list.append(transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.CIFAR10))
        else:
            # 기본 데이터 증강: RandomRotation
            train_transform_list.append(transforms.RandomRotation(15))
    
    # 공통 변환: ToTensor와 Normalize는 항상 적용
    train_transform_list.append(transforms.ToTensor())
    train_transform_list.append(transforms.Normalize(normalize_mean, normalize_std))
    
    train_transform = transforms.Compose(train_transform_list)

    # Validation: 데이터 증강 없이 기본 변환만
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])

    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform)
    
    # CutMix/Mixup 설정 (--augment가 활성화되어 있을 때만)
    # CutMix와 Mixup은 동시에 사용할 수 없음
    cutmix_collator = None
    cutmix_criterion = None 
    mixup_collator = None
    mixup_criterion = None
    
    if args.cutmix and args.augment:
        if args.mixup:
            raise ValueError("CutMix와 Mixup은 동시에 사용할 수 없습니다. 하나만 선택해주세요.")
        cutmix_collator = CutMixCollator(alpha=1.0, prob=0.5)
        cutmix_criterion = CutMixCriterion(reduction='mean', label_smoothing=args.label_smoothing)
    elif args.mixup and args.augment:
        mixup_collator = MixupCollator(alpha=1.0, prob=0.5)
        mixup_criterion = MixupCriterion(reduction='mean', label_smoothing=args.label_smoothing)
    
    # 시작 에포크 계산
    cutmix_start_epoch = int(args.cutmix_start_epoch_ratio * args.epochs) if args.cutmix and args.augment else args.epochs
    mixup_start_epoch = int(args.mixup_start_epoch_ratio * args.epochs) if args.mixup and args.augment else args.epochs
    
    # 초기 collate_fn 설정 (첫 에포크에서는 시작 비율에 따라 결정)
    collate_fn = None
    if args.cutmix and args.augment and cutmix_start_epoch == 0:
        collate_fn = cutmix_collator
    elif args.mixup and args.augment and mixup_start_epoch == 0:
        collate_fn = mixup_collator
    
    # num_workers 자동 설정: AutoAugment 사용 시 더 많은 워커 필요
    if args.num_workers is None:
        # AutoAugment는 CPU에서 무거운 작업이므로 더 많은 워커 사용
        num_workers = 4 if args.autoaugment and args.augment else 2
    else:
        num_workers = args.num_workers
    
    # GPU 사용 시 pin_memory 활성화로 전송 속도 향상
    pin_memory = torch.cuda.is_available()
    # persistent_workers로 워커 재생성 오버헤드 감소 (num_workers > 0일 때만)
    persistent_workers = num_workers > 0
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers, collate_fn=collate_fn)

    val_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=persistent_workers)

    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    net = get_net(args.net, init_weights=args.w_init)
    net = net.to(device)

    # Criterion 설정
    supcon_criterion = None
    supcon_weight = 0.1
    if args.criterion.lower() == 'supcon_ce':
        # SupCon + CrossEntropy 조합
        criterion = get_criterion('crossentropy', label_smoothing=args.label_smoothing)
        supcon_criterion = SupConLoss(temperature=0.07)
        # SupCon은 현재 baseline_bn 모델에서만 정식 지원
        if args.net != 'baseline_bn':
            print("[경고] --criterion supcon_ce는 현재 baseline_bn 모델에 최적화되어 있습니다.")
    else:
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
        eta_min=args.scheduler_eta_min, pct_start=args.scheduler_pct_start,
        final_lr_ratio=args.scheduler_final_lr_ratio
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
            'autoaugment': args.autoaugment and args.augment,
            'cutmix': args.cutmix and args.augment,
            'cutmix_start_epoch_ratio': args.cutmix_start_epoch_ratio if args.cutmix and args.augment else None,
            'mixup': args.mixup and args.augment,
            'mixup_start_epoch_ratio': args.mixup_start_epoch_ratio if args.mixup and args.augment else None,
            'normalize_mean': list(normalize_mean),
            'normalize_std': list(normalize_std),
            'seed': args.seed,
            'weight_init': args.w_init,
            'device': str(device),
            'early_stopping': args.early_stopping,
            'early_stopping_patience': args.early_stopping_patience if args.early_stopping else None,
            'num_workers': num_workers,
            'pin_memory': pin_memory
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
            history['hyperparameters']['scheduler_pct_start'] = args.scheduler_pct_start
            history['hyperparameters']['scheduler_final_lr_ratio'] = args.scheduler_final_lr_ratio
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

    # Training configuration 출력
    print_training_configuration(args, normalize_mean, normalize_std, SAVE_PATH, HISTORY_PATH)

    # 최고 검증 정확도 추적
    best_val_acc = -1.0
    
    # Early stopping 설정
    early_stopping_enabled = args.early_stopping
    early_stopping_patience = args.early_stopping_patience if early_stopping_enabled else None
    patience_counter = 0

    for epoch in range(args.epochs):
        # 현재 에포크에서 cutmix/mixup 적용 여부 확인
        use_cutmix = args.cutmix and args.augment and epoch >= cutmix_start_epoch
        use_mixup = args.mixup and args.augment and epoch >= mixup_start_epoch
        
        # collate_fn 동적 설정
        current_collate_fn = None
        if use_cutmix:
            current_collate_fn = cutmix_collator
        elif use_mixup:
            current_collate_fn = mixup_collator
        
        # collate_fn이 변경된 경우 DataLoader 재생성
        if current_collate_fn != collate_fn:
            collate_fn = current_collate_fn
            train_loader = torch.utils.data.DataLoader(
                train_set, batch_size=args.batch_size, shuffle=True, 
                num_workers=num_workers, pin_memory=pin_memory,
                persistent_workers=persistent_workers, collate_fn=collate_fn)
        
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

            optimizer.zero_grad()
            
            outputs = net(inputs)

            # 기본 CrossEntropy / CutMix / Mixup 손실
            if use_cutmix and isinstance(labels, tuple):
                loss_ce = cutmix_criterion(outputs, labels)
            elif use_mixup and isinstance(labels, tuple):
                loss_ce = mixup_criterion(outputs, labels)
            else:
                loss_ce = criterion(outputs, labels)

            # SupCon 손실 추가 (--criterion supcon_ce, CutMix/Mixup 미사용 시)
            if supcon_criterion is not None:
                if use_cutmix or use_mixup:
                    raise ValueError("SupCon CE(--criterion supcon_ce)는 CutMix/Mixup과 함께 사용할 수 없습니다.")
                if hasattr(net, "forward_features"):
                    features = net.forward_features(inputs)
                else:
                    # 안전 장치: 별도 feature 추출 메서드가 없을 경우 logits를 그대로 사용
                    features = outputs
                # SupCon은 hard label만 사용
                supcon_loss = supcon_criterion(features, labels)
                loss = loss_ce + supcon_weight * supcon_loss
            else:
                loss = loss_ce
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
            print(f"  Early Stopping: {patience_counter}/{early_stopping_patience} (no improvement)")
        print()

        # 매 epoch마다 히스토리 저장 (중간에 중단되어도 데이터 보존)
        with open(HISTORY_PATH, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Early stopping 체크 (활성화된 경우에만)
        if early_stopping_enabled and patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            print(f"No improvement in validation accuracy for {early_stopping_patience} consecutive epochs")
            history['early_stopped'] = True
            history['early_stopped_epoch'] = epoch + 1
            history['early_stopping_patience'] = early_stopping_patience
            # 히스토리 파일 업데이트
            with open(HISTORY_PATH, 'w') as f:
                json.dump(history, f, indent=2)
            break

    # Early stopping이 발생하지 않은 경우 히스토리 업데이트
    if not early_stopping_enabled or (early_stopping_enabled and patience_counter < early_stopping_patience):
        history['early_stopped'] = False
        if early_stopping_enabled:
            history['early_stopping_patience'] = early_stopping_patience
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
        TEMP_PATH = os.path.join(output_dir, f"{model_name}_temperature.json")
        with open(TEMP_PATH, 'w') as f:
            json.dump({'temperature': optimal_temperature}, f, indent=2)
        print(f"Temperature saved to: {TEMP_PATH}")


if __name__ == '__main__':
    main()
