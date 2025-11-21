"""학습 히스토리 관리 유틸리티"""
import json
import os


def create_history(args, normalize_mean, normalize_std, device, num_workers, pin_memory, steps_per_epoch):
    """
    학습 히스토리 딕셔너리 초기화
    
    Args:
        args: argparse 인자 객체
        normalize_mean: Normalize 평균 값
        normalize_std: Normalize 표준편차 값
        device: 사용 디바이스
        num_workers: DataLoader 워커 수
        pin_memory: pin_memory 설정
        steps_per_epoch: 에포크당 스텝 수
    
    Returns:
        history: 초기화된 히스토리 딕셔너리
    """
    history = {
        'hyperparameters': {
            'batch_size': args.batch_size,
            'criterion': args.criterion,
            'net': args.net,
            'optimizer': args.optimizer,
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'momentum': args.momentum,
            'nesterov': args.nesterov if args.optimizer.lower() == 'sgd' else None,
            'scheduler': args.scheduler,
            'label_smoothing': args.label_smoothing,
            'weighted_ce': args.weighted_ce,
            'data_augment': args.augment,
            'autoaugment': args.autoaugment and args.augment,
            'cutmix': args.cutmix and args.augment,
            'cutmix_alpha': args.cutmix_alpha if args.cutmix and args.augment else None,
            'cutmix_prob': args.cutmix_prob if args.cutmix and args.augment else None,
            'mixup': args.mixup and args.augment,
            'mixup_alpha': args.mixup_alpha if args.mixup and args.augment else None,
            'mixup_prob': args.mixup_prob if args.mixup and args.augment else None,
            'switch_prob': args.switch_prob if (args.cutmix and args.augment and args.mixup and args.augment) else None,
            'cutout': args.cutout and args.augment,
            'cutout_length': args.cutout_length if args.cutout and args.augment else None,
            'cutout_n_holes': args.cutout_n_holes if args.cutout and args.augment else None,
            'cutout_prob': args.cutout_prob if args.cutout and args.augment else None,
            'normalize_mean': list(normalize_mean),
            'normalize_std': list(normalize_std),
            'seed': args.seed,
            'weight_init': args.w_init,
            'device': str(device),
            'early_stopping': args.early_stopping,
            'early_stopping_patience': args.early_stopping_patience if args.early_stopping else None,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'ema': args.ema,
            'ema_decay': args.ema_decay if args.ema else None,
            'sam': args.sam,
            'sam_rho': args.sam_rho if args.sam else None,
            'sam_adaptive': args.sam_adaptive if args.sam else None
        },
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'best_val_accuracy': None
    }
    
    return history


def update_history_scheduler_params(history, args, steps_per_epoch):
    """
    Scheduler 관련 하이퍼파라미터를 히스토리에 추가
    
    Args:
        history: 히스토리 딕셔너리
        args: argparse 인자 객체
        steps_per_epoch: 에포크당 스텝 수
    """
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
        elif args.scheduler.lower() == 'cosineannealingwarmuprestarts':
            history['hyperparameters']['scheduler_first_cycle_steps'] = args.scheduler_first_cycle_steps if args.scheduler_first_cycle_steps else args.epochs * steps_per_epoch
            history['hyperparameters']['scheduler_cycle_mult'] = args.scheduler_cycle_mult
            history['hyperparameters']['scheduler_max_lr'] = args.scheduler_max_lr if args.scheduler_max_lr else args.lr
            history['hyperparameters']['scheduler_min_lr'] = args.scheduler_min_lr if args.scheduler_min_lr else (args.scheduler_eta_min if args.scheduler_eta_min > 0 else args.lr * 0.001)
            history['hyperparameters']['scheduler_warmup_epochs'] = args.scheduler_warmup_epochs
            history['hyperparameters']['scheduler_gamma_restarts'] = args.scheduler_gamma_restarts


def update_history_optimizer_params(history, args):
    """
    Optimizer 관련 하이퍼파라미터를 히스토리에 추가
    
    Args:
        history: 히스토리 딕셔너리
        args: argparse 인자 객체
    """
    if args.optimizer.lower() == 'adamw':
        history['hyperparameters']['weight_decay'] = args.weight_decay


def update_history_epoch(history, train_loss, val_loss, val_acc):
    """
    에포크마다 히스토리 업데이트
    
    Args:
        history: 히스토리 딕셔너리
        train_loss: 학습 손실
        val_loss: 검증 손실
        val_acc: 검증 정확도
    """
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_accuracy'].append(val_acc)


def update_history_best(history, best_val_acc):
    """
    최고 검증 정확도 업데이트
    
    Args:
        history: 히스토리 딕셔너리
        best_val_acc: 최고 검증 정확도
    """
    history['best_val_accuracy'] = best_val_acc


def update_history_early_stop(history, early_stopped, early_stopped_epoch=None, early_stopping_patience=None):
    """
    Early stopping 정보 업데이트
    
    Args:
        history: 히스토리 딕셔너리
        early_stopped: Early stopping 발생 여부
        early_stopped_epoch: Early stopping 발생 에포크
        early_stopping_patience: Early stopping patience 값
    """
    history['early_stopped'] = early_stopped
    if early_stopped_epoch is not None:
        history['early_stopped_epoch'] = early_stopped_epoch
    if early_stopping_patience is not None:
        history['early_stopping_patience'] = early_stopping_patience


def update_history_calibration(history, optimal_temperature, calibrated_val_accuracy):
    """
    캘리브레이션 정보 업데이트
    
    Args:
        history: 히스토리 딕셔너리
        optimal_temperature: 최적 온도 값
        calibrated_val_accuracy: 캘리브레이션 후 검증 정확도
    """
    history['hyperparameters']['temperature'] = optimal_temperature
    history['calibrated_val_accuracy'] = calibrated_val_accuracy


def save_history(history, history_path):
    """
    히스토리를 JSON 파일로 저장
    
    Args:
        history: 히스토리 딕셔너리
        history_path: 저장할 파일 경로
    """
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

