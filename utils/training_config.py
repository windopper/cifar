def print_training_configuration(args, normalize_mean, normalize_std, save_path, history_path):
    """
    학습 설정 정보를 출력합니다.
    
    Args:
        args: argparse.Namespace 객체, 학습 설정을 포함
        normalize_mean: 정규화에 사용되는 평균값 (tuple)
        normalize_std: 정규화에 사용되는 표준편차 (tuple)
        save_path: 모델 저장 경로 (str)
        history_path: 히스토리 저장 경로 (str)
    """
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
    if args.cutmix:
        if args.augment:
            cutmix_start_epoch = int(args.cutmix_start_epoch_ratio * args.epochs)
            print(f"  CutMix: Enabled (starts from epoch {cutmix_start_epoch + 1}, ratio: {args.cutmix_start_epoch_ratio})")
        else:
            print(f"  CutMix: Disabled (--augment must be enabled)")
    if args.mixup:
        if args.augment:
            mixup_start_epoch = int(args.mixup_start_epoch_ratio * args.epochs)
            print(f"  Mixup: Enabled (starts from epoch {mixup_start_epoch + 1}, ratio: {args.mixup_start_epoch_ratio})")
        else:
            print(f"  Mixup: Disabled (--augment must be enabled)")
    print(f"  Weight initialization: {args.w_init}")
    print(f"  EMA: {args.ema}")
    if args.ema:
        print(f"  EMA decay: {args.ema_decay}")
    print(f"  SAM: {args.sam}")
    if args.sam:
        print(f"  SAM rho: {args.sam_rho}")
        print(f"  SAM adaptive: {args.sam_adaptive}")
    print(f"  Normalize mean: {normalize_mean}")
    print(f"  Normalize std: {normalize_std}")
    print(f"  Scheduler: {args.scheduler}")
    if args.scheduler and args.scheduler.lower() != 'none':
        if args.scheduler.lower() == 'exponentiallr':
            print(f"  Scheduler gamma: {args.scheduler_gamma}")
        elif args.scheduler.lower() == 'onecyclelr':
            max_lr = args.scheduler_max_lr if args.scheduler_max_lr else args.lr * 10
            print(f"  Scheduler max_lr: {max_lr}")
            print(f"  Scheduler pct_start: {args.scheduler_pct_start}")
            print(f"  Scheduler final_lr_ratio: {args.scheduler_final_lr_ratio}")
            final_lr = args.lr * args.scheduler_final_lr_ratio
            print(f"  Scheduler final_lr: {final_lr}")
        elif args.scheduler.lower() == 'reducelronplateau':
            print(f"  Scheduler factor: {args.scheduler_factor}")
            print(f"  Scheduler patience: {args.scheduler_patience}")
            print(f"  Scheduler mode: {args.scheduler_mode}")
            print(f"  Scheduler metric: val_loss")
        elif args.scheduler.lower() == 'cosineannealinglr':
            t_max = args.scheduler_t_max if args.scheduler_t_max else args.epochs
            print(f"  Scheduler T_max: {t_max}")
            print(f"  Scheduler eta_min: {args.scheduler_eta_min}")
    print(f"  Save path: {save_path}")
    print(f"  History path: {history_path}")
    print()

