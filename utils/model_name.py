def get_model_name_parts(args):
    """
    학습 설정에 따라 모델 이름 파트 리스트를 생성합니다.
    
    Args:
        args: argparse.Namespace 객체, 학습 설정을 포함
        
    Returns:
        list: 모델 이름을 구성하는 파트들의 리스트
    """
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
    
    if args.autoaugment and args.augment:
        model_name_parts.append("autoaug")
    
    if args.cutmix and args.augment:
        model_name_parts.append("cutmix")
    
    if args.mixup and args.augment:
        model_name_parts.append("mixup")
    
    if args.w_init:
        model_name_parts.append("winit")
    
    if args.calibrate:
        model_name_parts.append("calibrated")
    
    if args.use_cifar_normalize:
        model_name_parts.append("cifar_normalize")
    
    return model_name_parts

