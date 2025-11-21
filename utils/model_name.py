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
    
    if args.optimizer.lower() == 'sgd' and args.nesterov:
        model_name_parts.append("nesterov")
    
    if args.optimizer.lower() == 'adamw':
        model_name_parts.append(f"wd{args.weight_decay}")
    
    if args.scheduler and args.scheduler.lower() != 'none':
        model_name_parts.append(f"sch{args.scheduler}")
        if args.scheduler.lower() == 'exponentiallr':
            model_name_parts.append(f"gamma{args.scheduler_gamma}")
        elif args.scheduler.lower() == 'onecyclelr':
            if args.scheduler_max_lr:
                model_name_parts.append(f"maxlr{args.scheduler_max_lr}")
            if args.scheduler_pct_start != 0.3:  # 기본값이 아닐 때만 추가
                model_name_parts.append(f"pct{args.scheduler_pct_start}")
            if args.scheduler_final_lr_ratio != 0.0001:  # 기본값이 아닐 때만 추가
                model_name_parts.append(f"finalr{args.scheduler_final_lr_ratio}")
        elif args.scheduler.lower() == 'reducelronplateau': 
            model_name_parts.append(f"factor{args.scheduler_factor}")
            model_name_parts.append(f"patience{args.scheduler_patience}")
        elif args.scheduler.lower() == 'cosineannealinglr':
            t_max = args.scheduler_t_max if args.scheduler_t_max else args.epochs
            model_name_parts.append(f"tmax{t_max}")
            if args.scheduler_eta_min > 0.0:
                model_name_parts.append(f"etamin{args.scheduler_eta_min}")
        elif args.scheduler.lower() == 'cosineannealingwarmuprestarts':
            if args.scheduler_first_cycle_steps:
                model_name_parts.append(f"firstcycle{args.scheduler_first_cycle_steps}")
            if args.scheduler_cycle_mult != 1.0:  # 기본값이 아닐 때만 추가
                model_name_parts.append(f"cyclemult{args.scheduler_cycle_mult}")
            if args.scheduler_max_lr:
                model_name_parts.append(f"maxlr{args.scheduler_max_lr}")
            if args.scheduler_min_lr:
                model_name_parts.append(f"minlr{args.scheduler_min_lr}")
            if args.scheduler_warmup_epochs > 0:
                model_name_parts.append(f"warmup{args.scheduler_warmup_epochs}")
            if args.scheduler_gamma_restarts != 1.0:  # 기본값이 아닐 때만 추가
                model_name_parts.append(f"gammar{args.scheduler_gamma_restarts}")
    
    if args.label_smoothing > 0.0:
        model_name_parts.append(f"ls{args.label_smoothing}")
    
    if args.weighted_ce:
        model_name_parts.append("weighted_ce")
    
    if args.augment:
        model_name_parts.append("aug")
    
    if args.autoaugment and args.augment:
        model_name_parts.append("autoaug")
    
    if args.cutmix and args.augment and args.mixup and args.augment:
        # 둘 다 활성화된 경우
        model_name_parts.append("cutmix_mixup")
        if args.cutmix_alpha != 1.0:  # 기본값이 아닐 때만 추가
            model_name_parts.append(f"cutmixa{args.cutmix_alpha}")
        if args.cutmix_prob != 0.5:  # 기본값이 아닐 때만 추가
            model_name_parts.append(f"cutmixp{args.cutmix_prob}")
        if args.mixup_alpha != 1.0:  # 기본값이 아닐 때만 추가
            model_name_parts.append(f"mixupa{args.mixup_alpha}")
        if args.mixup_prob != 0.5:  # 기본값이 아닐 때만 추가
            model_name_parts.append(f"mixupp{args.mixup_prob}")
        if args.switch_prob != 0.5:  # 기본값이 아닐 때만 추가
            model_name_parts.append(f"switchp{args.switch_prob}")
        if args.cutmix_start_epoch_ratio > 0.0:
            model_name_parts.append(f"cutmixstart{args.cutmix_start_epoch_ratio}")
        if args.mixup_start_epoch_ratio > 0.0:
            model_name_parts.append(f"mixupstart{args.mixup_start_epoch_ratio}")
    elif args.cutmix and args.augment:
        model_name_parts.append("cutmix")
        if args.cutmix_alpha != 1.0:  # 기본값이 아닐 때만 추가
            model_name_parts.append(f"cutmixa{args.cutmix_alpha}")
        if args.cutmix_prob != 0.5:  # 기본값이 아닐 때만 추가
            model_name_parts.append(f"cutmixp{args.cutmix_prob}")
        if args.cutmix_start_epoch_ratio > 0.0:
            model_name_parts.append(f"cutmixstart{args.cutmix_start_epoch_ratio}")
    elif args.mixup and args.augment:
        model_name_parts.append("mixup")
        if args.mixup_alpha != 1.0:  # 기본값이 아닐 때만 추가
            model_name_parts.append(f"mixupa{args.mixup_alpha}")
        if args.mixup_prob != 0.5:  # 기본값이 아닐 때만 추가
            model_name_parts.append(f"mixupp{args.mixup_prob}")
        if args.mixup_start_epoch_ratio > 0.0:
            model_name_parts.append(f"mixupstart{args.mixup_start_epoch_ratio}")
    
    if args.cutout and args.augment:
        model_name_parts.append("cutout")
        if args.cutout_length != 16:  # 기본값이 아닐 때만 추가
            model_name_parts.append(f"cutoutlen{args.cutout_length}")
        if args.cutout_n_holes != 1:  # 기본값이 아닐 때만 추가
            model_name_parts.append(f"cutoutn{args.cutout_n_holes}")
        if args.cutout_prob != 0.5:  # 기본값이 아닐 때만 추가
            model_name_parts.append(f"cutoutprob{args.cutout_prob}")
    
    if args.w_init:
        model_name_parts.append("winit")
    
    if args.calibrate:
        model_name_parts.append("calibrated")
    
    if args.use_cifar_normalize:
        model_name_parts.append("cifar_normalize")
    
    if args.ema:
        model_name_parts.append("ema")
        if args.ema_decay != 0.9999:  # 기본값이 아닐 때만 추가
            model_name_parts.append(f"emad{args.ema_decay}")
    
    if args.sam:
        model_name_parts.append("sam")
        if args.sam_rho != 0.05:  # 기본값이 아닐 때만 추가
            model_name_parts.append(f"samrho{args.sam_rho}")
        if args.sam_adaptive:
            model_name_parts.append("samadaptive")
    
    if args.shakedrop > 0.0:
        model_name_parts.append("shakedrop")
        if args.shakedrop != 0.5:  # 기본값이 아닐 때만 추가
            model_name_parts.append(f"shakedropp{args.shakedrop}")
    
    return model_name_parts

