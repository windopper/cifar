"""
모델 캘리브레이션 스크립트
학습된 모델에 Temperature Scaling을 적용하여 캘리브레이션을 수행합니다.
"""
import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.calibration import calibrate_temperature, ECELoss
from utils.dataset import get_cifar10_loaders
from utils.history import update_history_calibration, save_history, load_history
from inference import load_model_from_history, load_temperature
from train import CLASS_NAMES
from tqdm import tqdm


def get_net_from_history(history_path):
    """히스토리 파일에서 네트워크 정보 추출"""
    history = load_history(history_path)
    if history is None:
        return None
    return history.get('hyperparameters', {}).get('net')


def calibrate_single_model(history_path, model_path, device, use_cifar_normalize, 
                          cross_validate='ece', save_calibration=False):
    """
    단일 모델 캘리브레이션 수행
    
    Returns:
        (net, optimal_temperature, metrics, calibrated_acc, history, normalize_mean, normalize_std)
    """
    # 히스토리 로드
    history = load_history(history_path)
    if history is None:
        raise FileNotFoundError(f"히스토리 파일을 찾을 수 없습니다: {history_path}")
    
    # 네트워크 정보 가져오기
    net_name = get_net_from_history(history_path)
    if net_name is None:
        raise ValueError(f"히스토리에서 네트워크 정보를 찾을 수 없습니다: {history_path}")
    
    # 모델 경로 자동 추출 (지정되지 않은 경우)
    if model_path is None:
        if history_path.endswith('_history.json'):
            model_path = history_path.replace('_history.json', '.pth')
        else:
            base_path = history_path.rsplit('.json', 1)[0]
            model_path = f"{base_path.rsplit('_history', 1)[0]}.pth"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    # normalize 값 추출
    hp = history.get('hyperparameters', {})
    normalize_mean_val = hp.get('normalize_mean', [0.5, 0.5, 0.5])
    normalize_std_val = hp.get('normalize_std', [0.5, 0.5, 0.5])
    if isinstance(normalize_mean_val, list):
        normalize_mean = tuple(normalize_mean_val)
    else:
        normalize_mean = normalize_mean_val
    if isinstance(normalize_std_val, list):
        normalize_std = tuple(normalize_std_val)
    else:
        normalize_std = normalize_std_val
    
    # CIFAR-10 표준 normalize 확인
    cifar_mean = [0.4914, 0.4822, 0.4465]
    cifar_std = [0.2023, 0.1994, 0.2010]
    if 'normalize_mean' in hp and 'normalize_std' in hp:
        hist_mean = hp['normalize_mean']
        hist_std = hp['normalize_std']
        mean_match = all(abs(a - b) < 1e-4 for a, b in zip(hist_mean, cifar_mean))
        std_match = all(abs(a - b) < 1e-4 for a, b in zip(hist_std, cifar_std))
        use_cifar_normalize_model = mean_match and std_match
    else:
        use_cifar_normalize_model = False
    
    # 모델 로드
    from utils.net import get_net
    net = get_net(net_name, init_weights=False, shakedrop_prob=0.0)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net = net.to(device)
    net.eval()
    
    # 검증 데이터 로더 생성
    _, val_loader, _, _ = get_cifar10_loaders(
        batch_size=128,
        augment=False,
        autoaugment=False,
        cutout=False,
        use_cifar_normalize=use_cifar_normalize if use_cifar_normalize is not None else use_cifar_normalize_model,
        num_workers=2,
        collate_fn=None,
        data_root='./data'
    )
    
    # 캘리브레이션 전 검증 정확도 확인
    net.eval()
    before_correct = 0
    before_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = net(inputs)
            _, predicted = torch.max(logits.data, 1)
            before_total += labels.size(0)
            before_correct += (predicted == labels).sum().item()
    
    before_acc = 100 * before_correct / before_total
    
    # Temperature Scaling 캘리브레이션 수행
    optimal_temperature, metrics = calibrate_temperature(
        net, val_loader, device, cross_validate=cross_validate, log=False
    )
    
    # 캘리브레이션 후 검증 정확도 확인
    net.eval()
    calibrated_correct = 0
    calibrated_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = net(inputs)
            scaled_logits = logits / optimal_temperature
            _, predicted = torch.max(scaled_logits.data, 1)
            calibrated_total += labels.size(0)
            calibrated_correct += (predicted == labels).sum().item()
    
    calibrated_acc = 100 * calibrated_correct / calibrated_total
    
    # 저장 옵션이 활성화된 경우에만 저장
    if save_calibration:
        update_history_calibration(history, optimal_temperature, calibrated_acc)
        if 'calibration' not in history:
            history['calibration'] = {}
        history['calibration']['metrics'] = {
            'before_nll': metrics['before_nll'],
            'before_ece': metrics['before_ece'],
            'after_nll': metrics['after_nll'],
            'after_ece': metrics['after_ece'],
            'T_opt_nll': metrics['T_opt_nll'],
            'T_opt_ece': metrics['T_opt_ece'],
            'cross_validate': cross_validate
        }
        save_history(history, history_path)
        
        # Temperature를 별도 파일로도 저장
        model_dir = os.path.dirname(model_path)
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        temp_path = os.path.join(model_dir, f"{model_name}_temperature.json")
        temp_data = {
            'temperature': optimal_temperature,
            'T_opt_nll': metrics['T_opt_nll'],
            'T_opt_ece': metrics['T_opt_ece'],
            'cross_validate': cross_validate,
            'metrics': {
                'before_nll': metrics['before_nll'],
                'before_ece': metrics['before_ece'],
                'after_nll': metrics['after_nll'],
                'after_ece': metrics['after_ece']
            }
        }
        with open(temp_path, 'w') as f:
            json.dump(temp_data, f, indent=2)
    
    return net, optimal_temperature, metrics, calibrated_acc, history, normalize_mean, normalize_std


def calibrate_model(model_path, history_path, device=None, use_cifar_normalize=None, 
                   save_calibration=False, run_inference=False, cross_validate='ece',
                   tta=0, ensemble_history=None, ensemble_weights=None):
    """
    모델 캘리브레이션 수행
    
    Args:
        model_path: 모델 파일 경로 (.pth) 또는 리스트, None이면 히스토리 파일에서 자동 추출
        history_path: 히스토리 파일 경로 (.json) 또는 리스트, 첫 번째는 캘리브레이션 대상
        device: 디바이스 (None이면 자동 선택)
        use_cifar_normalize: CIFAR-10 표준 normalize 사용 여부 (None이면 히스토리에서 자동 추출)
        save_calibration: Temperature를 파일에 저장할지 여부 (기본값: False)
        run_inference: 캘리브레이션 후 테스트셋에서 추론 수행 여부 (기본값: False)
        cross_validate: 'ece' 또는 'nll' - 최적화할 metric (기본값: 'ece')
        tta: Test Time Augmentation 모드 (0: 비활성화, 2: 원본+Horizontal Flip, 3: 원본+Horizontal Flip+확대)
        ensemble_history: Ensemble용 히스토리 파일 경로 리스트 (None이면 단일 모델 또는 history_path의 나머지 사용)
        ensemble_weights: Ensemble 가중치 리스트 (None이면 균등 가중치)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # history_path와 model_path를 리스트로 정규화
    if isinstance(history_path, str):
        history_paths = [history_path]
    else:
        history_paths = list(history_path)
    
    if model_path is None:
        model_paths = [None] * len(history_paths)
    elif isinstance(model_path, str):
        # 단일 모델 경로인 경우, 첫 번째에만 할당하고 나머지는 None
        model_paths = [model_path] + [None] * (len(history_paths) - 1)
    else:
        model_paths = list(model_path)
        # 길이가 맞지 않으면 나머지를 None으로 채움
        if len(model_paths) < len(history_paths):
            model_paths.extend([None] * (len(history_paths) - len(model_paths)))
        elif len(model_paths) > len(history_paths):
            # 경고: model_path가 더 많으면 무시
            model_paths = model_paths[:len(history_paths)]
    
    # ensemble_history가 지정된 경우 추가
    if ensemble_history is not None and len(ensemble_history) > 0:
        history_paths.extend(ensemble_history)
        model_paths.extend([None] * len(ensemble_history))
    
    # 모든 모델 캘리브레이션 수행
    print("\n" + "="*50)
    print(f"총 {len(history_paths)}개 모델 캘리브레이션 시작...")
    print(f"최적화 기준: {cross_validate.upper()}")
    print("="*50)
    
    calibrated_models = []
    for i, (hist_path, mod_path) in enumerate(zip(history_paths, model_paths)):
        print(f"\n[모델 {i+1}/{len(history_paths)}]")
        print(f"히스토리: {hist_path}")
        
        try:
            net, opt_temp, metrics, cal_acc, hist, norm_mean, norm_std = calibrate_single_model(
                history_path=hist_path,
                model_path=mod_path,
                device=device,
                use_cifar_normalize=use_cifar_normalize,
                cross_validate=cross_validate,
                save_calibration=save_calibration
            )
            
            print(f"  Temperature: {opt_temp:.4f}")
            print(f"  검증 정확도: {cal_acc:.2f}%")
            
            calibrated_models.append({
                'net': net,
                'temperature': opt_temp,
                'metrics': metrics,
                'calibrated_acc': cal_acc,
                'history': hist,
                'normalize_mean': norm_mean,
                'normalize_std': norm_std,
                'history_path': hist_path,
                'model_path': mod_path if mod_path else (hist_path.replace('_history.json', '.pth') if hist_path.endswith('_history.json') else f"{hist_path.rsplit('_history', 1)[0]}.pth")
            })
        except Exception as e:
            print(f"  오류 발생: {e}")
            raise
    
    print("\n" + "="*50)
    print("모든 모델 캘리브레이션 완료!")
    print("="*50 + "\n")
    
    # 첫 번째 모델의 결과를 반환값으로 사용 (하위 호환성)
    main_model = calibrated_models[0]
    optimal_temperature = main_model['temperature']
    metrics = main_model['metrics']
    calibrated_acc = main_model['calibrated_acc']
    
    # 테스트셋에서 추론 수행
    if run_inference:
        print("\n" + "="*50)
        print("테스트셋에서 추론 수행 중...")
        print("="*50)
        
        # Ensemble 모드인지 확인 (모델이 2개 이상이면 ensemble)
        is_ensemble = len(calibrated_models) > 1
        
        if is_ensemble:
            print("Ensemble 모드 활성화")
            print(f"총 {len(calibrated_models)}개의 캘리브레이션된 모델을 사용합니다.\n")
            
            # Ensemble 가중치 설정
            if ensemble_weights:
                if len(ensemble_weights) != len(calibrated_models):
                    raise ValueError(f"Ensemble 가중치 개수({len(ensemble_weights)})가 모델 개수({len(calibrated_models)})와 일치하지 않습니다.")
                weights = [w / sum(ensemble_weights) for w in ensemble_weights]  # 정규화
            else:
                weights = [1.0 / len(calibrated_models)] * len(calibrated_models)  # 균등 가중치
            
            # Normalize 값 확인 (첫 번째 모델의 값 사용)
            normalize_mean = calibrated_models[0]['normalize_mean']
            normalize_std = calibrated_models[0]['normalize_std']
            
            nets = [model['net'] for model in calibrated_models]
            ensemble_temperatures = [model['temperature'] for model in calibrated_models]
        else:
            # 단일 모델 모드
            nets = [calibrated_models[0]['net']]
            ensemble_temperatures = [calibrated_models[0]['temperature']]
            weights = [1.0]
            normalize_mean = calibrated_models[0]['normalize_mean']
            normalize_std = calibrated_models[0]['normalize_std']
        
        # TTA 상태 출력
        if tta == 2:
            print("TTA 모드: 원본 + Horizontal Flip")
        elif tta == 3:
            print("TTA 모드: 원본 + Horizontal Flip + 확대")
        print()
        
        # 테스트 데이터 로더 생성 (첫 번째 모델의 normalize 설정 사용)
        first_model_normalize = calibrated_models[0]['normalize_mean']
        first_model_use_cifar = False
        cifar_mean = [0.4914, 0.4822, 0.4465]
        cifar_std = [0.2023, 0.1994, 0.2010]
        if isinstance(first_model_normalize, tuple):
            first_model_normalize_list = list(first_model_normalize)
        else:
            first_model_normalize_list = first_model_normalize
        if len(first_model_normalize_list) == 3:
            mean_match = all(abs(a - b) < 1e-4 for a, b in zip(first_model_normalize_list, cifar_mean))
            std_match = all(abs(a - b) < 1e-4 for a, b in zip(calibrated_models[0]['normalize_std'], cifar_std))
            first_model_use_cifar = mean_match and std_match
        
        _, test_loader, _, _ = get_cifar10_loaders(
            batch_size=128,
            augment=False,
            autoaugment=False,
            cutout=False,
            use_cifar_normalize=use_cifar_normalize if use_cifar_normalize is not None else first_model_use_cifar,
            num_workers=2,
            collate_fn=None,
            data_root='./data'
        )
        
        # Criterion 정의
        nll_criterion = nn.CrossEntropyLoss().to(device)
        ece_criterion = ECELoss().to(device)
        
        # 캘리브레이션 전 테스트 성능 (단일 모델, TTA 없음)
        print("캘리브레이션 전 테스트 성능 계산 중...")
        test_logits_list = []
        test_labels_list = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                logits = nets[0](inputs)
                test_logits_list.append(logits)
                test_labels_list.append(labels)
        
        test_logits = torch.cat(test_logits_list).to(device)
        test_labels = torch.cat(test_labels_list).to(device)
        
        test_nll_before = nll_criterion(test_logits, test_labels).item()
        test_ece_before = ece_criterion(test_logits, test_labels).item()
        _, predicted_before = torch.max(test_logits, 1)
        test_acc_before = 100 * predicted_before.eq(test_labels).float().mean().item()
        
        print(f"\n캘리브레이션 전:")
        print(f"  정확도: {test_acc_before:.2f}%")
        print(f"  NLL: {test_nll_before:.4f}")
        print(f"  ECE: {test_ece_before:.4f}")
        
        # 캘리브레이션 후 테스트 성능 (TTA 및 Ensemble 적용)
        print("\n캘리브레이션 후 테스트 성능 계산 중...")
        correct = 0
        total = 0
        test_logits_list = []
        test_labels_list = []
        
        with torch.no_grad():
            pbar = tqdm(test_loader, desc='Inference')
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Ensemble: 여러 모델의 출력을 가중 평균
                ensemble_outputs = None
                
                for net, weight, temp in zip(nets, weights, ensemble_temperatures):
                    if tta == 2:
                        # TTA: 원본 + Horizontal Flip 평균
                        outputs_original = net(inputs)
                        outputs_flipped = net(torch.flip(inputs, [3]))  # Horizontal flip (dim=3)
                        outputs = (outputs_original + outputs_flipped) / 2.0
                    elif tta == 3:
                        # TTA: 원본 + Horizontal Flip + 확대 평균
                        outputs_original = net(inputs)
                        outputs_flipped = net(torch.flip(inputs, [3]))  # Horizontal flip (dim=3)
                        
                        # 이미지 약간 확대 (1.1배) 후 center crop
                        scale_factor = 1.1
                        h, w = inputs.shape[2], inputs.shape[3]
                        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                        inputs_upscaled = F.interpolate(inputs, size=(new_h, new_w), mode='bilinear', align_corners=False)
                        # Center crop
                        start_h = (new_h - h) // 2
                        start_w = (new_w - w) // 2
                        inputs_cropped = inputs_upscaled[:, :, start_h:start_h+h, start_w:start_w+w]
                        outputs_upscaled = net(inputs_cropped)
                        
                        outputs = (outputs_original + outputs_flipped + outputs_upscaled) / 3.0
                    else:
                        outputs = net(inputs)
                    
                    # Temperature Scaling 적용
                    if temp is not None:
                        outputs = outputs / temp
                    
                    # 가중치 적용하여 ensemble에 추가
                    if ensemble_outputs is None:
                        ensemble_outputs = outputs * weight
                    else:
                        ensemble_outputs += outputs * weight
                
                test_logits_list.append(ensemble_outputs)
                test_labels_list.append(labels)
                
                _, predicted = torch.max(ensemble_outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 진행률 표시줄에 현재 정확도 업데이트
                current_acc = 100 * correct / total
                pbar.set_postfix({'accuracy': f'{current_acc:.2f}%'})
        
        test_logits_after = torch.cat(test_logits_list).to(device)
        test_labels_after = torch.cat(test_labels_list).to(device)
        
        test_nll_after = nll_criterion(test_logits_after, test_labels_after).item()
        test_ece_after = ece_criterion(test_logits_after, test_labels_after).item()
        test_acc_after = 100 * correct / total
        
        print(f"\n캘리브레이션 후:")
        print(f"  정확도: {test_acc_after:.2f}%")
        print(f"  NLL: {test_nll_after:.4f}")
        print(f"  ECE: {test_ece_after:.4f}")
        
        # 상태 메시지 구성
        tta_status = " (TTA 적용)" if tta in [2, 3] else ""
        ensemble_status = f" (Ensemble: {len(nets)} 모델)" if is_ensemble else ""
        print(f"\n최종 테스트셋 정확도{ensemble_status}{tta_status}: {test_acc_after:.2f}%")
        print("="*50 + "\n")
        
        test_metrics = {
            'before': {
                'accuracy': test_acc_before,
                'nll': test_nll_before,
                'ece': test_ece_before
            },
            'after': {
                'accuracy': test_acc_after,
                'nll': test_nll_after,
                'ece': test_ece_after
            }
        }
        
        return optimal_temperature, calibrated_acc, metrics, test_metrics
    
    return optimal_temperature, calibrated_acc, metrics


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='모델 캘리브레이션')
    parser.add_argument('--model-path', type=str, nargs='+', default=None,
                        help='학습된 모델 파일 경로 (.pth, 지정하지 않으면 히스토리 파일에서 자동 추출).')
    parser.add_argument('--history-path', type=str, nargs='+', required=True,
                        help='히스토리 파일 경로 (.json).')
    parser.add_argument('--use-cifar-normalize', action='store_true',
                        help='CIFAR-10 표준 Normalize 값 사용 (지정하지 않으면 히스토리에서 자동 추출)')
    parser.add_argument('--device', type=str, default=None,
                        help='사용할 디바이스 (cuda/cpu, default: 자동 선택)')
    parser.add_argument('--save-calibration', action='store_true',
                        help='Temperature를 파일에 저장 (히스토리 및 별도 JSON 파일)')
    parser.add_argument('--run-inference', action='store_true',
                        help='캘리브레이션 후 테스트셋에서 추론 수행')
    parser.add_argument('--cross-validate', type=str, default='ece', choices=['ece', 'nll'],
                        help='Temperature 최적화 기준 (ece 또는 nll, default: ece)')
    parser.add_argument('--tta', type=int, default=0, choices=[0, 2, 3],
                        help='Test Time Augmentation 활성화 (0: 비활성화, 2: 원본+Horizontal Flip, 3: 원본+Horizontal Flip+확대)')
    parser.add_argument('--ensemble-history', type=str, nargs='+', default=None,
                        help='Ensemble용 History 파일 경로들 (--history-path에 여러 개 지정 시 자동으로 사용되므로 선택사항)')
    parser.add_argument('--ensemble-weights', type=float, nargs='+', default=None,
                        help='Ensemble 가중치 (지정하지 않으면 균등 가중치 사용)')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 디바이스 설정
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # history_path와 model_path 처리
    # 리스트로 변환 (단일 값이어도 리스트로 처리)
    history_path = args.history_path if isinstance(args.history_path, list) else [args.history_path]
    model_path = args.model_path if args.model_path is not None else None
    
    # 캘리브레이션 수행
    result = calibrate_model(
        model_path=model_path,
        history_path=history_path,
        device=device,
        use_cifar_normalize=args.use_cifar_normalize if args.use_cifar_normalize else None,
        save_calibration=args.save_calibration,
        run_inference=args.run_inference,
        cross_validate=args.cross_validate,
        tta=args.tta,
        ensemble_history=args.ensemble_history,
        ensemble_weights=args.ensemble_weights
    )
    
    print("\n" + "="*50)
    print("캘리브레이션 완료!")
    print("="*50)
    
    if args.run_inference:
        optimal_temperature, calibrated_acc, cal_metrics, test_metrics = result
        print(f"\n최적 Temperature: {optimal_temperature:.4f} (기준: {args.cross_validate.upper()})")
        print(f"  - NLL 기준 최적값: {cal_metrics['T_opt_nll']:.4f}")
        print(f"  - ECE 기준 최적값: {cal_metrics['T_opt_ece']:.4f}")
        
        print(f"\n검증셋 정확도: {calibrated_acc:.2f}%")
        
        print(f"\n테스트셋 결과:")
        print(f"  캘리브레이션 전 - 정확도: {test_metrics['before']['accuracy']:.2f}%, "
              f"NLL: {test_metrics['before']['nll']:.4f}, ECE: {test_metrics['before']['ece']:.4f}")
        print(f"  캘리브레이션 후 - 정확도: {test_metrics['after']['accuracy']:.2f}%, "
              f"NLL: {test_metrics['after']['nll']:.4f}, ECE: {test_metrics['after']['ece']:.4f}")
    else:
        optimal_temperature, calibrated_acc, cal_metrics = result
        print(f"\n최적 Temperature: {optimal_temperature:.4f} (기준: {args.cross_validate.upper()})")
        print(f"  - NLL 기준 최적값: {cal_metrics['T_opt_nll']:.4f}")
        print(f"  - ECE 기준 최적값: {cal_metrics['T_opt_ece']:.4f}")
        print(f"\n검증셋 정확도: {calibrated_acc:.2f}%")


if __name__ == '__main__':
    main()

