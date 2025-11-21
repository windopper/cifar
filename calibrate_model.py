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
from calibration import calibrate_temperature, ECELoss
from utils.dataset import get_cifar10_loaders
from utils.history import update_history_calibration, save_history, load_history


def get_net_from_history(history_path):
    """히스토리 파일에서 네트워크 정보 추출"""
    history = load_history(history_path)
    if history is None:
        return None
    return history.get('hyperparameters', {}).get('net')


def calibrate_model(model_path, history_path, device=None, use_cifar_normalize=None, 
                   save_calibration=False, run_inference=False, cross_validate='ece'):
    """
    모델 캘리브레이션 수행
    
    Args:
        model_path: 모델 파일 경로 (.pth), None이면 히스토리 파일에서 자동 추출
        history_path: 히스토리 파일 경로 (.json)
        device: 디바이스 (None이면 자동 선택)
        use_cifar_normalize: CIFAR-10 표준 normalize 사용 여부 (None이면 히스토리에서 자동 추출)
        save_calibration: Temperature를 파일에 저장할지 여부 (기본값: False)
        run_inference: 캘리브레이션 후 테스트셋에서 추론 수행 여부 (기본값: False)
        cross_validate: 'ece' 또는 'nll' - 최적화할 metric (기본값: 'ece')
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # 히스토리 로드
    history = load_history(history_path)
    if history is None:
        raise FileNotFoundError(f"히스토리 파일을 찾을 수 없습니다: {history_path}")
    
    # 네트워크 정보 가져오기
    net_name = get_net_from_history(history_path)
    if net_name is None:
        raise ValueError("히스토리에서 네트워크 정보를 찾을 수 없습니다.")
    
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
    if use_cifar_normalize is None:
        # 히스토리에서 normalize 정보 확인
        hp = history.get('hyperparameters', {})
        # CIFAR-10 표준 normalize 값
        cifar_mean = [0.4914, 0.4822, 0.4465]
        cifar_std = [0.2023, 0.1994, 0.2010]
        
        if 'normalize_mean' in hp and 'normalize_std' in hp:
            # 히스토리의 normalize 값과 CIFAR-10 표준 값 비교
            hist_mean = hp['normalize_mean']
            hist_std = hp['normalize_std']
            # 리스트 비교 (소수점 오차 고려)
            mean_match = all(abs(a - b) < 1e-4 for a, b in zip(hist_mean, cifar_mean))
            std_match = all(abs(a - b) < 1e-4 for a, b in zip(hist_std, cifar_std))
            use_cifar_normalize = mean_match and std_match
        else:
            # 히스토리에 정보가 없으면 기본값 사용
            use_cifar_normalize = False
    
    # 모델 로드
    from main import get_net
    net = get_net(net_name, init_weights=False, shakedrop_prob=0.0)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net = net.to(device)
    net.eval()
    
    print(f"모델 로드 완료: {net_name}")
    print(f"모델 경로: {model_path}")
    print(f"Normalize 설정: {'CIFAR-10 표준' if use_cifar_normalize else '기본값'}")
    
    # 검증 데이터 로더 생성
    _, val_loader, _, _ = get_cifar10_loaders(
        batch_size=128,
        augment=False,
        autoaugment=False,
        cutout=False,
        use_cifar_normalize=use_cifar_normalize,
        num_workers=2,
        collate_fn=None,
        data_root='./data'
    )
    
    # Temperature Scaling 캘리브레이션 수행
    print("\n" + "="*50)
    print("Temperature Scaling 캘리브레이션 시작...")
    print(f"최적화 기준: {cross_validate.upper()}")
    print("="*50)

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
    print(f"캘리브레이션 전 검증 정확도: {before_acc:.2f}%")

    optimal_temperature, metrics = calibrate_temperature(
        net, val_loader, device, cross_validate=cross_validate, log=True
    )

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
    print(f"\n캘리브레이션 후 검증 정확도: {calibrated_acc:.2f}%")
    print(f"정확도 변화: {calibrated_acc - before_acc:+.2f}%")
    print("="*50 + "\n")
    
    # 저장 옵션이 활성화된 경우에만 저장
    if save_calibration:
        # 히스토리에 temperature와 calibration metrics 저장
        update_history_calibration(history, optimal_temperature, calibrated_acc)
        
        # calibration metrics도 저장
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
        
        # 히스토리 파일 업데이트
        save_history(history, history_path)
        print(f"히스토리 업데이트 완료: {history_path}")
        
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
        print(f"Temperature 저장 완료: {temp_path}")
    else:
        print("Temperature 저장 생략 (--save-calibration 플래그를 사용하면 저장됩니다)")
    
    # 테스트셋에서 추론 수행
    if run_inference:
        print("\n" + "="*50)
        print("테스트셋에서 추론 수행 중...")
        print("="*50)
        
        # 테스트 데이터 로더 생성 (val_loader가 test set입니다)
        _, test_loader, _, _ = get_cifar10_loaders(
            batch_size=128,
            augment=False,
            autoaugment=False,
            cutout=False,
            use_cifar_normalize=use_cifar_normalize,
            num_workers=2,
            collate_fn=None,
            data_root='./data'
        )
        
        # 모든 테스트 로짓과 레이블 수집
        test_logits_list = []
        test_labels_list = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                logits = net(inputs)
                test_logits_list.append(logits)
                test_labels_list.append(labels)
        
        test_logits = torch.cat(test_logits_list).to(device)
        test_labels = torch.cat(test_labels_list).to(device)
        
        # Criterion 정의
        nll_criterion = nn.CrossEntropyLoss().to(device)
        ece_criterion = ECELoss().to(device)
        
        # 캘리브레이션 전 테스트 성능
        with torch.no_grad():
            test_nll_before = nll_criterion(test_logits, test_labels).item()
            test_ece_before = ece_criterion(test_logits, test_labels).item()
            _, predicted_before = torch.max(test_logits, 1)
            test_acc_before = 100 * predicted_before.eq(test_labels).float().mean().item()
        
        print(f"\n캘리브레이션 전:")
        print(f"  정확도: {test_acc_before:.2f}%")
        print(f"  NLL: {test_nll_before:.4f}")
        print(f"  ECE: {test_ece_before:.4f}")
        
        # 캘리브레이션 후 테스트 성능
        with torch.no_grad():
            scaled_test_logits = test_logits / optimal_temperature
            test_nll_after = nll_criterion(scaled_test_logits, test_labels).item()
            test_ece_after = ece_criterion(scaled_test_logits, test_labels).item()
            _, predicted_after = torch.max(scaled_test_logits, 1)
            test_acc_after = 100 * predicted_after.eq(test_labels).float().mean().item()
        
        print(f"\n캘리브레이션 후:")
        print(f"  정확도: {test_acc_after:.2f}%")
        print(f"  NLL: {test_nll_after:.4f}")
        print(f"  ECE: {test_ece_after:.4f}")
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
    parser.add_argument('--model-path', type=str, default=None,
                        help='학습된 모델 파일 경로 (.pth, 지정하지 않으면 히스토리 파일에서 자동 추출)')
    parser.add_argument('--history-path', type=str, required=True,
                        help='히스토리 파일 경로 (.json)')
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
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 디바이스 설정
    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    # 캘리브레이션 수행
    result = calibrate_model(
        model_path=args.model_path if args.model_path else None,
        history_path=args.history_path,
        device=device,
        use_cifar_normalize=args.use_cifar_normalize if args.use_cifar_normalize else None,
        save_calibration=args.save_calibration,
        run_inference=args.run_inference,
        cross_validate=args.cross_validate
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

