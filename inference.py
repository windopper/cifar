import argparse
import torch
import torch.nn.functional as F
import json
import os
from torchvision.transforms import transforms
import torchvision
from tqdm import tqdm
from main import CLASS_NAMES, get_net

def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='CIFAR-10 Inference')
    parser.add_argument('--history', type=str, default=None,
                        help='History 파일 경로 (지정하면 모델 경로와 모델 이름을 자동으로 추출)')
    parser.add_argument('--ensemble-history', type=str, nargs='+', default=None,
                        help='Ensemble용 History 파일 경로들 (여러 모델을 결합하여 추론)')
    parser.add_argument('--path', '-p', type=str, default=None,
                        help='모델 파일 경로 (--history가 지정되지 않은 경우에만 사용)')
    parser.add_argument('--model', '-m', type=str, default=None,
                        choices=['baseline', 'deep_baseline', 'deep_baseline_silu', 'deep_baseline_bn', 
                                'deep_baseline_gap', 'deep_baseline_bn_dropout', 'deep_baseline_bn_dropout_resnet',
                                'deep_baseline_se', 'resnet18', 'vgg16', 'mobilenetv2', 'densenet121',
                                'deep_baseline2_bn', 'deep_baseline2_bn_residual', 'deep_baseline2_bn_residual_preact',
                                'deep_baseline3_bn', 'deep_baseline2_bn_resnext', 'deep_baseline2_bn_residual_se',
                                'mxresnet20', 'mxresnet32', 'mxresnet44', 'mxresnet56', 'dla', 'resnext29_4x64d'],
                        help='네트워크 모델 (--history가 지정되지 않은 경우에만 사용)')
    parser.add_argument('--batch-size', type=int, default=4, 
                        help='배치 크기 (default: 4)')
    parser.add_argument('--temperature', '-t', type=str, default=None,
                        help='Temperature 파일 경로 (지정하지 않으면 모델 경로에서 자동 검색)')
    parser.add_argument('--no-calibrate', action='store_true',
                        help='캘리브레이션 사용 안 함 (temperature 파일이 있어도 무시)')
    parser.add_argument('--tta', type=int, default=0, choices=[0, 2, 3],
                        help='Test Time Augmentation 활성화 (0: 비활성화, 2: 원본+Horizontal Flip, 3: 원본+Horizontal Flip+확대)')
    parser.add_argument('--ensemble-weights', type=float, nargs='+', default=None,
                        help='Ensemble 가중치 (지정하지 않으면 균등 가중치 사용)')
    return parser.parse_args()

def load_model_from_history(history_path):
    """History 파일에서 모델 정보를 추출하고 모델을 로드"""
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
    normalize_mean = (0.5, 0.5, 0.5)
    normalize_std = (0.5, 0.5, 0.5)
    if 'hyperparameters' in history_data:
        hp = history_data['hyperparameters']
        if 'normalize_mean' in hp and 'normalize_std' in hp:
            normalize_mean = tuple(hp['normalize_mean'])
            normalize_std = tuple(hp['normalize_std'])
    
    return model_name, model_path, normalize_mean, normalize_std

def load_temperature(model_path, no_calibrate=False, temperature_path=None):
    """Temperature 파일 로드"""
    if no_calibrate:
        return None
    
    if temperature_path:
        temp_path = temperature_path
    else:
        base_path = model_path.rsplit('.pth', 1)[0]
        temp_path = f"{base_path}_temperature.json"
    
    if os.path.exists(temp_path):
        with open(temp_path, 'r') as f:
            temp_data = json.load(f)
            return temp_data.get('temperature')
    return None

if __name__ == '__main__':
    args = parse_args()
    
    # Ensemble 모드인지 확인
    is_ensemble = args.ensemble_history is not None and len(args.ensemble_history) > 0
    
    if is_ensemble:
        # Ensemble 모드: 여러 모델 로드
        print("=" * 60)
        print("Ensemble 모드 활성화")
        print("=" * 60)
        
        models_info = []
        temperatures = []
        
        # Ensemble 가중치 확인
        if args.ensemble_weights:
            if len(args.ensemble_weights) != len(args.ensemble_history):
                raise ValueError(f"Ensemble 가중치 개수({len(args.ensemble_weights)})가 모델 개수({len(args.ensemble_history)})와 일치하지 않습니다.")
            weights = [w / sum(args.ensemble_weights) for w in args.ensemble_weights]  # 정규화
        else:
            weights = [1.0 / len(args.ensemble_history)] * len(args.ensemble_history)  # 균등 가중치
        
        print(f"\n총 {len(args.ensemble_history)}개의 모델을 로드합니다:\n")
        
        for i, history_path in enumerate(args.ensemble_history):
            model_name, model_path, normalize_mean, normalize_std = load_model_from_history(history_path)
            temperature = load_temperature(model_path, args.no_calibrate)
            
            print(f"[모델 {i+1}/{len(args.ensemble_history)}]")
            print(f"  History 파일: {history_path}")
            print(f"  모델 이름: {model_name}")
            print(f"  모델 경로: {model_path}")
            print(f"  가중치: {weights[i]:.4f}")
            if temperature:
                print(f"  Temperature: {temperature:.4f}")
            print(f"  Normalize: mean={normalize_mean}, std={normalize_std}")
            print()
            
            models_info.append({
                'name': model_name,
                'path': model_path,
                'normalize_mean': normalize_mean,
                'normalize_std': normalize_std,
                'weight': weights[i]
            })
            temperatures.append(temperature)
        
        # Normalize 값이 모두 동일한지 확인 (다르면 경고)
        normalize_means = [info['normalize_mean'] for info in models_info]
        normalize_stds = [info['normalize_std'] for info in models_info]
        if len(set(normalize_means)) > 1 or len(set(normalize_stds)) > 1:
            print("경고: 모델들의 Normalize 값이 다릅니다. 첫 번째 모델의 값을 사용합니다.")
        
        normalize_mean = models_info[0]['normalize_mean']
        normalize_std = models_info[0]['normalize_std']
        
    else:
        # 단일 모델 모드
        model_path = args.path
        model_name = args.model
        normalize_mean = (0.5, 0.5, 0.5)
        normalize_std = (0.5, 0.5, 0.5)
        
        if args.history:
            model_name, model_path, normalize_mean, normalize_std = load_model_from_history(args.history)
            print(f"History 파일에서 모델 이름 추출: {model_name}")
            print(f"모델 파일 경로: {model_path}")
            print(f"Normalize 값 사용: mean={normalize_mean}, std={normalize_std}")
        else:
            # History 파일이 없으면 기존 방식 사용
            if model_path is None:
                model_path = 'outputs/baseline_baseline_sgd_crossentropy.pth'
            if model_name is None:
                model_name = 'baseline'
        
        models_info = [{
            'name': model_name,
            'path': model_path,
            'normalize_mean': normalize_mean,
            'normalize_std': normalize_std,
            'weight': 1.0
        }]
        temperatures = [load_temperature(model_path, args.no_calibrate, args.temperature)]
        weights = [1.0]  # 단일 모델 모드
    
    # CUDA 호환성 문제로 인해 CPU 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(normalize_mean, normalize_std)]
    )

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 모든 모델 로드
    nets = []
    for info in models_info:
        net = get_net(info['name'])
        net.load_state_dict(torch.load(info['path'], map_location=device))
        net.to(device)
        net.eval()
        nets.append(net)
    
    if is_ensemble:
        print(f"\n{len(nets)}개의 모델이 모두 로드되었습니다.\n")
    else:
        net = nets[0]  # 단일 모델 모드에서 호환성을 위해
    
    # Temperature 정보 출력
    if not args.no_calibrate:
        if is_ensemble:
            calibrated_models = [i+1 for i, t in enumerate(temperatures) if t is not None]
            if calibrated_models:
                print(f"캘리브레이션 적용된 모델: {calibrated_models}")
            else:
                print("캘리브레이션 없이 추론합니다")
        else:
            if temperatures[0] is not None:
                print(f"Temperature 파일 로드: {models_info[0]['path'].rsplit('.pth', 1)[0]}_temperature.json")
                print(f"Temperature 값: {temperatures[0]:.4f}")
                print("캘리브레이션 적용됨")
            else:
                print("캘리브레이션 없이 추론합니다")
    else:
        print("캘리브레이션 비활성화됨")
    print()

    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in CLASS_NAMES}
    total_pred = {classname: 0 for classname in CLASS_NAMES}

    print("추론 시작...")
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Inference')
        for data in pbar:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            # Ensemble: 여러 모델의 출력을 가중 평균
            ensemble_outputs = None
            
            for i, (net, weight, temp) in enumerate(zip(nets, weights, temperatures)):
                if args.tta == 2:
                    # TTA: 원본 + Horizontal Flip 평균
                    outputs_original = net(images)
                    outputs_flipped = net(torch.flip(images, [3]))  # Horizontal flip (dim=3)
                    outputs = (outputs_original + outputs_flipped) / 2.0
                elif args.tta == 3:
                    # TTA: 원본 + Horizontal Flip + 확대 평균
                    outputs_original = net(images)
                    outputs_flipped = net(torch.flip(images, [3]))  # Horizontal flip (dim=3)
                    
                    # 이미지 약간 확대 (1.1배) 후 center crop
                    scale_factor = 1.1
                    h, w = images.shape[2], images.shape[3]
                    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                    images_upscaled = F.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    # Center crop
                    start_h = (new_h - h) // 2
                    start_w = (new_w - w) // 2
                    images_cropped = images_upscaled[:, :, start_h:start_h+h, start_w:start_w+w]
                    outputs_upscaled = net(images_cropped)
                    
                    outputs = (outputs_original + outputs_flipped + outputs_upscaled) / 3.0
                else:
                    outputs = net(images)
                
                # Temperature Scaling 적용
                if temp is not None:
                    outputs = outputs / temp
                
                # 가중치 적용하여 ensemble에 추가
                if ensemble_outputs is None:
                    ensemble_outputs = outputs * weight
                else:
                    ensemble_outputs += outputs * weight
            
            outputs = ensemble_outputs
            _, predicted = torch.max(outputs.data, 1)
            
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[CLASS_NAMES[label]] += 1
                total_pred[CLASS_NAMES[label]] += 1
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 진행률 표시줄에 현재 정확도 업데이트
            current_acc = 100 * correct / total
            pbar.set_postfix({'accuracy': f'{current_acc:.2f}%'})

    # 상태 메시지 구성
    calibration_status = ""
    if not args.no_calibrate:
        if is_ensemble:
            calibrated_count = sum(1 for t in temperatures if t is not None)
            if calibrated_count > 0:
                calibration_status = f" (캘리브레이션 적용: {calibrated_count}/{len(temperatures)} 모델)"
        else:
            if temperatures[0] is not None:
                calibration_status = " (캘리브레이션 적용)"
    
    tta_status = " (TTA 적용)" if args.tta in [2, 3] else ""
    ensemble_status = f" (Ensemble: {len(nets)} 모델)" if is_ensemble else ""
    
    print(f'Accuracy of the network on the 10000 test images{ensemble_status}{calibration_status}{tta_status}: {100 * correct / total:.2f}%')

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')