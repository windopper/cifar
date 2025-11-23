import argparse
import torch
import torch.nn.functional as F
import json
import os
import numpy as np
from torchvision.transforms import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
import torchvision
from tqdm import tqdm
from train import CLASS_NAMES
from utils.calibration import calibrate_temperature
from utils.net import get_available_nets, get_net
from utils.visualize_loss_samples import visualize_top_loss_samples

def parse_args():
    parser = argparse.ArgumentParser(description='CIFAR-10 Inference')
    parser.add_argument('--history', type=str, default=None,
                        help='History 파일 경로 (지정하면 모델 경로와 모델 이름을 자동으로 추출)')
    parser.add_argument('--ensemble-history', type=str, nargs='+', default=None,
                        help='Ensemble용 History 파일 경로들 (여러 모델을 결합하여 추론)')
    parser.add_argument('--path', '-p', type=str, default=None,
                        help='모델 파일 경로 (--history가 지정되지 않은 경우에만 사용)')
    parser.add_argument('--model', '-m', type=str, default=None,
                        choices=get_available_nets(),
                        help='네트워크 모델 (--history가 지정되지 않은 경우에만 사용)')
    parser.add_argument('--batch-size', type=int, default=128, 
                        help='배치 크기 (default: 128)')
    parser.add_argument('--temperature', '-t', type=str, default=None,
                        help='Temperature 파일 경로 (지정하지 않으면 모델 경로에서 자동 검색)')
    parser.add_argument('--no-calibrate', action='store_true',
                        help='캘리브레이션 사용 안 함 (temperature 파일이 있어도 무시)')
    parser.add_argument('--auto-calibrate', action='store_true',
                        help='각 모델에 대해 자동으로 캘리브레이션 수행 (temperature 파일이 있어도 재계산)')
    parser.add_argument('--cross-validate', type=str, default='ece', choices=['ece', 'nll'],
                        help='Auto-calibrate 시 Temperature 최적화 기준 (ece 또는 nll, default: ece)')
    parser.add_argument('--tta', type=int, default=0, choices=[0, 2, 3, 5],
                        help='Test Time Augmentation 활성화 (0: 비활성화, 2: 원본+Horizontal Flip, 3: 원본+Horizontal Flip+확대, 5: AutoAugment CIFAR-10 정책 5회)')
    parser.add_argument('--ensemble-weights', type=float, nargs='+', default=None,
                        help='Ensemble 가중치 (지정하지 않으면 균등 가중치 사용)')
    parser.add_argument('--voting', type=str, default='soft', choices=['soft', 'hard'],
                        help='앙상블 voting 방식 (soft: 확률 가중 평균, hard: 예측 클래스 투표, default: soft)')
    parser.add_argument('--confusion-matrix', action='store_true',
                        help='Confusion matrix 출력')
    parser.add_argument('--top-loss-samples', type=int, default=0,
                        help='Top loss samples 시각화 개수 (모델이 강하게 확신했으나 틀린 예측, default: 0)')
    parser.add_argument('--shakedrop-prob', type=float, default=0.0,
                        help='ShakeDrop 확률 (--path 옵션 사용 시 필요, default: 0.0)')
    return parser.parse_args()

def load_model_from_history(history_path):
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"History 파일을 찾을 수 없습니다: {history_path}")
    
    with open(history_path, 'r') as f:
        history_data = json.load(f)
    
    if 'hyperparameters' in history_data and 'net' in history_data['hyperparameters']:
        model_name = history_data['hyperparameters']['net']
    else:
        raise ValueError(f"History 파일에 'hyperparameters.net' 정보가 없습니다: {history_path}")
    
    if history_path.endswith('_history.json'):
        model_path = history_path.replace('_history.json', '.pth')
    else:
        base_path = history_path.rsplit('.json', 1)[0]
        model_path = f"{base_path.rsplit('_history', 1)[0]}.pth"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    normalize_mean = (0.5, 0.5, 0.5)
    normalize_std = (0.5, 0.5, 0.5)
    shakedrop_prob = 0.0
    
    if 'hyperparameters' in history_data:
        hp = history_data['hyperparameters']
        if 'normalize_mean' in hp and 'normalize_std' in hp:
            normalize_mean = tuple(hp['normalize_mean'])
            normalize_std = tuple(hp['normalize_std'])
        if 'shakedrop_prob' in hp and hp['shakedrop_prob'] is not None:
            shakedrop_prob = hp['shakedrop_prob']
    
    return model_name, model_path, normalize_mean, normalize_std, shakedrop_prob

def load_temperature(model_path, no_calibrate=False, temperature_path=None):
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

def auto_calibrate_model(net, normalize_mean, normalize_std, device, cross_validate='ece'):
    if isinstance(normalize_mean, (list, tuple)):
        normalize_mean_tuple = tuple(normalize_mean) if len(normalize_mean) == 3 else (0.5, 0.5, 0.5)
    else:
        normalize_mean_tuple = (0.5, 0.5, 0.5)
    
    if isinstance(normalize_std, (list, tuple)):
        normalize_std_tuple = tuple(normalize_std) if len(normalize_std) == 3 else (0.5, 0.5, 0.5)
    else:
        normalize_std_tuple = (0.5, 0.5, 0.5)
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean_tuple, normalize_std_tuple)
    ])
    
    val_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=val_transform
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=128, shuffle=False, num_workers=2
    )
    
    optimal_temperature, _ = calibrate_temperature(
        net, val_loader, device, cross_validate=cross_validate, log=True
    )
    
    return optimal_temperature

def print_confusion_matrix(confusion_matrix, class_names):
    num_classes = len(class_names)
    
    print("\n" + "=" * 80)
    print("Confusion Matrix")
    print("=" * 80)
    
    header = "실제 \\ 예측"
    for name in class_names:
        header += f"{name:>8}"
    header += "  Total"
    print(header)
    print("-" * 80)
    
    for i, class_name in enumerate(class_names):
        row_str = f"{class_name:>10}"
        row_total = 0
        for j in range(num_classes):
            count = confusion_matrix[i, j]
            row_str += f"{count:>8}"
            row_total += count
        row_str += f"  {row_total:>5}"
        print(row_str)
    
    print("-" * 80)
    col_sum_str = "     Total"
    for j in range(num_classes):
        col_total = confusion_matrix[:, j].sum()
        col_sum_str += f"{col_total:>8}"
    col_sum_str += f"  {confusion_matrix.sum():>5}"
    print(col_sum_str)
    
    print("\n" + "-" * 80)
    print("클래스별 성능 지표")
    print("-" * 80)
    print(f"{'클래스':>10} {'정확도':>10} {'정밀도':>10} {'재현율':>10} {'F1-Score':>10}")
    print("-" * 80)
    
    for i, class_name in enumerate(class_names):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        tn = confusion_matrix.sum() - tp - fp - fn
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{class_name:>10} {accuracy*100:>9.2f}% {precision*100:>9.2f}% {recall*100:>9.2f}% {f1_score*100:>9.2f}%")
    
    print("=" * 80 + "\n")

def apply_tta(net, images, tta_mode, normalize_mean, normalize_std, device):
    if tta_mode == 2:
        outputs_original = net(images)
        outputs_flipped = net(torch.flip(images, [3]))
        return (outputs_original + outputs_flipped) / 2.0
    elif tta_mode == 3:
        outputs_original = net(images)
        outputs_flipped = net(torch.flip(images, [3]))
        
        scale_factor = 1.1
        h, w = images.shape[2], images.shape[3]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        images_upscaled = F.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        images_cropped = images_upscaled[:, :, start_h:start_h+h, start_w:start_w+w]
        outputs_upscaled = net(images_cropped)
        
        return (outputs_original + outputs_flipped + outputs_upscaled) / 3.0
    elif tta_mode == 5:
        outputs_list = [net(images)]
        
        mean_tensor = torch.tensor(normalize_mean, device=device).view(1, 3, 1, 1)
        std_tensor = torch.tensor(normalize_std, device=device).view(1, 3, 1, 1)
        images_denorm = images * std_tensor + mean_tensor
        images_denorm = torch.clamp(images_denorm, 0.0, 1.0)
        
        to_pil = transforms.ToPILImage()
        autoaugment = AutoAugment(policy=AutoAugmentPolicy.CIFAR10)
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(normalize_mean, normalize_std)
        
        for _ in range(4):
            augmented_images = []
            for img in images_denorm:
                img_pil = to_pil(img.cpu())
                img_aug = autoaugment(img_pil)
                img_tensor = to_tensor(img_aug).to(device)
                img_normalized = normalize(img_tensor)
                augmented_images.append(img_normalized)
            augmented_batch = torch.stack(augmented_images)
            outputs_list.append(net(augmented_batch))
        
        return sum(outputs_list) / len(outputs_list)
    else:
        return net(images)

def load_single_model(args, device):
    model_path = args.path
    model_name = args.model
    normalize_mean = (0.5, 0.5, 0.5)
    normalize_std = (0.5, 0.5, 0.5)
    shakedrop_prob = 0.0
    
    if args.history:
        model_name, model_path, normalize_mean, normalize_std, shakedrop_prob = load_model_from_history(args.history)
        print(f"History 파일에서 모델 이름 추출: {model_name}")
        print(f"모델 파일 경로: {model_path}")
        print(f"Normalize 값 사용: mean={normalize_mean}, std={normalize_std}")
        print(f"ShakeDrop 확률: {shakedrop_prob}")
    else:
        if model_path is None:
            model_path = 'outputs/baseline_baseline_sgd_crossentropy.pth'
        if model_name is None:
            model_name = 'baseline'
        shakedrop_prob = args.shakedrop_prob
        if shakedrop_prob > 0.0:
            print(f"ShakeDrop 확률: {shakedrop_prob} (커맨드라인 인자에서 지정)")
    
    net = get_net(model_name, shakedrop_prob=shakedrop_prob)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()
    
    if args.auto_calibrate:
        print(f"자동 캘리브레이션 수행 중... (기준: {args.cross_validate.upper()})")
        temperature = auto_calibrate_model(net, normalize_mean, normalize_std, device, args.cross_validate)
        print(f"계산된 Temperature: {temperature:.4f}")
    else:
        temperature = load_temperature(model_path, args.no_calibrate, args.temperature)
        if temperature:
            print(f"Temperature 파일 로드: {temperature:.4f}")
    
    return {
        'name': model_name,
        'path': model_path,
        'normalize_mean': normalize_mean,
        'normalize_std': normalize_std,
        'shakedrop_prob': shakedrop_prob,
        'weight': 1.0,
        'net': net
    }, [temperature], [1.0]

def load_ensemble_models(args, device):
    print("=" * 60)
    print("Ensemble 모드 활성화")
    print(f"Voting 방식: {args.voting.upper()}")
    print("=" * 60)
    
    models_info = []
    temperatures = []
    
    if args.ensemble_weights:
        if len(args.ensemble_weights) != len(args.ensemble_history):
            raise ValueError(f"Ensemble 가중치 개수({len(args.ensemble_weights)})가 모델 개수({len(args.ensemble_history)})와 일치하지 않습니다.")
        weights = [w / sum(args.ensemble_weights) for w in args.ensemble_weights]
    else:
        weights = [1.0 / len(args.ensemble_history)] * len(args.ensemble_history)
    
    print(f"\n총 {len(args.ensemble_history)}개의 모델을 로드합니다:\n")
    
    for i, history_path in enumerate(args.ensemble_history):
        model_name, model_path, normalize_mean, normalize_std, shakedrop_prob = load_model_from_history(history_path)
        
        print(f"[모델 {i+1}/{len(args.ensemble_history)}]")
        print(f"  History 파일: {history_path}")
        print(f"  모델 이름: {model_name}")
        print(f"  모델 경로: {model_path}")
        print(f"  가중치: {weights[i]:.4f}")
        print(f"  Normalize: mean={normalize_mean}, std={normalize_std}")
        print(f"  ShakeDrop 확률: {shakedrop_prob}")
        
        net = get_net(model_name, shakedrop_prob=shakedrop_prob)
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.to(device)
        net.eval()
        
        if args.auto_calibrate:
            print(f"  자동 캘리브레이션 수행 중... (기준: {args.cross_validate.upper()})")
            temperature = auto_calibrate_model(net, normalize_mean, normalize_std, device, args.cross_validate)
            print(f"  계산된 Temperature: {temperature:.4f}")
        else:
            temperature = load_temperature(model_path, args.no_calibrate)
            if temperature:
                print(f"  Temperature: {temperature:.4f} (파일에서 로드)")
            else:
                print(f"  Temperature: 없음")
        
        print()
        
        models_info.append({
            'name': model_name,
            'path': model_path,
            'normalize_mean': normalize_mean,
            'normalize_std': normalize_std,
            'shakedrop_prob': shakedrop_prob,
            'weight': weights[i],
            'net': net
        })
        temperatures.append(temperature)
    
    normalize_means = [info['normalize_mean'] for info in models_info]
    normalize_stds = [info['normalize_std'] for info in models_info]
    if len(set(normalize_means)) > 1 or len(set(normalize_stds)) > 1:
        print("경고: 모델들의 Normalize 값이 다릅니다. 첫 번째 모델의 값을 사용합니다.")
    
    normalize_mean = models_info[0]['normalize_mean']
    normalize_std = models_info[0]['normalize_std']
    
    return models_info, temperatures, weights, normalize_mean, normalize_std

def infer_batch(images, nets, weights, temperatures, is_ensemble, voting, tta_mode, normalize_mean, normalize_std, device):
    num_classes = len(CLASS_NAMES)
    
    if is_ensemble and voting == 'hard':
        batch_size = images.size(0)
        vote_counts = torch.zeros(batch_size, num_classes, device=device)
        
        for net, weight, temp in zip(nets, weights, temperatures):
            outputs = apply_tta(net, images, tta_mode, normalize_mean, normalize_std, device)
            
            if temp is not None:
                outputs = outputs / temp
            
            _, predicted_class = torch.max(outputs.data, 1)
            
            for j in range(batch_size):
                vote_counts[j, predicted_class[j]] += weight
        
        _, predicted = torch.max(vote_counts, 1)
        outputs = vote_counts
    else:
        ensemble_outputs = None
        
        for net, weight, temp in zip(nets, weights, temperatures):
            outputs = apply_tta(net, images, tta_mode, normalize_mean, normalize_std, device)
            
            if temp is not None:
                outputs = outputs / temp
            
            if ensemble_outputs is None:
                ensemble_outputs = outputs * weight
            else:
                ensemble_outputs += outputs * weight
        
        outputs = ensemble_outputs
        _, predicted = torch.max(outputs.data, 1)
    
    return predicted, outputs

if __name__ == '__main__':
    args = parse_args()
    
    if args.auto_calibrate and args.no_calibrate:
        raise ValueError("--auto-calibrate와 --no-calibrate를 동시에 지정할 수 없습니다.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    is_ensemble = args.ensemble_history is not None and len(args.ensemble_history) > 0
    
    if not is_ensemble and args.voting != 'soft':
        print(f"경고: --voting 옵션은 앙상블 모드(--ensemble-history)에서만 적용됩니다. 현재는 무시됩니다.")
    
    if is_ensemble:
        models_info, temperatures, weights, normalize_mean, normalize_std = load_ensemble_models(args, device)
    else:
        model_info, temperatures, weights = load_single_model(args, device)
        models_info = [model_info]
        normalize_mean = model_info['normalize_mean']
        normalize_std = model_info['normalize_std']
    
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(normalize_mean, normalize_std)]
    )
    
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    nets = []
    for info in models_info:
        if 'net' in info:
            nets.append(info['net'])
        else:
            shakedrop_prob = info.get('shakedrop_prob', 0.0)
            net = get_net(info['name'], shakedrop_prob=shakedrop_prob)
            net.load_state_dict(torch.load(info['path'], map_location=device))
            net.to(device)
            net.eval()
            nets.append(net)
    
    if is_ensemble:
        print(f"\n{len(nets)}개의 모델이 모두 로드되었습니다.\n")
    else:
        net = nets[0]
    
    if args.auto_calibrate:
        if is_ensemble:
            print(f"자동 캘리브레이션 완료: {len([t for t in temperatures if t is not None])}/{len(temperatures)}개 모델")
        else:
            print("자동 캘리브레이션 완료")
    elif not args.no_calibrate:
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
    
    num_classes = len(CLASS_NAMES)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    all_labels = []
    all_predictions = []
    
    # Top loss samples 수집을 위한 리스트
    top_loss_samples = [] if args.top_loss_samples > 0 else None
    
    print("추론 시작...")
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Inference')
        for data in pbar:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
            predicted, outputs = infer_batch(
                images, nets, weights, temperatures, is_ensemble, 
                args.voting, args.tta, normalize_mean, normalize_std, device
            )
            
            # Top loss samples 수집 (틀린 예측 중 확신도가 높은 것들)
            if args.top_loss_samples > 0:
                probs = F.softmax(outputs, dim=1)
                pred_probs = probs.gather(1, predicted.unsqueeze(1)).squeeze(1)
                loss_values = F.cross_entropy(outputs, labels, reduction='none')
                
                for i in range(len(labels)):
                    if predicted[i] != labels[i]:  # 틀린 예측만
                        top_loss_samples.append({
                            'image': images[i].cpu(),
                            'true_label': labels[i].item(),
                            'predicted_label': predicted[i].item(),
                            'probability': pred_probs[i].item(),
                            'loss': loss_values[i].item()
                        })
            
            if args.confusion_matrix:
                labels_cpu = labels.cpu().numpy()
                predicted_cpu = predicted.cpu().numpy()
                all_labels.extend(labels_cpu)
                all_predictions.extend(predicted_cpu)
                for label, prediction in zip(labels_cpu, predicted_cpu):
                    confusion_matrix[label, prediction] += 1
            
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[CLASS_NAMES[label]] += 1
                total_pred[CLASS_NAMES[label]] += 1
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            current_acc = 100 * correct / total
            pbar.set_postfix({'accuracy': f'{current_acc:.2f}%'})
    
    calibration_status = ""
    if args.auto_calibrate:
        if is_ensemble:
            calibrated_count = sum(1 for t in temperatures if t is not None)
            calibration_status = f" (자동 캘리브레이션: {calibrated_count}/{len(temperatures)} 모델)"
        else:
            calibration_status = " (자동 캘리브레이션 적용)"
    elif not args.no_calibrate:
        if is_ensemble:
            calibrated_count = sum(1 for t in temperatures if t is not None)
            if calibrated_count > 0:
                calibration_status = f" (캘리브레이션 적용: {calibrated_count}/{len(temperatures)} 모델)"
        else:
            if temperatures[0] is not None:
                calibration_status = " (캘리브레이션 적용)"
    
    tta_status = " (TTA 적용)" if args.tta in [2, 3, 5] else ""
    voting_status = f" ({args.voting.upper()} voting)" if is_ensemble else ""
    ensemble_status = f" (Ensemble: {len(nets)} 모델)" if is_ensemble else ""
    
    print(f'Accuracy of the network on the 10000 test images{ensemble_status}{voting_status}{calibration_status}{tta_status}: {100 * correct / total:.2f}%')
    
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
    
    if args.confusion_matrix:
        print_confusion_matrix(confusion_matrix, CLASS_NAMES)
    
    # Top loss samples 시각화
    if args.top_loss_samples > 0 and top_loss_samples:
        print(f"\nTop loss samples 시각화 중... (총 {len(top_loss_samples)}개 틀린 예측 중 상위 {args.top_loss_samples}개)")
        
        # Loss 기준으로 정렬하여 상위 N개 선택
        top_loss_samples.sort(key=lambda x: x['loss'], reverse=True)
        selected_samples = top_loss_samples[:args.top_loss_samples]
        
        images_list = [sample['image'] for sample in selected_samples]
        true_labels_list = [sample['true_label'] for sample in selected_samples]
        predicted_labels_list = [sample['predicted_label'] for sample in selected_samples]
        probabilities_list = [sample['probability'] for sample in selected_samples]
        losses_list = [sample['loss'] for sample in selected_samples]
        
        visualize_top_loss_samples(
            images=images_list,
            true_labels=true_labels_list,
            predicted_labels=predicted_labels_list,
            probabilities=probabilities_list,
            losses=losses_list,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
            num_samples=args.top_loss_samples,
            save_path=None
        )
    elif args.top_loss_samples > 0:
        print(f"\n경고: Top loss samples를 수집할 틀린 예측이 없습니다.")
