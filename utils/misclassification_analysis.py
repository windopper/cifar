"""
오분류 분석 유틸리티 함수

주요 기능:
1. 특정 모델의 오분류 데이터 수집
2. 실제 라벨별로 그룹화하여 잘못 예측된 클래스 정보 수집
"""
import os
import json
import torch
import torchvision
from torchvision.transforms import transforms
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from main import CLASS_NAMES, get_net


def load_model_from_history(history_path: str) -> Tuple[str, str, tuple, tuple]:
    """
    History 파일에서 모델 정보를 추출하고 모델 경로를 반환합니다.
    
    Args:
        history_path: History 파일 경로
        
    Returns:
        (model_name, model_path, normalize_mean, normalize_std) 튜플
    """
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


def load_temperature(model_path: str, temperature_path: Optional[str] = None) -> Optional[float]:
    """
    Temperature 파일을 로드합니다.
    
    Args:
        model_path: 모델 파일 경로
        temperature_path: Temperature 파일 경로 (None이면 자동 검색)
        
    Returns:
        Temperature 값 또는 None
    """
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


def collect_misclassifications(
    history_path: str,
    batch_size: int = 4,
    temperature_path: Optional[str] = None,
    use_calibration: bool = True,
    tta: int = 0,
    device: Optional[torch.device] = None
) -> Dict[str, Dict[str, List[Dict]]]:
    """
    특정 모델의 오분류 데이터를 수집합니다.
    
    실제 라벨별로 그룹화하여, 각 라벨에서 어떤 클래스로 잘못 예측했는지 수집합니다.
    
    Args:
        history_path: History 파일 경로
        batch_size: 배치 크기
        temperature_path: Temperature 파일 경로 (None이면 자동 검색)
        use_calibration: Temperature scaling 사용 여부
        tta: Test Time Augmentation (0: 비활성화, 2: 원본+Horizontal Flip)
        device: 사용할 디바이스 (None이면 자동 선택)
        
    Returns:
        딕셔너리 형태의 오분류 데이터:
        {
            'actual_label': {
                'predicted_label': [
                    {
                        'image_index': int,
                        'confidence': float,
                        'probabilities': List[float]
                    },
                    ...
                ],
                ...
            },
            ...
        }
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 정보 로드
    model_name, model_path, normalize_mean, normalize_std = load_model_from_history(history_path)
    
    # Temperature 로드
    temperature = None
    if use_calibration:
        temperature = load_temperature(model_path, temperature_path)
    
    # 데이터 변환 및 로더 설정
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std)
    ])
    
    test_set = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # 모델 로드
    net = get_net(model_name)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device)
    net.eval()
    
    # 오분류 데이터 수집을 위한 딕셔너리
    # 구조: {actual_label: {predicted_label: [misclassification_info, ...]}}
    misclassifications = defaultdict(lambda: defaultdict(list))
    
    # 이미지 인덱스 추적
    image_index = 0
    
    print(f"모델: {model_name}")
    print(f"모델 경로: {model_path}")
    if temperature:
        print(f"Temperature: {temperature:.4f}")
    print(f"Device: {device}")
    print("오분류 데이터 수집 시작...\n")
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Collecting misclassifications')
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            # 추론 수행
            if tta == 2:
                # TTA: 원본 + Horizontal Flip 평균
                outputs_original = net(images)
                outputs_flipped = net(torch.flip(images, [3]))  # Horizontal flip (dim=3)
                outputs = (outputs_original + outputs_flipped) / 2.0
            else:
                outputs = net(images)
            
            # Temperature Scaling 적용
            if temperature is not None:
                outputs = outputs / temperature
            
            # 확률 계산
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            # 오분류 데이터 수집
            for i in range(len(labels)):
                actual_label_idx = labels[i].item()
                predicted_label_idx = predicted[i].item()
                actual_label = CLASS_NAMES[actual_label_idx]
                predicted_label = CLASS_NAMES[predicted_label_idx]
                
                # 틀린 예측인 경우만 수집
                if actual_label_idx != predicted_label_idx:
                    confidence = probabilities[i][predicted_label_idx].item()
                    prob_list = probabilities[i].cpu().tolist()
                    
                    misclassifications[actual_label][predicted_label].append({
                        'image_index': image_index,
                        'confidence': confidence,
                        'probabilities': prob_list
                    })
                
                image_index += 1
            
            # 진행률 표시
            total_misclassified = sum(
                len(pred_list)
                for actual_dict in misclassifications.values()
                for pred_list in actual_dict.values()
            )
            pbar.set_postfix({'misclassified': total_misclassified})
    
    # defaultdict를 일반 dict로 변환
    result = {
        actual_label: dict(predicted_dict)
        for actual_label, predicted_dict in misclassifications.items()
    }
    
    # 통계 출력
    print("\n" + "=" * 60)
    print("오분류 통계")
    print("=" * 60)
    
    total_misclassified = 0
    for actual_label in CLASS_NAMES:
        if actual_label in result:
            label_misclassified = sum(
                len(pred_list) for pred_list in result[actual_label].values()
            )
            total_misclassified += label_misclassified
            
            print(f"\n[{actual_label}]")
            print(f"  총 오분류 수: {label_misclassified}")
            
            # 잘못 예측된 클래스별 통계
            for predicted_label, misclass_list in sorted(
                result[actual_label].items(),
                key=lambda x: len(x[1]),
                reverse=True
            ):
                avg_confidence = sum(m['confidence'] for m in misclass_list) / len(misclass_list)
                print(f"    -> {predicted_label}: {len(misclass_list)}개 (평균 confidence: {avg_confidence:.3f})")
        else:
            print(f"\n[{actual_label}]")
            print(f"  총 오분류 수: 0")
    
    print("\n" + "=" * 60)
    print(f"전체 오분류 수: {total_misclassified}")
    print("=" * 60)
    
    return result


def save_misclassifications(
    misclassifications: Dict[str, Dict[str, List[Dict]]],
    output_path: str
):
    """
    오분류 데이터를 JSON 파일로 저장합니다.
    
    Args:
        misclassifications: collect_misclassifications 함수의 반환값
        output_path: 저장할 파일 경로
    """
    # numpy 타입을 Python 기본 타입으로 변환
    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (int, float)):
            return float(obj) if isinstance(obj, (torch.Tensor,)) else obj
        else:
            return obj
    
    serializable_data = convert_to_serializable(misclassifications)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n오분류 데이터가 저장되었습니다: {output_path}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='CIFAR-10 오분류 데이터 수집')
    parser.add_argument('--history', type=str, required=True,
                        help='History 파일 경로')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='배치 크기 (default: 4)')
    parser.add_argument('--temperature', '-t', type=str, default=None,
                        help='Temperature 파일 경로 (지정하지 않으면 모델 경로에서 자동 검색)')
    parser.add_argument('--no-calibrate', action='store_true',
                        help='캘리브레이션 사용 안 함')
    parser.add_argument('--tta', type=int, default=0, choices=[0, 2],
                        help='Test Time Augmentation 활성화 (0: 비활성화, 2: 원본+Horizontal Flip)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='결과 저장 경로 (JSON 파일, 지정하지 않으면 저장하지 않음)')
    
    args = parser.parse_args()
    
    # 오분류 데이터 수집
    misclassifications = collect_misclassifications(
        history_path=args.history,
        batch_size=args.batch_size,
        temperature_path=args.temperature,
        use_calibration=not args.no_calibrate,
        tta=args.tta
    )
    
    # 결과 저장
    if args.output:
        save_misclassifications(misclassifications, args.output)

