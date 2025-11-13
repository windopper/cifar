import argparse
import torch
import json
import os
from torchvision.transforms import transforms
import torchvision
from main import CLASS_NAMES, get_net

def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='CIFAR-10 Inference')
    parser.add_argument('--path', '-p', type=str, 
                        default='outputs/baseline_baseline_sgd_crossentropy.pth',
                        help='모델 파일 경로 (default: outputs/baseline_baseline_sgd_crossentropy.pth)')
    parser.add_argument('--model', '-m', type=str, default='baseline',
                        choices=['baseline'],
                        help='네트워크 모델 (default: baseline)')
    parser.add_argument('--batch-size', type=int, default=4, 
                        help='배치 크기 (default: 4)')
    parser.add_argument('--temperature', '-t', type=str, default=None,
                        help='Temperature 파일 경로 (지정하지 않으면 모델 경로에서 자동 검색)')
    parser.add_argument('--no-calibrate', action='store_true',
                        help='캘리브레이션 사용 안 함 (temperature 파일이 있어도 무시)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # CUDA 호환성 문제로 인해 CPU 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    net = get_net(args.model)
    net.load_state_dict(torch.load(args.path))
    net.to(device)
    net.eval()

    # Temperature 파일 로드
    temperature = None
    if not args.no_calibrate:
        if args.temperature:
            # 사용자가 직접 지정한 경로
            temp_path = args.temperature
        else:
            # 모델 경로에서 자동으로 temperature 파일 찾기
            # 예: outputs/model.pth -> outputs/model_temperature.json
            base_path = args.path.rsplit('.pth', 1)[0]
            temp_path = f"{base_path}_temperature.json"
        
        if os.path.exists(temp_path):
            with open(temp_path, 'r') as f:
                temp_data = json.load(f)
                temperature = temp_data.get('temperature')
            print(f"Temperature 파일 로드: {temp_path}")
            print(f"Temperature 값: {temperature:.4f}")
            print("캘리브레이션 적용됨\n")
        else:
            print(f"Temperature 파일을 찾을 수 없습니다: {temp_path}")
            print("캘리브레이션 없이 추론합니다\n")
    else:
        print("캘리브레이션 비활성화됨\n")

    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in CLASS_NAMES}
    total_pred = {classname: 0 for classname in CLASS_NAMES}

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            
            # Temperature Scaling 적용
            if temperature is not None:
                outputs = outputs / temperature
            
            _, predicted = torch.max(outputs.data, 1)
            
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[CLASS_NAMES[label]] += 1
                total_pred[CLASS_NAMES[label]] += 1
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    calibration_status = " (캘리브레이션 적용)" if temperature is not None else ""
    print(f'Accuracy of the network on the 10000 test images{calibration_status}: {100 * correct / total:.2f}%')

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')import argparse
import torch
import json
import os
from torchvision.transforms import transforms
import torchvision
from main import CLASS_NAMES, get_net

def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='CIFAR-10 Inference')
    parser.add_argument('--path', '-p', type=str, 
                        default='outputs/baseline_baseline_sgd_crossentropy.pth',
                        help='모델 파일 경로 (default: outputs/baseline_baseline_sgd_crossentropy.pth)')
    parser.add_argument('--model', '-m', type=str, default='baseline',
                        choices=['baseline'],
                        help='네트워크 모델 (default: baseline)')
    parser.add_argument('--batch-size', type=int, default=4, 
                        help='배치 크기 (default: 4)')
    parser.add_argument('--temperature', '-t', type=str, default=None,
                        help='Temperature 파일 경로 (지정하지 않으면 모델 경로에서 자동 검색)')
    parser.add_argument('--no-calibrate', action='store_true',
                        help='캘리브레이션 사용 안 함 (temperature 파일이 있어도 무시)')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # CUDA 호환성 문제로 인해 CPU 사용
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    net = get_net(args.model)
    net.load_state_dict(torch.load(args.path))
    net.to(device)
    net.eval()

    # Temperature 파일 로드
    temperature = None
    if not args.no_calibrate:
        if args.temperature:
            # 사용자가 직접 지정한 경로
            temp_path = args.temperature
        else:
            # 모델 경로에서 자동으로 temperature 파일 찾기
            # 예: outputs/model.pth -> outputs/model_temperature.json
            base_path = args.path.rsplit('.pth', 1)[0]
            temp_path = f"{base_path}_temperature.json"
        
        if os.path.exists(temp_path):
            with open(temp_path, 'r') as f:
                temp_data = json.load(f)
                temperature = temp_data.get('temperature')
            print(f"Temperature 파일 로드: {temp_path}")
            print(f"Temperature 값: {temperature:.4f}")
            print("캘리브레이션 적용됨\n")
        else:
            print(f"Temperature 파일을 찾을 수 없습니다: {temp_path}")
            print("캘리브레이션 없이 추론합니다\n")
    else:
        print("캘리브레이션 비활성화됨\n")

    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in CLASS_NAMES}
    total_pred = {classname: 0 for classname in CLASS_NAMES}

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            
            # Temperature Scaling 적용
            if temperature is not None:
                outputs = outputs / temperature
            
            _, predicted = torch.max(outputs.data, 1)
            
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    correct_pred[CLASS_NAMES[label]] += 1
                total_pred[CLASS_NAMES[label]] += 1
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    calibration_status = " (캘리브레이션 적용)" if temperature is not None else ""
    print(f'Accuracy of the network on the 10000 test images{calibration_status}: {100 * correct / total:.2f}%')

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')