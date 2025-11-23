import sys
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from collections import OrderedDict
import subprocess
import shutil
from utils.net import _get_nets_dict 

try:
    from torchviz import make_dot
    TORCHVIZ_AVAILABLE = True
except ImportError:
    TORCHVIZ_AVAILABLE = False

try:
    from torchsummary import summary
    TORCHSUMMARY_AVAILABLE = True
except ImportError:
    TORCHSUMMARY_AVAILABLE = False


def check_graphviz_available() -> bool:
    """
    Graphviz가 설치되어 있고 PATH에 있는지 확인합니다.
    Windows에서는 일반적인 설치 경로도 확인합니다.
    
    Returns:
        Graphviz 사용 가능 여부
    """
    # PATH에서 확인
    if shutil.which('dot') is not None:
        return True
    
    # Windows에서 일반적인 설치 경로 확인
    import platform
    if platform.system() == 'Windows':
        common_paths = [
            r"C:\Program Files\Graphviz\bin\dot.exe",
            r"C:\Program Files (x86)\Graphviz\bin\dot.exe",
        ]
        for path in common_paths:
            if Path(path).exists():
                # PATH에 추가 (현재 세션에만 적용)
                import os
                graphviz_bin = str(Path(path).parent)
                if graphviz_bin not in os.environ.get('PATH', ''):
                    os.environ['PATH'] = os.environ.get('PATH', '') + os.pathsep + graphviz_bin
                return True
    
    return False


def count_parameters(model: nn.Module) -> int:
    """
    모델의 학습 가능한 파라미터 개수를 계산합니다.
    
    Args:
        model: PyTorch 모델
        
    Returns:
        학습 가능한 파라미터의 총 개수
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_all(model: nn.Module) -> int:
    """
    모델의 모든 파라미터 개수를 계산합니다 (requires_grad=False 포함).
    
    Args:
        model: PyTorch 모델
        
    Returns:
        모든 파라미터의 총 개수
    """
    return sum(p.numel() for p in model.parameters())


def get_layer_info(model: nn.Module, input_size: Tuple[int, ...] = (3, 32, 32)) -> List[dict]:
    """
    모델의 각 레이어별 정보를 수집합니다.
    
    Args:
        model: PyTorch 모델
        input_size: 입력 텐서 크기 (C, H, W)
        
    Returns:
        레이어 정보 리스트
    """
    model.eval()
    layers_info = []
    
    def register_hook(name):
        def hook(module, input, output):
            layer_info = {
                'name': name,
                'type': type(module).__name__,
                'input_shape': tuple(input[0].shape) if isinstance(input, tuple) else tuple(input.shape),
                'output_shape': tuple(output.shape) if hasattr(output, 'shape') else str(type(output)),
                'parameters': sum(p.numel() for p in module.parameters()),
                'trainable_parameters': sum(p.numel() for p in module.parameters() if p.requires_grad),
            }
            layers_info.append(layer_info)
        return hook
    
    # 모든 서브모듈에 hook 등록
    hooks = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 리프 노드만
            hook = module.register_forward_hook(register_hook(name))
            hooks.append(hook)
    
    # 더미 입력으로 forward pass 실행
    dummy_input = torch.randn(1, *input_size)
    with torch.no_grad():
        _ = model(dummy_input)
    
    # hook 제거
    for hook in hooks:
        hook.remove()
    
    return layers_info


def print_model_summary(
    model: nn.Module,
    input_size: Tuple[int, ...] = (3, 32, 32),
    device: str = 'cpu',
    detailed: bool = True
):
    """
    모델 구조 요약을 출력합니다.
    
    Args:
        model: PyTorch 모델
        input_size: 입력 텐서 크기 (C, H, W)
        device: 디바이스 ('cpu' 또는 'cuda')
        detailed: 상세 정보 출력 여부
    """
    model = model.to(device)
    model.eval()
    
    print("=" * 80)
    print(f"모델: {model.__class__.__name__}")
    print("=" * 80)
    
    # 기본 정보
    total_params = count_parameters(model)
    total_params_all = count_parameters_all(model)
    
    print(f"\n[기본 정보]")
    print(f"  학습 가능한 파라미터: {total_params:,} 개")
    print(f"  전체 파라미터: {total_params_all:,} 개")
    print(f"  입력 크기: {input_size}")
    print(f"  디바이스: {device}")
    
    # 모델 구조 출력
    print(f"\n[모델 구조]")
    print(model)
    
    # 출력 크기 확인
    print(f"\n[출력 크기 확인]")
    dummy_input = torch.randn(1, *input_size).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"  입력: {tuple(dummy_input.shape)}")
    print(f"  출력: {tuple(output.shape)}")
    
    # 상세 정보
    if detailed:
        print(f"\n[레이어별 상세 정보]")
        layers_info = get_layer_info(model, input_size)
        
        print(f"{'레이어 이름':<40} {'타입':<25} {'입력 크기':<20} {'출력 크기':<20} {'파라미터':<15}")
        print("-" * 120)
        
        for layer in layers_info:
            input_shape_str = str(layer['input_shape'])[:18]
            output_shape_str = str(layer['output_shape'])[:18]
            params_str = f"{layer['parameters']:,}"
            
            print(f"{layer['name']:<40} {layer['type']:<25} {input_shape_str:<20} {output_shape_str:<20} {params_str:<15}")
    
    # 파라미터 통계
    print(f"\n[파라미터 통계]")
    param_stats = {}
    for name, param in model.named_parameters():
        module_type = name.split('.')[0] if '.' in name else name
        if module_type not in param_stats:
            param_stats[module_type] = {'count': 0, 'size': 0}
        param_stats[module_type]['count'] += param.numel()
        param_stats[module_type]['size'] += param.numel() * param.element_size()
    
    print(f"{'모듈 타입':<30} {'파라미터 개수':<20} {'메모리 (MB)':<15}")
    print("-" * 65)
    for module_type, stats in sorted(param_stats.items()):
        memory_mb = stats['size'] / (1024 * 1024)
        print(f"{module_type:<30} {stats['count']:>15,} {memory_mb:>12.2f}")
    
    print("=" * 80)


def visualize_model_graph(
    model: nn.Module,
    input_size: Tuple[int, ...] = (3, 32, 32),
    save_path: Optional[str] = None,
    device: str = 'cpu'
):
    """
    모델의 계산 그래프를 시각화합니다 (torchviz 사용).
    
    Args:
        model: PyTorch 모델
        input_size: 입력 텐서 크기 (C, H, W)
        save_path: 저장 경로 (None이면 표시만)
        device: 디바이스 ('cpu' 또는 'cuda')
    """
    if not TORCHVIZ_AVAILABLE:
        print("경고: torchviz가 설치되지 않았습니다. 계산 그래프 시각화를 사용할 수 없습니다.")
        print("설치 방법: uv add torchviz")
        return
    
    if not check_graphviz_available():
        print("경고: Graphviz가 설치되지 않았거나 PATH에 없습니다.")
        print("계산 그래프 시각화를 사용하려면 Graphviz를 설치해야 합니다.")
        print("\n설치 방법:")
        print("1. Windows:")
        print("   - https://graphviz.org/download/ 에서 Windows용 Graphviz 다운로드")
        print("   - 설치 후 시스템 PATH에 Graphviz bin 디렉토리 추가")
        print("   - 또는: winget install graphviz")
        print("   - 또는: choco install graphviz (Chocolatey 사용 시)")
        print("\n2. Linux:")
        print("   - sudo apt-get install graphviz (Ubuntu/Debian)")
        print("   - sudo yum install graphviz (CentOS/RHEL)")
        print("\n3. macOS:")
        print("   - brew install graphviz")
        print("\n설치 후 터미널을 재시작하거나 PATH를 새로고침하세요.")
        return
    
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(1, *input_size).to(device)
    
    try:
        output = model(dummy_input)
        dot = make_dot(output, params=dict(model.named_parameters()))
        
        if save_path:
            dot.render(save_path, format='png', cleanup=True)
            print(f"계산 그래프가 저장되었습니다: {save_path}.png")
        else:
            print("계산 그래프를 생성했습니다. (저장하려면 save_path를 지정하세요)")
            print(dot.source)
    except Exception as e:
        print(f"계산 그래프 생성 중 오류 발생: {e}")
        if "dot" in str(e).lower() or "graphviz" in str(e).lower():
            print("\nGraphviz 관련 오류입니다. Graphviz가 올바르게 설치되고 PATH에 있는지 확인하세요.")


def visualize_model(
    model: nn.Module,
    input_size: Tuple[int, ...] = (3, 32, 32),
    device: str = 'cpu',
    detailed: bool = True,
    save_graph: Optional[str] = None
):
    """
    모델을 시각화하는 통합 함수입니다.
    
    Args:
        model: PyTorch 모델
        input_size: 입력 텐서 크기 (C, H, W)
        device: 디바이스 ('cpu' 또는 'cuda')
        detailed: 상세 정보 출력 여부
        save_graph: 계산 그래프 저장 경로 (None이면 저장하지 않음)
    """
    # 모델 요약 출력
    print_model_summary(model, input_size=input_size, device=device, detailed=detailed)
    
    # 계산 그래프 시각화 (선택적)
    if save_graph:
        visualize_model_graph(model, input_size=input_size, save_path=save_graph, device=device)


def print_model_architecture(model: nn.Module, indent: int = 0):
    """
    모델의 계층 구조를 트리 형태로 출력합니다.
    
    Args:
        model: PyTorch 모델
        indent: 들여쓰기 레벨
    """
    prefix = "  " * indent
    
    # 현재 모듈 정보
    module_name = model.__class__.__name__
    num_params = sum(p.numel() for p in model.parameters())
    
    print(f"{prefix}{module_name} ({num_params:,} params)")
    
    # 자식 모듈 출력
    for name, child in model.named_children():
        print(f"{prefix}  └─ {name}:")
        print_model_architecture(child, indent + 2)


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='CIFAR-10 모델 시각화')
    parser.add_argument('--net', type=str, default='resnet18',
                        choices=list(_get_nets_dict().keys()),
                        help='네트워크 모델 (default: resnet18)')
    parser.add_argument('--w-init', action='store_true',
                        help='Weight initialization 사용 (default: False)')
    parser.add_argument('--input-size', type=int, nargs=3, default=[3, 32, 32],
                        metavar=('C', 'H', 'W'),
                        help='입력 크기 (C, H, W) (default: 3 32 32)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='디바이스 (default: cpu)')
    parser.add_argument('--no-detailed', action='store_true',
                        help='상세 정보 출력 비활성화 (default: False)')
    parser.add_argument('--save-graph', type=str, default=None,
                        help='계산 그래프 저장 경로 (저장하지 않으려면 지정하지 않음)')
    parser.add_argument('--show-architecture', action='store_true',
                        help='계층 구조 트리 출력 (default: False)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # main.py에서 get_net 함수 import
    from utils.net import get_net
    
    # 모델 로드
    print(f"모델 로드 중: {args.net}\n")
    model = get_net(args.net, init_weights=args.w_init)
    
    # 입력 크기 설정
    input_size = tuple(args.input_size)
    
    # 모델 시각화
    visualize_model(
        model,
        input_size=input_size,
        device=args.device,
        detailed=not args.no_detailed,
        save_graph=args.save_graph
    )
    
    # 계층 구조 트리 출력 (옵션)
    # if args.show_architecture:
    #     print("\n" + "=" * 80)
    #     print("계층 구조 트리")
    #     print("=" * 80)
    #     print_model_architecture(model)

