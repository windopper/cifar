"""
thop을 사용하여 모델의 GFLOPs를 측정하는 유틸 함수
"""
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.net import get_available_nets, get_net

import torch
import torch.nn as nn
from thop import profile, clever_format

def measure_gflops(model: nn.Module, input_size: tuple = (1, 3, 32, 32), device: torch.device = None):
    """
    모델의 GFLOPs를 측정합니다.
    
    Args:
        model: PyTorch 모델 (nn.Module)
        input_size: 입력 텐서 크기 (batch_size, channels, height, width)
        device: 디바이스 (None이면 CPU 사용)
        
    Returns:
        tuple: (FLOPs, GFLOPs, 파라미터 개수)
    """
    if device is None:
        device = torch.device('cpu')
    
    model = model.to(device)
    model.eval()
    
    # 더미 입력 생성
    dummy_input = torch.randn(input_size).to(device)
    
    try:
        # thop을 사용하여 FLOPs와 파라미터 개수 측정
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        
        # GFLOPs 계산 (1 GFLOP = 10^9 FLOP)
        gflops = flops / 1e9
        
        return flops, gflops, params
    except Exception as e:
        print(f"[오류] GFLOPs 측정 실패: {e}")
        return None, None, None


def get_model_gflops(model_name: str, init_weights: bool = False, shakedrop_prob: float = 0.0,
                     input_size: tuple = (1, 3, 32, 32), device: torch.device = None):
    """
    특정 모델의 GFLOPs를 측정합니다.
    
    Args:
        model_name: 모델 이름
        init_weights: 가중치 초기화 여부 (기본값: False)
        shakedrop_prob: ShakeDrop 확률 (기본값: 0.0)
        input_size: 입력 텐서 크기 (batch_size, channels, height, width)
        device: 디바이스 (None이면 CPU 사용)
        
    Returns:
        tuple: (FLOPs, GFLOPs, 파라미터 개수) 또는 (None, None, None) (실패 시)
    """
    try:
        model = get_net(model_name, init_weights=init_weights, shakedrop_prob=shakedrop_prob)
        return measure_gflops(model, input_size=input_size, device=device)
    except Exception as e:
        print(f"[오류] 모델 '{model_name}' 생성 실패: {e}")
        return None, None, None


def get_models_gflops(model_names: list = None, init_weights: bool = False, shakedrop_prob: float = 0.0,
                      input_size: tuple = (1, 3, 32, 32), device: torch.device = None):
    """
    지정된 모델들의 GFLOPs를 측정합니다.
    
    Args:
        model_names: 모델 이름 목록 (None이면 모든 모델)
        init_weights: 가중치 초기화 여부 (기본값: False)
        shakedrop_prob: ShakeDrop 확률 (기본값: 0.0)
        input_size: 입력 텐서 크기 (batch_size, channels, height, width)
        device: 디바이스 (None이면 CPU 사용)
        
    Returns:
        dict: 모델 이름을 키로 하고 (FLOPs, GFLOPs, 파라미터 개수) 튜플을 값으로 하는 딕셔너리
    """
    results = {}
    
    # 모델 이름 목록이 없으면 모든 모델 사용
    if model_names is None:
        model_names = get_available_nets()
    
    for model_name in model_names:
        flops, gflops, params = get_model_gflops(
            model_name, init_weights=init_weights, shakedrop_prob=shakedrop_prob,
            input_size=input_size, device=device
        )
        if flops is not None:
            results[model_name] = (flops, gflops, params)
    
    return results


def print_model_gflops(init_weights: bool = False, model_name: str = None, shakedrop_prob: float = 0.0,
                       input_size: tuple = (1, 3, 32, 32), device: torch.device = None):
    """
    모델의 GFLOPs를 출력합니다.
    
    Args:
        init_weights: 가중치 초기화 여부 (기본값: False)
        model_name: 특정 모델 이름 (None이면 모든 모델)
        shakedrop_prob: ShakeDrop 확률 (기본값: 0.0)
        input_size: 입력 텐서 크기 (batch_size, channels, height, width)
        device: 디바이스 (None이면 CPU 사용)
    """
    if model_name:
        # 특정 모델만 계산
        model_names = [model_name]
    else:
        # 모든 모델 계산
        model_names = None
    
    results = get_models_gflops(
        model_names=model_names, init_weights=init_weights, shakedrop_prob=shakedrop_prob,
        input_size=input_size, device=device
    )
    
    print("=" * 100)
    if model_name:
        print(f"모델 '{model_name}' GFLOPs 측정 결과")
    else:
        print("모델 GFLOPs 측정 결과")
    print("=" * 100)
    print(f"{'모델 이름':<50s} {'FLOPs':>20s} {'GFLOPs':>15s} {'파라미터':>20s}")
    print("-" * 100)
    
    # GFLOPs로 정렬
    sorted_results = sorted(results.items(), key=lambda x: x[1][1] if x[1][1] is not None else 0)
    
    for model_name, (flops, gflops, params) in sorted_results:
        if flops is not None:
            # FLOPs를 읽기 쉬운 형식으로 변환
            flops_str = clever_format([flops], "%.2f")[0]
            gflops_str = f"{gflops:.4f}"
            params_str = clever_format([params], "%.2f")[0]
            
            print(f"{model_name:<50s} {flops_str:>20s} {gflops_str:>15s} {params_str:>20s}")
        else:
            print(f"{model_name:<50s} {'측정 실패':>20s} {'-':>15s} {'-':>20s}")
    
    print("=" * 100)
    print(f"총 모델 수: {len(results)}")
    print("=" * 100)


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='모델 GFLOPs 측정')
    parser.add_argument('--model', type=str, default=None,
                        help='측정할 모델 이름 (지정하지 않으면 모든 모델)')
    parser.add_argument('--init-weights', action='store_true',
                        help='가중치 초기화 사용 (default: False)')
    parser.add_argument('--shakedrop', type=float, default=0.0,
                        help='ShakeDrop 확률 (0.0~1.0, WideResNet/PyramidNet 모델에만 적용, default: 0.0)')
    parser.add_argument('--input-size', type=int, nargs=4, default=[1, 3, 32, 32],
                        metavar=('BATCH', 'CHANNELS', 'HEIGHT', 'WIDTH'),
                        help='입력 텐서 크기 (default: 1 3 32 32)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='측정에 사용할 디바이스 (default: cpu)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    # 디바이스 설정
    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        if args.device == 'cuda':
            print("[경고] CUDA를 사용할 수 없습니다. CPU를 사용합니다.")
    
    # 모델 이름 검증
    if args.model:
        available_models = get_available_nets()
        if args.model.lower() not in [m.lower() for m in available_models]:
            print(f"[오류] 알 수 없는 모델: {args.model}")
            print(f"사용 가능한 모델: {', '.join(available_models)}")
            sys.exit(1)
        # 대소문자 구분 없이 정확한 모델 이름 찾기
        model_name = next(m for m in available_models if m.lower() == args.model.lower())
    else:
        model_name = None
    
    # 입력 크기 튜플로 변환
    input_size = tuple(args.input_size)
    
    # GFLOPs 출력
    print_model_gflops(
        init_weights=args.init_weights,
        model_name=model_name,
        shakedrop_prob=args.shakedrop,
        input_size=input_size,
        device=device
    )



