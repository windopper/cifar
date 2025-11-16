"""
deep_baseline2 기반 모든 모델의 파라미터 개수를 계산하는 유틸 함수
"""
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from models.baseline import BaselineNet
from models.baseline_bn import BaselineNetBN
from models.deep_baseline_bn import DeepBaselineNetBN
from models.deep_baseline2_bn import DeepBaselineNetBN2
from models.deep_baseline2_bn_residual import DeepBaselineNetBN2Residual
from models.deep_baseline2_bn_residual_preact import DeepBaselineNetBN2ResidualPreAct
from models.deep_baseline2_bn_resnext import DeepBaselineNetBN2ResNeXt
from models.deep_baseline2_bn_residual_se import DeepBaselineNetBN2ResidualSE
from models.deep_baseline2_bn_residual_grn import DeepBaselineNetBN2ResidualGRN
from models.deep_baseline3_bn import DeepBaselineNetBN3
from models.deep_baseline3_bn_residual import DeepBaselineNetBN3Residual
from models.deep_baseline3_bn_residual_dla import DeepBaselineNetBN3ResidualDLA
from models.mxresnet import MXResNet20, MXResNet32, MXResNet44, MXResNet56
from models.resnext import ResNeXt29_2x64d, ResNeXt29_4x64d, ResNeXt29_8x64d, ResNeXt29_32x4d
from models.dla import DLA
from models.convnext_step3_full import ConvNeXtCIFAR


def count_parameters(model):
    """
    모델의 파라미터 개수를 계산합니다.
    
    Args:
        model: PyTorch 모델 (nn.Module)
        
    Returns:
        int: 학습 가능한 파라미터의 총 개수
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters_all(model):
    """
    모델의 모든 파라미터 개수를 계산합니다 (requires_grad=False 포함).
    
    Args:
        model: PyTorch 모델 (nn.Module)
        
    Returns:
        int: 모든 파라미터의 총 개수
    """
    return sum(p.numel() for p in model.parameters())


def get_deep_baseline2_parameter_counts(init_weights=False):
    """
    deep_baseline, deep_baseline2/3, MXResNet, ResNeXt, DLA 및 ConvNeXt 모델의 파라미터 개수를 계산합니다.
    
    Args:
        init_weights: 가중치 초기화 여부 (기본값: False)
        
    Returns:
        dict: 모델 이름을 키로 하고 파라미터 개수를 값으로 하는 딕셔너리
    """
    results = {}
    
    model_baseline = BaselineNet()
    results['baseline'] = count_parameters(model_baseline)
    
    # BaselineNetBN
    model_baseline_bn = BaselineNetBN(init_weights=init_weights)
    results['baseline_bn'] = count_parameters(model_baseline_bn)
    
    # DeepBaselineNetBN
    model_bn = DeepBaselineNetBN(init_weights=init_weights)
    results['deep_baseline_bn'] = count_parameters(model_bn)
    
    # DeepBaselineNetBN2
    model_bn2 = DeepBaselineNetBN2(init_weights=init_weights)
    results['deep_baseline2_bn'] = count_parameters(model_bn2)
    
    # DeepBaselineNetBN2Residual
    model_residual = DeepBaselineNetBN2Residual(init_weights=init_weights)
    results['deep_baseline2_bn_residual'] = count_parameters(model_residual)
    
    # DeepBaselineNetBN2ResidualPreAct
    model_preact = DeepBaselineNetBN2ResidualPreAct()
    results['deep_baseline2_bn_residual_preact'] = count_parameters(model_preact)
    
    # DeepBaselineNetBN2ResNeXt (기본값 사용: cardinality=8, bottleneck_width=4)
    model_resnext = DeepBaselineNetBN2ResNeXt(init_weights=init_weights)
    results['deep_baseline2_bn_resnext'] = count_parameters(model_resnext)
    
    # DeepBaselineNetBN2ResidualSE (기본값 사용: se_reduction=16)
    model_se = DeepBaselineNetBN2ResidualSE(init_weights=init_weights)
    results['deep_baseline2_bn_residual_se'] = count_parameters(model_se)
    
    # DeepBaselineNetBN2ResidualGRN (기본값 사용: drop_path_rate=0.1, layer_scale_init_value=1e-5)
    model_grn = DeepBaselineNetBN2ResidualGRN(init_weights=init_weights)
    results['deep_baseline2_bn_residual_grn'] = count_parameters(model_grn)
    
    # DeepBaselineNetBN3
    model_bn3 = DeepBaselineNetBN3(init_weights=init_weights)
    results['deep_baseline3_bn'] = count_parameters(model_bn3)
    
    # DeepBaselineNetBN3Residual
    model_bn3_residual = DeepBaselineNetBN3Residual(init_weights=init_weights)
    results['deep_baseline3_bn_residual'] = count_parameters(model_bn3_residual)
    
    # DeepBaselineNetBN3ResidualDLA
    model_bn3_residual_dla = DeepBaselineNetBN3ResidualDLA(init_weights=init_weights)
    results['deep_baseline3_bn_residual_dla'] = count_parameters(model_bn3_residual_dla)
    
    # MXResNet models
    model_mxresnet20 = MXResNet20(init_weights=init_weights)
    results['mxresnet20'] = count_parameters(model_mxresnet20)
    
    model_mxresnet32 = MXResNet32(init_weights=init_weights)
    results['mxresnet32'] = count_parameters(model_mxresnet32)
    
    model_mxresnet44 = MXResNet44(init_weights=init_weights)
    results['mxresnet44'] = count_parameters(model_mxresnet44)
    
    model_mxresnet56 = MXResNet56(init_weights=init_weights)
    results['mxresnet56'] = count_parameters(model_mxresnet56)
    
    # ResNeXt models
    model_resnext29_2x64d = ResNeXt29_2x64d()
    results['resnext29_2x64d'] = count_parameters(model_resnext29_2x64d)
    
    model_resnext29_4x64d = ResNeXt29_4x64d()
    results['resnext29_4x64d'] = count_parameters(model_resnext29_4x64d)
    
    model_resnext29_8x64d = ResNeXt29_8x64d()
    results['resnext29_8x64d'] = count_parameters(model_resnext29_8x64d)
    
    model_resnext29_32x4d = ResNeXt29_32x4d()
    results['resnext29_32x4d'] = count_parameters(model_resnext29_32x4d)
    
    # DLA model
    model_dla = DLA()
    results['dla'] = count_parameters(model_dla)
    
    # ConvNeXtCIFAR model
    model_convnext = ConvNeXtCIFAR(init_weights=init_weights)
    results['convnext_step3_full'] = count_parameters(model_convnext)
    
    return results


def print_deep_baseline2_parameter_counts(init_weights=False):
    """
    deep_baseline, deep_baseline2/3, MXResNet, ResNeXt, DLA 및 ConvNeXt 모델의 파라미터 개수를 출력합니다.
    
    Args:
        init_weights: 가중치 초기화 여부 (기본값: False)
    """
    results = get_deep_baseline2_parameter_counts(init_weights=init_weights)
    
    print("=" * 70)
    print("deep_baseline, deep_baseline2/3, MXResNet, ResNeXt, DLA 및 ConvNeXt 모델 파라미터 개수")
    print("=" * 70)
    
    # 파라미터 개수로 정렬
    sorted_results = sorted(results.items(), key=lambda x: x[1])
    
    for model_name, param_count in sorted_results:
        # 천 단위 구분자로 포맷팅
        param_count_str = f"{param_count:,}"
        print(f"{model_name:40s}: {param_count_str:>15s} 개")
    
    print("=" * 70)
    print(f"총 모델 수: {len(results)}")
    print("=" * 70)


if __name__ == '__main__':
    # 기본 설정으로 파라미터 개수 출력
    print_deep_baseline2_parameter_counts()

