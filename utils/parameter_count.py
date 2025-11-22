"""
deep_baseline2 기반 모든 모델의 파라미터 개수를 계산하는 유틸 함수
"""
import sys
import argparse
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
from models.deep_baseline3_bn_residual_15 import DeepBaselineNetBN3Residual15
from models.deep_baseline3_bn_residual_18 import DeepBaselineNetBN3Residual18
from models.deep_baseline3_bn_residual_15_convnext import DeepBaselineNetBN3Residual15ConvNeXt
from models.deep_baseline3_bn_residual_15_convnext_ln_classifier import DeepBaselineNetBN3Residual15ConvNeXtLNClassifier
from models.deep_baseline3_bn_residual_15_convnext_ln_classifier_stem import DeepBaselineNetBN3Residual15ConvNeXtLNClassifierStem
from models.deep_baseline3_bn_residual_15_ln import DeepBaselineNetBN3Residual15LN
from models.deep_baseline3_bn_residual_15_attention import (
    DeepBaselineNetBN3Residual15Attention,
    make_deep_baseline3_bn_residual_15_attention_tiny,
)
from models.residual_attention_92_32input import (
    ResidualAttentionModel_92_32input,
    make_residual_attention_92_32input_tiny,
)
from models.residual_attention_92_32input_preact import (
    ResidualAttentionModel_92_32input_PreAct,
    ResidualAttentionModel_92_32input_PreAct_Tiny,
    make_residual_attention_92_32input_preact_tiny,
)
from models.residual_attention_92_32input_se import (
    ResidualAttentionModel_92_32input_SE,
    ResidualAttentionModel_92_32input_SE_Tiny,
    make_residual_attention_92_32input_se_tiny,
)
from models.residual_attention_92_32input_gelu import (
    ResidualAttentionModel_92_32input_GELU,
    ResidualAttentionModel_92_32input_GELU_Tiny,
    make_residual_attention_92_32input_gelu_tiny,
)
from models.residual_attention_92_32input_gelu_medium import (
    ResidualAttentionModel_92_32input_GELU_Medium,
    make_residual_attention_92_32input_gelu_medium,
)
from models.residual_attention_92_32input_gelu_tiny_dla import (
    ResidualAttentionModel_92_32input_GELU_Tiny_DLA,
    make_residual_attention_92_32input_gelu_tiny_dla,
    make_residual_attention_92_32input_gelu_tiny_dla_tiny,
    make_residual_attention_92_32input_gelu_tiny_dla_small,
    make_residual_attention_92_32input_gelu_tiny_dla_base,
    make_residual_attention_92_32input_gelu_tiny_dla_large,
    make_residual_attention_92_32input_gelu_tiny_dla_wide,
    make_residual_attention_92_32input_gelu_tiny_dla_deep,
)
from models.deep_baseline3_bn_residual_bottleneck import DeepBaselineNetBN3ResidualBottleneck
from models.deep_baseline3_convnext_stride import DeepBaselineNetBN3ResidualConvNeXt
from models.deep_baseline3_bn_residual_wide import DeepBaselineNetBN3ResidualWide
from models.deep_baseline3_bn_residual_4x import DeepBaselineNetBN3Residual4X
from models.deep_baseline3_bn_residual_deep import DeepBaselineNetBN3ResidualDeep
from models.deep_baseline3_bn_residual_dla import DeepBaselineNetBN3ResidualDLA
from models.deep_baseline3_bn_residual_dla_tree import DeepBaselineNetBN3ResidualDLATree
from models.deep_baseline3_bn_residual_shakedrop import DeepBaselineNetBN3ResidualShakeDrop
from models.deep_baseline3_bn_residual_gap_gmp import (
    DeepBaselineNetBN3ResidualGAPGMP,
    DeepBaselineNetBN3ResidualGAPGMP_S3_F8_16_32_B2,
    DeepBaselineNetBN3ResidualGAPGMP_S3_F16_32_64_B3,
    DeepBaselineNetBN3ResidualGAPGMP_S3_F32_64_128_B5,
    DeepBaselineNetBN3ResidualGAPGMP_S3_F64_128_256_B5,
    DeepBaselineNetBN3ResidualGAPGMP_S4_F64_128_256_512_B5
)
from models.deep_baseline4_bn_residual import ResNet18
from models.mxresnet import MXResNet20, MXResNet32, MXResNet44, MXResNet56
from models.resnext import ResNeXt29_2x64d, ResNeXt29_4x64d, ResNeXt29_8x64d, ResNeXt29_32x4d
from models.wideresnet import wideresnet28_10, wideresnet16_8, WideResNet
from models.wideresnet_pyramid import (
    wideresnet28_10_pyramid, wideresnet16_8_pyramid,
    pyramidnet110_270, pyramidnet110_150, pyramidnet272_200_bottleneck
)
from models.dla import DLA
from models.convnext_step3_full import ConvNeXtCIFAR
from models.convnextv2 import convnext_v2_cifar_nano, convnext_v2_cifar_nano_k3
from models.rdnet import rdnet_tiny, rdnet_small, rdnet_base, rdnet_large
from main import get_net, get_available_nets


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
    deep_baseline, deep_baseline2/3, MXResNet, ResNeXt, WideResNet, DLA, ConvNeXt 및 RDNet 모델의 파라미터 개수를 계산합니다.
    
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
    
    # DeepBaselineNetBN3Residual15
    model_bn3_residual_15 = DeepBaselineNetBN3Residual15(init_weights=init_weights)
    results['deep_baseline3_bn_residual_15'] = count_parameters(model_bn3_residual_15)
    
    # DeepBaselineNetBN3Residual20
    model_bn3_residual_18 = DeepBaselineNetBN3Residual18(init_weights=init_weights)
    results['deep_baseline3_bn_residual_18'] = count_parameters(model_bn3_residual_18)
    
    # DeepBaselineNetBN3Residual15ConvNeXt
    model_bn3_residual_15_convnext = DeepBaselineNetBN3Residual15ConvNeXt(init_weights=init_weights)
    results['deep_baseline3_bn_residual_15_convnext'] = count_parameters(model_bn3_residual_15_convnext)
    
    # DeepBaselineNetBN3Residual15ConvNeXtLNClassifier
    model_bn3_residual_15_convnext_ln_classifier = DeepBaselineNetBN3Residual15ConvNeXtLNClassifier(init_weights=init_weights)
    results['deep_baseline3_bn_residual_15_convnext_ln_classifier'] = count_parameters(model_bn3_residual_15_convnext_ln_classifier)
    
    # DeepBaselineNetBN3Residual15ConvNeXtLNClassifierStem
    model_bn3_residual_15_convnext_ln_classifier_stem = DeepBaselineNetBN3Residual15ConvNeXtLNClassifierStem(init_weights=init_weights)
    results['deep_baseline3_bn_residual_15_convnext_ln_classifier_stem'] = count_parameters(model_bn3_residual_15_convnext_ln_classifier_stem)
    
    # DeepBaselineNetBN3Residual15LN
    model_bn3_residual_15_ln = DeepBaselineNetBN3Residual15LN(init_weights=init_weights)
    results['deep_baseline3_bn_residual_15_ln'] = count_parameters(model_bn3_residual_15_ln)
    
    # DeepBaselineNetBN3Residual15Attention
    model_bn3_residual_15_attention = DeepBaselineNetBN3Residual15Attention(init_weights=init_weights)
    results['deep_baseline3_bn_residual_15_attention'] = count_parameters(model_bn3_residual_15_attention)
    
    model_bn3_residual_15_attention_tiny = make_deep_baseline3_bn_residual_15_attention_tiny(init_weights=init_weights)
    results['deep_baseline3_bn_residual_15_attention_tiny'] = count_parameters(model_bn3_residual_15_attention_tiny)
    
    # ResidualAttentionModel_92_32input
    model_residual_attention_92_32input = ResidualAttentionModel_92_32input(init_weights=init_weights)
    results['residual_attention_92_32input'] = count_parameters(model_residual_attention_92_32input)
    
    model_residual_attention_92_32input_tiny = make_residual_attention_92_32input_tiny(init_weights=init_weights)
    results['residual_attention_92_32input_tiny'] = count_parameters(model_residual_attention_92_32input_tiny)
    
    # ResidualAttentionModel_92_32input_PreAct
    model_residual_attention_92_32input_preact = ResidualAttentionModel_92_32input_PreAct(init_weights=init_weights)
    results['residual_attention_92_32input_preact'] = count_parameters(model_residual_attention_92_32input_preact)
    
    model_residual_attention_92_32input_preact_tiny = make_residual_attention_92_32input_preact_tiny(init_weights=init_weights)
    results['residual_attention_92_32input_preact_tiny'] = count_parameters(model_residual_attention_92_32input_preact_tiny)
    
    # ResidualAttentionModel_92_32input_SE
    model_residual_attention_92_32input_se = ResidualAttentionModel_92_32input_SE(init_weights=init_weights)
    results['residual_attention_92_32input_se'] = count_parameters(model_residual_attention_92_32input_se)
    
    model_residual_attention_92_32input_se_tiny = make_residual_attention_92_32input_se_tiny(init_weights=init_weights)
    results['residual_attention_92_32input_se_tiny'] = count_parameters(model_residual_attention_92_32input_se_tiny)
    
    # ResidualAttentionModel_92_32input_GELU
    model_residual_attention_92_32input_gelu = ResidualAttentionModel_92_32input_GELU(init_weights=init_weights)
    results['residual_attention_92_32input_gelu'] = count_parameters(model_residual_attention_92_32input_gelu)
    
    model_residual_attention_92_32input_gelu_tiny = make_residual_attention_92_32input_gelu_tiny(init_weights=init_weights)
    results['residual_attention_92_32input_gelu_tiny'] = count_parameters(model_residual_attention_92_32input_gelu_tiny)
    
    model_residual_attention_92_32input_gelu_medium = make_residual_attention_92_32input_gelu_medium(init_weights=init_weights)
    results['residual_attention_92_32input_gelu_medium'] = count_parameters(model_residual_attention_92_32input_gelu_medium)
    
    model_residual_attention_92_32input_gelu_tiny_dla = make_residual_attention_92_32input_gelu_tiny_dla(init_weights=init_weights)
    results['residual_attention_92_32input_gelu_tiny_dla'] = count_parameters(model_residual_attention_92_32input_gelu_tiny_dla)
    
    model_residual_attention_92_32input_gelu_tiny_dla_tiny = make_residual_attention_92_32input_gelu_tiny_dla_tiny(init_weights=init_weights)
    results['residual_attention_92_32input_gelu_tiny_dla_tiny'] = count_parameters(model_residual_attention_92_32input_gelu_tiny_dla_tiny)
    
    model_residual_attention_92_32input_gelu_tiny_dla_small = make_residual_attention_92_32input_gelu_tiny_dla_small(init_weights=init_weights)
    results['residual_attention_92_32input_gelu_tiny_dla_small'] = count_parameters(model_residual_attention_92_32input_gelu_tiny_dla_small)
    
    model_residual_attention_92_32input_gelu_tiny_dla_base = make_residual_attention_92_32input_gelu_tiny_dla_base(init_weights=init_weights)
    results['residual_attention_92_32input_gelu_tiny_dla_base'] = count_parameters(model_residual_attention_92_32input_gelu_tiny_dla_base)
    
    model_residual_attention_92_32input_gelu_tiny_dla_large = make_residual_attention_92_32input_gelu_tiny_dla_large(init_weights=init_weights)
    results['residual_attention_92_32input_gelu_tiny_dla_large'] = count_parameters(model_residual_attention_92_32input_gelu_tiny_dla_large)
    
    model_residual_attention_92_32input_gelu_tiny_dla_wide = make_residual_attention_92_32input_gelu_tiny_dla_wide(init_weights=init_weights)
    results['residual_attention_92_32input_gelu_tiny_dla_wide'] = count_parameters(model_residual_attention_92_32input_gelu_tiny_dla_wide)
    
    model_residual_attention_92_32input_gelu_tiny_dla_deep = make_residual_attention_92_32input_gelu_tiny_dla_deep(init_weights=init_weights)
    results['residual_attention_92_32input_gelu_tiny_dla_deep'] = count_parameters(model_residual_attention_92_32input_gelu_tiny_dla_deep)
    
    # DeepBaselineNetBN3ResidualBottleneck
    model_bn3_residual_bottleneck = DeepBaselineNetBN3ResidualBottleneck(init_weights=init_weights)
    results['deep_baseline3_bn_residual_bottleneck'] = count_parameters(model_bn3_residual_bottleneck)
    
    # DeepBaselineNetBN3ResidualConvNeXt
    model_bn3_residual_convnext_stride = DeepBaselineNetBN3ResidualConvNeXt(init_weights=init_weights)
    results['deep_baseline3_bn_residual_convnext_stride'] = count_parameters(model_bn3_residual_convnext_stride)
    
    model_bn3_residual_wide = DeepBaselineNetBN3ResidualWide(init_weights=init_weights)
    results['deep_baseline3_bn_residual_wide'] = count_parameters(model_bn3_residual_wide)
    
    model_bn3_residual_4x = DeepBaselineNetBN3Residual4X(init_weights=init_weights)
    results['deep_baseline3_bn_residual_4x'] = count_parameters(model_bn3_residual_4x)
    
    model_bn3_residual_deep = DeepBaselineNetBN3ResidualDeep(init_weights=init_weights)
    results['deep_baseline3_bn_residual_deep'] = count_parameters(model_bn3_residual_deep)
    
    # DeepBaselineNetBN3ResidualShakeDrop
    model_bn3_residual_shakedrop = DeepBaselineNetBN3ResidualShakeDrop(init_weights=init_weights)
    results['deep_baseline3_bn_residual_shakedrop'] = count_parameters(model_bn3_residual_shakedrop)
    
    # DeepBaselineNetBN3ResidualDLA
    model_bn3_residual_dla = DeepBaselineNetBN3ResidualDLA(init_weights=init_weights)
    results['deep_baseline3_bn_residual_dla'] = count_parameters(model_bn3_residual_dla)

    # DeepBaselineNetBN3ResidualDLATree
    model_bn3_residual_dla_tree = DeepBaselineNetBN3ResidualDLATree(init_weights=init_weights)
    results['deep_baseline3_bn_residual_dla_tree'] = count_parameters(model_bn3_residual_dla_tree)
    
    # DeepBaselineNetBN3ResidualGAPGMP
    model_bn3_residual_gap_gmp = DeepBaselineNetBN3ResidualGAPGMP(init_weights=init_weights)
    results['deep_baseline3_bn_residual_gap_gmp'] = count_parameters(model_bn3_residual_gap_gmp)
    
    # DeepBaselineNetBN3ResidualGAPGMP 프리셋 모델들
    model_bn3_residual_gap_gmp_s3_f8_16_32_b2 = DeepBaselineNetBN3ResidualGAPGMP_S3_F8_16_32_B2(init_weights=init_weights)
    results['deep_baseline3_bn_residual_gap_gmp_s3_f8_16_32_b2'] = count_parameters(model_bn3_residual_gap_gmp_s3_f8_16_32_b2)
    
    model_bn3_residual_gap_gmp_s3_f16_32_64_b3 = DeepBaselineNetBN3ResidualGAPGMP_S3_F16_32_64_B3(init_weights=init_weights)
    results['deep_baseline3_bn_residual_gap_gmp_s3_f16_32_64_b3'] = count_parameters(model_bn3_residual_gap_gmp_s3_f16_32_64_b3)
    
    model_bn3_residual_gap_gmp_s3_f32_64_128_b5 = DeepBaselineNetBN3ResidualGAPGMP_S3_F32_64_128_B5(init_weights=init_weights)
    results['deep_baseline3_bn_residual_gap_gmp_s3_f32_64_128_b5'] = count_parameters(model_bn3_residual_gap_gmp_s3_f32_64_128_b5)
    
    model_bn3_residual_gap_gmp_s3_f64_128_256_b5 = DeepBaselineNetBN3ResidualGAPGMP_S3_F64_128_256_B5(init_weights=init_weights)
    results['deep_baseline3_bn_residual_gap_gmp_s3_f64_128_256_b5'] = count_parameters(model_bn3_residual_gap_gmp_s3_f64_128_256_b5)
    
    model_bn3_residual_gap_gmp_s4_f64_128_256_512_b5 = DeepBaselineNetBN3ResidualGAPGMP_S4_F64_128_256_512_B5(init_weights=init_weights)
    results['deep_baseline3_bn_residual_gap_gmp_s4_f64_128_256_512_b5'] = count_parameters(model_bn3_residual_gap_gmp_s4_f64_128_256_512_b5)
    
    # DeepBaselineNetBN4Residual
    model_bn4_residual = ResNet18(init_weights=init_weights)
    results['deep_baseline4_bn_residual'] = count_parameters(model_bn4_residual)
        
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
    
    # WideResNet models
    model_wideresnet28_10 = wideresnet28_10()
    results['wideresnet28_10'] = count_parameters(model_wideresnet28_10)
    
    model_wideresnet16_8 = wideresnet16_8()
    results['wideresnet16_8'] = count_parameters(model_wideresnet16_8)
    
    # WideResNet with remove_first_relu
    model_wideresnet28_10_remove_first_relu = wideresnet28_10(remove_first_relu=True)
    results['wideresnet28_10_remove_first_relu'] = count_parameters(model_wideresnet28_10_remove_first_relu)
    
    model_wideresnet16_8_remove_first_relu = wideresnet16_8(remove_first_relu=True)
    results['wideresnet16_8_remove_first_relu'] = count_parameters(model_wideresnet16_8_remove_first_relu)
    
    # WideResNet with last_batch_norm and remove_first_relu
    model_wideresnet28_10_last_bn_remove_first_relu = wideresnet28_10(last_batch_norm=True, remove_first_relu=True)
    results['wideresnet28_10_last_bn_remove_first_relu'] = count_parameters(model_wideresnet28_10_last_bn_remove_first_relu)
    
    model_wideresnet16_8_last_bn_remove_first_relu = wideresnet16_8(last_batch_norm=True, remove_first_relu=True)
    results['wideresnet16_8_last_bn_remove_first_relu'] = count_parameters(model_wideresnet16_8_last_bn_remove_first_relu)
    
    model_wideresnet28_10_fullpyramid = wideresnet28_10_pyramid()
    results['wideresnet28_10_pyramid'] = count_parameters(model_wideresnet28_10_fullpyramid)
    
    model_wideresnet16_8_fullpyramid = wideresnet16_8_pyramid()
    results['wideresnet16_8_pyramid'] = count_parameters(model_wideresnet16_8_fullpyramid)
    
    # PyramidNet-110
    model_pyramidnet110_270 = pyramidnet110_270()
    results['pyramidnet110_270'] = count_parameters(model_pyramidnet110_270)
    
    # PyramidNet-110 with alpha=150 (~10M parameters, depth=110으로 변경됨)
    model_pyramidnet110_150 = pyramidnet110_150()
    results['pyramidnet110_150'] = count_parameters(model_pyramidnet110_150)
    
    # PyramidNet-272 with bottleneck structure and alpha=200
    model_pyramidnet272_200_bottleneck = pyramidnet272_200_bottleneck()
    results['pyramidnet272_200_bottleneck'] = count_parameters(model_pyramidnet272_200_bottleneck)
    
    # DLA model
    model_dla = DLA()
    results['dla'] = count_parameters(model_dla)
    
    # ConvNeXtCIFAR model
    model_convnext = ConvNeXtCIFAR(init_weights=init_weights)
    results['convnext_step3_full'] = count_parameters(model_convnext)
    
    # ConvNeXt V2 model
    model_convnext_v2_nano = convnext_v2_cifar_nano()
    results['convnext_v2_cifar_nano'] = count_parameters(model_convnext_v2_nano)
    
    model_convnext_v2_nano_k3 = convnext_v2_cifar_nano_k3()
    results['convnext_v2_cifar_nano_k3'] = count_parameters(model_convnext_v2_nano_k3)
    
    # RDNet models
    model_rdnet_tiny = rdnet_tiny(pretrained=False, num_classes=10)
    results['rdnet_tiny'] = count_parameters(model_rdnet_tiny)
    
    model_rdnet_small = rdnet_small(pretrained=False, num_classes=10)
    results['rdnet_small'] = count_parameters(model_rdnet_small)
    
    model_rdnet_base = rdnet_base(pretrained=False, num_classes=10)
    results['rdnet_base'] = count_parameters(model_rdnet_base)
    
    model_rdnet_large = rdnet_large(pretrained=False, num_classes=10)
    results['rdnet_large'] = count_parameters(model_rdnet_large)
    
    return results


def get_model_parameter_count(model_name: str, init_weights: bool = False, shakedrop_prob: float = 0.0):
    """
    특정 모델의 파라미터 개수를 계산합니다.
    
    Args:
        model_name: 모델 이름
        init_weights: 가중치 초기화 여부 (기본값: False)
        shakedrop_prob: ShakeDrop 확률 (기본값: 0.0)
        
    Returns:
        int: 학습 가능한 파라미터의 총 개수
    """
    try:
        model = get_net(model_name, init_weights=init_weights, shakedrop_prob=shakedrop_prob)
        return count_parameters(model)
    except Exception as e:
        print(f"[오류] 모델 '{model_name}' 생성 실패: {e}")
        return None


def get_models_parameter_counts(model_names: list = None, init_weights: bool = False, shakedrop_prob: float = 0.0):
    """
    지정된 모델들의 파라미터 개수를 계산합니다.
    
    Args:
        model_names: 모델 이름 목록 (None이면 모든 모델)
        init_weights: 가중치 초기화 여부 (기본값: False)
        shakedrop_prob: ShakeDrop 확률 (기본값: 0.0)
        
    Returns:
        dict: 모델 이름을 키로 하고 파라미터 개수를 값으로 하는 딕셔너리
    """
    results = {}
    
    # 모델 이름 목록이 없으면 모든 모델 사용
    if model_names is None:
        model_names = get_available_nets()
    
    for model_name in model_names:
        param_count = get_model_parameter_count(model_name, init_weights=init_weights, shakedrop_prob=shakedrop_prob)
        if param_count is not None:
            results[model_name] = param_count
    
    return results


def print_deep_baseline2_parameter_counts(init_weights=False, model_name: str = None, shakedrop_prob: float = 0.0):
    """
    deep_baseline, deep_baseline2/3, MXResNet, ResNeXt, WideResNet, DLA, ConvNeXt 및 RDNet 모델의 파라미터 개수를 출력합니다.
    
    Args:
        init_weights: 가중치 초기화 여부 (기본값: False)
        model_name: 특정 모델 이름 (None이면 모든 모델)
        shakedrop_prob: ShakeDrop 확률 (기본값: 0.0)
    """
    if model_name:
        # 특정 모델만 계산
        model_names = [model_name]
    else:
        # 모든 모델 계산
        model_names = None
    
    results = get_models_parameter_counts(model_names=model_names, init_weights=init_weights, shakedrop_prob=shakedrop_prob)
    
    print("=" * 70)
    if model_name:
        print(f"모델 '{model_name}' 파라미터 개수")
    else:
        print("deep_baseline, deep_baseline2/3, MXResNet, ResNeXt, WideResNet, DLA, ConvNeXt 및 RDNet 모델 파라미터 개수")
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


def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(description='모델 파라미터 개수 계산')
    parser.add_argument('--model', type=str, default=None,
                        help='계산할 모델 이름 (지정하지 않으면 모든 모델)')
    parser.add_argument('--init-weights', action='store_true',
                        help='가중치 초기화 사용 (default: False)')
    parser.add_argument('--shakedrop', type=float, default=0.0,
                        help='ShakeDrop 확률 (0.0~1.0, WideResNet/PyramidNet 모델에만 적용, default: 0.0)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
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
    
    # 파라미터 개수 출력
    print_deep_baseline2_parameter_counts(init_weights=args.init_weights, model_name=model_name, shakedrop_prob=args.shakedrop)

