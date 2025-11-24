from models.baseline import BaselineNet
from models.baseline_bn import BaselineNetBN
from models.convnext_step1_patchify import ConvNeXtPatchifyClassifier
from models.convnext_step2_local_block import ConvNeXtLocalBlockClassifier
from models.convnext_step3_full import ConvNeXtCIFAR, convnext_tiny
from models.convnextv2 import convnext_v2_cifar_nano, convnext_v2_cifar_nano_k3
from models.cvt import cvt_10m
from models.deep_baseline import DeepBaselineNet
from models.deep_baseline2_bn import DeepBaselineNetBN2
from models.deep_baseline2_bn_residual import DeepBaselineNetBN2Residual
from models.deep_baseline2_bn_residual_grn import DeepBaselineNetBN2ResidualGRN
from models.deep_baseline2_bn_residual_preact import DeepBaselineNetBN2ResidualPreAct
from models.deep_baseline2_bn_residual_se import DeepBaselineNetBN2ResidualSE
from models.deep_baseline2_bn_resnext import DeepBaselineNetBN2ResNeXt
from models.deep_baseline3_bn import DeepBaselineNetBN3
from models.deep_baseline3_bn_residual import DeepBaselineNetBN3Residual
from models.deep_baseline3_bn_residual_15 import DeepBaselineNetBN3Residual15
from models.deep_baseline3_bn_residual_15_attention import DeepBaselineNetBN3Residual15Attention, make_deep_baseline3_bn_residual_15_attention_tiny
from models.deep_baseline3_bn_residual_15_convnext import DeepBaselineNetBN3Residual15ConvNeXt
from models.deep_baseline3_bn_residual_15_convnext_ln_classifier import DeepBaselineNetBN3Residual15ConvNeXtLNClassifier
from models.deep_baseline3_bn_residual_15_convnext_ln_classifier_stem import DeepBaselineNetBN3Residual15ConvNeXtLNClassifierStem
from models.deep_baseline3_bn_residual_15_ln import DeepBaselineNetBN3Residual15LN
from models.deep_baseline3_bn_residual_18 import DeepBaselineNetBN3Residual18
from models.deep_baseline3_bn_residual_4x import DeepBaselineNetBN3Residual4X
from models.deep_baseline3_bn_residual_bottleneck import DeepBaselineNetBN3ResidualBottleneck
from models.deep_baseline3_bn_residual_deep import DeepBaselineNetBN3ResidualDeep
from models.deep_baseline3_bn_residual_dla import DeepBaselineNetBN3ResidualDLA
from models.deep_baseline3_bn_residual_dla_tree import DeepBaselineNetBN3ResidualDLATree
from models.deep_baseline3_bn_residual_gap_gmp import DeepBaselineNetBN3ResidualGAPGMP, DeepBaselineNetBN3ResidualGAPGMP_S3_F16_32_64_B3, DeepBaselineNetBN3ResidualGAPGMP_S3_F32_64_128_B5, DeepBaselineNetBN3ResidualGAPGMP_S3_F64_128_256_B5, DeepBaselineNetBN3ResidualGAPGMP_S3_F8_16_32_B2, DeepBaselineNetBN3ResidualGAPGMP_S4_F64_128_256_512_B5
from models.deep_baseline3_bn_residual_group import DeepBaselineNetBN3ResidualGroup
from models.deep_baseline3_bn_residual_mish import DeepBaselineNetBN3ResidualMish
from models.deep_baseline3_bn_residual_preact import DeepBaselineNetBN3ResidualPreAct
from models.deep_baseline3_bn_residual_shakedrop import DeepBaselineNetBN3ResidualShakeDrop
from models.deep_baseline3_bn_residual_swiglu import DeepBaselineNetBN3ResidualSwiGLU
from models.deep_baseline3_bn_residual_swish import DeepBaselineNetBN3ResidualSwish
from models.deep_baseline3_bn_residual_wide import DeepBaselineNetBN3ResidualWide
from models.deep_baseline3_convnext_stride import DeepBaselineNetBN3ResidualConvNeXt
from models.deep_baseline4_bn_residual import ResNet18 as DeepBaselineNetBN4Residual
from models.deep_baseline4_bn_residual_shakedrop import ResNet18ShakeDrop as DeepBaselineNetBN4ResidualShakeDrop
from models.deep_baseline_bn import DeepBaselineNetBN
from models.deep_baseline_bn_dropout import DeepBaselineNetBNDropout
from models.deep_baseline_bn_dropout_resnet import DeepBaselineNetBNDropoutResNet
from models.deep_baseline_gap import DeepBaselineNetGAP
from models.deep_baseline_se import DeepBaselineNetSE
from models.deep_baseline_silu import DeepBaselineNetSilu
from models.densenet import DenseNet121
from models.dla import DLA
from models.mobilenetv2 import MobileNetV2
from models.mxresnet import MXResNet20, MXResNet32, MXResNet44, MXResNet56
from models.rdnet import rdnet_base, rdnet_large, rdnet_small, rdnet_tiny
from models.residual_attention_92_32input import ResidualAttentionModel_92_32input, make_residual_attention_92_32input_tiny
from models.residual_attention_92_32input_gelu import ResidualAttentionModel_92_32input_GELU, make_residual_attention_92_32input_gelu_tiny
from models.residual_attention_92_32input_gelu_medium import make_residual_attention_92_32input_gelu_medium
from models.residual_attention_92_32input_gelu_tiny_dla import make_residual_attention_92_32input_gelu_tiny_dla, make_residual_attention_92_32input_gelu_tiny_dla_base, make_residual_attention_92_32input_gelu_tiny_dla_deep, make_residual_attention_92_32input_gelu_tiny_dla_large, make_residual_attention_92_32input_gelu_tiny_dla_small, make_residual_attention_92_32input_gelu_tiny_dla_tiny, make_residual_attention_92_32input_gelu_tiny_dla_wide
from models.residual_attention_92_32input_preact import ResidualAttentionModel_92_32input_PreAct, make_residual_attention_92_32input_preact_tiny
from models.residual_attention_92_32input_se import ResidualAttentionModel_92_32input_SE, make_residual_attention_92_32input_se_tiny
from models.resnet import ResNet18
from models.resnext import ResNeXt29_4x64d
from models.vgg import VGG
from models.wideresnet import wideresnet16_8, wideresnet28_10
from models.wideresnet_pyramid import pyramidnet110_150, pyramidnet110_270, pyramidnet272_200_bottleneck, wideresnet16_8_pyramid, wideresnet28_10_pyramid


def _get_nets_dict(init_weights: bool = False, shakedrop_prob: float = 0.0):
    return {
        'baseline': BaselineNet(init_weights=init_weights),
        'baseline_bn': BaselineNetBN(init_weights=init_weights),
        'deep_baseline': DeepBaselineNet(init_weights=init_weights),
        'deep_baseline_silu': DeepBaselineNetSilu(),
        'deep_baseline_bn': DeepBaselineNetBN(init_weights=init_weights),
        'deep_baseline_gap': DeepBaselineNetGAP(),
        'deep_baseline_bn_dropout': DeepBaselineNetBNDropout(),
        'deep_baseline_bn_dropout_resnet': DeepBaselineNetBNDropoutResNet(),
        'deep_baseline2_bn': DeepBaselineNetBN2(init_weights=init_weights),
        'deep_baseline2_bn_residual': DeepBaselineNetBN2Residual(init_weights=init_weights),
        'deep_baseline2_bn_residual_se': DeepBaselineNetBN2ResidualSE(init_weights=init_weights),
        'deep_baseline2_bn_resnext': DeepBaselineNetBN2ResNeXt(init_weights=init_weights),
        'deep_baseline2_bn_residual_preact': DeepBaselineNetBN2ResidualPreAct(),
        'deep_baseline2_bn_residual_grn': DeepBaselineNetBN2ResidualGRN(init_weights=init_weights),
        'deep_baseline3_bn': DeepBaselineNetBN3(init_weights=init_weights),
        'deep_baseline3_bn_residual': DeepBaselineNetBN3Residual(init_weights=init_weights),
        'deep_baseline3_bn_residual_15': DeepBaselineNetBN3Residual15(init_weights=init_weights),
        'deep_baseline3_bn_residual_18': DeepBaselineNetBN3Residual18(init_weights=init_weights),
        'deep_baseline3_bn_residual_15_convnext': DeepBaselineNetBN3Residual15ConvNeXt(init_weights=init_weights),
        'deep_baseline3_bn_residual_15_convnext_ln_classifier': DeepBaselineNetBN3Residual15ConvNeXtLNClassifier(init_weights=init_weights),
        'deep_baseline3_bn_residual_15_convnext_ln_classifier_stem': DeepBaselineNetBN3Residual15ConvNeXtLNClassifierStem(init_weights=init_weights),
        'deep_baseline3_bn_residual_15_attention': DeepBaselineNetBN3Residual15Attention(init_weights=init_weights),
        'deep_baseline3_bn_residual_15_attention_tiny': make_deep_baseline3_bn_residual_15_attention_tiny(init_weights=init_weights),
        'residual_attention_92_32input': ResidualAttentionModel_92_32input(init_weights=init_weights),
        'residual_attention_92_32input_tiny': make_residual_attention_92_32input_tiny(init_weights=init_weights),
        'residual_attention_92_32input_preact': ResidualAttentionModel_92_32input_PreAct(init_weights=init_weights),
        'residual_attention_92_32input_preact_tiny': make_residual_attention_92_32input_preact_tiny(init_weights=init_weights),
        'residual_attention_92_32input_se': ResidualAttentionModel_92_32input_SE(init_weights=init_weights),
        'residual_attention_92_32input_se_tiny': make_residual_attention_92_32input_se_tiny(init_weights=init_weights),
        'residual_attention_92_32input_gelu': ResidualAttentionModel_92_32input_GELU(init_weights=init_weights),
        'residual_attention_92_32input_gelu_tiny': make_residual_attention_92_32input_gelu_tiny(init_weights=init_weights),
        'residual_attention_92_32input_gelu_medium': make_residual_attention_92_32input_gelu_medium(init_weights=init_weights),
        'residual_attention_92_32input_gelu_tiny_dla': make_residual_attention_92_32input_gelu_tiny_dla(init_weights=init_weights),
        'residual_attention_92_32input_gelu_tiny_dla_tiny': make_residual_attention_92_32input_gelu_tiny_dla_tiny(init_weights=init_weights),
        'residual_attention_92_32input_gelu_tiny_dla_small': make_residual_attention_92_32input_gelu_tiny_dla_small(init_weights=init_weights),
        'residual_attention_92_32input_gelu_tiny_dla_base': make_residual_attention_92_32input_gelu_tiny_dla_base(init_weights=init_weights),
        'residual_attention_92_32input_gelu_tiny_dla_large': make_residual_attention_92_32input_gelu_tiny_dla_large(init_weights=init_weights),
        'residual_attention_92_32input_gelu_tiny_dla_wide': make_residual_attention_92_32input_gelu_tiny_dla_wide(init_weights=init_weights),
        'residual_attention_92_32input_gelu_tiny_dla_deep': make_residual_attention_92_32input_gelu_tiny_dla_deep(init_weights=init_weights),
        'deep_baseline3_bn_residual_15_ln': DeepBaselineNetBN3Residual15LN(init_weights=init_weights),
        'deep_baseline3_bn_residual_bottleneck': DeepBaselineNetBN3ResidualBottleneck(init_weights=init_weights),
        'deep_baseline3_bn_residual_convnext_stride': DeepBaselineNetBN3ResidualConvNeXt(init_weights=init_weights),
        'deep_baseline3_bn_residual_convnext_stride_k3': DeepBaselineNetBN3ResidualConvNeXt(kernel_size=3, init_weights=init_weights),
        'deep_baseline3_bn_residual_wide': DeepBaselineNetBN3ResidualWide(init_weights=init_weights),
        'deep_baseline3_bn_residual_4x': DeepBaselineNetBN3Residual4X(init_weights=init_weights),
        'deep_baseline3_bn_residual_deep': DeepBaselineNetBN3ResidualDeep(init_weights=init_weights),
        'deep_baseline3_bn_residual_preact': DeepBaselineNetBN3ResidualPreAct(init_weights=init_weights),
        'deep_baseline3_bn_residual_swish': DeepBaselineNetBN3ResidualSwish(init_weights=init_weights),
        'deep_baseline3_bn_residual_swiglu': DeepBaselineNetBN3ResidualSwiGLU(init_weights=init_weights),
        'deep_baseline3_bn_residual_dla': DeepBaselineNetBN3ResidualDLA(init_weights=init_weights),
        'deep_baseline3_bn_residual_dla_tree': DeepBaselineNetBN3ResidualDLATree(init_weights=init_weights),
        'deep_baseline3_bn_residual_group': DeepBaselineNetBN3ResidualGroup(init_weights=init_weights),
        'deep_baseline3_bn_residual_shakedrop': DeepBaselineNetBN3ResidualShakeDrop(init_weights=init_weights),
        'deep_baseline3_bn_residual_mish': DeepBaselineNetBN3ResidualMish(init_weights=init_weights),
        'deep_baseline3_bn_residual_gap_gmp': DeepBaselineNetBN3ResidualGAPGMP(init_weights=init_weights),
        'deep_baseline3_bn_residual_gap_gmp_s3_f8_16_32_b2': DeepBaselineNetBN3ResidualGAPGMP_S3_F8_16_32_B2(init_weights=init_weights),
        'deep_baseline3_bn_residual_gap_gmp_s3_f16_32_64_b3': DeepBaselineNetBN3ResidualGAPGMP_S3_F16_32_64_B3(init_weights=init_weights),
        'deep_baseline3_bn_residual_gap_gmp_s3_f32_64_128_b5': DeepBaselineNetBN3ResidualGAPGMP_S3_F32_64_128_B5(init_weights=init_weights),
        'deep_baseline3_bn_residual_gap_gmp_s3_f64_128_256_b5': DeepBaselineNetBN3ResidualGAPGMP_S3_F64_128_256_B5(init_weights=init_weights),
        'deep_baseline3_bn_residual_gap_gmp_s4_f64_128_256_512_b5': DeepBaselineNetBN3ResidualGAPGMP_S4_F64_128_256_512_B5(init_weights=init_weights),
        'deep_baseline4_bn_residual': DeepBaselineNetBN4Residual(init_weights=init_weights),
        'deep_baseline4_bn_residual_shakedrop': DeepBaselineNetBN4ResidualShakeDrop(init_weights=init_weights),
        'deep_baseline_se': DeepBaselineNetSE(),
        'convnext_patchify': ConvNeXtPatchifyClassifier(init_weights=init_weights),
        'convnext_local': ConvNeXtLocalBlockClassifier(init_weights=init_weights),
        'convnext_cifar': ConvNeXtCIFAR(init_weights=init_weights),
        'convnext_tiny': convnext_tiny(init_weights=init_weights),
        'convnext_v2_cifar_nano': convnext_v2_cifar_nano(),
        'convnext_v2_cifar_nano_k3': convnext_v2_cifar_nano_k3(),
        'cvt_10m': cvt_10m(init_weights=init_weights),
        'resnet18': ResNet18(),
        'vgg16': VGG('VGG16'),
        'mobilenetv2': MobileNetV2(),
        'densenet121': DenseNet121(),
        'mxresnet20': MXResNet20(init_weights=init_weights),
        'mxresnet32': MXResNet32(init_weights=init_weights),
        'mxresnet44': MXResNet44(init_weights=init_weights),
        'mxresnet56': MXResNet56(init_weights=init_weights),
        'dla': DLA(),
        'resnext29_4x64d': ResNeXt29_4x64d(),
        'wideresnet28_10': wideresnet28_10(shakedrop_prob=shakedrop_prob),
        'wideresnet16_8': wideresnet16_8(shakedrop_prob=shakedrop_prob),
        'wideresnet28_10_remove_first_relu': wideresnet28_10(shakedrop_prob=shakedrop_prob, remove_first_relu=True),
        'wideresnet28_10_last_bn_remove_first_relu': wideresnet28_10(shakedrop_prob=shakedrop_prob, last_batch_norm=True, remove_first_relu=True),
        'wideresnet16_8_remove_first_relu': wideresnet16_8(shakedrop_prob=shakedrop_prob, remove_first_relu=True),
        'wideresnet16_8_last_bn_remove_first_relu': wideresnet16_8(shakedrop_prob=shakedrop_prob, last_batch_norm=True, remove_first_relu=True),
        'wideresnet28_10_pyramid': wideresnet28_10_pyramid(shakedrop_prob=shakedrop_prob),
        'wideresnet16_8_pyramid': wideresnet16_8_pyramid(shakedrop_prob=shakedrop_prob),
        'pyramidnet110_270': pyramidnet110_270(shakedrop_prob=shakedrop_prob),
        'pyramidnet110_150': pyramidnet110_150(shakedrop_prob=shakedrop_prob),
        'pyramidnet272_200_bottleneck': pyramidnet272_200_bottleneck(shakedrop_prob=shakedrop_prob),
        'rdnet_tiny': rdnet_tiny(pretrained=False, num_classes=10),
        'rdnet_small': rdnet_small(pretrained=False, num_classes=10),
        'rdnet_base': rdnet_base(pretrained=False, num_classes=10),
        'rdnet_large': rdnet_large(pretrained=False, num_classes=10),
    }


def get_available_nets():
    """사용 가능한 네트워크 이름 목록 반환"""
    nets_dict = _get_nets_dict(init_weights=False)
    return list(nets_dict.keys())


def get_net(name: str, init_weights: bool = False, shakedrop_prob: float = 0.0):
    """Network 팩토리 함수"""
    nets = _get_nets_dict(init_weights=init_weights, shakedrop_prob=shakedrop_prob)
    if name.lower() not in nets:
        raise ValueError(
            f"Unknown net: {name}. Available: {list(nets.keys())}")
    return nets[name.lower()]