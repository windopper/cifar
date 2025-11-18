"""
ResidualAttentionModel_92_32input_GELU_Tiny_DLA
===============================================

Residual Attention Tiny 모델에 DLA(Tree) 구조를 결합한 변형.
각 stage는 ResidualBlock + AttentionModule(GELU) 구성으로 유지하면서,
DLA의 계층적 집계를 통해 다중 해상도 특징을 효율적으로 통합합니다.

설계 개요:
- conv1: 3 -> base_channels, 32x32 유지
- stage1: ResidualBlock(base_channels -> stage1_channels) + Stage1 Attention (32x32)
- DLA Stages: Tree 기반 채널 집계 + Stage2/3 Attention
- Global head: BN + GELU + AdaptiveAvgPool2d + FC(num_classes)

매개변수화 가능한 요소:
- base_channels: 초기 채널 수 (기본값: 32)
- stage1_channels: Stage1 채널 수 (기본값: 64)
- dla_stage_channels: 각 DLA stage의 채널 수 리스트 (기본값: [128, 256, 512])
- dla_levels: 각 DLA stage의 level 값 리스트 (기본값: [2, 2, 1])
- dla_strides: 각 DLA stage의 stride 값 리스트 (기본값: [1, 2, 1])
- use_attention: 각 stage에서 attention 사용 여부 리스트 (기본값: 모두 True)
- stage1_stride: Stage1의 stride (기본값: 1)
- pre_stage2_stride: Stage2 전 다운샘플링 stride (기본값: 2)
"""

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.residual_block_gelu import ResidualBlock
from .modules import (
    AttentionModuleStage1CIFAR_GELU,
    AttentionModuleStage2CIFAR_GELU,
    AttentionModuleStage3CIFAR_GELU,
)


class RootGELU(nn.Module):
    """여러 노드 출력을 concat 후 1x1 Conv + GELU로 집계."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        x = torch.cat(xs, dim=1)
        out = self.bn(self.conv(x))
        return F.gelu(out)


class DLATree(nn.Module):
    """
    ResidualBlock 기반 DLA Tree (GELU 버전).

    attention_module이 주어지면 root 집계 이후 attention을 적용한다.
    """

    def __init__(
        self,
        block: type[nn.Module],
        in_channels: int,
        out_channels: int,
        level: int = 1,
        stride: int = 1,
        attention_module: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.level = level
        self.attention_module = attention_module

        if level == 1:
            self.root = RootGELU(2 * out_channels, out_channels)
            self.left_node = block(in_channels, out_channels, stride=stride)
            self.right_node = block(out_channels, out_channels, stride=1)
        else:
            self.root = RootGELU((level + 2) * out_channels, out_channels)
            self.sub_trees = nn.ModuleList(
                [
                    DLATree(block, in_channels, out_channels, level=i, stride=stride)
                    for i in reversed(range(1, level))
                ]
            )
            self.prev_root = block(in_channels, out_channels, stride=stride)
            self.left_node = block(out_channels, out_channels, stride=1)
            self.right_node = block(out_channels, out_channels, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = []
        out = x
        if self.level > 1:
            prev = self.prev_root(x)
            xs.append(prev)
            for tree in self.sub_trees:
                out = tree(out)
                xs.append(out)
        out = self.left_node(out)
        xs.append(out)
        out = self.right_node(out)
        xs.append(out)
        out = self.root(xs)
        if self.attention_module is not None:
            out = self.attention_module(out)
        return out


class ResidualAttentionModel_92_32input_GELU_Tiny_DLA(nn.Module):
    """
    Residual Attention Tiny + DLA(Tree) 하이브리드 모델 (매개변수화 버전).

    - Stage1: ResidualBlock + Stage1 Attention (32x32)
    - DLA Stages: Tree 기반 채널 집계 + Stage2/3 Attention
    - 마지막은 Global AvgPool + FC 로 CIFAR-10 분류

    Args:
        num_classes: 분류 클래스 수 (기본값: 10)
        init_weights: 가중치 초기화 여부 (기본값: False)
        base_channels: 초기 채널 수 (기본값: 32)
        stage1_channels: Stage1 채널 수 (기본값: 64)
        dla_stage_channels: 각 DLA stage의 채널 수 리스트 (기본값: [128, 256, 512])
        dla_levels: 각 DLA stage의 level 값 리스트 (기본값: [2, 2, 1])
        dla_strides: 각 DLA stage의 stride 값 리스트 (기본값: [1, 2, 1])
        use_attention: 각 stage에서 attention 사용 여부 리스트 (기본값: 모두 True)
        stage1_stride: Stage1의 stride (기본값: 1)
        pre_stage2_stride: Stage2 전 다운샘플링 stride (기본값: 2)
    """

    def __init__(
        self,
        num_classes: int = 10,
        init_weights: bool = False,
        base_channels: int = 32,
        stage1_channels: int = 64,
        dla_stage_channels: List[int] = None,
        dla_levels: List[int] = None,
        dla_strides: List[int] = None,
        use_attention: List[bool] = None,
        stage1_stride: int = 1,
        pre_stage2_stride: int = 2,
    ):
        super().__init__()
        
        # 기본값 설정
        if dla_stage_channels is None:
            dla_stage_channels = [128, 256, 512]
        if dla_levels is None:
            dla_levels = [2, 2, 1]
        if dla_strides is None:
            dla_strides = [1, 2, 1]
        if use_attention is None:
            use_attention = [True] * (1 + len(dla_stage_channels))  # Stage1 + DLA stages
        
        # 검증
        num_dla_stages = len(dla_stage_channels)
        if len(dla_levels) != num_dla_stages:
            raise ValueError(f"dla_levels 길이({len(dla_levels)})가 dla_stage_channels 길이({num_dla_stages})와 일치하지 않습니다.")
        if len(dla_strides) != num_dla_stages:
            raise ValueError(f"dla_strides 길이({len(dla_strides)})가 dla_stage_channels 길이({num_dla_stages})와 일치하지 않습니다.")
        if len(use_attention) != 1 + num_dla_stages:
            raise ValueError(f"use_attention 길이({len(use_attention)})가 1 + dla_stage_channels 길이({1 + num_dla_stages})와 일치하지 않습니다.")
        
        self.base_channels = base_channels
        self.stage1_channels = stage1_channels
        self.dla_stage_channels = dla_stage_channels
        self.num_dla_stages = num_dla_stages
        
        # 초기 feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
        )

        # Stage1: ResidualBlock + Stage1 Attention
        self.residual_block1 = ResidualBlock(base_channels, stage1_channels, stride=stage1_stride)
        
        # Stage1의 공간 크기 계산 (stride에 따라)
        stage1_size = (32, 32) if stage1_stride == 1 else (16, 16)
        stage1_size2 = (16, 16) if stage1_stride == 1 else (8, 8)
        
        if use_attention[0]:
            self.attention_module1 = AttentionModuleStage1CIFAR_GELU(
                stage1_channels, stage1_channels, size1=stage1_size, size2=stage1_size2
            )
        else:
            self.attention_module1 = None

        # Stage2 전 다운샘플링
        current_channels = stage1_channels
        current_size = stage1_size[0] // pre_stage2_stride
        
        if pre_stage2_stride > 1:
            self.residual_block2 = ResidualBlock(
                stage1_channels, dla_stage_channels[0], stride=pre_stage2_stride
            )
            current_channels = dla_stage_channels[0]
        else:
            # stride=1이면 채널만 변경
            if stage1_channels != dla_stage_channels[0]:
                self.residual_block2 = ResidualBlock(
                    stage1_channels, dla_stage_channels[0], stride=1
                )
                current_channels = dla_stage_channels[0]
            else:
                self.residual_block2 = None

        # DLA Stages
        self.dla_stages = nn.ModuleList()
        for i in range(num_dla_stages):
            in_channels = current_channels
            out_channels = dla_stage_channels[i]
            level = dla_levels[i]
            stride = dla_strides[i]
            
            # DLA Tree 출력 후의 실제 크기 계산 (stride 적용 후)
            output_size_after_stride = current_size // stride if stride > 1 else current_size
            
            # Attention module 생성 (DLA Tree 출력 후의 실제 크기 사용)
            attention_module = None
            if use_attention[1 + i]:
                if output_size_after_stride == 32 or output_size_after_stride == 16:
                    # Stage2 Attention (16x16 또는 32x32)
                    attention_module = AttentionModuleStage2CIFAR_GELU(
                        out_channels, out_channels, size=(output_size_after_stride, output_size_after_stride)
                    )
                else:
                    # Stage3 Attention (8x8 이하)
                    attention_module = AttentionModuleStage3CIFAR_GELU(
                        out_channels, out_channels
                    )
            
            dla_stage = DLATree(
                ResidualBlock,
                in_channels=in_channels,
                out_channels=out_channels,
                level=level,
                stride=stride,
                attention_module=attention_module,
            )
            self.dla_stages.append(dla_stage)
            
            # 다음 stage를 위한 채널 및 크기 업데이트
            current_channels = out_channels
            current_size = output_size_after_stride

        # Global pooling 및 분류기
        final_channels = dla_stage_channels[-1]
        self.global_pool = nn.Sequential(
            nn.BatchNorm2d(final_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(final_channels, num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.residual_block1(out)
        if self.attention_module1 is not None:
            out = self.attention_module1(out)

        if self.residual_block2 is not None:
            out = self.residual_block2(out)
        
        for dla_stage in self.dla_stages:
            out = dla_stage(out)

        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def make_residual_attention_92_32input_gelu_tiny_dla(
    num_classes: int = 10,
    init_weights: bool = False,
    base_channels: int = 32,
    stage1_channels: int = 64,
    dla_stage_channels: List[int] = None,
    dla_levels: List[int] = None,
    dla_strides: List[int] = None,
    use_attention: List[bool] = None,
    stage1_stride: int = 1,
    pre_stage2_stride: int = 2,
) -> ResidualAttentionModel_92_32input_GELU_Tiny_DLA:
    """
    Residual Attention Tiny DLA 모델 생성 팩토리 함수.
    
    Args:
        num_classes: 분류 클래스 수 (기본값: 10)
        init_weights: 가중치 초기화 여부 (기본값: False)
        base_channels: 초기 채널 수 (기본값: 32)
        stage1_channels: Stage1 채널 수 (기본값: 64)
        dla_stage_channels: 각 DLA stage의 채널 수 리스트 (기본값: [128, 256, 512])
        dla_levels: 각 DLA stage의 level 값 리스트 (기본값: [2, 2, 1])
        dla_strides: 각 DLA stage의 stride 값 리스트 (기본값: [1, 2, 1])
        use_attention: 각 stage에서 attention 사용 여부 리스트 (기본값: 모두 True)
        stage1_stride: Stage1의 stride (기본값: 1)
        pre_stage2_stride: Stage2 전 다운샘플링 stride (기본값: 2)
    """
    return ResidualAttentionModel_92_32input_GELU_Tiny_DLA(
        num_classes=num_classes,
        init_weights=init_weights,
        base_channels=base_channels,
        stage1_channels=stage1_channels,
        dla_stage_channels=dla_stage_channels,
        dla_levels=dla_levels,
        dla_strides=dla_strides,
        use_attention=use_attention,
        stage1_stride=stage1_stride,
        pre_stage2_stride=pre_stage2_stride,
    )


# 프리셋 모델들
def make_residual_attention_92_32input_gelu_tiny_dla_tiny(
    num_classes: int = 10, init_weights: bool = False
) -> ResidualAttentionModel_92_32input_GELU_Tiny_DLA:
    """경량 버전 (~5M 파라미터)."""
    return make_residual_attention_92_32input_gelu_tiny_dla(
        num_classes=num_classes,
        init_weights=init_weights,
        base_channels=16,
        stage1_channels=32,
        dla_stage_channels=[64, 128, 256],
        dla_levels=[1, 1, 1],
        dla_strides=[1, 2, 1],
    )


def make_residual_attention_92_32input_gelu_tiny_dla_small(
    num_classes: int = 10, init_weights: bool = False
) -> ResidualAttentionModel_92_32input_GELU_Tiny_DLA:
    """소형 버전 (~10M 파라미터)."""
    return make_residual_attention_92_32input_gelu_tiny_dla(
        num_classes=num_classes,
        init_weights=init_weights,
        base_channels=32,
        stage1_channels=64,
        dla_stage_channels=[128, 256, 512],
        dla_levels=[2, 2, 1],
        dla_strides=[1, 2, 1],
    )


def make_residual_attention_92_32input_gelu_tiny_dla_base(
    num_classes: int = 10, init_weights: bool = False
) -> ResidualAttentionModel_92_32input_GELU_Tiny_DLA:
    """기본 버전 (~20M 파라미터)."""
    return make_residual_attention_92_32input_gelu_tiny_dla(
        num_classes=num_classes,
        init_weights=init_weights,
        base_channels=32,
        stage1_channels=64,
        dla_stage_channels=[128, 256, 512],
        dla_levels=[2, 3, 2],
        dla_strides=[1, 2, 1],
    )


def make_residual_attention_92_32input_gelu_tiny_dla_large(
    num_classes: int = 10, init_weights: bool = False
) -> ResidualAttentionModel_92_32input_GELU_Tiny_DLA:
    """대형 버전 (~40M 파라미터)."""
    return make_residual_attention_92_32input_gelu_tiny_dla(
        num_classes=num_classes,
        init_weights=init_weights,
        base_channels=64,
        stage1_channels=128,
        dla_stage_channels=[256, 512, 1024],
        dla_levels=[2, 3, 2],
        dla_strides=[1, 2, 1],
    )


def make_residual_attention_92_32input_gelu_tiny_dla_wide(
    num_classes: int = 10, init_weights: bool = False
) -> ResidualAttentionModel_92_32input_GELU_Tiny_DLA:
    """와이드 버전 (더 많은 채널, 더 얕은 깊이)."""
    return make_residual_attention_92_32input_gelu_tiny_dla(
        num_classes=num_classes,
        init_weights=init_weights,
        base_channels=64,
        stage1_channels=128,
        dla_stage_channels=[256, 512, 1024],
        dla_levels=[1, 1, 1],  # 얕은 깊이
        dla_strides=[1, 2, 1],
    )


def make_residual_attention_92_32input_gelu_tiny_dla_deep(
    num_classes: int = 10, init_weights: bool = False
) -> ResidualAttentionModel_92_32input_GELU_Tiny_DLA:
    """딥 버전 (더 깊은 DLA Tree)."""
    return make_residual_attention_92_32input_gelu_tiny_dla(
        num_classes=num_classes,
        init_weights=init_weights,
        base_channels=32,
        stage1_channels=64,
        dla_stage_channels=[128, 256, 512],
        dla_levels=[3, 3, 2],  # 더 깊은 Tree
        dla_strides=[1, 2, 1],
    )


__all__ = [
    "ResidualAttentionModel_92_32input_GELU_Tiny_DLA",
    "make_residual_attention_92_32input_gelu_tiny_dla",
    "make_residual_attention_92_32input_gelu_tiny_dla_tiny",
    "make_residual_attention_92_32input_gelu_tiny_dla_small",
    "make_residual_attention_92_32input_gelu_tiny_dla_base",
    "make_residual_attention_92_32input_gelu_tiny_dla_large",
    "make_residual_attention_92_32input_gelu_tiny_dla_wide",
    "make_residual_attention_92_32input_gelu_tiny_dla_deep",
]


