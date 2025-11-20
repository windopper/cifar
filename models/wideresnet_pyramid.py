# -*- coding: utf-8 -*-
"""
WideResNet with PyramidNet-style improvements
- Zero-padding shortcut instead of 1x1 convolution
- Gradual channel increase option (pyramid style)
- Improved ShakeDrop integration
- Original PyramidNet block structure (BN-Conv-BN-ReLU-Conv-BN)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import ShakeDrop


class PyramidBasicBlock(nn.Module):
    """
    PyramidNet 스타일의 BasicBlock (원본 ShakePyramidNet 구조 반영)
    - BN → Conv → BN → ReLU → Conv → BN 구조
    - Zero-padding shortcut (파라미터 효율적)
    - ShakeDrop을 residual branch에 적용
    """
    def __init__(self, in_planes, out_planes, stride=1, dropRate=0.0, shakedrop_prob=0.0):
        super(PyramidBasicBlock, self).__init__()
        
        self.downsampled = stride == 2
        self.branch = self._make_branch(in_planes, out_planes, stride=stride)
        self.shortcut = None if not self.downsampled else nn.AvgPool2d(2)
        
        # ShakeDrop 모듈 (원본과 동일하게 항상 생성)
        self.shake_drop = ShakeDrop(p_drop=shakedrop_prob)
        
        self.droprate = dropRate
        self.in_planes = in_planes
        self.out_planes = out_planes

    def _make_branch(self, in_ch, out_ch, stride=1):
        """원본 ShakePyramidNet의 _make_branch 구조"""
        return nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch))

    def forward(self, x):
        h = self.branch(x)
        h = self.shake_drop(h)
        
        # Shortcut 처리
        h0 = x if not self.downsampled else self.shortcut(x)
        
        # Zero-padding (device-aware)
        if h.size(1) > h0.size(1):
            pad_size = h.size(1) - h0.size(1)
            pad_zero = torch.zeros(h0.size(0), pad_size, h0.size(2), h0.size(3), 
                                  dtype=h0.dtype, device=h0.device)
            h0 = torch.cat([h0, pad_zero], dim=1)
        
        return h + h0


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_chs, start_idx, block, stride, dropRate=0.0, 
                 shakedrop_probs=None):
        super(NetworkBlock, self).__init__()
        self.start_idx = start_idx
        self.in_chs = in_chs
        self.layer = self._make_layer(block, nb_layers, stride, dropRate, shakedrop_probs)

    def _make_layer(self, block, nb_layers, stride, dropRate, shakedrop_probs):
        layers = []
        u_idx = self.start_idx
        
        for i in range(int(nb_layers)):
            # ShakeDrop 확률
            prob = shakedrop_probs[u_idx] if shakedrop_probs and u_idx < len(shakedrop_probs) else 0.0
            
            # 채널은 미리 계산된 in_chs 리스트에서 가져옴 (원본 방식)
            current_in = self.in_chs[u_idx]
            current_out = self.in_chs[u_idx + 1]
            
            # 첫 번째 블록만 stride 적용
            current_stride = stride if i == 0 else 1
            
            layers.append(block(current_in, current_out, current_stride, dropRate, shakedrop_prob=prob))
            u_idx += 1
        
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNetPyramid(nn.Module):
    """
    PyramidNet 스타일로 개선된 WideResNet (원본 ShakePyramidNet 구조 완전 반영)
    
    Args:
        depth: 네트워크 깊이 (원본: (depth-2)//6, WideResNet 호환: (depth-4)//6)
        num_classes: 클래스 수
        widen_factor: width 배수 (use_pyramid=False일 때만 사용)
        dropRate: dropout 비율
        shakedrop_prob: 마지막 블록의 최대 ShakeDrop 확률 (기본값: 0.5)
        use_pyramid: True이면 pyramid 스타일 채널 증가 사용
        alpha: pyramid 스타일 총 채널 증가량 (use_pyramid=True일 때)
        use_original_depth: True이면 원본 방식 (depth-2)//6 사용, False면 WideResNet 방식 (depth-4)//6
    """
    def __init__(self, depth=28, num_classes=10, widen_factor=10, dropRate=0.3, 
                 shakedrop_prob=0.5, use_pyramid=False, alpha=48, use_original_depth=False):
        super(WideResNetPyramid, self).__init__()
        
        in_ch = 16
        block = PyramidBasicBlock
        
        # Depth 계산: 원본 방식 또는 WideResNet 방식
        if use_original_depth:
            # 원본 ShakePyramidNet 방식
            n_units = (depth - 2) // 6
            assert (depth - 2) % 6 == 0, f"depth must satisfy (depth-2) % 6 == 0, got depth={depth}"
        else:
            # WideResNet 호환 방식
            n_units = (depth - 4) // 6
            assert (depth - 4) % 6 == 0, f"depth must satisfy (depth-4) % 6 == 0, got depth={depth}"
        
        # 채널 리스트 계산 (원본 방식: 미리 모든 채널 계산)
        if use_pyramid:
            # Pyramid 스타일: 각 블록마다 채널이 점진적으로 증가
            # 원본 공식: in_ch + math.ceil((alpha / (3 * n_units)) * (i + 1))
            in_chs = [in_ch] + [in_ch + math.ceil((alpha / (3 * n_units)) * (i + 1)) 
                                for i in range(3 * n_units)]
        else:
            # WideResNet 스타일: 각 stage마다 채널이 고정
            nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
            in_chs = [nChannels[0]]
            # 각 stage의 블록 수만큼 채널 반복
            for stage_idx in range(3):
                stage_ch = nChannels[stage_idx + 1]
                for _ in range(n_units):
                    in_chs.append(stage_ch)
        
        # ShakeDrop 확률 스케줄 생성 (원본 공식 반영)
        # 원본: [1 - (1.0 - (0.5 / (3 * n_units)) * (i + 1)) for i in range(3 * n_units)]
        # 단순화: (shakedrop_prob / total_blocks) * (i + 1)
        total_blocks = 3 * n_units
        if shakedrop_prob > 0 and total_blocks > 1:
            # 원본 공식과 동일하게 구현
            probs = [1 - (1.0 - (shakedrop_prob / total_blocks) * (i + 1)) 
                    for i in range(total_blocks)]
        else:
            probs = [0.0] * total_blocks
        
        # Initial convolution + BatchNorm (원본 구조 반영)
        self.c_in = nn.Conv2d(3, in_chs[0], 3, padding=1)
        self.bn_in = nn.BatchNorm2d(in_chs[0])
        
        # Network blocks (원본 방식: 채널 리스트와 시작 인덱스 전달)
        self.block1 = NetworkBlock(n_units, in_chs, 0, block, 1, dropRate, 
                                   shakedrop_probs=probs)
        self.block2 = NetworkBlock(n_units, in_chs, n_units, block, 2, dropRate, 
                                   shakedrop_probs=probs)
        self.block3 = NetworkBlock(n_units, in_chs, 2*n_units, block, 2, dropRate, 
                                   shakedrop_probs=probs)
        
        # Classifier
        self.bn_out = nn.BatchNorm2d(in_chs[-1])
        self.fc_out = nn.Linear(in_chs[-1], num_classes)
        self.nChannels = in_chs[-1]

        # Weight initialization (원본과 동일)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        # 원본 구조: Conv -> BN
        h = self.bn_in(self.c_in(x))
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = F.relu(self.bn_out(h))
        h = F.avg_pool2d(h, 8)
        h = h.view(h.size(0), -1)
        h = self.fc_out(h)
        return h

# PyramidNet 완전 스타일 (pyramid 채널 증가 + zero-padding shortcut)
def wideresnet28_10_pyramid(shakedrop_prob=0.5, alpha=48):
    """
    WideResNet-28-10 with full PyramidNet style
    """
    return WideResNetPyramid(depth=28, num_classes=10, widen_factor=10, dropRate=0.3,
                            shakedrop_prob=shakedrop_prob, use_pyramid=True, alpha=alpha)


def wideresnet16_8_pyramid(shakedrop_prob=0.5, alpha=32):
    """
    WideResNet-16-8 with full PyramidNet style
    """
    return WideResNetPyramid(depth=16, num_classes=10, widen_factor=8, dropRate=0.3,
                            shakedrop_prob=shakedrop_prob, use_pyramid=True, alpha=alpha)


def pyramidnet110_270(shakedrop_prob=0.5, alpha=270):
    """
    PyramidNet-110 with alpha=270 (Original ShakePyramidNet configuration)
    
    depth=110: (110 - 2) / 6 = 18 (원본 방식)
    최종 채널: 16 + math.ceil(270) = 286
    파라미터: ~28.5M
    
    Args:
        shakedrop_prob: 최대 ShakeDrop 확률 (기본값: 0.5)
        alpha: pyramid 채널 증가량 (기본값: 270)
    """
    return WideResNetPyramid(depth=110, num_classes=10, widen_factor=1, dropRate=0.0,
                            shakedrop_prob=shakedrop_prob, use_pyramid=True, alpha=alpha,
                            use_original_depth=True)


# 10M 파라미터 수준의 PyramidNet 모델들
def pyramidnet164_118(shakedrop_prob=0.5, alpha=118):
    """
    PyramidNet-164 with alpha=118 (~10M parameters)
    
    depth=164: (164 - 2) / 6 = 27 (원본 방식)
    최종 채널: 16 + math.ceil(118) = 134
    파라미터: ~10.12M
    
    Args:
        shakedrop_prob: 최대 ShakeDrop 확률 (기본값: 0.5)
        alpha: pyramid 채널 증가량 (기본값: 118)
    """
    return WideResNetPyramid(depth=164, num_classes=10, widen_factor=1, dropRate=0.0,
                            shakedrop_prob=shakedrop_prob, use_pyramid=True, alpha=alpha,
                            use_original_depth=True)

