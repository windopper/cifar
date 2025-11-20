# -*- coding: utf-8 -*-
"""
WideResNet with PyramidNet-style improvements
- Zero-padding shortcut instead of 1x1 convolution
- Gradual channel increase option (pyramid style)
- Improved ShakeDrop integration
- Pre-activation style blocks
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import ShakeDrop


class PyramidBasicBlock(nn.Module):
    """
    PyramidNet 스타일의 BasicBlock
    - Pre-activation 구조
    - Zero-padding shortcut (파라미터 효율적)
    - ShakeDrop을 residual branch에 적용
    """
    def __init__(self, in_planes, out_planes, stride=1, dropRate=0.0, shakedrop_prob=0.0):
        super(PyramidBasicBlock, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        
        self.droprate = dropRate
        self.stride = stride
        self.in_planes = in_planes
        self.out_planes = out_planes
        
        # Shortcut: stride=2일 때만 average pooling 사용
        self.shortcut = None
        if stride == 2:
            self.shortcut = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # ShakeDrop 모듈
        self.shake_drop = ShakeDrop(p_drop=shakedrop_prob) if shakedrop_prob > 0.0 else None

    def forward(self, x):
        # Pre-activation
        out = self.relu1(self.bn1(x))
        
        # First convolution (stride가 여기서 적용됨)
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        
        # Dropout (ShakeDrop 미사용 시에만)
        if self.droprate > 0 and self.shake_drop is None:
            out = F.dropout(out, p=self.droprate, training=self.training)
        
        # Second convolution
        out = self.conv2(out)
        out = self.bn3(out)
        
        # ShakeDrop 적용
        if self.shake_drop is not None:
            out = self.shake_drop(out)
        
        # Shortcut connection with zero-padding
        shortcut = x
        if self.shortcut is not None:
            shortcut = self.shortcut(shortcut)
        
        # 채널 수가 다르면 zero-padding
        if self.in_planes != self.out_planes:
            pad_size = self.out_planes - self.in_planes
            shortcut = F.pad(shortcut, (0, 0, 0, 0, 0, pad_size), mode='constant', value=0)
        
        return out + shortcut


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, 
                 shakedrop_probs=None, use_pyramid=False, alpha=0):
        super(NetworkBlock, self).__init__()
        self.use_pyramid = use_pyramid
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, 
                                       dropRate, shakedrop_probs, alpha)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, 
                    shakedrop_probs, alpha):
        layers = []
        
        for i in range(int(nb_layers)):
            # ShakeDrop 확률
            prob = shakedrop_probs[i] if shakedrop_probs and i < len(shakedrop_probs) else 0.0
            
            # Pyramid 스타일: 각 블록마다 채널을 점진적으로 증가
            if self.use_pyramid and alpha > 0:
                # 각 블록당 증가량
                add_per_block = alpha / nb_layers
                current_in = in_planes + int(add_per_block * i)
                current_out = in_planes + int(add_per_block * (i + 1))
            else:
                # 기존 WideResNet 스타일: 첫 블록만 in_planes, 나머지는 out_planes
                current_in = in_planes if i == 0 else out_planes
                current_out = out_planes
            
            # 첫 번째 블록만 stride 적용
            current_stride = stride if i == 0 else 1
            
            layers.append(block(current_in, current_out, current_stride, dropRate, shakedrop_prob=prob))
        
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNetPyramid(nn.Module):
    """
    PyramidNet 스타일로 개선된 WideResNet
    
    Args:
        depth: 네트워크 깊이 (28, 16 등)
        num_classes: 클래스 수
        widen_factor: width 배수
        dropRate: dropout 비율
        shakedrop_prob: 마지막 블록의 최대 ShakeDrop 확률
        use_pyramid: True이면 pyramid 스타일 채널 증가 사용
        alpha: pyramid 스타일 총 채널 증가량 (use_pyramid=True일 때)
    """
    def __init__(self, depth=28, num_classes=10, widen_factor=10, dropRate=0.3, 
                 shakedrop_prob=0.0, use_pyramid=False, alpha=48):
        super(WideResNetPyramid, self).__init__()
        
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = PyramidBasicBlock
        
        # ShakeDrop 확률 스케줄 생성 (PyramidNet 스타일: 선형 증가)
        total_blocks = int(n * 3)
        if shakedrop_prob > 0 and total_blocks > 1:
            # 0부터 shakedrop_prob까지 선형 증가
            probs = [shakedrop_prob * (i + 1) / total_blocks for i in range(total_blocks)]
        else:
            probs = [0.0] * total_blocks
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        
        # Pyramid 스타일에서는 전체 블록에 걸쳐 채널을 점진적으로 증가
        if use_pyramid:
            # 각 블록당 증가량 계산
            add_channel = alpha / (3 * n)
            
            # 각 stage의 시작과 끝 채널 계산
            stage1_out = nChannels[0] + int(add_channel * n)
            stage2_out = stage1_out + int(add_channel * n)
            stage3_out = stage2_out + int(add_channel * n)
            
            n_int = int(n)
            self.block1 = NetworkBlock(n, nChannels[0], stage1_out, block, 1, dropRate, 
                                       shakedrop_probs=probs[0:n_int],
                                       use_pyramid=use_pyramid, alpha=int(add_channel * n))
            self.block2 = NetworkBlock(n, stage1_out, stage2_out, block, 2, dropRate, 
                                       shakedrop_probs=probs[n_int:2*n_int],
                                       use_pyramid=use_pyramid, alpha=int(add_channel * n))
            self.block3 = NetworkBlock(n, stage2_out, stage3_out, block, 2, dropRate, 
                                       shakedrop_probs=probs[2*n_int:],
                                       use_pyramid=use_pyramid, alpha=int(add_channel * n))
            
            final_channels = stage3_out
        else:
            # 기존 WideResNet 스타일
            n_int = int(n)
            self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, 
                                       shakedrop_probs=probs[0:n_int],
                                       use_pyramid=False, alpha=0)
            self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, 
                                       shakedrop_probs=probs[n_int:2*n_int],
                                       use_pyramid=False, alpha=0)
            self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, 
                                       shakedrop_probs=probs[2*n_int:],
                                       use_pyramid=False, alpha=0)
            
            final_channels = nChannels[3]
        
        # Classifier
        self.bn1 = nn.BatchNorm2d(final_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(final_channels, num_classes)
        self.nChannels = final_channels

        # Weight initialization
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
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


# 기본 WideResNet 스타일 (zero-padding shortcut만 적용)
def wideresnet28_10_pyramid(shakedrop_prob=0.0, use_pyramid=False, alpha=48):
    """
    WideResNet-28-10 with PyramidNet improvements
    
    Args:
        shakedrop_prob: 최대 ShakeDrop 확률
        use_pyramid: True이면 pyramid 스타일 채널 증가 활성화
        alpha: pyramid 채널 증가량
    """
    return WideResNetPyramid(depth=28, num_classes=10, widen_factor=10, dropRate=0.3,
                            shakedrop_prob=shakedrop_prob, use_pyramid=use_pyramid, alpha=alpha)


def wideresnet16_8_pyramid(shakedrop_prob=0.0, use_pyramid=False, alpha=32):
    """
    WideResNet-16-8 with PyramidNet improvements
    
    Args:
        shakedrop_prob: 최대 ShakeDrop 확률
        use_pyramid: True이면 pyramid 스타일 채널 증가 활성화
        alpha: pyramid 채널 증가량
    """
    return WideResNetPyramid(depth=16, num_classes=10, widen_factor=8, dropRate=0.3,
                            shakedrop_prob=shakedrop_prob, use_pyramid=use_pyramid, alpha=alpha)


# PyramidNet 완전 스타일 (pyramid 채널 증가 + zero-padding shortcut)
def wideresnet28_10_fullpyramid(shakedrop_prob=0.5, alpha=48):
    """
    WideResNet-28-10 with full PyramidNet style
    """
    return WideResNetPyramid(depth=28, num_classes=10, widen_factor=10, dropRate=0.3,
                            shakedrop_prob=shakedrop_prob, use_pyramid=True, alpha=alpha)


def wideresnet16_8_fullpyramid(shakedrop_prob=0.5, alpha=32):
    """
    WideResNet-16-8 with full PyramidNet style
    """
    return WideResNetPyramid(depth=16, num_classes=10, widen_factor=8, dropRate=0.3,
                            shakedrop_prob=shakedrop_prob, use_pyramid=True, alpha=alpha)


def pyramidnet110_270(shakedrop_prob=0.5, alpha=270):
    """
    PyramidNet-110 with alpha=270 (Original ShakePyramidNet configuration)
    
    depth=110: (110 - 4) / 6 = 17.67, 가장 가까운 depth=112 사용 (n=18)
    최종 채널: 16 + 270 = 286
    
    Args:
        shakedrop_prob: 최대 ShakeDrop 확률 (기본값: 0.5)
        alpha: pyramid 채널 증가량 (기본값: 270)
    """
    return WideResNetPyramid(depth=112, num_classes=10, widen_factor=1, dropRate=0.0,
                            shakedrop_prob=shakedrop_prob, use_pyramid=True, alpha=alpha)

