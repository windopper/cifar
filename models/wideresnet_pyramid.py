import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import ShakeDrop


class PyramidBasicBlock(nn.Module):
    """
    """
    def __init__(self, in_planes, out_planes, stride=1, dropRate=0.0, shakedrop_prob=0.0):
        super(PyramidBasicBlock, self).__init__()
        
        self.downsampled = stride == 2
        self.branch = self._make_branch(in_planes, out_planes, stride=stride)
        self.shortcut = None if not self.downsampled else nn.AvgPool2d(2)
        
        self.shake_drop = ShakeDrop(p_drop=shakedrop_prob)
        
        self.droprate = dropRate
        self.in_planes = in_planes
        self.out_planes = out_planes

    def _make_branch(self, in_ch, out_ch, stride=1):
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
        
        h0 = x if not self.downsampled else self.shortcut(x)
        
        # torch.compile 최적화: F.pad 사용
        if h.size(1) > h0.size(1):
            pad_size = h.size(1) - h0.size(1)
            h0 = F.pad(h0, (0, 0, 0, 0, 0, pad_size))
        
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
            prob = shakedrop_probs[u_idx] if shakedrop_probs and u_idx < len(shakedrop_probs) else 0.0
            
            current_in = self.in_chs[u_idx]
            current_out = self.in_chs[u_idx + 1]
            
            current_stride = stride if i == 0 else 1
            
            layers.append(block(current_in, current_out, current_stride, dropRate, shakedrop_prob=prob))
            u_idx += 1
        
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNetPyramid(nn.Module):
    """
    
    Args:
        depth: 네트워크 깊이
        num_classes: 클래스 수
        widen_factor: width 배수
        dropRate: dropout 비율
        shakedrop_prob: 마지막 블록의 최대 ShakeDrop 확률
        use_pyramid: True이면 pyramid 스타일 채널 증가 사용
        alpha: pyramid 스타일 총 채널 증가량
    """
    def __init__(self, depth=28, num_classes=10, widen_factor=10, dropRate=0.3, 
                 shakedrop_prob=0.5, use_pyramid=False, alpha=48, use_original_depth=False):
        super(WideResNetPyramid, self).__init__()
        
        in_ch = 16
        block = PyramidBasicBlock
        
        if use_original_depth:
            n_units = (depth - 2) // 6
            assert (depth - 2) % 6 == 0, f"depth must satisfy (depth-2) % 6 == 0, got depth={depth}"
        else:
            n_units = (depth - 4) // 6
            assert (depth - 4) % 6 == 0, f"depth must satisfy (depth-4) % 6 == 0, got depth={depth}"
        
        if use_pyramid:
            in_chs = [in_ch] + [in_ch + math.ceil((alpha / (3 * n_units)) * (i + 1)) 
                                for i in range(3 * n_units)]
        else:
            nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
            in_chs = [nChannels[0]]
            for stage_idx in range(3):
                stage_ch = nChannels[stage_idx + 1]
                for _ in range(n_units):
                    in_chs.append(stage_ch)
        
        total_blocks = 3 * n_units
        if shakedrop_prob > 0 and total_blocks > 1:
            probs = [1 - (1.0 - (shakedrop_prob / total_blocks) * (i + 1)) 
                    for i in range(total_blocks)]
        else:
            probs = [0.0] * total_blocks
        
        self.c_in = nn.Conv2d(3, in_chs[0], 3, padding=1)
        self.bn_in = nn.BatchNorm2d(in_chs[0])
        
        self.block1 = NetworkBlock(n_units, in_chs, 0, block, 1, dropRate, 
                                   shakedrop_probs=probs)
        self.block2 = NetworkBlock(n_units, in_chs, n_units, block, 2, dropRate, 
                                   shakedrop_probs=probs)
        self.block3 = NetworkBlock(n_units, in_chs, 2*n_units, block, 2, dropRate, 
                                   shakedrop_probs=probs)
        
        self.bn_out = nn.BatchNorm2d(in_chs[-1])
        self.fc_out = nn.Linear(in_chs[-1], num_classes)
        self.nChannels = in_chs[-1]

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
        h = self.bn_in(self.c_in(x))
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = F.relu(self.bn_out(h))
        h = F.avg_pool2d(h, 8)
        h = h.view(h.size(0), -1)
        h = self.fc_out(h)
        return h

def wideresnet28_10_pyramid(shakedrop_prob=0.5, alpha=48):
    """
    """
    return WideResNetPyramid(depth=28, num_classes=10, widen_factor=10, dropRate=0.3,
                            shakedrop_prob=shakedrop_prob, use_pyramid=True, alpha=alpha)


def wideresnet16_8_pyramid(shakedrop_prob=0.5, alpha=32):
    """
    """
    return WideResNetPyramid(depth=16, num_classes=10, widen_factor=8, dropRate=0.3,
                            shakedrop_prob=shakedrop_prob, use_pyramid=True, alpha=alpha)


def pyramidnet110_270(shakedrop_prob=0.5, alpha=270):
    """
    Args:
        shakedrop_prob: 최대 ShakeDrop 확률
        alpha: pyramid 채널 증가량
    """
    return WideResNetPyramid(depth=110, num_classes=10, widen_factor=1, dropRate=0.0,
                            shakedrop_prob=shakedrop_prob, use_pyramid=True, alpha=alpha,
                            use_original_depth=True)


def pyramidnet110_150(shakedrop_prob=0.5, alpha=150):
    """
    Args:
        shakedrop_prob: 최대 ShakeDrop 확률
        alpha: pyramid 채널 증가량
    """
    return WideResNetPyramid(depth=110, num_classes=10, widen_factor=1, dropRate=0.0,
                            shakedrop_prob=shakedrop_prob, use_pyramid=True, alpha=alpha,
                            use_original_depth=True)

