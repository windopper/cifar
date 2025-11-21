import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import ShakeDrop

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, shakedrop_prob=0.0, last_batch_norm=False, remove_first_relu=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        
        self.shake_drop = ShakeDrop(p_drop=shakedrop_prob) if shakedrop_prob > 0.0 else None
        self.bn3 = nn.BatchNorm2d(out_planes) if last_batch_norm else None
        
        self.remove_first_relu = remove_first_relu
        
        self.last_batch_norm = last_batch_norm

    def forward(self, x):
        if not self.equalInOut:
            x = self.bn1(x) if self.remove_first_relu else self.relu1(self.bn1(x))
        else:
            out = self.bn1(x) if self.remove_first_relu else self.relu1(self.bn1(x))
        
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0 and self.shake_drop is None:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        
        if self.last_batch_norm:
            out = self.bn3(out)
        
        if self.shake_drop is not None:
            out = self.shake_drop(out)
        
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, shakedrop_probs=None, last_batch_norm=False, remove_first_relu=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, shakedrop_probs, last_batch_norm, remove_first_relu)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, shakedrop_probs, last_batch_norm, remove_first_relu):
        layers = []
        for i in range(int(nb_layers)):
            prob = shakedrop_probs[i] if shakedrop_probs and i < len(shakedrop_probs) else 0.0
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, dropRate, shakedrop_prob=prob,
                                last_batch_norm=last_batch_norm, remove_first_relu=remove_first_relu))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth=28, num_classes=10, widen_factor=10, dropRate=0.3, shakedrop_prob=0.0, last_batch_norm=False, remove_first_relu=False):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        
        total_blocks = int(n * 3)
        if shakedrop_prob > 0 and total_blocks > 1:
            step = shakedrop_prob / (total_blocks - 1)
            probs = [i * step for i in range(total_blocks)]
        elif shakedrop_prob > 0:
            probs = [shakedrop_prob]
        else:
            probs = [0.0] * total_blocks
        
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        
        n_int = int(n)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, shakedrop_probs=probs[0:n_int], last_batch_norm=last_batch_norm, remove_first_relu=remove_first_relu)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, shakedrop_probs=probs[n_int:2*n_int], last_batch_norm=last_batch_norm, remove_first_relu=remove_first_relu)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, shakedrop_probs=probs[2*n_int:], last_batch_norm=last_batch_norm, remove_first_relu=remove_first_relu)
        
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
    
def wideresnet28_10(shakedrop_prob=0.0, last_batch_norm=False, remove_first_relu=False):
    return WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.3, shakedrop_prob=shakedrop_prob, last_batch_norm=last_batch_norm, remove_first_relu=remove_first_relu)

def wideresnet16_8(shakedrop_prob=0.0, last_batch_norm=False, remove_first_relu=False):
    return WideResNet(depth=16, num_classes=10, widen_factor=8, dropRate=0.3, shakedrop_prob=shakedrop_prob, last_batch_norm=last_batch_norm, remove_first_relu=remove_first_relu)
    