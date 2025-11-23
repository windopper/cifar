
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNeXtBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, cardinality=8, bottleneck_width=4, stride=1):
        super(ResNeXtBlock, self).__init__()
        
        bottleneck_channels = cardinality * bottleneck_width
        
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, 
                               stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3,
                               stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1,
                               stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        out += identity
        
        out = F.relu(out)
        
        return out


class DeepBaselineNetBN2ResNeXt(nn.Module):
    def __init__(self, cardinality=8, bottleneck_width=4, init_weights=False):
        super(DeepBaselineNetBN2ResNeXt, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.resnext_block1 = ResNeXtBlock(64, 64, cardinality=cardinality, 
                                           bottleneck_width=8, stride=1)
        
        self.resnext_block2 = ResNeXtBlock(64, 128, cardinality=cardinality,
                                           bottleneck_width=16, stride=1)
        
        self.resnext_block3 = ResNeXtBlock(128, 256, cardinality=cardinality,
                                           bottleneck_width=32, stride=1)
        
        self.resnext_block4 = ResNeXtBlock(256, 256, cardinality=cardinality,
                                           bottleneck_width=32, stride=1)
        
        self.resnext_block5 = ResNeXtBlock(256, 512, cardinality=cardinality,
                                           bottleneck_width=64, stride=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(512 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        
        if init_weights:
            self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.resnext_block1(x)
        
        x = self.resnext_block2(x)
        x = self.pool(x)
        
        x = self.resnext_block3(x)
        
        x = self.resnext_block4(x)
        x = self.pool(x)
        
        x = self.resnext_block5(x)
        x = self.pool(x)
        
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x

