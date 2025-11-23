import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        
        reduced_channels = max(1, channels // reduction)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc1 = nn.Linear(channels, reduced_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(reduced_channels, channels, bias=False)
        self.sigmoid = nn.Sigmoid()
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.fc2.weight)
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        y = self.avg_pool(x).view(b, c)
        
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        
        return x * y.expand_as(x)


class ResidualBlockSE(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, reduction=16):
        super(ResidualBlockSE, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
                    
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
        
        out = self.se(out)
        
        out += identity
        
        out = F.relu(out)
        
        return out


class DeepBaselineNetBN2ResidualSE(nn.Module):
    def __init__(self, init_weights=False, se_reduction=16):
        super(DeepBaselineNetBN2ResidualSE, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.res_block1 = ResidualBlockSE(64, 64, stride=1, reduction=se_reduction)
        
        self.res_block2 = ResidualBlockSE(64, 128, stride=1, reduction=se_reduction)
        
        self.res_block3 = ResidualBlockSE(128, 256, stride=1, reduction=se_reduction)
        
        self.res_block4 = ResidualBlockSE(256, 256, stride=1, reduction=se_reduction)
        
        self.res_block5 = ResidualBlockSE(256, 512, stride=1, reduction=se_reduction)
        
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
        
        x = self.res_block1(x)
        
        x = self.res_block2(x)
        x = self.pool(x)
        
        x = self.res_block3(x)
        
        x = self.res_block4(x)
        x = self.pool(x)
        
        x = self.res_block5(x)
        x = self.pool(x)
        
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x

