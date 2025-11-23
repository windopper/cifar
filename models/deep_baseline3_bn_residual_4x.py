import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
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
        
        out += identity
        
        out = F.relu(out)
        
        return out


class DeepBaselineNetBN3Residual4X(nn.Module):
    def __init__(self, init_weights=False):
        super(DeepBaselineNetBN3Residual4X, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        
        self.res_block1 = ResidualBlock(128, 128, stride=1)
        
        self.res_block2 = ResidualBlock(128, 256, stride=1)
        
        self.res_block3 = ResidualBlock(256, 512, stride=1)
        
        self.res_block4 = ResidualBlock(512, 512, stride=1)
        
        self.res_block5 = ResidualBlock(512, 1024, stride=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 4 * 4, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10),
        )
        
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
        
        x = self.classifier(x)
        
        return x

