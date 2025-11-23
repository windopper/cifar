import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        identity = x
        
        # 첫 번째 Conv-BN-ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # 두 번째 Conv-BN
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = F.relu(out)
        
        return out


class DeepBaselineNetBNDropoutResNet(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(DeepBaselineNetBNDropoutResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.block1 = BasicBlock(32, 32)
        
        self.downsample2 = nn.Sequential(
            nn.Conv2d(32, 64, 1, stride=2, bias=False),
            nn.BatchNorm2d(64)
        )
        self.block2 = BasicBlock(32, 64, stride=2, downsample=self.downsample2)
        
        self.downsample3 = nn.Sequential(
            nn.Conv2d(64, 128, 1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        )
        self.block3 = BasicBlock(64, 128, stride=2, downsample=self.downsample3)
        
        self.downsample4 = nn.Sequential(
            nn.Conv2d(128, 256, 1, stride=2, bias=False),
            nn.BatchNorm2d(256)
        )
        self.block4 = BasicBlock(128, 256, stride=2, downsample=self.downsample4)
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        x = torch.flatten(x, 1)
        
        x = self.dropout(x)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x

