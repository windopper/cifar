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

class DeepBaselineNetSE(nn.Module):
    def __init__(self):
        super(DeepBaselineNetSE, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.se1 = SEBlock(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.se2 = SEBlock(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.se3 = SEBlock(128)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.se4 = SEBlock(128)
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.se5 = SEBlock(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.se1(x)
        x = F.relu(self.conv2(x))
        x = self.se2(x)
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.se3(x)
        x = F.relu(self.conv4(x))
        x = self.se4(x)
        x = self.pool(x)
        
        x = F.relu(self.conv5(x))
        x = self.se5(x)
        x = self.pool(x)
        
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x

