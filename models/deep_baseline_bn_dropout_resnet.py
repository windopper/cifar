import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """ResNet BasicBlock with Conv-BN-ReLU"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 첫 번째 Conv-BN-ReLU 블록
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 두 번째 Conv-BN 블록 (ReLU는 잔차 연결 후에 적용)
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
        
        # Downsample이 필요한 경우 identity도 변환
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # 잔차 연결
        out += identity
        out = F.relu(out)
        
        return out


class DeepBaselineNetBNDropoutResNet(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(DeepBaselineNetBNDropoutResNet, self).__init__()
        
        # 초기 Conv-BN-ReLU (3 -> 32)
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # ResNet BasicBlock들
        # Block 1: 32 -> 32 (같은 채널 수, 잔차 연결 가능)
        self.block1 = BasicBlock(32, 32)
        
        # Block 2: 32 -> 64 (채널 수 증가, downsample 필요)
        self.downsample2 = nn.Sequential(
            nn.Conv2d(32, 64, 1, stride=2, bias=False),
            nn.BatchNorm2d(64)
        )
        self.block2 = BasicBlock(32, 64, stride=2, downsample=self.downsample2)
        
        # Block 3: 64 -> 128 (채널 수 증가, downsample 필요)
        self.downsample3 = nn.Sequential(
            nn.Conv2d(64, 128, 1, stride=2, bias=False),
            nn.BatchNorm2d(128)
        )
        self.block3 = BasicBlock(64, 128, stride=2, downsample=self.downsample3)
        
        # Block 4: 128 -> 256 (채널 수 증가, downsample 필요)
        self.downsample4 = nn.Sequential(
            nn.Conv2d(128, 256, 1, stride=2, bias=False),
            nn.BatchNorm2d(256)
        )
        self.block4 = BasicBlock(128, 256, stride=2, downsample=self.downsample4)
        
        # 분류기 앞에 Dropout 추가
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
    
    def forward(self, x):
        # 초기 Conv-BN-ReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # ResNet BasicBlock들
        x = self.block1(x)  # 32x32x32 -> 32x32x32
        x = self.block2(x)  # 32x32x32 -> 16x16x64
        x = self.block3(x)  # 16x16x64 -> 8x8x128
        x = self.block4(x)  # 8x8x128 -> 4x4x256
        
        x = torch.flatten(x, 1)
        
        # 분류기 앞에 Dropout 적용
        x = self.dropout(x)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x

