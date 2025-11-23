
import torch
import torch.nn as nn
import torch.nn.functional as F


class PreActivationResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(PreActivationResidualBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                      stride=stride, bias=False)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: 입력 텐서 [batch, in_channels, H, W]

        Returns:
            out: 출력 텐서 [batch, out_channels, H', W']
        """
        identity = x
        if isinstance(self.shortcut, nn.Conv2d):
            identity = self.shortcut(x)
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)

        out += identity

        return out


class DeepBaselineNetBN2ResidualPreAct(nn.Module):
    def __init__(self):
        super(DeepBaselineNetBN2ResidualPreAct, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.res_block1 = PreActivationResidualBlock(64, 64, stride=1)

        self.res_block2 = PreActivationResidualBlock(64, 128, stride=1)

        self.res_block3 = PreActivationResidualBlock(128, 256, stride=1)

        self.res_block4 = PreActivationResidualBlock(256, 256, stride=1)

        self.res_block5 = PreActivationResidualBlock(256, 512, stride=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(512 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: 입력 이미지 [batch, 3, 32, 32] (CIFAR-10)

        Returns:
            out: 분류 로짓 [batch, 10]
        """
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
        x = self.fc4(x)  # 마지막 레이어는 활성화 함수 없음 (CrossEntropyLoss 사용)

        return x
