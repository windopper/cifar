import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleneckBlock(nn.Module):
    expansion = 2  # 4에서 2로 줄여 파라미터 감소

    def __init__(self, in_channels, out_channels, stride=1):
        super(BottleneckBlock, self).__init__()

        mid_channels = out_channels * self.expansion

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1,
                               stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels * self.expansion,
                               kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
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


class DeepBaselineNetBN3ResidualBottleneck(nn.Module):
    def __init__(self, init_weights=False):
        super(DeepBaselineNetBN3ResidualBottleneck, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = nn.Sequential(
            BottleneckBlock(64, 64, stride=1),  # 64 -> 128
        )

        self.layer2 = nn.Sequential(
            BottleneckBlock(128, 128, stride=1),  # 128 -> 256
            BottleneckBlock(256, 128, stride=1),  # 256 -> 256
        )

        self.layer3 = nn.Sequential(
            BottleneckBlock(256, 256, stride=1),  # 256 -> 512
            BottleneckBlock(512, 256, stride=1),  # 512 -> 512
        )

        self.layer4 = nn.Sequential(
            BottleneckBlock(512, 256, stride=1),  # 512 -> 512
        )

        self.pool = nn.MaxPool2d(2, 2)

        self.classifier = nn.Linear(512 * 4 * 4, 10)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.layer1(x)  # [batch, 128, 32, 32]

        x = self.layer2(x)  # [batch, 256, 32, 32]
        x = self.pool(x)  # 32x32 -> 16x16

        x = self.layer3(x)  # [batch, 512, 16, 16]
        x = self.pool(x)  # 16x16 -> 8x8

        x = self.layer4(x)  # [batch, 512, 8, 8]
        x = self.pool(x)  # 8x8 -> 4x4

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x
