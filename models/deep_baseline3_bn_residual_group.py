import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlockGroup(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, groups=2):
        super(ResidualBlockGroup, self).__init__()

        assert (
            in_channels % groups == 0 and out_channels % groups == 0
        ), f"groups({groups}) must divide in_channels({in_channels}) and out_channels({out_channels})"

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            groups=groups,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=groups,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
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


class DeepBaselineNetBN3ResidualGroup(nn.Module):
    def __init__(self, init_weights: bool = False, groups: int = 2, stem_groups: int = 1):
        super(DeepBaselineNetBN3ResidualGroup, self).__init__()

        assert 3 % stem_groups == 0 and 64 % stem_groups == 0, (
            f"stem_groups({stem_groups}) must divide 3 and 64"
        )

        self.groups = groups
        self.stem_groups = stem_groups

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            padding=1,
            bias=False,
            groups=stem_groups,
        )
        self.bn1 = nn.BatchNorm2d(64)

        self.res_block1 = ResidualBlockGroup(64, 64, stride=1, groups=groups)

        self.res_block2 = ResidualBlockGroup(64, 128, stride=1, groups=groups)

        self.res_block3 = ResidualBlockGroup(128, 256, stride=1, groups=groups)

        self.res_block4 = ResidualBlockGroup(256, 256, stride=1, groups=groups)

        self.res_block5 = ResidualBlockGroup(256, 512, stride=1, groups=groups)

        self.pool = nn.MaxPool2d(2, 2)

        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 10),
        )

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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
        x = self.pool(x)  # 32x32 -> 16x16

        x = self.res_block3(x)

        x = self.res_block4(x)
        x = self.pool(x)  # 16x16 -> 8x8

        x = self.res_block5(x)
        x = self.pool(x)  # 8x8 -> 4x4

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x



