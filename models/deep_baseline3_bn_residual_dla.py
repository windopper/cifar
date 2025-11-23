import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
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


class DeepLayerAggregator(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(DeepLayerAggregator, self).__init__()
        assert len(in_channels_list) > 0, "in_channels_list는 비어 있을 수 없습니다."

        self.out_channels = out_channels
        self.proj_convs = nn.ModuleList()
        self.proj_bns = nn.ModuleList()

        for c in in_channels_list:
            self.proj_convs.append(nn.Conv2d(c, out_channels, kernel_size=1, bias=False))
            self.proj_bns.append(nn.BatchNorm2d(out_channels))

    def forward(self, features):
        assert len(features) == len(
            self.proj_convs
        ), "features 길이와 in_channels_list 길이는 동일해야 합니다."

        target_h, target_w = features[-1].shape[2], features[-1].shape[3]

        agg = None
        for x, conv, bn in zip(features, self.proj_convs, self.proj_bns):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)

            if x.shape[2] != target_h or x.shape[3] != target_w:
                x = F.adaptive_avg_pool2d(x, (target_h, target_w))

            if agg is None:
                agg = x
            else:
                agg = agg + x

        return agg


class DeepBaselineNetBN3ResidualDLA(nn.Module):
    def __init__(self, init_weights: bool = False):
        super(DeepBaselineNetBN3ResidualDLA, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.res_block1 = ResidualBlock(64, 64, stride=1)
        self.res_block2 = ResidualBlock(64, 128, stride=1)
        self.res_block3 = ResidualBlock(128, 256, stride=1)
        self.res_block4 = ResidualBlock(256, 256, stride=1)
        self.res_block5 = ResidualBlock(256, 512, stride=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.aggregator = DeepLayerAggregator(
            in_channels_list=[64, 128, 256, 256, 512],
            out_channels=512,
        )

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
        f1 = x  # [B, 64, 32, 32]

        x = self.res_block2(x)
        x = self.pool(x)  # 32x32 -> 16x16
        f2 = x  # [B, 128, 16, 16]

        x = self.res_block3(x)
        f3 = x  # [B, 256, 16, 16]

        x = self.res_block4(x)
        x = self.pool(x)  # 16x16 -> 8x8
        f4 = x  # [B, 256, 8, 8]

        x = self.res_block5(x)
        x = self.pool(x)  # 8x8 -> 4x4
        f5 = x  # [B, 512, 4, 4]

        agg = self.aggregator([f1, f2, f3, f4, f5])  # [B, 512, 4, 4]

        agg = torch.flatten(agg, 1)

        out = self.classifier(agg)

        return out



