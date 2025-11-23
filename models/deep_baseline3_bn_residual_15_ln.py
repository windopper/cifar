import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super(LayerNorm2d, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.ln = nn.LayerNorm(num_channels, eps=eps)
    
    def forward(self, x):
        # (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        # (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return x


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.ln1 = LayerNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.ln2 = LayerNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                LayerNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.ln1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.ln2(out)

        out += identity

        out = F.relu(out)

        return out


def _make_layer(in_channels, out_channels, num_blocks, stride=1):
    layers = []
    layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
    for _ in range(1, num_blocks):
        layers.append(ResidualBlock(out_channels, out_channels, stride=1))
    return nn.Sequential(*layers)


class DeepBaselineNetBN3Residual15LN(nn.Module):

    def __init__(self, init_weights=False):
        super(DeepBaselineNetBN3Residual15LN, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.ln1 = LayerNorm2d(64)

        self.stage1 = _make_layer(64, 64, num_blocks=2, stride=1)

        self.stage2 = _make_layer(64, 128, num_blocks=2, stride=2)

        self.stage3 = _make_layer(128, 256, num_blocks=4, stride=2)

        self.stage4 = _make_layer(256, 512, num_blocks=2, stride=2)

        self.classifier = nn.Sequential(
            nn.Linear(512, 10),
        )

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
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ln1(x)
        x = F.relu(x)

        x = self.stage1(x)

        x = self.stage2(x)

        x = self.stage3(x)

        x = self.stage4(x)

        x = F.avg_pool2d(x, kernel_size=4)

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x

