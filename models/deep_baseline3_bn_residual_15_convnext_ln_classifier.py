import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.convnext_block import ConvNeXtBlock, StridedConvNeXtBlock


def _make_layer(in_channels, out_channels, num_blocks, stride=1, kernel_size=3):
    layers = []
    layers.append(StridedConvNeXtBlock(
        in_channels, out_channels, stride=stride, kernel_size=kernel_size
    ))
    for _ in range(1, num_blocks):
        layers.append(ConvNeXtBlock(
            out_channels, kernel_size=kernel_size
        ))
    return nn.Sequential(*layers)


class DeepBaselineNetBN3Residual15ConvNeXtLNClassifier(nn.Module):
    def __init__(self, init_weights=False, num_classes=10):
        super(DeepBaselineNetBN3Residual15ConvNeXtLNClassifier, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        kernel_size = 7
        
        self.stage1 = _make_layer(64, 64, num_blocks=3, stride=1, kernel_size=kernel_size)
        self.stage2 = _make_layer(64, 128, num_blocks=3, stride=2, kernel_size=kernel_size)
        self.stage3 = _make_layer(128, 256, num_blocks=9, stride=2, kernel_size=kernel_size)
        self.stage4 = _make_layer(256, 512, num_blocks=3, stride=2, kernel_size=kernel_size)
        
        in_channels = 512
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, num_classes),
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
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.stage1(x)
        
        x = self.stage2(x)
        
        x = self.stage3(x)
        
        x = self.stage4(x)
        
        x = self.classifier(x)
        
        return x

