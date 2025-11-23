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
            pool_stride = stride if stride > 1 else 1
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=pool_stride, stride=pool_stride),
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=1, bias=False),
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


class DeepBaselineNetBN3ResidualGAPGMP(nn.Module):
    def __init__(self, base_filter=16, filters=None, num_blocks_per_stage=5, 
                 num_classes=10, init_weights=False):
        super(DeepBaselineNetBN3ResidualGAPGMP, self).__init__()
        
        if filters is None:
            filters = [16, 32, 64]
        
        if isinstance(num_blocks_per_stage, int):
            num_blocks_per_stage = [num_blocks_per_stage] * len(filters)
        elif len(num_blocks_per_stage) != len(filters):
            raise ValueError(f"num_blocks_per_stage의 길이({len(num_blocks_per_stage)})가 "
                           f"filters의 길이({len(filters)})와 일치하지 않습니다.")
        
        self.base_filter = base_filter
        self.filters = filters
        self.num_blocks_per_stage = num_blocks_per_stage
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv2d(3, base_filter, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_filter)
        
        self.stages = nn.ModuleList()
        in_channels = base_filter
        
        for i, (out_channels, num_blocks) in enumerate(zip(filters, num_blocks_per_stage)):
            self.stages.append(self._make_stage(in_channels, out_channels, num_blocks))
            in_channels = out_channels
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        final_filter = filters[-1] if filters else base_filter
        self.classifier = nn.Linear(final_filter * 2, num_classes)
        
        if init_weights:
            self._initialize_weights()
    
    def _make_stage(self, in_channels, out_channels, num_blocks):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=1))
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
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
        
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.stages) - 1:
                x = self.pool(x)
        
        x = self.pool(x)
        
        final_filter = self.filters[-1] if self.filters else self.base_filter
        avg_pool = self.global_avg_pool(x)  # [batch, final_filter, 1, 1]
        max_pool = self.global_max_pool(x)  # [batch, final_filter, 1, 1]
        
        x = torch.cat([avg_pool, max_pool], dim=1)
        
        x = torch.flatten(x, 1)
        
        x = self.classifier(x)
        
        return x


def make_deep_baseline3_bn_residual_gap_gmp(base_filter=16, filters=None, 
                                            num_blocks_per_stage=5, 
                                            num_classes=10, init_weights=False):
    return DeepBaselineNetBN3ResidualGAPGMP(
        base_filter=base_filter,
        filters=filters,
        num_blocks_per_stage=num_blocks_per_stage,
        num_classes=num_classes,
        init_weights=init_weights
    )


def DeepBaselineNetBN3ResidualGAPGMP_S3_F8_16_32_B2(num_classes=10, init_weights=False):
    return make_deep_baseline3_bn_residual_gap_gmp(
        base_filter=8,
        filters=[8, 16, 32],
        num_blocks_per_stage=[2, 2, 2],
        num_classes=num_classes,
        init_weights=init_weights
    )


def DeepBaselineNetBN3ResidualGAPGMP_S3_F16_32_64_B3(num_classes=10, init_weights=False):
    return make_deep_baseline3_bn_residual_gap_gmp(
        base_filter=16,
        filters=[16, 32, 64],
        num_blocks_per_stage=[3, 3, 3],
        num_classes=num_classes,
        init_weights=init_weights
    )


def DeepBaselineNetBN3ResidualGAPGMP_S3_F32_64_128_B5(num_classes=10, init_weights=False):
    return make_deep_baseline3_bn_residual_gap_gmp(
        base_filter=32,
        filters=[32, 64, 128],
        num_blocks_per_stage=[5, 5, 5],
        num_classes=num_classes,
        init_weights=init_weights
    )


def DeepBaselineNetBN3ResidualGAPGMP_S3_F64_128_256_B5(num_classes=10, init_weights=False):
    return make_deep_baseline3_bn_residual_gap_gmp(
        base_filter=64,
        filters=[64, 128, 256],
        num_blocks_per_stage=[5, 5, 5],
        num_classes=num_classes,
        init_weights=init_weights
    )


def DeepBaselineNetBN3ResidualGAPGMP_S4_F64_128_256_512_B5(num_classes=10, init_weights=False):
    return make_deep_baseline3_bn_residual_gap_gmp(
        base_filter=64,
        filters=[64, 128, 256, 512],
        num_blocks_per_stage=[5, 5, 5, 5],
        num_classes=num_classes,
        init_weights=init_weights
    )

