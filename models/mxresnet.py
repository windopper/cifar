"""
MXResNet: Modified ResNet with Mish activation function

참고 레포지토리: https://github.com/iamVarunAnand/image_classification
- Mish 활성화 함수 사용 (ReLU 대신)
- Global Average Pooling과 Global Max Pooling을 concatenate
- CIFAR-10용 최적화된 구조
- 파라미터 효율적인 설계 (약 0.27M 파라미터로 높은 성능 달성)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    """
    Mish 활성화 함수
    Mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
    """
    def __init__(self):
        super(Mish, self).__init__()
    
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class MXResNet:
    """
    Modified ResNet with Mish activation function
    
    주요 특징:
    1. Mish 활성화 함수 사용
    2. Global Average Pooling과 Global Max Pooling concatenate
    3. CIFAR-10용 최적화된 구조
    4. Basic Block 또는 Bottleneck Block 지원
    """
    
    @staticmethod
    def residual_module(data, K, stride, chan_dim=-1, red=False, reg=1e-4, 
                        bn_eps=2e-5, bn_mom=0.9, bottleneck=True, name="res_block"):
        """
        Residual block 생성
        
        Args:
            data: 입력 텐서
            K: 출력 채널 수
            stride: stride 값
            chan_dim: BatchNorm channel dimension (-1 for NHWC)
            red: 공간 차원 축소 여부
            reg: L2 정규화 계수 (PyTorch에서는 weight_decay로 처리)
            bn_eps: BatchNorm epsilon
            bn_mom: BatchNorm momentum
            bottleneck: Bottleneck 구조 사용 여부
            name: 블록 이름
        
        Returns:
            Residual block 출력
        """
        shortcut = data
        
        if bottleneck:
            # Bottleneck 구조: 1x1 -> 3x3 -> 1x1
            # 첫 번째 bottleneck block - 1x1
            bn1 = nn.BatchNorm2d(data.size(1), eps=bn_eps, momentum=bn_mom)
            act1 = Mish()
            conv1 = nn.Conv2d(data.size(1), int(K * 0.25), (1, 1), 
                            bias=False, padding=0)
            conv1.weight.data = nn.init.kaiming_normal_(conv1.weight.data, mode='fan_out', nonlinearity='relu')
            
            # conv block - 3x3
            bn2 = nn.BatchNorm2d(int(K * 0.25), eps=bn_eps, momentum=bn_mom)
            act2 = Mish()
            conv2 = nn.Conv2d(int(K * 0.25), int(K * 0.25), (3, 3), 
                             stride=stride, padding=1, bias=False)
            conv2.weight.data = nn.init.kaiming_normal_(conv2.weight.data, mode='fan_out', nonlinearity='relu')
            
            # 두 번째 bottleneck block - 1x1
            bn3 = nn.BatchNorm2d(int(K * 0.25), eps=bn_eps, momentum=bn_mom)
            act3 = Mish()
            conv3 = nn.Conv2d(int(K * 0.25), K, (1, 1), bias=False, padding=0)
            conv3.weight.data = nn.init.kaiming_normal_(conv3.weight.data, mode='fan_out', nonlinearity='relu')
            
            # shortcut 처리
            if red:
                shortcut_pool = nn.AvgPool2d(kernel_size=2, stride=stride, padding=0)
                shortcut_conv = nn.Conv2d(data.size(1), K, (1, 1), stride=1, bias=False)
                shortcut_conv.weight.data = nn.init.kaiming_normal_(shortcut_conv.weight.data, mode='fan_out', nonlinearity='relu')
                shortcut_bn = nn.BatchNorm2d(K, eps=bn_eps, momentum=bn_mom)
                shortcut = shortcut_bn(shortcut_conv(shortcut_pool(act1(bn1(data)))))
            
            # Forward pass
            out = bn1(data)
            out = act1(out)
            out = conv1(out)
            
            out = bn2(out)
            out = act2(out)
            out = conv2(out)
            
            out = bn3(out)
            out = act3(out)
            out = conv3(out)
            
            x = out + shortcut
            
        else:
            # Basic Block 구조: 3x3 -> 3x3
            # conv block 1 - 3x3
            bn1 = nn.BatchNorm2d(data.size(1), eps=bn_eps, momentum=bn_mom)
            act1 = Mish()
            conv1 = nn.Conv2d(data.size(1), K, (3, 3), stride=stride, 
                             padding=1, bias=False)
            conv1.weight.data = nn.init.kaiming_normal_(conv1.weight.data, mode='fan_out', nonlinearity='relu')
            
            # conv block 2 - 3x3
            bn2 = nn.BatchNorm2d(K, eps=bn_eps, momentum=bn_mom)
            act2 = Mish()
            conv2 = nn.Conv2d(K, K, (3, 3), padding=1, bias=False)
            conv2.weight.data = nn.init.kaiming_normal_(conv2.weight.data, mode='fan_out', nonlinearity='relu')
            
            # shortcut 처리
            if red and stride != (1, 1):
                shortcut_pool = nn.AvgPool2d(kernel_size=2, stride=stride, padding=0)
                shortcut_conv = nn.Conv2d(data.size(1), K, (1, 1), stride=1, bias=False)
                shortcut_conv.weight.data = nn.init.kaiming_normal_(shortcut_conv.weight.data, mode='fan_out', nonlinearity='relu')
                shortcut_bn = nn.BatchNorm2d(K, eps=bn_eps, momentum=bn_mom)
                shortcut = shortcut_bn(shortcut_conv(shortcut_pool(act1(bn1(data)))))
            
            # Forward pass
            out = bn1(data)
            out = act1(out)
            out = conv1(out)
            
            out = bn2(out)
            out = act2(out)
            out = conv2(out)
            
            x = out + shortcut
        
        return x
    
    @staticmethod
    def build(height, width, depth, classes, stages, filters, stem_type="imagenet", 
              bottleneck=True, reg=1e-4, bn_eps=2e-5, bn_mom=0.9):
        """
        MXResNet 모델 빌드
        
        Args:
            height: 입력 이미지 높이
            width: 입력 이미지 너비
            depth: 입력 채널 수
            classes: 클래스 수
            stages: 각 stage의 residual block 개수 리스트
            filters: 각 stage의 필터 수 리스트
            stem_type: 'imagenet' 또는 'cifar'
            bottleneck: Bottleneck 구조 사용 여부
            reg: L2 정규화 계수
            bn_eps: BatchNorm epsilon
            bn_mom: BatchNorm momentum
        
        Returns:
            MXResNet 모델
        """
        layers = []
        
        # Input block
        inputs = nn.Identity()  # 실제로는 forward에서 처리
        
        # Stem
        if stem_type == "imagenet":
            stem_conv1 = nn.Conv2d(depth, filters[0], (3, 3), stride=2, 
                                   padding=1, bias=False)
            stem_conv1.weight.data = nn.init.kaiming_normal_(stem_conv1.weight.data, mode='fan_out', nonlinearity='relu')
            stem_conv2 = nn.Conv2d(filters[0], filters[0], (3, 3), stride=1, 
                                  padding=1, bias=False)
            stem_conv2.weight.data = nn.init.kaiming_normal_(stem_conv2.weight.data, mode='fan_out', nonlinearity='relu')
            stem_conv3 = nn.Conv2d(filters[0], filters[0], (3, 3), stride=1, 
                                  padding=1, bias=False)
            stem_conv3.weight.data = nn.init.kaiming_normal_(stem_conv3.weight.data, mode='fan_out', nonlinearity='relu')
            stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            layers.extend([stem_conv1, stem_conv2, stem_conv3, stem_pool])
        elif stem_type == "cifar":
            stem_conv = nn.Conv2d(depth, filters[0], (3, 3), padding=1, bias=False)
            stem_conv.weight.data = nn.init.kaiming_normal_(stem_conv.weight.data, mode='fan_out', nonlinearity='relu')
            layers.append(stem_conv)
        
        # Stages
        n_layers = 1
        for i in range(len(stages)):
            stride = 2 if i > 0 else 1
            
            # 첫 번째 block (reduction)
            name = f"stage{i+1}_res_block1"
            # 실제 구현에서는 Sequential로 처리해야 함
            
            # 나머지 blocks
            for j in range(stages[i] - 1):
                name = f"stage{i+1}_res_block{j+2}"
                # 실제 구현에서는 Sequential로 처리해야 함
            
            if bottleneck:
                n_layers += (3 * stages[i])
            else:
                n_layers += (2 * stages[i])
        
        # Final BN => Mish -> Pool -> Concatenate -> Dense
        final_bn = nn.BatchNorm2d(filters[-1], eps=bn_eps, momentum=bn_mom)
        final_mish = Mish()
        global_avg_pool = nn.AdaptiveAvgPool2d(1)
        global_max_pool = nn.AdaptiveMaxPool2d(1)
        # Concatenate는 forward에서 처리
        classifier = nn.Linear(filters[-1] * 2, classes)  # Avg + Max = 2배
        classifier.weight.data = nn.init.kaiming_normal_(classifier.weight.data, mode='fan_out', nonlinearity='relu')
        
        n_layers += 1
        
        print(f"[INFO] MXResNet{n_layers} built successfully!")
        
        # 실제로는 nn.Module을 상속받은 클래스로 구현해야 함
        # 여기서는 구조만 보여줌
        return {
            'layers': layers,
            'stages': stages,
            'filters': filters,
            'final_bn': final_bn,
            'final_mish': final_mish,
            'global_avg_pool': global_avg_pool,
            'global_max_pool': global_max_pool,
            'classifier': classifier,
            'n_layers': n_layers
        }


class MXResidualBlock(nn.Module):
    """
    MXResNet Residual Block (Basic Block)
    Mish 활성화 함수 사용
    """
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1, bottleneck=False, 
                 bn_eps=2e-5, bn_mom=0.9):
        super(MXResidualBlock, self).__init__()
        
        self.bottleneck = bottleneck
        
        if bottleneck:
            # Bottleneck: 1x1 -> 3x3 -> 1x1
            bottleneck_channels = int(out_channels * 0.25)
            
            self.bn1 = nn.BatchNorm2d(in_channels, eps=bn_eps, momentum=bn_mom)
            self.mish1 = Mish()
            self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, 
                                   stride=1, bias=False)
            
            self.bn2 = nn.BatchNorm2d(bottleneck_channels, eps=bn_eps, momentum=bn_mom)
            self.mish2 = Mish()
            self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            
            self.bn3 = nn.BatchNorm2d(bottleneck_channels, eps=bn_eps, momentum=bn_mom)
            self.mish3 = Mish()
            self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1,
                                   stride=1, bias=False)
        else:
            # Basic Block: 3x3 -> 3x3
            self.bn1 = nn.BatchNorm2d(in_channels, eps=bn_eps, momentum=bn_mom)
            self.mish1 = Mish()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                                   stride=stride, padding=1, bias=False)
            
            self.bn2 = nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_mom)
            self.mish2 = Mish()
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=1, padding=1, bias=False)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            if stride != 1:
                # Average pooling for spatial reduction
                # stride=2일 때 kernel_size=2, padding=0으로 정확히 절반 크기로 축소
                self.shortcut = nn.Sequential(
                    nn.AvgPool2d(kernel_size=stride, stride=stride, padding=0),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                             stride=1, bias=False),
                    nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_mom)
                )
            else:
                # Only channel projection
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1,
                             stride=1, bias=False),
                    nn.BatchNorm2d(out_channels, eps=bn_eps, momentum=bn_mom)
                )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        if self.bottleneck:
            out = self.bn1(x)
            out = self.mish1(out)
            out = self.conv1(out)
            
            out = self.bn2(out)
            out = self.mish2(out)
            out = self.conv2(out)
            
            out = self.bn3(out)
            out = self.mish3(out)
            out = self.conv3(out)
        else:
            out = self.bn1(x)
            out = self.mish1(out)
            out = self.conv1(out)
            
            out = self.bn2(out)
            out = self.mish2(out)
            out = self.conv2(out)
        
        out = out + identity
        return out


class MXResNet(nn.Module):
    """
    MXResNet: Modified ResNet with Mish activation
    
    구조:
    - Stem (CIFAR: 단일 3x3 conv)
    - Stages (각 stage는 여러 residual blocks)
    - Global Average Pooling + Global Max Pooling (concatenate)
    - Classifier
    """
    
    def __init__(self, stages, filters, num_classes=10, stem_type="cifar", 
                 bottleneck=False, bn_eps=2e-5, bn_mom=0.9, init_weights=False):
        super(MXResNet, self).__init__()
        
        self.stem_type = stem_type
        self.bottleneck = bottleneck
        
        # Stem
        if stem_type == "imagenet":
            self.stem_conv1 = nn.Conv2d(3, filters[0], kernel_size=3, stride=2, 
                                        padding=1, bias=False)
            self.stem_bn1 = nn.BatchNorm2d(filters[0], eps=bn_eps, momentum=bn_mom)
            self.stem_conv2 = nn.Conv2d(filters[0], filters[0], kernel_size=3, 
                                       stride=1, padding=1, bias=False)
            self.stem_bn2 = nn.BatchNorm2d(filters[0], eps=bn_eps, momentum=bn_mom)
            self.stem_conv3 = nn.Conv2d(filters[0], filters[0], kernel_size=3, 
                                       stride=1, padding=1, bias=False)
            self.stem_bn3 = nn.BatchNorm2d(filters[0], eps=bn_eps, momentum=bn_mom)
            self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        elif stem_type == "cifar":
            self.stem_conv = nn.Conv2d(3, filters[0], kernel_size=3, 
                                       padding=1, bias=False)
            self.stem_bn = nn.BatchNorm2d(filters[0], eps=bn_eps, momentum=bn_mom)
        
        # Stages
        self.stages = nn.ModuleList()
        in_channels = filters[0]
        
        for i in range(len(stages)):
            stride = 2 if i > 0 else 1
            out_channels = filters[i + 1]
            
            # 첫 번째 block (reduction)
            blocks = []
            blocks.append(MXResidualBlock(in_channels, out_channels, stride=stride,
                                         bottleneck=bottleneck, bn_eps=bn_eps, bn_mom=bn_mom))
            
            # 나머지 blocks
            for _ in range(stages[i] - 1):
                blocks.append(MXResidualBlock(out_channels, out_channels, stride=1,
                                            bottleneck=bottleneck, bn_eps=bn_eps, bn_mom=bn_mom))
            
            self.stages.append(nn.Sequential(*blocks))
            in_channels = out_channels
        
        # Final layers
        self.final_bn = nn.BatchNorm2d(filters[-1], eps=bn_eps, momentum=bn_mom)
        self.final_mish = Mish()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Linear(filters[-1] * 2, num_classes)  # Avg + Max = 2배
        
        if init_weights:
            self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Stem
        if self.stem_type == "imagenet":
            x = self.stem_conv1(x)
            x = self.stem_bn1(x)
            x = self.stem_conv2(x)
            x = self.stem_bn2(x)
            x = self.stem_conv3(x)
            x = self.stem_bn3(x)
            x = self.stem_pool(x)
        elif self.stem_type == "cifar":
            x = self.stem_conv(x)
            x = self.stem_bn(x)
        
        # Stages
        for stage in self.stages:
            x = stage(x)
        
        # Final BN => Mish
        x = self.final_bn(x)
        x = self.final_mish(x)
        
        # Global Pooling (Average + Max)
        avg_pool = self.global_avg_pool(x)
        max_pool = self.global_max_pool(x)
        
        # Concatenate
        x = torch.cat([avg_pool, max_pool], dim=1)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.classifier(x)
        
        return x


def MXResNet20(num_classes=10, init_weights=False):
    """MXResNet20: stages=[3, 3, 3], filters=[16, 16, 32, 64]"""
    return MXResNet(stages=[3, 3, 3], filters=[16, 16, 32, 64],
                   num_classes=num_classes, stem_type="cifar", 
                   bottleneck=False, init_weights=init_weights)


def MXResNet32(num_classes=10, init_weights=False):
    """MXResNet32: stages=[5, 5, 5], filters=[16, 16, 32, 64]"""
    return MXResNet(stages=[5, 5, 5], filters=[16, 16, 32, 64],
                   num_classes=num_classes, stem_type="cifar",
                   bottleneck=False, init_weights=init_weights)


def MXResNet44(num_classes=10, init_weights=False):
    """MXResNet44: stages=[7, 7, 7], filters=[16, 16, 32, 64]"""
    return MXResNet(stages=[7, 7, 7], filters=[16, 16, 32, 64],
                   num_classes=num_classes, stem_type="cifar",
                   bottleneck=False, init_weights=init_weights)


def MXResNet56(num_classes=10, init_weights=False):
    """MXResNet56: stages=[9, 9, 9], filters=[16, 16, 32, 64]"""
    return MXResNet(stages=[9, 9, 9], filters=[16, 16, 32, 64],
                   num_classes=num_classes, stem_type="cifar",
                   bottleneck=False, init_weights=init_weights)

