"""
DeepBaselineNetBN3ResidualDLATree:

DeepBaselineNetBN3Residual에서 사용한 ResidualBlock을 기반으로 DLA(Tree) 구조를
적용한 변형 모델.

설계 핵심:
1. ResidualBlock 재사용
   - 기존 DeepBaselineNetBN3Residual과 동일한 ResidualBlock을 사용하여 학습 안정성 유지
2. DLA(Tree) 계층 도입
   - dla.py의 Root/Tree 아이디어를 차용하여 다중 깊이의 특징을 계층적으로 집계
   - ResidualBlock을 Tree의 기본 block으로 사용
3. 채널 구성
   - (64 -> 64 -> 128 -> 256 -> 256 -> 512 -> 512) 순으로 채널 확장
   - Tree 단계마다 stride를 통해 공간 크기를 점진적으로 축소 (32 -> 16 -> 8 -> 4)
4. 분류기
   - AdaptiveAvgPool2d(1) 이후 전형적인 FC 레이어로 CIFAR-10 분류
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .deep_baseline3_bn_residual import ResidualBlock


class Root(nn.Module):
    """여러 노드 출력을 concat 후 1x1 Conv로 집계"""

    def __init__(self, in_channels, out_channels, kernel_size=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, xs):
        x = torch.cat(xs, dim=1)
        out = F.relu(self.bn(self.conv(x)))
        return out


class Tree(nn.Module):
    """
    DLA에서 사용되는 계층적 Aggregation 트리.

    Args:
        block: 기본 블록 (여기서는 ResidualBlock)
        in_channels (int): 입력 채널 수
        out_channels (int): 출력 채널 수
        level (int): 트리 깊이
        stride (int): 다운샘플링 stride
    """

    def __init__(self, block, in_channels, out_channels, level=1, stride=1):
        super().__init__()
        self.level = level

        if level == 1:
            self.root = Root(2 * out_channels, out_channels)
            self.left_node = block(in_channels, out_channels, stride=stride)
            self.right_node = block(out_channels, out_channels, stride=1)
        else:
            self.root = Root((level + 2) * out_channels, out_channels)
            for i in reversed(range(1, level)):
                subtree = Tree(block, in_channels, out_channels, level=i, stride=stride)
                setattr(self, f"level_{i}", subtree)
            self.prev_root = block(in_channels, out_channels, stride=stride)
            self.left_node = block(out_channels, out_channels, stride=1)
            self.right_node = block(out_channels, out_channels, stride=1)

    def forward(self, x):
        xs = [self.prev_root(x)] if self.level > 1 else []
        for i in reversed(range(1, self.level)):
            level_i = getattr(self, f"level_{i}")
            x = level_i(x)
            xs.append(x)
        x = self.left_node(x)
        xs.append(x)
        x = self.right_node(x)
        xs.append(x)
        out = self.root(xs)
        return out


class DeepBaselineNetBN3ResidualDLATree(nn.Module):
    """
    DeepBaselineNetBN3Residual + DLA(Tree) 하이브리드 모델.

    - 초기 두 단계는 ResidualBlock으로 특징 추출
    - 이후 Tree 모듈을 통해 계층적 집계를 수행
    """

    def __init__(self, num_classes: int = 10, init_weights: bool = False):
        super().__init__()

        self.base = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # stage1-2: 원본 ResidualBlock 단계
        self.stage1 = ResidualBlock(64, 64, stride=1)
        self.stage2 = ResidualBlock(64, 128, stride=1)

        # DLA Tree stages
        self.layer3 = Tree(ResidualBlock, in_channels=128, out_channels=256, level=1, stride=1)
        self.layer4 = Tree(ResidualBlock, in_channels=256, out_channels=256, level=2, stride=2)
        self.layer5 = Tree(ResidualBlock, in_channels=256, out_channels=512, level=2, stride=2)
        self.layer6 = Tree(ResidualBlock, in_channels=512, out_channels=512, level=1, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(256, num_classes),
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
        out = self.base(x)
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def test():
    model = DeepBaselineNetBN3ResidualDLATree(init_weights=True)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print("output shape:", y.shape)


if __name__ == "__main__":
    test()

