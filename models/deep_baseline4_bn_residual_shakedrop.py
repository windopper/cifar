"""
ResNet-18 with ShakeDrop regularization.

이 구현은 `models/deep_baseline4_bn_residual.py`의 CIFAR-10 전용 ResNet-18을 기반으로
ShakeDrop 규제를 각 BasicBlock에 적용한 변형입니다.
ShakeDrop: https://arxiv.org/abs/1705.07485
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

__all__ = ["ResNet18ShakeDrop"]


class ShakeDropFunction(Function):
    """Autograd function that applies ShakeDrop during forward/backward."""

    @staticmethod
    def forward(ctx, x, training: bool, drop_prob: float):
        ctx.drop_prob = drop_prob

        if drop_prob <= 0.0:
            ctx.training_mode = False
            return x

        if not training:
            ctx.training_mode = False
            return (1.0 - drop_prob) * x

        ctx.training_mode = True
        batch_size = x.size(0)
        reduce_dims = [1] * (x.dim() - 1)

        gate = torch.empty(
            (batch_size, *reduce_dims),
            device=x.device,
            dtype=x.dtype
        ).bernoulli_(1.0 - drop_prob)
        alpha = torch.empty_like(gate).uniform_(-1.0, 1.0)

        ctx.save_for_backward(gate)
        return (gate + (1.0 - gate) * alpha) * x

    @staticmethod
    def backward(ctx, grad_output):
        drop_prob = ctx.drop_prob

        if not ctx.training_mode:
            if drop_prob <= 0.0:
                return grad_output, None, None
            return (1.0 - drop_prob) * grad_output, None, None

        (gate,) = ctx.saved_tensors
        beta = torch.empty_like(gate).uniform_(-1.0, 1.0)
        grad_input = (gate + (1.0 - gate) * beta) * grad_output
        return grad_input, None, None


class ShakeDrop(nn.Module):
    """Module wrapper for the ShakeDrop autograd function."""

    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob <= 0.0:
            return x
        return ShakeDropFunction.apply(x, self.training, self.drop_prob)


class BasicBlock(nn.Module):
    """
    ResNet BasicBlock with optional ShakeDrop on the residual branch.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, drop_prob: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride,
            padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes)
            )

        self.shake_drop = ShakeDrop(drop_prob) if drop_prob > 0.0 else None

    def forward(self, x):
        identity = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.shake_drop is not None:
            out = self.shake_drop(out)

        out += identity
        out = F.relu(out)
        return out


class ResNet18ShakeDrop(nn.Module):
    """
    ResNet-18 variant where each residual block is regularized with ShakeDrop.

    Args:
        num_classes (int): 출력 클래스 수.
        init_weights (bool): Kaiming 초기화 수행 여부.
        min_shakedrop_prob (float): 첫 번째 블록에 적용할 최소 drop 확률.
        max_shakedrop_prob (float): 마지막 블록에 적용할 최대 drop 확률.
    """

    def __init__(
        self,
        num_classes: int = 10,
        init_weights: bool = False,
        min_shakedrop_prob: float = 0.0,
        max_shakedrop_prob: float = 0.5
    ):
        super().__init__()

        if max_shakedrop_prob < min_shakedrop_prob:
            raise ValueError("max_shakedrop_prob는 min_shakedrop_prob보다 크거나 같아야 합니다.")

        self.in_planes = 64
        self.layer_config = [2, 2, 2, 2]
        self.total_blocks = sum(self.layer_config)

        self.drop_schedule = self._generate_shakedrop_probs(
            total_blocks=self.total_blocks,
            min_prob=min_shakedrop_prob,
            max_prob=max_shakedrop_prob
        )
        self._block_ptr = 0

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(BasicBlock, 64, self.layer_config[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, self.layer_config[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, self.layer_config[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, self.layer_config[3], stride=2)

        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

        if init_weights:
            self._initialize_weights()

    def _generate_shakedrop_probs(self, total_blocks: int, min_prob: float, max_prob: float):
        if total_blocks <= 1:
            return [max_prob]

        step = (max_prob - min_prob) / (total_blocks - 1)
        return [min_prob + i * step for i in range(total_blocks)]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            drop_prob = self.drop_schedule[self._block_ptr]
            layers.append(block(self.in_planes, planes, stride, drop_prob=drop_prob))
            self.in_planes = planes * block.expansion
            self._block_ptr += 1
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
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

