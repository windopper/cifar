
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

__all__ = ["DeepBaselineNetBN3ResidualShakeDrop"]


class ShakeDropFunction(Function):
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

    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob <= 0.0:
            return x
        return ShakeDropFunction.apply(x, self.training, self.drop_prob)


class ResidualBlockShakeDrop(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, drop_prob: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.shake_drop = ShakeDrop(drop_prob) if drop_prob > 0.0 else None

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.shake_drop is not None:
            out = self.shake_drop(out)

        out += identity
        out = F.relu(out)
        return out


class DeepBaselineNetBN3ResidualShakeDrop(nn.Module):
    def __init__(
        self,
        init_weights: bool = False,
        min_shakedrop_prob: float = 0.0,
        max_shakedrop_prob: float = 0.5
    ):
        super().__init__()

        if max_shakedrop_prob < min_shakedrop_prob:
            raise ValueError("max_shakedrop_prob는 min_shakedrop_prob보다 크거나 같아야 합니다.")

        self.total_blocks = 5
        self.drop_schedule = self._generate_shakedrop_probs(
            total_blocks=self.total_blocks,
            min_prob=min_shakedrop_prob,
            max_prob=max_shakedrop_prob
        )
        self._block_ptr = 0

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.res_block1 = self._make_block(64, 64, stride=1)
        self.res_block2 = self._make_block(64, 128, stride=1)
        self.res_block3 = self._make_block(128, 256, stride=1)
        self.res_block4 = self._make_block(256, 256, stride=1)
        self.res_block5 = self._make_block(256, 512, stride=1)

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

    def _next_drop_prob(self):
        prob = self.drop_schedule[self._block_ptr]
        self._block_ptr += 1
        return prob

    def _make_block(self, in_channels, out_channels, stride):
        return ResidualBlockShakeDrop(
            in_channels,
            out_channels,
            stride=stride,
            drop_prob=self._next_drop_prob()
        )

    def _generate_shakedrop_probs(self, total_blocks: int, min_prob: float, max_prob: float):
        if total_blocks <= 1:
            return [max_prob]
        step = (max_prob - min_prob) / (total_blocks - 1)
        return [min_prob + i * step for i in range(total_blocks)]

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

