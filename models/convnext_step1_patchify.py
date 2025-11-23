

import torch
import torch.nn as nn

__all__ = [
    "LayerNormChannels",
    "ConvNeXtStem",
    "ConvNeXtHead",
    "ConvNeXtPatchifyClassifier",
]


class LayerNormChannels(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = self.norm(x)
        return x.transpose(-1, 1)


class ConvNeXtStem(nn.Sequential):

    def __init__(self, in_channels: int = 3, embed_dim: int = 96, patch_size: int = 4):
        super().__init__(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            LayerNormChannels(embed_dim),
        )


class ConvNeXtHead(nn.Sequential):

    def __init__(self, in_channels: int, num_classes: int = 10):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, num_classes),
        )


class ConvNeXtPatchifyClassifier(nn.Module):

    def __init__(
        self,
        *,
        patch_size: int = 4,
        embed_dim: int = 96,
        num_classes: int = 10,
        init_weights: bool = False,
    ):
        super().__init__()
        self.stem = ConvNeXtStem(3, embed_dim, patch_size)
        self.head = ConvNeXtHead(embed_dim, num_classes)

        if init_weights:
            self._reset_parameters()

    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.head(x)



