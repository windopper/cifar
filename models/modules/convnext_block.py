"""
ConvNeXt Block 모듈

ConvNeXt 스타일의 레지듀얼 블록을 제공합니다.
이 모듈은 여러 모델에서 공통으로 사용되는 ConvNeXt 블록 구현입니다.

구조:
- 입력 x
- Main path: Depthwise Conv (7x7) -> LayerNorm -> Pointwise Conv (expansion) -> GELU -> Pointwise Conv (projection)
- Skip connection: identity
- 출력: main_path + shortcut
"""

import torch
import torch.nn as nn


class LayerNormChannels(nn.Module):
    """
    채널 차원 기준 LayerNorm.
    
    ConvNeXt는 채널 우선(C, H, W) 텐서를 다루므로, PyTorch 기본 LayerNorm을 활용하려면
    (B, H, W, C) 형태로 전치했다가 되돌리는 보조 모듈이 필요합니다.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, -1)
        x = self.norm(x)
        return x.transpose(-1, 1)


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt 스타일의 레지듀얼 블록
    
    구조:
    - Depthwise Convolution (7x7)
    - Layer Normalization
    - Pointwise Convolution (expansion, 1x1) - 채널 확장
    - GELU 활성화 함수
    - Pointwise Convolution (projection, 1x1) - 채널 축소
    - Residual connection
    
    Args:
        channels: 입력/출력 채널 수
        kernel_size: Depthwise convolution의 커널 크기 (기본값=7)
        mlp_ratio: MLP 확장 비율 (기본값=4, hidden_channels = channels * mlp_ratio)
        drop_prob: Dropout 확률 (기본값=0.0)
        layer_scale_init_value: Layer scale 초기화 값 (기본값=1e-6, 0이면 사용 안 함)
    """

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 7,
        mlp_ratio: int = 4,
        drop_prob: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        hidden_channels = channels * mlp_ratio

        # Depthwise Convolution (7x7)
        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,  # depthwise convolution
            bias=False,
        )
        
        # Layer Normalization
        self.norm = LayerNormChannels(channels)
        
        # Pointwise Convolution (expansion, 1x1) - 채널 확장
        self.pwconv1 = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)
        
        # GELU 활성화 함수
        self.act = nn.GELU()
        
        # Pointwise Convolution (projection, 1x1) - 채널 축소
        self.pwconv2 = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(drop_prob) if drop_prob > 0 else nn.Identity()

        # Layer Scale (선택적)
        if layer_scale_init_value > 0:
            gamma = layer_scale_init_value * torch.ones(channels)
            self.gamma = nn.Parameter(gamma.view(1, -1, 1, 1))
        else:
            self.gamma = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: 입력 텐서 [batch, channels, H, W]
            
        Returns:
            out: 출력 텐서 [batch, channels, H, W]
        """
        residual = x

        # Depthwise Convolution
        out = self.dwconv(x)
        
        # Layer Normalization
        out = self.norm(out)
        
        # Pointwise Convolution (expansion)
        out = self.pwconv1(out)
        
        # GELU 활성화
        out = self.act(out)
        
        # Pointwise Convolution (projection)
        out = self.pwconv2(out)
        
        # Dropout
        out = self.dropout(out)

        # Layer Scale (선택적)
        if self.gamma is not None:
            out = self.gamma * out

        # Residual connection
        return residual + out


class StridedConvNeXtBlock(nn.Module):
    """
    Stride를 지원하는 ConvNeXt 블록
    
    다운샘플링이 필요한 경우 depthwise convolution에 stride를 적용하고,
    채널 수가 변경되는 경우 projection shortcut을 사용합니다.
    
    Args:
        in_channels: 입력 채널 수
        out_channels: 출력 채널 수
        stride: Depthwise convolution의 stride (기본값=1)
        kernel_size: Depthwise convolution의 커널 크기 (기본값=7)
        mlp_ratio: MLP 확장 비율 (기본값=4)
        drop_prob: Dropout 확률 (기본값=0.0)
        layer_scale_init_value: Layer scale 초기화 값 (기본값=1e-6)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        *,
        kernel_size: int = 7,
        mlp_ratio: int = 4,
        drop_prob: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        padding = (kernel_size - 1) // 2
        hidden_channels = out_channels * mlp_ratio
        
        # Depthwise Convolution with stride
        self.dwconv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # depthwise convolution
            bias=False,
        )
        
        # Layer Normalization
        self.norm = LayerNormChannels(in_channels)
        
        # Pointwise Convolution (expansion, 1x1) - 채널 확장
        self.pwconv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False)
        
        # GELU 활성화 함수
        self.act = nn.GELU()
        
        # Pointwise Convolution (projection, 1x1) - 채널 축소
        self.pwconv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(drop_prob) if drop_prob > 0 else nn.Identity()
        
        # Layer Scale (선택적)
        if layer_scale_init_value > 0:
            gamma = layer_scale_init_value * torch.ones(out_channels)
            self.gamma = nn.Parameter(gamma.view(1, -1, 1, 1))
        else:
            self.gamma = None
        
        # Shortcut connection
        # 채널 수가 다르거나 stride가 1이 아닌 경우 projection 필요
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: 입력 텐서 [batch, in_channels, H, W]
            
        Returns:
            out: 출력 텐서 [batch, out_channels, H', W']
        """
        residual = self.shortcut(x)
        
        # Depthwise Convolution
        out = self.dwconv(x)
        
        # Layer Normalization
        out = self.norm(out)
        
        # Pointwise Convolution (expansion)
        out = self.pwconv1(out)
        
        # GELU 활성화
        out = self.act(out)
        
        # Pointwise Convolution (projection)
        out = self.pwconv2(out)
        
        # Dropout
        out = self.dropout(out)
        
        # Layer Scale (선택적)
        if self.gamma is not None:
            out = self.gamma * out
        
        # Residual connection
        return residual + out

