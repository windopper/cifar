"""
Convolutional Vision Transformer (CvT) for CIFAR-10

Reference: https://github.com/microsoft/CvT/blob/main/lib/models/cls_cvt.py
Paper: "CvT: Introducing Convolutions to Vision Transformers"
       (Wu et al., ICCV 2021)

This implementation adapts the CvT architecture for CIFAR-10 (32x32 images, 10 classes).
The model uses a 3-stage hierarchy with convolutional token embeddings and 
convolutional projection in the multi-head self-attention mechanism.

Architecture:
- Stage 1: 32x32 -> 8x8, 64 channels, 1 transformer block
- Stage 2: 8x8 -> 4x4, 128 channels, 2 transformer blocks  
- Stage 3: 4x4 -> 2x2, 256 channels, 12 transformer blocks
- Global average pooling + linear classifier

Total parameters: ~10.3M (comparable to WideResNet-16-8 and PyramidNet-110-150)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import DropPath, trunc_normal_


class Mlp(nn.Module):
    """Multi-layer perceptron."""
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvEmbed(nn.Module):
    """Image to Patch Embedding using Conv2D."""
    def __init__(self, patch_size=7, in_chans=3, embed_dim=64, stride=4, padding=2, norm_layer=None):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, 
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


class Attention(nn.Module):
    """Convolutional Multi-head Self-Attention."""
    def __init__(self, dim_in, dim_out, num_heads, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 kernel_size=3, stride_kv=1, stride_q=1, padding_kv=1, padding_q=1):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        self.scale = dim_out ** -0.5

        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q, stride_q, 'linear' if qkv_bias else 'conv'
        )
        self.conv_proj_k = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv, stride_kv, 'linear' if qkv_bias else 'conv'
        )
        self.conv_proj_v = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv, stride_kv, 'linear' if qkv_bias else 'conv'
        )

        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(self, dim_in, dim_out, kernel_size, padding, stride, method):
        if method == 'dw_bn':
            proj = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size, stride, padding, bias=False, groups=dim_in),
                nn.BatchNorm2d(dim_in),
                nn.ReLU(inplace=True),
                nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(dim_out),
            )
        elif method == 'avg':
            proj = nn.Sequential(
                nn.AvgPool2d(kernel_size, stride, padding),
            )
        elif method == 'linear':
            proj = None
        else:
            proj = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(dim_out),
            )
        return proj

    def forward_conv(self, x, h, w):
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        
        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = x
        q = rearrange(q, 'b c h w -> b (h w) c')
        
        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = x
        k = rearrange(k, 'b c h w -> b (h w) c')
        
        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = x
        v = rearrange(v, 'b c h w -> b (h w) c')
        
        return q, k, v

    def forward(self, x, h, w):
        B, N, C = x.shape
        
        q, k, v = self.forward_conv(x, h, w)
        
        q = self.proj_q(q).reshape(B, -1, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)
        k = self.proj_k(k).reshape(B, -1, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)
        v = self.proj_v(v).reshape(B, -1, self.num_heads, self.dim // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class Block(nn.Module):
    """Transformer Block with Convolutional Attention."""
    def __init__(self, dim_in, dim_out, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., 
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 kernel_size=3, stride_kv=1, stride_q=1, padding_kv=1, padding_q=1):
        super().__init__()
        self.norm1 = norm_layer(dim_in)
        self.attn = Attention(
            dim_in, dim_out, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, kernel_size=kernel_size,
            stride_kv=stride_kv, stride_q=stride_q, padding_kv=padding_kv, padding_q=padding_q
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim_out)
        mlp_hidden_dim = int(dim_out * mlp_ratio)
        self.mlp = Mlp(in_features=dim_out, hidden_features=mlp_hidden_dim, 
                       act_layer=act_layer, drop=drop)

    def forward(self, x, h, w):
        x = x + self.drop_path(self.attn(self.norm1(x), h, w))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionTransformerStage(nn.Module):
    """A single stage of Vision Transformer."""
    def __init__(self, patch_size, patch_stride, patch_padding, in_chans, embed_dim, depth,
                 num_heads, mlp_ratio, qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 kernel_size=3, padding_kv=1, stride_kv=1, padding_q=1, stride_q=1):
        super().__init__()
        
        self.patch_embed = ConvEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            stride=patch_stride,
            padding=patch_padding,
            norm_layer=None
        )
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        blocks = []
        for j in range(depth):
            blocks.append(
                Block(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    kernel_size=kernel_size,
                    stride_kv=stride_kv,
                    stride_q=stride_q,
                    padding_kv=padding_kv,
                    padding_q=padding_q
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.size()
        
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x, H, W)
        
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


class CvTCIFAR(nn.Module):
    """Convolutional Vision Transformer for CIFAR-10.
    
    This is a 3-stage architecture adapted for 32x32 images with ~10M parameters.
    """
    def __init__(self, in_chans=3, num_classes=10, init_weights=False):
        super().__init__()
        
        # Stage 1: 32x32 -> 8x8, 64 channels
        self.stage1 = VisionTransformerStage(
            patch_size=7,
            patch_stride=4,
            patch_padding=2,
            in_chans=in_chans,
            embed_dim=64,
            depth=1,
            num_heads=1,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            kernel_size=3,
            padding_kv=1,
            stride_kv=1,
            padding_q=1,
            stride_q=1
        )
        
        # Stage 2: 8x8 -> 4x4, 128 channels
        self.stage2 = VisionTransformerStage(
            patch_size=3,
            patch_stride=2,
            patch_padding=1,
            in_chans=64,
            embed_dim=128,
            depth=2,
            num_heads=2,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            kernel_size=3,
            padding_kv=1,
            stride_kv=1,
            padding_q=1,
            stride_q=1
        )
        
        # Stage 3: 4x4 -> 2x2, 256 channels
        self.stage3 = VisionTransformerStage(
            patch_size=3,
            patch_stride=2,
            patch_padding=1,
            in_chans=128,
            embed_dim=256,
            depth=12,
            num_heads=4,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            kernel_size=3,
            padding_kv=1,
            stride_kv=1,
            padding_q=1,
            stride_q=1
        )
        
        self.norm = nn.LayerNorm(256)
        self.head = nn.Linear(256, num_classes)
        
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Stage 1: 32x32 -> 8x8
        x = self.stage1(x)
        
        # Stage 2: 8x8 -> 4x4
        x = self.stage2(x)
        
        # Stage 3: 4x4 -> 2x2
        x = self.stage3(x)
        
        # Global average pooling
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        x = torch.mean(x, dim=1)
        
        # Classification head
        x = self.head(x)
        
        return x


def cvt_cifar_10m(init_weights=False):
    """Create a CvT model for CIFAR-10 with approximately 10M parameters."""
    return CvTCIFAR(num_classes=10, init_weights=init_weights)
