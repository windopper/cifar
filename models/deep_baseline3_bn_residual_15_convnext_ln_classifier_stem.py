"""
DeepBaselineNetBN3Residual15ConvNeXtLNClassifierStem: 새로운 ConvNeXt 구조에 맞춘 버전

설계 의도:
1. Residual 클래스 기반 구조
   - gamma 파라미터를 사용한 스케일된 residual connection
   - 초기에는 gamma=0으로 시작하여 identity mapping에 가깝게 학습 시작

2. ConvNeXt 블록 구조
   - Depthwise Convolution -> LayerNorm -> Pointwise Conv (expansion) -> GELU -> Pointwise Conv (projection)
   - Residual connection을 통한 그래디언트 흐름 개선
   - Layer Normalization 사용으로 배치 크기에 덜 민감한 학습

3. Stage 기반 구조
   - DownsampleBlock으로 다운샘플링과 채널 변경을 분리
   - Stage 클래스로 각 스테이지의 블록들을 관리
   - ConvNeXtBody로 전체 스테이지들을 관리

4. 가중치 초기화
   - Conv2d, Linear: normal(0, 0.02)
   - LayerNorm: weight=1, bias=0
   - Residual.gamma: 0으로 초기화

5. 하이퍼파라미터
   - channel_list = [64, 128, 256, 512]
   - num_blocks_list = [3, 3, 9, 3] (총 15개 블록)
   - kernel_size = 7 (일반적인 ConvNeXt 설정)
   - patch_size = 1 (기본값)
   - res_p_drop = 0. (기본값)

네트워크 구조:
- Stem: 패치 임베딩 (3 -> 64, patch_size로 다운샘플링)
- Stage 1: 3개의 ConvNeXt block (64 -> 64)
- Stage 2: 3개의 ConvNeXt block (64 -> 128, DownsampleBlock으로 다운샘플링)
- Stage 3: 9개의 ConvNeXt block (128 -> 256, DownsampleBlock으로 다운샘플링)
- Stage 4: 3개의 ConvNeXt block (256 -> 512, DownsampleBlock으로 다운샘플링)
- Head: AdaptiveAvgPool2d(1) -> Flatten -> LayerNorm -> Linear

참고:
- ConvNeXt 논문: "A ConvNet for the 2020s" (Liu et al., 2022)
- ConvNeXt는 Transformer의 디자인 원칙을 ConvNet에 적용한 모델
- DownsampleBlock은 LayerNorm을 먼저 적용한 후 다운샘플링
"""
import torch
import torch.nn as nn


class LayerNormChannels(nn.Module):
    """
    채널 차원 기준 LayerNorm.
    
    ConvNeXt는 채널 우선(C, H, W) 텐서를 다루므로, PyTorch 기본 LayerNorm을 활용하려면
    (B, H, W, C) 형태로 전치했다가 되돌리는 보조 모듈이 필요합니다.
    """
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
    
    def forward(self, x):
        x = x.transpose(1, -1)
        x = self.norm(x)
        x = x.transpose(-1, 1)
        return x


class Residual(nn.Module):
    """
    스케일된 residual connection을 제공하는 기본 클래스.
    
    gamma 파라미터를 사용하여 residual path를 스케일링하며,
    초기값이 0이므로 학습 초기에는 identity mapping에 가깝게 시작합니다.
    """
    def __init__(self, *layers):
        super().__init__()
        self.residual = nn.Sequential(*layers)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        return x + self.gamma * self.residual(x)


class ConvNeXtBlock(Residual):
    """
    ConvNeXt 스타일의 레지듀얼 블록.
    
    구조:
    - Depthwise Convolution (kernel_size x kernel_size)
    - Layer Normalization
    - Pointwise Convolution (expansion, 1x1) - 채널 확장
    - GELU 활성화 함수
    - Pointwise Convolution (projection, 1x1) - 채널 축소
    - Dropout (선택적)
    - Residual connection (gamma 스케일링)
    
    Args:
        channels: 입력/출력 채널 수
        kernel_size: Depthwise convolution의 커널 크기
        mult: MLP 확장 비율 (기본값=4, hidden_channels = channels * mult)
        p_drop: Dropout 확률 (기본값=0.)
    """
    def __init__(self, channels, kernel_size, mult=4, p_drop=0.):
        padding = (kernel_size - 1) // 2
        hidden_channels = channels * mult
        super().__init__(
            nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels),
            LayerNormChannels(channels),
            nn.Conv2d(channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, 1),
            nn.Dropout(p_drop)
        )


class DownsampleBlock(nn.Sequential):
    """
    다운샘플링을 위한 블록.
    
    LayerNorm을 먼저 적용한 후 stride를 사용하여 다운샘플링하고 채널을 변경합니다.
    
    Args:
        in_channels: 입력 채널 수
        out_channels: 출력 채널 수
        stride: 다운샘플링 stride (기본값=2)
    """
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__(
            LayerNormChannels(in_channels),
            nn.Conv2d(in_channels, out_channels, stride, stride=stride)
        )


class Stage(nn.Sequential):
    """
    하나의 ConvNeXt 스테이지를 구성하는 클래스.
    
    채널이 변경되는 경우 DownsampleBlock을 먼저 추가하고,
    그 다음 지정된 개수만큼 ConvNeXtBlock을 추가합니다.
    
    Args:
        in_channels: 입력 채널 수
        out_channels: 출력 채널 수
        num_blocks: 이 스테이지에 포함될 ConvNeXt block의 개수
        kernel_size: Depthwise convolution의 커널 크기
        p_drop: Dropout 확률 (기본값=0.)
    """
    def __init__(self, in_channels, out_channels, num_blocks, kernel_size, p_drop=0.):
        layers = [] if in_channels == out_channels else [DownsampleBlock(in_channels, out_channels)]
        layers += [ConvNeXtBlock(out_channels, kernel_size, p_drop=p_drop) for _ in range(num_blocks)]
        super().__init__(*layers)


class ConvNeXtBody(nn.Sequential):
    """
    ConvNeXt의 본체를 구성하는 클래스.
    
    여러 스테이지를 순차적으로 연결합니다.
    
    Args:
        in_channels: 첫 번째 스테이지의 입력 채널 수
        channel_list: 각 스테이지의 출력 채널 수 리스트
        num_blocks_list: 각 스테이지의 블록 개수 리스트
        kernel_size: Depthwise convolution의 커널 크기
        p_drop: Dropout 확률 (기본값=0.)
    """
    def __init__(self, in_channels, channel_list, num_blocks_list, kernel_size, p_drop=0.):
        layers = []
        for out_channels, num_blocks in zip(channel_list, num_blocks_list):
            layers.append(Stage(in_channels, out_channels, num_blocks, kernel_size, p_drop))
            in_channels = out_channels
        super().__init__(*layers)


class Stem(nn.Sequential):
    """
    패치 임베딩을 위한 Stem 클래스.
    
    ConvNeXt 스타일의 패치 임베딩으로, 입력 이미지를 패치 크기만큼 다운샘플링하고
    채널 수를 변경한 후 LayerNorm을 적용합니다.
    
    Args:
        in_channels: 입력 채널 수
        out_channels: 출력 채널 수
        patch_size: 패치 크기 (커널 크기 및 stride로 사용)
    """
    def __init__(self, in_channels, out_channels, patch_size):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, patch_size, stride=patch_size),
            LayerNormChannels(out_channels)
        )


class Head(nn.Sequential):
    """
    분류 헤드 클래스.
    
    AdaptiveAvgPool2d로 공간 차원을 1x1로 축소하고,
    Flatten으로 1D 벡터로 변환한 후,
    LayerNorm으로 정규화하고 Linear로 최종 분류합니다.
    
    Args:
        in_channels: 입력 채널 수
        classes: 출력 클래스 수
    """
    def __init__(self, in_channels, classes):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, classes)
        )


class ConvNeXt(nn.Sequential):
    """
    전체 ConvNeXt 모델.
    
    Stem -> ConvNeXtBody -> Head 순서로 구성됩니다.
    
    Args:
        classes: 출력 클래스 수
        channel_list: 각 스테이지의 채널 수 리스트
        num_blocks_list: 각 스테이지의 블록 개수 리스트
        kernel_size: Depthwise convolution의 커널 크기
        patch_size: Stem의 패치 크기
        in_channels: 입력 채널 수 (기본값=3)
        res_p_drop: Residual block의 Dropout 확률 (기본값=0.)
    """
    def __init__(self, classes, channel_list, num_blocks_list, kernel_size, patch_size,
                 in_channels=3, res_p_drop=0.):
        super().__init__(
            Stem(in_channels, channel_list[0], patch_size),
            ConvNeXtBody(channel_list[0], channel_list, num_blocks_list, kernel_size, res_p_drop),
            Head(channel_list[-1], classes)
        )
        self.reset_parameters()
    
    def reset_parameters(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.)
                nn.init.zeros_(m.bias)
            elif isinstance(m, Residual):
                nn.init.zeros_(m.gamma)
    
    def separate_parameters(self):
        """
        Weight decay를 다르게 적용하기 위해 파라미터를 분리합니다.
        
        Returns:
            parameters_decay: Weight decay를 적용할 파라미터 이름 집합
            parameters_no_decay: Weight decay를 적용하지 않을 파라미터 이름 집합
        """
        parameters_decay = set()
        parameters_no_decay = set()
        modules_weight_decay = (nn.Linear, nn.Conv2d)
        modules_no_weight_decay = (nn.LayerNorm,)
        
        for m_name, m in self.named_modules():
            for param_name, param in m.named_parameters():
                full_param_name = f"{m_name}.{param_name}" if m_name else param_name
                
                if isinstance(m, modules_no_weight_decay):
                    parameters_no_decay.add(full_param_name)
                elif param_name.endswith("bias"):
                    parameters_no_decay.add(full_param_name)
                elif isinstance(m, Residual) and param_name.endswith("gamma"):
                    parameters_no_decay.add(full_param_name)
                elif isinstance(m, modules_weight_decay):
                    parameters_decay.add(full_param_name)
        
        # Sanity check
        all_params = set(p[0] for p in self.named_parameters())
        assert len(parameters_decay & parameters_no_decay) == 0
        assert len(parameters_decay) + len(parameters_no_decay) == len(all_params)
        
        return parameters_decay, parameters_no_decay


class DeepBaselineNetBN3Residual15ConvNeXtLNClassifierStem(nn.Module):
    """
    DeepBaselineNetBN3Residual15ConvNeXtLNClassifierStem: 새로운 ConvNeXt 구조에 맞춘 네트워크 (총 15개 ConvNeXt block)
    
    구조:
    1. Stem: 패치 임베딩 (3 -> 64, patch_size로 다운샘플링)
    2. Stage 1: 3개의 ConvNeXt Block (64 -> 64)
    3. Stage 2: 3개의 ConvNeXt Block (64 -> 128, DownsampleBlock으로 다운샘플링)
    4. Stage 3: 9개의 ConvNeXt Block (128 -> 256, DownsampleBlock으로 다운샘플링)
    5. Stage 4: 3개의 ConvNeXt Block (256 -> 512, DownsampleBlock으로 다운샘플링)
    6. Head: AdaptiveAvgPool2d(1) -> Flatten -> LayerNorm -> Linear
    
    하이퍼파라미터:
    - channel_list = [64, 128, 256, 512]
    - num_blocks_list = [3, 3, 9, 3]
    - kernel_size = 7
    - patch_size = 1
    - res_p_drop = 0.
    """
    
    def __init__(self, init_weights=False, num_classes=10, patch_size=1, kernel_size=7, res_p_drop=0.):
        super(DeepBaselineNetBN3Residual15ConvNeXtLNClassifierStem, self).__init__()
        
        # 하이퍼파라미터 설정
        channel_list = [64, 128, 256, 512]
        num_blocks_list = [3, 3, 9, 3]  # 총 15개 블록
        
        # ConvNeXt 모델 생성
        self.model = ConvNeXt(
            classes=num_classes,
            channel_list=channel_list,
            num_blocks_list=num_blocks_list,
            kernel_size=kernel_size,
            patch_size=patch_size,
            in_channels=3,
            res_p_drop=res_p_drop
        )
        
        if init_weights:
            # reset_parameters는 이미 __init__에서 호출됨
            pass
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: 입력 이미지 [batch, 3, 32, 32] (CIFAR-10)
        
        Returns:
            out: 분류 로짓 [batch, num_classes]
        """
        return self.model(x)
    
    def separate_parameters(self):
        """
        Weight decay를 다르게 적용하기 위해 파라미터를 분리합니다.
        
        Returns:
            parameters_decay: Weight decay를 적용할 파라미터 이름 집합
            parameters_no_decay: Weight decay를 적용하지 않을 파라미터 이름 집합
        """
        return self.model.separate_parameters()

