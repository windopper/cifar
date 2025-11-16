"""
DeepBaselineNetBN3ResidualGAPGMP: DeepBaselineNetBN3Residual의 개선 버전

주요 변경사항:
1. Shortcut Connection 개선
   - 기존: 1x1 Conv + BN (채널/크기 조정 시)
   - 변경: Average Pool -> Conv2d -> BatchNorm2d
   - Average Pooling을 통해 공간 정보를 압축한 후 채널 조정

2. Classifier 개선
   - 기존: Flatten -> FC(512*4*4, 512) -> ReLU -> Dropout -> FC(512, 256) -> ReLU -> Dropout -> FC(256, 10)
   - 변경: Global Average Pooling + Global Max Pooling -> Concatenate -> FC(512*2, 10)
   - GAP와 GMP를 결합하여 평균과 최대값 정보를 모두 활용
   - 단일 FC 레이어로 파라미터 수 감소 및 효율성 향상

설계 의도:
1. Shortcut에 Average Pooling 추가
   - 공간 차원을 먼저 압축하여 연산량 감소
   - 채널 조정 전에 공간 정보를 요약하여 더 효율적인 변환

2. Dual Pooling (GAP + GMP)
   - Global Average Pooling: 전체 feature map의 평균 정보 (전역적인 특징)
   - Global Max Pooling: 전체 feature map의 최대값 정보 (강한 활성화 영역)
   - 두 정보를 결합하여 더 풍부한 표현력 확보

참고:
- MXResNet에서 GAP + GMP concatenate 방식 사용
- ResNet의 GAP 기반 분류기에서 영감을 받음
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    ResNet 스타일의 Basic Residual Block (Shortcut 개선 버전)
    
    구조:
    - 입력 x
    - Main path: Conv -> BN -> ReLU -> Conv -> BN
    - Shortcut: Average Pool -> Conv2d -> BN (채널/크기 다름) 또는 identity (같음)
    - 출력: ReLU(main_path + shortcut)
    
    Args:
        in_channels: 입력 채널 수
        out_channels: 출력 채널 수
        stride: 첫 번째 conv의 stride (다운샘플링용, 기본값=1)
    """
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Main path: 두 개의 Conv-BN 블록
        # 첫 번째 Conv-BN-ReLU 블록
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # 두 번째 Conv-BN 블록 (ReLU는 residual connection 후에 적용)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection (개선 버전)
        # 채널 수가 다르거나 stride가 1이 아닌 경우 projection 필요
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # Average Pool -> Conv2d -> BatchNorm2d
            # stride가 1이 아니면 다운샘플링을 위해 stride 사용
            pool_stride = stride if stride > 1 else 1
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=pool_stride, stride=pool_stride),
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: 입력 텐서 [batch, in_channels, H, W]
            
        Returns:
            out: 출력 텐서 [batch, out_channels, H', W']
        """
        # Identity 또는 projection shortcut 저장
        identity = self.shortcut(x)
        
        # Main path: 첫 번째 Conv-BN-ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # Main path: 두 번째 Conv-BN
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Residual connection: F(x) + x
        out += identity
        
        # 최종 ReLU 활성화
        out = F.relu(out)
        
        return out


class DeepBaselineNetBN3ResidualGAPGMP(nn.Module):
    """
    DeepBaselineNetBN3Residual의 개선 버전 (커스터마이징 가능)
    
    구조:
    1. 초기 Conv-BN-ReLU (3 -> base_filter)
    2. 여러 스테이지: 각 스테이지마다 지정된 필터 수와 블록 개수
    3. Global Average Pooling + Global Max Pooling -> Concatenate -> FC
    
    Args:
        base_filter: 초기 필터 수 (기본값: 16)
        filters: 각 스테이지의 필터 수 리스트 (기본값: [16, 32, 64])
        num_blocks_per_stage: 각 스테이지당 블록 개수 (정수 또는 리스트, 기본값: 5)
        num_classes: 분류 클래스 수 (기본값: 10)
        init_weights: 가중치 초기화 여부 (기본값: False)
    """
    
    def __init__(self, base_filter=16, filters=None, num_blocks_per_stage=5, 
                 num_classes=10, init_weights=False):
        super(DeepBaselineNetBN3ResidualGAPGMP, self).__init__()
        
        # 기본값 설정
        if filters is None:
            filters = [16, 32, 64]
        
        # num_blocks_per_stage가 정수면 리스트로 변환
        if isinstance(num_blocks_per_stage, int):
            num_blocks_per_stage = [num_blocks_per_stage] * len(filters)
        elif len(num_blocks_per_stage) != len(filters):
            raise ValueError(f"num_blocks_per_stage의 길이({len(num_blocks_per_stage)})가 "
                           f"filters의 길이({len(filters)})와 일치하지 않습니다.")
        
        self.base_filter = base_filter
        self.filters = filters
        self.num_blocks_per_stage = num_blocks_per_stage
        self.num_classes = num_classes
        
        # 초기 feature extraction
        # 입력: 3채널 (RGB), 출력: base_filter 채널
        self.conv1 = nn.Conv2d(3, base_filter, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_filter)
        
        # 스테이지 생성
        self.stages = nn.ModuleList()
        in_channels = base_filter
        
        for i, (out_channels, num_blocks) in enumerate(zip(filters, num_blocks_per_stage)):
            self.stages.append(self._make_stage(in_channels, out_channels, num_blocks))
            in_channels = out_channels
        
        # MaxPooling (스테이지 간 다운샘플링)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Global Pooling (GAP + GMP)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Classifier: GAP + GMP concatenate 후 단일 FC 레이어
        # 마지막 필터 수 * 2 (GAP + GMP)
        final_filter = filters[-1] if filters else base_filter
        self.classifier = nn.Linear(final_filter * 2, num_classes)
        
        if init_weights:
            self._initialize_weights()
    
    def _make_stage(self, in_channels, out_channels, num_blocks):
        """
        하나의 스테이지를 생성합니다.
        
        Args:
            in_channels: 입력 채널 수
            out_channels: 출력 채널 수
            num_blocks: Residual Block 개수
            
        Returns:
            nn.Sequential: 스테이지 레이어들
        """
        layers = []
        # 첫 번째 block: 채널 변경 및 다운샘플링 (stride=1, MaxPool로 다운샘플링)
        layers.append(ResidualBlock(in_channels, out_channels, stride=1))
        # 나머지 blocks: 같은 채널 유지
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """가중치 초기화 - ReLU를 사용하므로 Kaiming initialization 사용"""
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
        """
        Forward pass
        
        Args:
            x: 입력 이미지 [batch, 3, 32, 32] (CIFAR-10)
            
        Returns:
            out: 분류 로짓 [batch, num_classes]
        """
        # 초기 Conv-BN-ReLU: 3 -> base_filter
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # 각 스테이지 통과
        for i, stage in enumerate(self.stages):
            x = stage(x)
            # 마지막 스테이지가 아니면 MaxPool 적용
            if i < len(self.stages) - 1:
                x = self.pool(x)
        
        # 마지막 스테이지 후에도 MaxPool 적용 (원래 구조 유지)
        x = self.pool(x)
        
        # Global Average Pooling과 Global Max Pooling
        final_filter = self.filters[-1] if self.filters else self.base_filter
        avg_pool = self.global_avg_pool(x)  # [batch, final_filter, 1, 1]
        max_pool = self.global_max_pool(x)  # [batch, final_filter, 1, 1]
        
        # Concatenate: [batch, final_filter*2, 1, 1]
        x = torch.cat([avg_pool, max_pool], dim=1)
        
        # Flatten: [batch, final_filter*2]
        x = torch.flatten(x, 1)
        
        # Classifier: 단일 FC 레이어
        x = self.classifier(x)
        
        return x


def make_deep_baseline3_bn_residual_gap_gmp(base_filter=16, filters=None, 
                                            num_blocks_per_stage=5, 
                                            num_classes=10, init_weights=False):
    """
    DeepBaselineNetBN3ResidualGAPGMP 모델을 생성하는 팩토리 함수
    
    Args:
        base_filter: 초기 필터 수 (기본값: 16)
        filters: 각 스테이지의 필터 수 리스트 (기본값: [16, 32, 64])
                 None이면 기본값 사용
        num_blocks_per_stage: 각 스테이지당 블록 개수 (기본값: 5)
                            - 정수: 모든 스테이지에 동일하게 적용
                            - 리스트: 각 스테이지별로 지정
        num_classes: 분류 클래스 수 (기본값: 10)
        init_weights: 가중치 초기화 여부 (기본값: False)
    
    Returns:
        DeepBaselineNetBN3ResidualGAPGMP: 생성된 모델 인스턴스
    
    Examples:
        # 기본 설정 (base_filter=16, filters=[16, 32, 64], num_blocks_per_stage=5)
        model = make_deep_baseline3_bn_residual_gap_gmp()
        
        # 커스텀 필터 설정
        model = make_deep_baseline3_bn_residual_gap_gmp(
            base_filter=32, 
            filters=[32, 64, 128, 256],
            num_blocks_per_stage=3
        )
        
        # 각 스테이지별 다른 블록 개수
        model = make_deep_baseline3_bn_residual_gap_gmp(
            filters=[16, 32, 64],
            num_blocks_per_stage=[3, 5, 7]
        )
    """
    return DeepBaselineNetBN3ResidualGAPGMP(
        base_filter=base_filter,
        filters=filters,
        num_blocks_per_stage=num_blocks_per_stage,
        num_classes=num_classes,
        init_weights=init_weights
    )


# 프리셋 모델 함수들 (스테이지, 필터, 블록 구성 기준)
def DeepBaselineNetBN3ResidualGAPGMP_S3_F8_16_32_B2(num_classes=10, init_weights=False):
    """
    3 스테이지, 필터 [8, 16, 32], 각 스테이지당 2개 블록
    작은 모델 (약 0.4M 파라미터)
    """
    return make_deep_baseline3_bn_residual_gap_gmp(
        base_filter=8,
        filters=[8, 16, 32],
        num_blocks_per_stage=[2, 2, 2],
        num_classes=num_classes,
        init_weights=init_weights
    )


def DeepBaselineNetBN3ResidualGAPGMP_S3_F16_32_64_B3(num_classes=10, init_weights=False):
    """
    3 스테이지, 필터 [16, 32, 64], 각 스테이지당 3개 블록
    중간 모델 (약 2M 파라미터)
    """
    return make_deep_baseline3_bn_residual_gap_gmp(
        base_filter=16,
        filters=[16, 32, 64],
        num_blocks_per_stage=[3, 3, 3],
        num_classes=num_classes,
        init_weights=init_weights
    )


def DeepBaselineNetBN3ResidualGAPGMP_S3_F32_64_128_B5(num_classes=10, init_weights=False):
    """
    3 스테이지, 필터 [32, 64, 128], 각 스테이지당 5개 블록
    중대형 모델 (약 10M 파라미터)
    """
    return make_deep_baseline3_bn_residual_gap_gmp(
        base_filter=32,
        filters=[32, 64, 128],
        num_blocks_per_stage=[5, 5, 5],
        num_classes=num_classes,
        init_weights=init_weights
    )


def DeepBaselineNetBN3ResidualGAPGMP_S3_F64_128_256_B5(num_classes=10, init_weights=False):
    """
    3 스테이지, 필터 [64, 128, 256], 각 스테이지당 5개 블록
    대형 모델 (약 30M 파라미터)
    """
    return make_deep_baseline3_bn_residual_gap_gmp(
        base_filter=64,
        filters=[64, 128, 256],
        num_blocks_per_stage=[5, 5, 5],
        num_classes=num_classes,
        init_weights=init_weights
    )


def DeepBaselineNetBN3ResidualGAPGMP_S4_F64_128_256_512_B5(num_classes=10, init_weights=False):
    """
    4 스테이지, 필터 [64, 128, 256, 512], 각 스테이지당 5개 블록
    초대형 모델 (약 60M 파라미터)
    """
    return make_deep_baseline3_bn_residual_gap_gmp(
        base_filter=64,
        filters=[64, 128, 256, 512],
        num_blocks_per_stage=[5, 5, 5, 5],
        num_classes=num_classes,
        init_weights=init_weights
    )

