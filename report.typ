#import "@preview/wrap-it:0.1.1": wrap-content
#show table.cell.where(y: 0): strong
#set table(
  stroke: (x, y) => if y == 0 {
    (bottom: 0.7pt + black)
  },
  align: (x, y) => (
    if x > 0 { center }
    else { left }
  )
)

#align(center)[
  = CIFAR-10 분류모델 개선하기
  202135713 권영훈
]


= 개요
CIFAR-10 데이터셋은 10가지 종류의 사물 이미지가 담긴 머신러닝용 이미지 데이터셋으로 각 32x32 픽셀 크기의 컬러 이미지 60000장으로 구성된다. 총 60,000장 중 50,000장은 훈련용으로, 10,000장은 테스트용으로 사용된다.
#image("random_images.png")
\
= Baseline
#figure(
  image("baseline.png"),
  caption: [
    Baseline
  ],
)

LeNet-5를 기반으로 한 CNN 모델을 베이스라인으로 선정하였다. 옵티마이저(Optimizer)로는 수렴 속도를 고려하여 *Adam*을 채택하였으며, 구체적인 학습 하이퍼파라미터는 다음과 같다.

- *Epochs:* 20
- *Batch Size:* 128
- *Learning Rate:* 3e-4 #footnote[카파시는 블로그에서 Adam은 베이스라인 초기 단계에서 3e-4의 학습률을 사용하는 것을 경험적으로 선호한다고 한다. https://karpathy.github.io/2019/04/25/recipe/]


=== 실험 결과
#figure(
  caption: [베이스라인 및 초기화 기법 적용 성능 비교],
  table(
    columns: (2fr, 1fr), // 첫 열을 좀 더 넓게
    align: (x, y) => (left, center).at(x),
    stroke: none,
    table.hline(y: 0, stroke: 1pt),
    table.hline(y: 1, stroke: 0.5pt),
    
    table.header(
      [*Model Setup*], [*Best Val Acc (%)*]
    ),
    
    [Baseline (Adam Optimizer)], [59.48],
    [\+ Weight Initialization (Kaiming) @he2015delvingdeeprectifierssurpassing], [*61.25*],
    
    table.hline(stroke: 1pt),
  )
)

실험 결과, 단순 베이스라인 모델 대비 가중치 초기화(Weight Initialization)를 적용했을 때 성능 향상이 관찰되었다.

#image("baseline_comparison.png")

\
= 개선 전략 및 설계

== Deeper Baseline
#image("deep_baseline.png")
\
모델 학습을 위해 다음과 같은 환경을 설정하였다.
=== 실험 설정
#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
    - *Network:* DeepBaselineNet
    - *Optimizer:* Adam
  ],
  [
    - *Epochs:* 20
    - *Batch Size:* 128
    - *Learning Rate:* 3e-4
  ]
)
\
기본 Deep Baseline 모델에 가중치 초기화, 학습률 스케줄링, 배치 정규화(Batch Normalization)를 순차적으로 적용하여 성능 변화를 관찰하였다.
=== 실험 결과
#figure(
  caption: [Deep Baseline 모델 개선 실험 결과 요약],
  table(
    columns: (2fr, 1fr),
    align: (x, y) => if x == 0 { left } else { center },
    stroke: none,
    table.hline(y: 0, stroke: 1pt),
    table.hline(y: 1, stroke: 0.5pt),
    
    table.header(
      [*실험 조건 (Modifications)*], [*Best Val Accuracy (%)*]
    ),
    
    [Deep Baseline (Adam, 3e-4)], [76.22],
    [\+ Weight Initialization (Kaiming)], [77.63],
    [\+ Cosine Annealing LR],
[*78.81*],
    [\+ Batch Normalization], [71.93],
    
    table.hline(stroke: 1pt),
  )
)

\
== Augmentation

#figure(
  image("augmentation_visualization.png"),
  caption: [이미지 증강]
)

\
모델의 구조적 개선 이후, 과적합을 방지하고 일반화 성능을 극대화하기 위해 다양한 데이터 증강 기법을 적용하였다. 본 실험부터는 모델의 수렴을 충분히 보장하기 위해 학습 에포크를 *100 Epochs*로 확장하였다.

사용된 주요 증강 기법의 정의는 다음과 같다.

/ Standard Augmentation: 기본적인 기하학적 변환으로, `RandomCrop(padding=4)`, `RandomHorizontalFlip`, `RandomRotation(15)`를 포함한다.
/ CutMix: 이미지의 일정 영역을 잘라내어 다른 이미지의 패치로 채우고, 라벨 또한 면적 비율에 따라 혼합하는 기법이다. @yun2019cutmixregularizationstrategytrain
/ Mixup: 두 이미지의 픽셀 값을 비율에 따라 선형적으로 섞고 라벨도 동일하게 섞는 기법이다. @zhang2018mixupempiricalriskminimization
/ AutoAugment: 강화학습(RL)을 통해 데이터셋(CIFAR-10)에 가장 적합한 증강 정책(Policy)을 자동으로 탐색하여 적용하는 기법이다. @cubuk2019autoaugmentlearningaugmentationpolicies
/ Cutout: 이미지의 임의의 사각형 영역을 검은색(0) 등으로 마스킹하여 모델이 특정 특징에만 의존하지 않도록 하는 정규화 기법이다. @devries2017improvedregularizationconvolutionalneural



=== 실험 설정
#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  inset: 5pt,
  [
    *Model Settings:*
    - *Architecture:* DeepBaselineBN
    - *Parameters:* 10.4 Million
    - *Weight Init:* Kaiming Normal
  ],
  [
    *Training Strategy:*
    - *Optimizer:* Adam (LR 3e-4)
    - *Scheduler:* Cosine Annealing (100 Epochs)
  ]
)
=== 실험 결과
#figure(
  caption: [다양한 데이터 증강 및 스케줄러 조합에 따른 성능 비교],
  table(
    columns: (3fr, 1fr),
    align: (col, row) => if col == 0 { left } else { center },
    stroke: none,
    table.hline(y: 0, stroke: 1pt),
    table.hline(y: 1, stroke: 0.5pt),
    
    table.header(
      [*어그먼테이션 조합*], [*Best Val Acc (%)*]
    ),
    
    [Baseline (No Augmentation)], [78.43],
    table.hline(stroke: (dash: "dotted")),
    
    [*Standard Augmentation (SA)*], [90.01],
    [SA + CutMix], [90.26],
    [SA + CutMix (start at 75%)], [89.49],
    [SA + Mixup], [89.85],
    [SA + Cutout], [90.11],
    table.hline(stroke: (dash: "dotted")),
    
    [SA + Cutout + AutoAugment], [90.52],
    [SA + Cutout (Len 8) + AutoAugment], [89.86],
    [SA + CutMix + AutoAugment], [90.88],
    [SA + Mixup + AutoAugment], [90.43],
    table.hline(stroke: (dash: "dotted")),
    [*SA + AutoAugment*], [*91.17*],
    
    table.hline(stroke: 1pt),
  )
)
아무런 증강을 하지 않았을 때(78.43%) 대비, 기본적인 *Standard Augmentation*만 적용해도 정확도가 *90.01%*로 급격히 상승(약 +11.5%p)하였다.

*AutoAugment*를 단독으로 추가했을 때 *91.17%*로 가장 높은 성능을 기록하였다. 이는 사람이 수동으로 설정한 정책보다 데이터셋에 최적화된 정책이 효과적임을 보여준다.

== Deeper Model And Residual Connection
#image("deep_baseline3_bn_residual.png")

모델의 층이 깊어질수록 학습 데이터에 대한 오차가 오히려 증가하는 *Degradation 문제*와 *기울기 소실* 문제를 해결하기 위해, ResNet(He et al., 2015)@he2015deepresiduallearningimage 의 핵심 개념인 *잔차 학습*을 도입하였다.

기존 모델이 입력 $x$를 출력 $H(x)$로 직접 매핑하려 했다면, 개선된 모델은 잔차함수 $F(x) := H(x) - x$를 학습하는 것을 목표로 한다. 최종 출력은 $H(x) = F(x) + x$가 되며, 이는 다음과 같은 이점을 제공한다.

1. *Gradient Flow:* 역전파 시 덧셈 연산을 통해 그래디언트가 하위 레이어로 직접 전달(Shortcut)되어 학습이 안정적이다.
2. *Identity Mapping:* 모델이 더 깊어져도, 추가된 레이어가 항등 함수(Identity Mapping)를 쉽게 학습할 수 있어 성능 저하를 방지한다.

기존의 단순 적층형 구조를 `ResidualBlock` 단위로 재설계하였다. 각 블록은 *Conv-BN-ReLU-Conv-BN* 구조를 가지며, 입력값을 출력에 더하는 *Skip Connection*을 포함한다.

#grid(
  columns: (1.5fr, 1fr),
  gutter: 1em,
  [
    *Skip Connection 전략:*
    - *Identity Shortcut:* 입출력의 채널 수와 해상도가 같을 경우, 입력 $x$를 그대로 더한다.
    - *Projection Shortcut:* 채널 수가 증가하거나 해상도가 변할 경우, $1 times 1$ Convolution을 통해 차원을 일치시킨 후 더한다.
  ],
  [
    #block(fill: luma(245), inset: 10pt, radius: 5pt, width: 100%)[
      *Forward Logic:*
      ```python
      # F(x) 계산
      out = conv1(x) -> bn -> relu
      out = conv2(out) -> bn
      
      # F(x) + x
      out += identity(x)
      
      # Final Activation
      return relu(out)
      ```
    ]
  ]
)
=== 아키텍처 변화
기존 `DeepBaselineNetBN` 대비 채널 수를 대폭 확장(Max 256 $->$ 512)하였으며, 단순 Conv 레이어를 Residual Block으로 대체하였다.

#figure(
  caption: [계층 구조 상세],
  table(
    columns: (auto, 2fr, auto),
    align: (center, left, center),
    stroke: none,
    table.hline(y: 0, stroke: 1pt),
    table.hline(y: 1, stroke: 0.5pt),
    
    table.header(
      [*Layer Stage*], [*Component / Block Type*], [*Channels*]
    ),
    
    [Stem], [Conv ($3 times 3$) + BN + ReLU], [$3 -> 64$],
    [Block 1], [Residual Block (Identity)], [$64 -> 64$],
    [Block 2], [Residual Block (Identity) + MaxPool], [$64 -> 128$],
    [Block 3], [Residual Block (Identity)], [$128 -> 256$],
    [Block 4], [Residual Block (Identity) + MaxPool], [$256 -> 256$],
    [Block 5], [Residual Block (Identity) + MaxPool], [$256 -> 512$],
    [Classifier], [Flatten to FC Layers (Dropout 0.1)], [$512 -> 10$],
    
    table.hline(stroke: 1pt),
  )
)

=== 실험 설정
앞선 실험들을 통해 검증된 최적의 학습 전략과 구조적으로 개선된 Residual 모델을 결합하여 최종 성능을 측정하였다.

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  inset: 5pt,
  [
    *Model Settings:*
    - *Architecture:* DeepBaselineNetBNResidual
    - *Parameters:* 10.4 Million
    - *Weight Init:* Kaiming Normal
  ],
  [
    *Training Strategy:*
    - *Optimizer:* Adam (LR 3e-4)
    - *Scheduler:* Cosine Annealing (100 Epochs)
    - *Augmentation:* Standard + *AutoAugment*
  ]
)
=== 실험 결과
#figure(
  caption: [최종 모델(Residual + AutoAugment) 성능 측정 결과],
  table(
    columns: (2fr, 1.5fr, 1fr),
    align: (center, center, center),
    stroke: none,
    table.hline(y: 0, stroke: 1pt),
    table.hline(y: 1, stroke: 0.5pt),
    
    table.header(
      [*Model*], [*Best Acc (%)*], [*Params*]
    ),
    [DeepBaselineBN], [91.17], [1.9 M],
    [DeepBaselineNetBN3Residual], [*94.65*], [10.4 M],
    
    table.hline(stroke: 1pt),
  )
)
이전 단계에서 `DeepBaselineNetBN`에 `AutoAugment`를 적용했을 때의 최고 성능은 *91.17%*였다. 여기에 Residual Connection 구조를 도입하고 채널 수를 확장한 결과, 정확도가 *94.65%*로 대폭 향상(*+3.48%p*)되었다.

== Deeper And Deeper
=== 모델 설계
1. *Multi-Stage Architecture:*
   단순히 레이어를 나열하던 방식에서 벗어나, 채널 수(Feature Map size)에 따라 4개의 *Stage*로 구분하고, 각 Stage마다 다수의 Residual Block을 적층하는 방식을 채택하였다.
   - *Stage 구성:* (64ch $times$ 2) $\to$ (128ch $times$ 2) $\to$ (256ch $times$ 4) $\to$ (512ch $times$ 2)
   - 총 15개의 Residual Block을 사용하여 고차원적인 특징 추출 능력을 강화하였다.

2. *Global Average Pooling 도입:*
   기존 모델은 `Flatten` 후 거대한 FC Layer(512$times$4$times$4 $\to$ 512)를 사용하였으나, 이번 모델에서는 마지막 Stage 출력값(4$times$4)에 평균 풀링을 적용하여 ($1times$1) 크기로 압축하였다.
   - 이를 통해 공간 정보를 요약하고, 과적합 위험을 줄이며, 입력 이미지 크기에 유연한 구조를 확보하였다.

=== 실험 설정
기존의 `DeepBaselineNetBN3Residual` 모델과 동일한 실험 조건하에 진행했다.
=== 실험 결과
#figure(
  caption: [네트워크 깊이 확장에 따른 최종 성능 비교],
  table(
  columns: (2fr, 1fr, 1fr, 1fr),
  align: (left, center, center, center),
  stroke: none,
  table.hline(y: 0, stroke: 1pt),
  table.hline(y: 1, stroke: 0.5pt),
  
  table.header(
    [*Model Architecture*], [*Block Count*], [*Best Acc (%)*], [*Params*]
  ),

  [DeepBaselineBN], [5 Blocks], [91.17], [1.9 M],
  [DeepBaselineNetBN3Residual], [5 Blocks], [94.65], [10.4 M],
  [*DeepBaselineNetBN3Residual15*], [*15 Blocks*], [*94.84*], [13.5 M],
  
  table.hline(stroke: 1pt),
)
)

이전 모델 대비 파라미터 수는 약 *30% 증가*(10.4M to 13.5M)하고 연산량 또한 크게 늘었으나, 정확도 향상은 *+0.19%p*에 그쳤다. 현재의 데이터 증강기법과 모델의 용량이 데이터셋의 복잡도를 충분히 커버하고 있음을 알 수 있다. 더 이상의 단순한 깊이 확장은 연산 비용 대비 효율이 떨어지므로 다른 접근법이 유효할 것으로 판단된다.

== Wider Model
#image("wideresnet16_8.png")

ResNet 구조 실험을 통해 깊이가 성능에 기여함을 확인했으나, 층이 너무 깊어지면 학습 속도가 저하되고 파라미터 효율이 떨어지는 문제가 있었다. 이를 해결하기 위해 Zagoruyko et al.(2016)@zagoruyko2017wideresidualnetworks 이 제안한 *Wide Residual Network (WRN)*를 도입하였다.

=== Pre-activation
He et al.(2016)@he2016identitymappingsdeepresidual 이 제안한 *Pre-activation ResNet* 구조를 따르고 있다. 이는 `Conv` 이전에 `BN`과 `ReLU`를 먼저 수행하는 방식(BN $->$ ReLU $->$ Conv)으로, 그래디언트 전파를 원활하게 한다.

또한, 넓어진 채널로 인한 파라미터 과적합을 막기 위해 두 Convolution 사이에 *Dropout*이 삽입된 것이 특징이다.

=== Widen Factor
기본 ResNet 블록의 채널 수에 *확장 계수 $k=8$*을 곱하여 필터(Filter)의 수를 대폭 늘렸다. 이는 모델이 더 풍부한 특징을 학습할 수 있게 하며, 동일한 파라미터 수를 가진 얇고 깊은 모델보다 병렬 연산 효율이 좋다.

- *Stem Layer:* 16 channels
- *Group 1:* $16 times k = 128$ channels
- *Group 2:* $32 times k = 256$ channels
- *Group 3:* $64 times k = 512$ channels
  
=== 아키텍처 구성
`WideResNet-16-8`은 전체 깊이가 16이며, 3개의 주요 그룹으로 구성된다. 각 그룹은 $N=2$개의 `PreActResidualBlock`을 포함한다.

#figure(
  caption: [WideResNet-16-8 계층 구성 상세],
  table(
    columns: (auto, 2fr, 1fr),
    align: (center, left, center),
    stroke: none,
    table.hline(y: 0, stroke: 1pt),
    table.hline(y: 1, stroke: 0.5pt),
    
    table.header(
      [*Group*], [*Block Type / Config*], [*Output Size*]
    ),
    
    [Stem], [Conv $3 times 3$], [$32 times 32$],
    [Group 1], [PreActResidualBlock $times$ 2 ($128$ ch)], [$32times 32$],
    [Group 2], [PreActResidualBlock $times$ 2 ($256$ ch) + Stride 2], [$16times 16$],
    [Group 3], [PreActResidualBlock $times$ 2 ($512$ ch) + Stride 2], [$8times 8$],
    [Global], [BN $->$ ReLU $->$ AvgPool ($8times 8$)], [$1 times 1$],
    [Output], [Linear ($512 -> 10$)], [10],
    
    table.hline(stroke: 1pt),
  )
)

=== 모델 설정
#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  inset: 5pt,
  [
    *Model Settings:*
    - *Architecture:* WideResnet
    - *Parameters:* 10.9 Million
    - *Weight Init:* Kaiming Normal

  ],
  [
    \
    - *Depth:* 16
    - *Widen Factor:* 8
    - *Dropout:* 0.3
  ]
)

=== 실험 결과
#figure(
  caption: [네트워크 깊이 확장에 따른 최종 성능 비교],
  table(
  columns: (2fr, 1fr, 1fr, 1fr),
  align: (left, center, center, center),
  stroke: none,
  table.hline(y: 0, stroke: 1pt),
  table.hline(y: 1, stroke: 0.5pt),
  
  table.header(
    [*Model Architecture*], [*Block Count*], [*Best Acc (%)*], [*Params*]
  ),

  [DeepBaselineBN], [5 Blocks], [91.17], [1.9 M],
  [DeepBaselineNetBN3Residual], [5 Blocks], [94.65], [10.4 M],
  [DeepBaselineNetBN3Residual15], [15 Blocks], [94.84], [13.5 M],
  [*WideResNet-16-8*], [*5 Blocks*], [*95.22*], [*10.9 M*],
  
  table.hline(stroke: 1pt),
)
)
*95.22*%의 정확도를 달성하여 더 적은 파라미터 수로 `DeepBaselineNetBN3Residual15`보다 높은 성능을 기록했다. 단순히 네트워크의 깊이를 늘리는 것보다 넓은 구조가 더 효율적임을 알 수 있었다.


== 성능 끌어올리기
=== Optimizer 변경
Adaptive method인 Adam 대신, Momentum이 적용된 *SGD (Nesterov)*로 변경. CIFAR-10과 같은 이미지 분류 태스크에서 CNN은 SGD가 Adam보다 더 나은 일반화 해를 찾는다는 연구 결과@1705.08292 를 찾을 수 있었다. Adam은 초반 수렴은 빠르지만, Local Minima에 갇힐 가능성이 높다.

정확도가 *95.22% $->$ 95.89%*로 향상되었다. 이는 단순한 Optimizer 교체만으로도 큰 성능 이득을 얻을 수 있음을 보여준다.

=== ASAM (Adaptive Sharpness-Aware Minimization) 적용
단순히 Loss가 낮은 지점을 찾는 것이 아니라, Loss Landscape가 평탄한 지점을 찾도록 하기 위해, SAM @2010.01412 을 도입한다. 또한 ASAM @2102.11600 은 파라미터 스케일을 고려하여 Flat Minima를 탐색하기 때문에 SAM보다 일반화가 우수하다.

ASAM을 적용하고 탐색 반경 ρ=2.0 을 설정하여 학습한 결과, 기존 대비 정확도가 *95.89%*에서 *96.34%*로 (+0.45%p) 향상되었다.

=== EMA (Exponential Moving Average)
학습 중 모델 가중치의 지수 이동 평균을 별도로 저장하여 추론시에 사용하는 기법. 학습 경로의 노이즈를 상쇄하는 효과가 있다.
정확도가 *96.34% $->$ 96.40%*로 소폭 상승하였다.

=== Label Smoothing 적용 @müller2020doeslabelsmoothinghelp
레이블을 그대로 사용하는 것이 아니라 부드럽게 만들어서 일반화 성능을 높이는 방법이다.
$alpha=0.1$을 설정하여 학습한 결과, 정확도가 *96.40% $->$ 96.86%*로 유의미하게 향상되었다.

=== Epoch 확장
학습 기간을 100 Epoch에서 *200 Epoch*로 2배 연장하여 모델이 충분히 수렴할 수 있도록 하였다. 그 결과 *96.40% $->$ 97.07%* 의 성능을 향상시킬 수 있었다.

=== 실험 결과
#figure(
  caption: [WideResNet-16-8 기반의 단계별 최적화 실험 결과],
  table(
    columns: (1.5fr, 3fr, 1fr, 1fr),
    align: (left, left, center, center),
    stroke: none,
    table.hline(y: 0, stroke: 1pt),
    table.hline(y: 1, stroke: 0.5pt),
    
    table.header(
      [*Model Variant*], [*Optimization & Strategy*], [*Best Acc*], [*Params*]
    ),
    
    [WRN-16-8], [--], [95.22%], [10.9 M],
    [WRN-16-8], [SGD + Nesterov (LR 0.1)], [*95.89%*], [10.9 M],
    [\+ Remove 1st ReLU], [SGD + Nesterov], [94.78%], [10.9 M],
    [\+ Last BN], [SGD + Nesterov], [95.08%], [10.9 M],
    table.hline(stroke: (dash: "dotted")),
    [WRN-16-8], [\+ ASAM (rho=$2.0$)], [96.34%], [10.9 M],
    [WRN-16-8], [\+ ASAM + EMA], [96.40%], [10.9 M],
    [WRN-16-8], [\+ ASAM + EMA + Label Smoothing], [*96.86%*], [10.9 M],
    [WRN-16-8], [\+ Use CIFAR Normalize], [96.61%], [10.9 M],
    table.hline(stroke: (dash: "dotted")),
    
    [*WRN-16-8*], [*ASAM + EMA + LS + Epoch 200*], [*97.07%*], [10.9 M],
    
    table.hline(stroke: 1pt),
  )
)

=== 더 많은 아키텍처 탐색


#bibliography("refs.bib")