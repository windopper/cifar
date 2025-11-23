#import "@preview/wrap-it:0.1.1": wrap-content
#import "@preview/cetz:0.4.2"
#import "@preview/cetz-plot:0.1.3": plot, chart
#import "@preview/octique:0.1.0": *

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

#show link: underline

#align(center)[
  = CIFAR-10 분류모델 개선하기
  202135713 권영훈
]

#outline()
#pagebreak()

= 초록
본 실험은 CIFAR-10 데이터셋을 이용한 이미지 분류 태스크에서 모델의 일반화 성능을 극대화하기 위한 단계별 최적화 과정을 기술한다. LeNet-5 기반의 베이스라인 모델(정확도 약 61.25%)에서 출발하여, 모델의 표현력을 높이기 위해 ResNet, WideResNet, PyramidNet 등 다양한 심층 신경망 아키텍처를 도입하고 그 효율성을 비교 분석하였다.
단순한 모델의 깊이 확장보다 넓은 채널 폭을 가진 구조가 효율적임을 확인하였으며, 과적합 방지와 학습 안정성을 위해 AutoAugment, CutMix 등의 데이터 증강 기법과 Cosine Annealing 스케줄러를 적용하였다. 특히, 최적화 관점에서 Adam 대신 Nesterov Momentum을 적용한 SGD로의 전환, 손실 지형의 평탄함을 찾는 ASAM(Adaptive Sharpness-Aware Minimization), 학습 가중치의 이동 평균을 사용하는 EMA(Exponential Moving Average), 그리고 Label Smoothing 기법을 결합하여 성능을 대폭 향상시켰다.
최종적으로 PyramidNet-110 아키텍처에 ShakeDrop 정규화를 적용하고 400 Epoch 동안 학습하여 단일 모델 성능을 극대화하였으며, 앙상블(Ensemble) 및 메타 러너를 활용한 스태킹(Stacking) 기법을 도입하여 97.78%의 최종 정확도를 달성하였다. 본 실험을 통해 아키텍처의 구조적 개선뿐만 아니라, 최적화 알고리즘과 정규화 전략의 조화가 고성능 모델 구현에 필수적임을 입증하였다. 

본 실험에 사용된 전체 소스 코드와 실험 로그는 #link("https://github.com/windopper/cifar")[#octique-inline("mark-github") GitHub 저장소]에서 확인할 수 있다.

= 개요
CIFAR-10 데이터셋은 10가지 종류의 사물 이미지가 담긴 머신러닝용 이미지 데이터셋으로 각 32x32 픽셀 크기의 컬러 이미지 60000장으로 구성된다. 총 60,000장 중 50,000장은 훈련용으로, 10,000장은 테스트용으로 사용된다.
#image("random_images.png")
\
= Baseline
#figure(
  image("baseline.png"),
  caption: [
    Baseline #link(<baseline_code>)[Code]
  ],
)

LeNet-5를 기반으로 한 CNN 모델을 베이스라인으로 선정하였다. 합성곱층과 풀링층을 반복하여 특징을 추출하고, FC 층을 통해 분류를 수행하는 기본적인 구조이다. 최적화 알고리즘으로는 수렴 속도를 고려하여 *Adam*을 채택하였으며, 구체적인 학습 하이퍼파라미터는 다음과 같다.

- *Epochs:* 20
- *Batch Size:* 128
- *Learning Rate:* 3e-4 #footnote[Andrej Karpathy는 블로그에서 "A Recipe for Training Neural Networks"를 통해 Adam 옵티마이저 사용 시 $3e-4$를 초기 학습률로 설정하는 것이 경험적으로 안전한 시작점임을 언급하였다. https://karpathy.github.io/2019/04/25/recipe/]
- *Seed*: 42


== 실험 결과
베이스라인 모델의 성능을 측정하고, 가중치 초기화 기법의 영향력을 검증했다. ReLU 활성화 함수를 사용하는 신경망에서는 가중치가 적절히 초기화되지 않을 경우 그레디언트 소실이나 폭주가 발생할 수 있다.@pmlr-v9-glorot10a 이를 방지하기 위해 *He Initialization (Kaiming Normal)*@he2015delvingdeeprectifierssurpassing 을 적용했다.

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
    [\+ Weight Initialization (Kaiming)], [*61.25*],
    
    table.hline(stroke: 1pt),
  )
)

실험 결과, 기본 초기화 방법 대비 Kaiming Initialization을 적용했을 때 약 *1.77%p*의 성능 향상이 관찰되었다. 

#image("baseline_comparison.png")

\
= 개선 전략 및 설계
베이스라인 모델은 얕은 구조로 인해 CIFAR-10 데이터의 복잡한 패턴을 충분히 학습하지 못한다. 이에 따라 모델의 표현력을 높이기 위해 다음 전략을 수립하였다.

== Deeper Baseline
성능 개선의 핵심은 모델의 깊이와 너비를 확장하는 것이다. 컨볼루션 층을 더 추가하여 고차원 특징을 추출하도록 유도하였으며, 정보 손실을 최소화 하기 위해 필터수를 늘렸다.
#figure(
  image("deep_baseline.png"),
  caption: [DeepBaselineNet #link(<deep_baseline_bn_code>)[Code]]
)
\
성능 개선을 위해 모델의 깊이와 채널의 수를 대폭 증가 시켰다.
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
    - *Batch Size:* 32
    - *Learning Rate:* 3e-4
  ]
)
\
모델이 깊어짐에 따라 발생할 수 있는 학습 불안정성과 과적합 문제를 해결하기 위해 단계적으로 개선 기법을 적용하였다.
\

- *Weight Initialization*: 앞선 실험에서 유효성이 입증된 Kaiming Initialization을 기본으로 적용하였다.
- *Batch Normalization*: 학습 과정에서 각 층의 입력 분포가 변하는 현상을 완화하여, 학습 속도를 높이고 초기화 민감도를 낮추었다.
- *Cosine Annealing*: 학습이 진행됨에 따라 학습률을 점진적으로 감소시켜, 최적점 부근에서 안정적으로 수렴하도록 유도하였다.


=== 실험 결과
#image("deep_baseline_bs32.png")

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
    
    [Deep Baseline (Adam, 3e-4)], [77.27],
    [\+ Weight Initialization (Kaiming)], [80.52],
    [\+ Cosine Annealing LR],
[82.05],
    [\+ Batch Normalization], [*82.70*],
    
    table.hline(stroke: 1pt),
  )
)
기법을 하나씩 추가할 때마다 정확도가 뚜렷하게 상승했다. 가중치 초기화를 통해 베이스라인 대비 큰 폭의 성능 향상을 보였으며 이는 초기 가중치 설정이 중요함을 입증했다. 
*Cosine Annealing* 적용 후 발생했던 *Validation Loss*의 변동을 *Batch Normalization*을 통해 어느 정도 억제할 수 있었다.

모든 모델에서 *10 Epoch* 이후로 *Validation Loss*가 다시 증가하거나 정체되는 과적합 문제가 발견되었다.


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
/ CutMix: 이미지의 일정 영역을 잘라내어 다른 이미지의 패치로 채우고, 라벨 또한 면적 비율에 따라 혼합하는 기법이다. @yun2019cutmixregularizationstrategytrain #link(<cutmix_code>)[Code]
/ Mixup: 두 이미지의 픽셀 값을 비율에 따라 선형적으로 섞고 라벨도 동일하게 섞는 기법이다. @zhang2018mixupempiricalriskminimization #link(<mixup_code>)[Code] 
/ AutoAugment: 강화학습(RL)을 통해 데이터셋(CIFAR-10)에 가장 적합한 증강 정책(Policy)을 자동으로 탐색하여 적용하는 기법이다. @cubuk2019autoaugmentlearningaugmentationpolicies #link(<autoaugment_code>)[Code]
/ Cutout: 이미지의 임의의 사각형 영역을 검은색(0) 등으로 마스킹하여 모델이 특정 특징에만 의존하지 않도록 하는 정규화 기법이다. @devries2017improvedregularizationconvolutionalneural #link(<cutout_code>)[Code]



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
#figure(
  image("deep_baseline3_bn_residual.png"),
  caption: [DeepBaseline3BNResidual #link(<deep_baseline3_bn_residual_code>)[Code]]
)

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
    - *Augmentation:* Standard + AutoAugment
  ]
)
=== 실험 결과
#figure(
  caption: [최종 모델(Residual + AutoAugment) 성능 측정 결과],
  table(
    columns: (2fr, 1.5fr, 1fr, 1fr),
    align: (center, center, center),
    stroke: none,
    table.hline(y: 0, stroke: 1pt),
    table.hline(y: 1, stroke: 0.5pt),
    
    table.header(
      [*Model*], [*Best Acc (%)*], [*Params*], [*GFLOPs*]
    ),
    [DeepBaselineBN], [91.17], [1.9 M], [0.1 GFLOPs],
    [DeepBaselineNetBN3Residual], [*94.65*], [10.4 M], [1.1 GFLOPs],
    
    table.hline(stroke: 1pt),
  )
)
이전 단계에서 `DeepBaselineNetBN`에 `AutoAugment`를 적용했을 때의 최고 성능은 *91.17%*였다. 여기에 Residual Connection 구조를 도입하고 채널 수를 확장한 결과, 정확도가 *94.65%*로 대폭 향상(*+3.48%p*)되었다.

== Deeper And Deeper
=== 모델 설계
1. *Multi-Stage Architecture:*
   단순히 레이어를 나열하던 방식에서 벗어나, 채널 수(Feature Map size)에 따라 4개의 *Stage*로 구분하고, 각 Stage마다 다수의 Residual Block을 적층하는 방식을 채택하였다.
   - *Stage 구성:* (64ch $times$ 2) $->$ (128ch $times$ 2) $->$ (256ch $times$ 4) $->$ (512ch $times$ 2)
   - 총 15개의 Residual Block을 사용하여 고차원적인 특징 추출 능력을 강화하였다.

2. *Global Average Pooling 도입:*
   기존 모델은 `Flatten` 후 거대한 FC Layer(512$times$4$times$4 $\to$ 512)를 사용하였으나, 이번 모델에서는 마지막 Stage 출력값(4$times$4)에 평균 풀링을 적용하여 ($1times$1) 크기로 압축하였다.

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
      [*Model*], [*Best Acc (%)*], [*Params*], [*GFLOPs*]
    ),
    [DeepBaselineBN], [91.17], [1.9 M], [0.1 GFLOPs],
    [DeepBaselineNetBN3Residual], [94.65], [10.4 M], [1.1 GFLOPs],
  [*DeepBaselineNetBN3Residual15*], [*94.84*], [*13.5 M*], [0.71 GFLOPs],
  
  table.hline(stroke: 1pt),
)
)

이전 모델 대비 파라미터 수는 약 *30%* 증가(10.4M $->$ 13.5M)하고 연산량 또한 크게 늘었으나, 정확도 향상은 *+0.19%p*에 그쳤다. 현재의 데이터 증강기법과 모델의 용량이 데이터셋의 복잡도를 충분히 커버하고 있음을 알 수 있다. 더 이상의 단순한 깊이 확장은 연산 비용 대비 효율이 떨어지므로 다른 접근법이 유효할 것으로 판단된다.

== Wider Model
#figure(image("wideresnet16_8.png"), caption: [WideResnet16-8, #link(<wide_resnet_code>)[Code]])

ResNet 구조 실험을 통해 깊이가 성능에 기여함을 확인했으나, 층이 너무 깊어지면 학습 속도가 저하되고 파라미터 효율이 떨어지는 문제가 있었다. 이를 해결하기 위해 Zagoruyako et al.(2016)@zagoruyko2017wideresidualnetworks 이 제안한 *Wide Residual Network (WRN)*를 도입하였다.

=== Pre-activation
#grid(
  columns: (1fr, auto),
  gutter: 1cm,
  [
    He et al.(2016)@he2016identitymappingsdeepresidual 이 제안한 *Pre-activation ResNet* 구조를 따르고 있다. 이는 `Conv` 이전에 `BN`과 `ReLU`를 먼저 수행하는 방식(BN $->$ ReLU $->$ Conv)으로, 그래디언트 전파를 원활하게 한다.

또한, 넓어진 채널로 인한 파라미터 과적합을 막기 위해 두 Convolution 사이에 *Dropout*이 삽입된 것이 특징이다.
  ],
  figure(
    cetz.canvas(length: 0.5cm, {
      import cetz.draw: *
  
      // --- 스타일 정의 ---
      // frame: "rect"를 사용하여 content 자체가 박스 역할을 하도록 설정
      let node-style = (
        frame: "rect",
        stroke: 1pt + gray,
        rx: 0.2, ry: 0.2, // 둥근 모서리
        padding: 0.2,     // 내부 여백
        width: 2,       // 너비 고정 (내용이 짧아도 박스 크기 유지)
      )
      
      // 색상 테마
      let c-bn = rgb("#e1f5fe")   // Light Blue
      let c-act = rgb("#fff9c4")  // Light Yellow
      let c-conv = rgb("#ffebee") // Light Red
      let c-drop = rgb("#f3e5f5") // Light Purple
  
      // --- 노드 배치 ---
      
      // 1. Input
      content((0, 0), [$x_l$], name: "input")
      
      // 2. 첫 번째 그룹 (BN -> ReLU -> Conv)
      content((0, -1.5), [Batch Norm], name: "bn1", fill: c-bn, ..node-style)
      content((0, -3.0), [ReLU],       name: "relu1", fill: c-act, ..node-style)
      content((0, -4.5), [Conv 3x3],   name: "conv1", fill: c-conv, ..node-style)
  
      // 3. 두 번째 그룹 (BN -> ReLU -> Dropout -> Conv)
      content((0, -6.5), [Batch Norm], name: "bn2", fill: c-bn, ..node-style)
      content((0, -8.0), [ReLU],       name: "relu2", fill: c-act, ..node-style)
      content((0, -9.5), [Dropout],    name: "drop", fill: c-drop, ..node-style)
      content((0, -11.0), [Conv 3x3],  name: "conv2", fill: c-conv, ..node-style)
  
      // 4. Addition Node (원형)
      circle((0, -13.0), radius: 0.3, name: "add")
      content((0, -13.0), [+]) 
  
      // 5. Output
      content((0, -14.5), [$x_{l+1}$], name: "output")
  
      // --- 연결 선 그리기 ---
  
      // 메인 경로 (Main Branch)
      line("input", "bn1", mark: (end: ">"))
      line("bn1", "relu1", mark: (end: ">"))
      line("relu1", "conv1", mark: (end: ">"))
      line("conv1", "bn2", mark: (end: ">"))
      line("bn2", "relu2", mark: (end: ">"))
      line("relu2", "drop", mark: (end: ">"))
      line("drop", "conv2", mark: (end: ">"))
      line("conv2", "add", mark: (end: ">")) // add 노드의 경계(anchor)로 자동 연결
      line("add", "output", mark: (end: ">"))
  
      // 스킵 연결 (Skip Connection)
      // rel: (dx, dy)를 사용하여 상대 좌표 이동
      line("input", (rel: (-2.5, 0)), (rel: (0, -13.0)), "add", mark: (end: ">"))
  
      // 캡션 텍스트 (각도를 deg 단위로 명시하여 오류 해결)
      content((-2.8, -6.5), [Identity Mapping], angle: 90deg)
    }),
    caption: [WideResNet Pre-activation Block]
  )
)




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
#image("model_comparison.png")

#figure(
  caption: [네트워크 깊이 확장에 따른 최종 성능 비교],
  table(
  columns: (2fr, 1fr, 1fr, 1fr),
  align: (left, center, center, center),
  stroke: none,
  table.hline(y: 0, stroke: 1pt),
  table.hline(y: 1, stroke: 0.5pt),
  
    table.header(
      [*Model*], [*Best Acc (%)*], [*Params*], [*GFLOPs*]
    ),
    [DeepBaselineBN], [91.17], [1.9 M], [0.1 GFLOPs],
    [DeepBaselineNetBN3Residual], [94.65], [10.4 M], [1.1 GFLOPs],
  [DeepBaselineNetBN3Residual15], [94.84], [13.5 M], [0.71 GFLOPs],
  [*WideResNet-16-8*], [*95.08*], [*10.9 M*], [*1.55 GFLOPs*],
  
  table.hline(stroke: 1pt),
)
)
*95.08*%의 정확도를 달성하여 더 적은 파라미터 수로 `DeepBaselineNetBN3Residual15`보다 높은 성능을 기록했다. 단순히 네트워크의 깊이를 늘리는 것보다 넓은 구조가 더 효율적임을 알 수 있었다.

= 성능 끌어올리기
== Optimizer 변경
Adaptive method인 Adam 대신, Momentum이 적용된 *SGD (Nesterov)*로 변경. CIFAR-10과 같은 이미지 분류 태스크에서 CNN은 SGD가 Adam보다 더 나은 일반화 해를 찾는다는 연구 결과@1705.08292 를 찾을 수 있었다. Adam은 초반 수렴은 빠르지만, Local Minima에 갇힐 가능성이 높다. 옵티마이저 변경에 맞춰서 학습률을 *0.1*로 설정하였다.

정확도가 *95.22% $->$ 95.89%*로 향상되었다. 이는 단순한 Optimizer 교체만으로도 큰 성능 이득을 얻을 수 있음을 보여준다.

== Residual Block 구조 변경
Pre-activation Block에서 첫번째 ReLU 층을 제거하고, 마지막에 BN층을 추가하는 구조 변화를 시도해보았지만 현재 아키텍처에서는 오히려 성능이 하락했다.

#grid(
  columns: (1fr, 1fr),
  [\+ Remove 1st ReLU (*95.89 $->$ 94.78*)],
  [\+ Last BN (*94.78 $->$ 95.08*)]
)

== ASAM (Adaptive Sharpness-Aware Minimization) 적용
일반적인 SGD 기반의 학습 방법은 훈련 데이터에 대한 손실 함수 $L(w)$를 최소화하는 파라미터 $w$를 찾는 것에 집중한다. 하지만 단순히 손실값이 낮은 지점, 즉 Global Minima를 찾더라도, 해당 지점의 손실 곡선이 가파른 경우, 테스트 데이터의 분포가 조금만 달라져도 성능이 급격히 하락할 수 있다.

반면, 손실 곡선이 평탄한 지점은 파라미터의 미세한 변동에도 손실값이 크게 변하지 않아 일반화 성능이 우수하다. 이를 달성하기 위해 주변 파라미터 공간의 손실값까지 고려하여 최적화를 수행하는 *SAM (Sharpness-Aware Minimization)* @2010.01412 을 도입하였다.

SAM은 파라미터 $w$의 주변 반경 $rho$ 내에서 가장 손실이 큰 지점을 찾아 그 지점의 손실을 최소화하는 Min-Max 문제를 푼다.

$ min_w L^("SAM")(w) approx min_w ( max_(||epsilon||_2 <= rho) L(w + epsilon) ) $

기존 SAM은 모든 파라미터에 대해 동일한 반경 $rho$를 적용한다. 그러나 신경망의 스케일이 서로 다르기 때문에 이런 문제를 해결하기 위해 *ASAM (Adaptive SAM)* @2102.11600 을 도입하여 파라미터 크기 $|w|$에 비례하여 탐색영역을 조정한다. 즉, 정규화 연산자 $T_w = "diag"(|w|)$를 도입하여, $epsilon$의 크기를 파라미터별로 적응적으로 제어한다. ASAM의 목적함수는 다음과 같이 정의된다.

$ min_w max_(|| T_w^(-1) epsilon ||_2 <= rho) L(w + epsilon) $

이를 통해 학습 과정에서 파라미터의 스케일과 무관하게 일관된 평탄함을 최적화할 수 있으며, 결과적으로 우수한 일반화 성능을 기대할 수 있다.

본 실험에서는 ASAM을 적용하고, 평탄함의 탐색 반경을 결정하는 하이퍼파라미터 $rho$를 *2.0*으로 설정하였다. 
일반적인 SAM이 $rho approx 0.05$ 수준의 작은 값을 사용하는 것과 달리 큰 값을 설정한 이유는, ASAM의 구조상 가중치 크기로 섭동이 정규화되므로 실질적인 탐색 반경을 확보하기 위해 더 큰 스칼라 값이 요구되기 때문이다. 이는 ASAM 구현체의 저자가 수행한 실험 결과에서도 최적의 값으로 보고된 바 있다#footnote[ASAM 저자의 GitHub Issue 논의 참고: https://github.com/davda54/sam/issues/37]. 

최종 성능 평가 결과, 기존 최고 성능 모델 대비 유의미한 정확도 향상이 관찰되었다. #link(<asam_code>)[Code]

#figure(
  table(
    columns: (2fr, 1fr, 1fr),
    align: (left, center, center),
    stroke: none,
    table.hline(y: 0, stroke: 1pt),
    table.hline(y: 1, stroke: 0.5pt),
    
    table.header([*Method*], [*Test Accuracy*], [*Gain*]),
    
    [Baseline + Augmentation], [95.89%], [-],
    [*+ ASAM* ($rho=2.0$)], [*96.34%*], [*+0.45%p*],
    
    table.hline(stroke: 1pt),
  ),
  caption: [ASAM 적용에 따른 성능 향상 비교]
)

== EMA (Exponential Moving Average)
SGD를 사용하는 딥러닝 학습 과정에서는 미니 배치의 노이즈로 인핸 파라미터가 최적점 부근에서 수렴하지 않고 진동하는 현상이 발생한다. 이러한 학습 불안정성을 보완하기 위해, 학습 중 발생하는 파라미터의 변동을 평활화하는 *EMA* 기법을 도입했다.

*EMA*는 학습 과정에서 모델의 가중치 $theta$가 업데이트 될 때마다, 별도의 가중치 $theta_("EMA")$를 지수 이동 평균 방식으로 갱신한다.

$ theta_("EMA")^(t) = beta dot theta_("EMA")^(t-1) + (1 - beta) dot theta^(t) $

여기서 $t$는 현재 스텝, $beta$는 Decay Rate를 의미한다. 이 실험에서는 $beta$를 0.999로 설정하였으며 이를 통해 단일 시점의 가중치보다 Flat Minima에 위치할 확률을 높인다.

EMA 가중치는 학습에는 관여하지 않고 갱신만 되며, 최종 추론 단계에서만 사용될 수 있다. 이는 추론 비용 증가 없이 성능을 높일 수 있는 효율적인 방법이다.

실험 결과, 정확도가 *96.34%*에서 *96.40%*로 소폭 상승하였다. #link(<ema_code>)[Code])

== Label Smoothing 적용
분류 문제에서 일반적으로 사용하는 CrossEntropy Loss는 정답 클래스의 확률은 1, 나머지를 0으로 설정된 원핫 벡터를 타겟으로 한다. 모델이 이를 학습하는 과정에서 손실을 0으로 만들기 위해 정답 클래스의 로짓을 과도하게 키우는 현상이 발생기 쉽다. 이는 학습 데이터에 대한 과적합을 유발하고, 잘못된 예측에 대해서도 높은 확신을 갖게 하여 일반화 성능을 저해한다. 

이러한 문제를 완화하기 위해 타겟 분포를 부드럽게 조정하는 *Label Smoothing*@müller2020doeslabelsmoothinghelp 기법을 적용하였다.

Label Smoothing은 정답 레이블 $y$를 $1$이 아닌 조금 더 작은 값으로, 오답 레이블을 $0$이 아닌 작은 양수로 설정하여 모델이 너무 확신하지 않도록 규제한다. 스무딩 계수(Smoothing Factor)가 $alpha$, 클래스의 개수가 $K$일 때, 보정된 레이블 $y_k^("LS")$는 다음과 같이 정의된다.

$ y_k^("LS") = (1 - alpha) dot y_k + alpha / K $

여기서 $y_k$는 기존의 원-핫 인코딩된 값이다. 본 실험에서는 $K=10$ (CIFAR-10), $alpha=0.1$을 적용하였다. 이는 정답 클래스의 타겟값을 $1.0 -> 0.9$로 낮추고, 나머지 오답 클래스들에 $0.01$씩 확률을 분배함으로써, 모델이 결정 경계 주변에서 더 부드러운 분포를 학습하도록 유도한다.

$alpha=0.1$을 설정하여 학습한 결과, 정확도가 *96.40% $->$ 96.86%*로 유의미하게 향상되었다.

// --- 파라미터 및 데이터 ---
#let K = 5
#let eps = 0.2 
#let smooth-true = 1 - eps + (eps / K)
#let smooth-false = eps / K

// 데이터
#let data-hard = ((0, 0), (1, 0), (2, 1.0), (3, 0), (4, 0))
#let data-soft = (
  (0, smooth-false), (1, smooth-false), 
  (2, smooth-true), 
  (3, smooth-false), (4, smooth-false)
)

// X축 라벨 매핑
#let x-ticks-def = ((0, "C1"), (1, "C2"), (2, "C3"), (3, "C4"), (4, "C5"))

// 공통 플롯 설정 (가로로 넓게 수정)
#let my-plot-settings = (
  size: (11, 3), // [수정] 가로 11, 세로 3 -> 와이드 비율
  x-label: text(size: 9pt)[Classes],
  y-label: text(size: 9pt)[Probability],
  y-min: 0, y-max: 1.25,
  x-min: -0.6, x-max: 4.6,
  axis-style: "scientific",
  x-grid: false,
  y-grid: true,
  x-ticks: x-ticks-def,
  y-tick-step: 0.2,
  tick-style: (stroke: 0.5pt), 
)

// 1. Hard Label Figure
#figure(
  block({
    set text(size: 8pt)
    cetz.canvas({
      import plot: *
      plot(..my-plot-settings, {
        add-bar(
          data-hard, x-key: 0, y-key: 1, bar-width: 0.5,
          style: (fill: rgb("#ffcdd2"), stroke: none)
        )
        
        annotate({
          import draw: *
          let arrow-style = (mark: (end: "stealth", fill: black, size: 0.15), stroke: 0.5pt)
          
          // 1.0 표시
          content((2, 1.05), anchor: "south", text(fill: red, weight: "bold", size: 9pt)[$1.0$])
          
          // 설명 박스 (위치를 약간 오른쪽으로)
          content((3.8, 0.8), box(align(center)[Hard Target\n(Overconfident)], width: 2.5cm), name: "lbl-hard")
          // 화살표 연결
          line("lbl-hard.west", (2.1, 0.95), ..arrow-style)
        })
      })
    })
  }),
  caption: [Without Smoothing (Hard Target)]
)

#v(0.2cm) // 그래프 사이 간격

// 2. Label Smoothing Figure
#figure(
  block({
    set text(size: 8pt)
    cetz.canvas({
      import plot: *
      plot(..my-plot-settings, {
        add-bar(
          data-soft, x-key: 0, y-key: 1, bar-width: 0.5,
          style: (fill: rgb("#c8e6c9"), stroke: none)
        )

        annotate({
          import draw: *
          let arrow-style = (mark: (end: "stealth", fill: black, size: 0.15), stroke: 0.5pt)

          // 정답 (Soft True)
          content((2, smooth-true + 0.05), anchor: "south", 
            text(fill: green.darken(30%), weight: "bold", size: 9pt)[$1-epsilon$])

          // 오답 (Soft False) 설명
          // 와이드 비율에 맞춰 화살표 시작점 조정
          let text-pos = (0.5, 0.4) 
          let target-pos = (0.1, smooth-false + 0.02)
          
          content(text-pos, [$epsilon/K$], name: "lbl-eps")
          line("lbl-eps.south", target-pos, ..arrow-style)
              
          // 전체 설명
          content((3.8, 0.8), box(align(center)[Soft Target\n(Generalization)], width: 2.5cm))
        })
      })
    })
  }),
  caption: [With Label Smoothing ($epsilon$ =0.2)]
)

== CIFAR Normalize
Train Set에 대한 평균과 표준편차를 계산하여 정규화시에 학습 효율이 오를 것으로 기대하였지만 정확도가 *96.86 $->$ 96.61* 로 하락하였다.


== Epoch 확장
학습 기간을 100 Epoch에서 *200 Epoch*로 2배 연장하여 모델이 충분히 수렴할 수 있도록 하였다. 그 결과 *96.40% $->$ 97.07%* 의 성능을 향상시킬 수 있었다.

== 실험 결과

#image("wideresnet_comparison.png")

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
    
    [WRN-16-8], [--], [95.08%], [10.9 M],
    [WRN-16-8], [SGD + Nesterov (LR 0.1)], [95.89%], [10.9 M],
    [\+ Remove 1st ReLU], [SGD + Nesterov], [94.78%], [10.9 M],
    [\+ Last BN], [SGD + Nesterov], [95.08%], [10.9 M],
    table.hline(stroke: (dash: "dotted")),
    [WRN-16-8], [\+ ASAM (rho=$2.0$)], [96.34%], [10.9 M],
    [WRN-16-8], [\+ ASAM + EMA], [96.40%], [10.9 M],
    [WRN-16-8], [\+ ASAM + EMA + Label Smoothing], [96.86%], [10.9 M],
    [WRN-16-8], [\+ Use CIFAR Normalize], [96.61%], [10.9 M],
    table.hline(stroke: (dash: "dotted")),
    
    [*WRN-16-8*], [*ASAM + EMA + LS + Epoch 200*], [*97.07%*], [10.9 M],
    
    table.hline(stroke: 1pt),
  )
)
\
= 더 많은 아키텍처 탐색
Han et al. (2017)@han2017deeppyramidalresidualnetworks 에서 채널 수를 급격하게 늘리는 것보다 조금씩 선형적으로 늘리는 것이 최적화 관점에서 훨씬 유리함을 실험으로 증명했다.

또한, 기존의 아키텍처는 차원이 변화하는 구간에서 Identity Mapping을 유지하기 위해 $1 times 1$ Convolution을 사용하였지만
$ y = F(x, \{W_i\}) + W_s x $

반면 PyramidNet은 늘어난 채널만큼을 0으로 채우는 *Zero-padded Shortcut*을 사용한다.

$ y = F(x, \{W_i\}) + ["pad"(x)] $

논문의 모델 구조를 유지한채 이전에 사용했던 모델의 파라미터 (10M)와 비슷한 PyramidNet-110 ($alpha=150$) (10.9M)를 사용하였다.

== 아키텍처 구조
#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 10pt,
    align: (col, row) => (center + horizon, center + horizon, center + horizon, center + horizon).at(col),
    stroke: none,
    
    table.header(
      table.hline(stroke: 1.5pt),
      [*Group Name*], 
      [*Output Size*], 
      [*Block Configuration*], 
      [*Channel Range* \ ($C_{\i\n} arrow.r C_{\o\u\t}$)],
      table.hline(stroke: 0.7pt)
    ),

    // Conv1
    [Conv1], 
    [$32 times 32$], 
    [$3 times 3$ Conv], 
    [16],

    table.hline(stroke: 0.5pt + gray),

    // Conv2 (Group 1)
    [Conv2_x], 
    [$32 times 32$], 
    [
      $ mat(3 times 3, C_k; 3 times 3, C_k) times 18 $
    ], 
    [$16 arrow.r 66$],

    table.hline(stroke: 0.5pt + gray),

    // Conv3 (Group 2)
    [Conv3_x], 
    [$16 times 16$], 
    [
      $ mat(3 times 3, C_k; 3 times 3, C_k) times 18 $
    ], 
    [$66 arrow.r 116$],

    table.hline(stroke: 0.5pt + gray),

    // Conv4 (Group 3)
    [Conv4_x], 
    [$8 times 8$], 
    [
      $ mat(3 times 3, C_k; 3 times 3, C_k) times 18 $
    ], 
    [$116 arrow.r 166$],

    table.hline(stroke: 0.5pt + gray),

    // Classification
    [Classifier], 
    [$1 times 1$], 
    [
      $8 times 8$ Global Avg Pool \
      Softmax Linear
    ], 
    [$166 arrow.r 10$],
    
    table.hline(stroke: 1.5pt)
  ),
  caption: [PyramidNet-110 ($alpha=150$) 아키텍처. 채널 너비 $C_k$는 각 블록마다 약 2.8씩 선형적으로 증가합니다.]
)
=== 실험 결과
#figure(
  table(
    columns: (1.0fr, 1.0fr, 0.6fr, 0.6fr, 1fr),
    align: (center, center, center, center),
    inset: 10pt,
    stroke: none,
    table.hline(y: 0, stroke: 1pt),
    table.hline(y: 1, stroke: 0.5pt),
    table.header(
      [*Model Variant*], [*Optimization & Strategy*], [*Best Acc*], [*Params*], [GFLOPs]
    ),
    [WRN-16-8], [--], [95.08%], [10.9 M], [1.5519 GFLOPs],
    [*WRN-16-8*], [*ASAM + EMA + LS + Epoch 200*], [*97.07%*], [*10.9 M*], [*1.5519 GFLOPs*],
    [PyramidNet-110 ($alpha=150$)], [--], [96.82%], [10.9 M], [1.8197 GFLOPs],
  ),
)
추가적인 기법없이 WRN-16-8에 비해 *+1.74%*의 성능 향상을 이끌어냈다.

== ShakeDrop Regularization
*ShakeDrop*@Shakedrop 은 Residual Block을 확률적으로 삭제하는 *Stochastic Depth*@huang2016deepnetworksstochasticdepth 와 *ResNext*@xie2017aggregatedresidualtransformationsdeep 와 같이 2개의 *Residual Branch*를 가진 네트워크에서, 두 가지의 출력 합을 구할 난수를 곱하는 방식인 *Shake-Shake*@gastaldi2017shakeshakeregularization 의 한계를 극복하기 위해 고안되었다.

*ShakeDrop*은 Branch가 하나인 경우에도 $b_l$(bernoulli) 게이트를 가상의 두 번째 경로를 만드는 효과를 내어 *Shake-Shake*와 유사한 정규화 효과를 얻었다.

표준적인 ResNet의 $l$번째 블록의 출력 $y_l$은 입력 $x_l$과 잔차 함수 $cal(F)(x_l)$의 합으로 정의된다.

$ y_l = x_l + cal(F)(x_l, cal(W)_l) $

*ShakeDrop*은 여기에 확률 변수 $M_l$을 도입하여 $cal(F)$의 출력을 스케일링한다.

$ y_l = x_l + M_l dot cal(F)(x_l, cal(W)_l) $

여기서 $M_l$은 베르누이 확률 변수 $b_l$에 따라 $alpha$ 또는 $beta$의 값을 가지는 랜덤 변수이다.

$ M_l = cases(
  alpha & "if" b_l = 1 " (forward path activated)",
  beta & "if" b_l = 0 " (forward path dropped)"
) $

이를 하나의 수식으로 표현하면 다음과 같다.

$ M_l = b_l dot alpha + (1 - b_l) dot beta $

#figure(
  cetz.canvas(length: 0.5cm, {
    import cetz.draw: *

    // --- 스타일 정의 (제공해주신 스타일 활용) ---
    let node-style = (
      frame: "rect",
      stroke: 1pt + gray,
      rx: 0.2, ry: 0.2, // 둥근 모서리
      padding: 0.4,     // 내부 여백 (수식 공간 확보를 위해 약간 늘림)
      width: 3.5,       // 너비 (수식 길이에 맞춰 조정)
      align: center     // 텍스트 가운데 정렬
    )
    
    // 색상 테마
    let c-bn = rgb("#e1f5fe")   // Light Blue (Batch Norm)
    let c-act = rgb("#fff9c4")  // Light Yellow (Activation)
    let c-conv = rgb("#ffebee") // Light Red (Convolution)
    let c-noise = rgb("#e0f2f1") // Light Teal (ShakeDrop Noise)
    let c-mult = rgb("#fff3e0")  // Light Orange (Multiplication Node)

    // --- 노드 배치 ---
    
    // 1. Input
    content((0, 0), text(weight: "bold")[$x_l$], name: "input")
    
    // 2. Residual Function F(x) 그룹
    // 실제 ResNet 블록처럼 Conv -> BN -> ReLU -> Conv 구조를 단순화하여 표현
    content((0, -2.0), [Conv], name: "layer1", fill: c-conv, ..node-style)
    content((0, -4.0), [Conv], name: "layer2", fill: c-conv, ..node-style)
    
    // 주석: 이 부분이 F(x)임을 표시
    content((2.5, -3.0), text(fill: gray, size: 0.8em)[$cal(F)(x_l)$])

    // 3. ShakeDrop Mechanism
    // 곱셈 연산 (Circle 형태이나 텍스트 배치를 위해 content 사용하되 모양을 원으로)
    circle((0, -6.5), radius: 0.5, fill: c-mult, stroke: 1pt + gray, name: "mult_circle")
    content((0, -6.5), [$times$], name: "mult_text")

    // 노이즈 생성기 (오른쪽 배치)
    content((5.5, -6.5), 
      [$M_l = b_l alpha + (1-b_l)beta$ \ Noise Generator], 
      name: "noise", fill: c-noise, ..node-style, width: 4.5)

    // 4. Addition Node
    circle((0, -9.0), radius: 0.3, name: "add", fill: white, stroke: 1pt + black)
    content((0, -9.0), [$+$]) 

    // 5. Output
    content((0, -10.5), text(weight: "bold")[$y_l$], name: "output")

    // --- 연결 선 그리기 ---

    // 메인 경로 (Residual Branch)
    line("input", "layer1", mark: (end: ">"))
    line("layer1", "layer2", mark: (end: ">"))
    line("layer2", "mult_circle.north", mark: (end: ">")) // Anchor: north 사용
    
    // 노이즈 주입 경로
    line("noise.west", "mult_circle.east", mark: (end: ">"))

    // 곱셈 후 덧셈으로
    line("mult_circle.south", "add.north", mark: (end: ">"))

    // 스킵 연결 (Skip Connection / Identity Mapping)
    // 왼쪽으로 우회
    line("input", (rel: (-3.0, 0)), (rel: (0, -9.0)), "add.west", mark: (end: ">"))

    // Output 연결
    line("add.south", "output", mark: (end: ">"))

    // --- 추가 캡션 ---
    content((-3.3, -4.5), [Identity Mapping], angle: 90deg)

  }),
  caption: [ShakeDrop Regularization Block Diagram]
)

PyramidNet-110 ($alpha=150$)에 적용시키기 위해서 얕은 층은 기본적인 특징을 추출하므로 보존하고, 과적합 위험이 높은 깊은 층일수록 강한 규제를 적용하기 위해 선형적으로 확률을 조절하였다.

// 수식 부분
$ p_l = 1 - l / L dot (1 - P_"max") $

// 변수 설명 부분
/ $l$: 현재 블록의 인덱스 ($l = 1, 2, ..., L$)
/ $L$: 전체 블록의 개수 (Total blocks)
/ $P_"max"$: 마지막 블록에서의 목표 ShakeDrop 확률
/ $p_l$: $l$번째 블록에 적용되는 ShakeDrop 확률

$P_"max"$로 $0.5$와 $1.0$로 실험한 결과는 다음과 같다.

=== 실험 결과
#figure(
  table(
    columns: (1.5fr, 1.5fr, 0.6fr, 0.6fr),
    align: (left, center, center, center),
    inset: 10pt,
    stroke: none,
    table.hline(y: 0, stroke: 1pt),
    table.hline(y: 1, stroke: 0.5pt),
    table.header(
      [*Model Variant*], [*Optimization & Strategy*], [*Best Acc*], [*Params*]
    ),
    [PyramidNet-110 ($alpha=150$)], [--], [96.82%], [10.9 M],
    [PyramidNet-110 ($alpha=150$)], [$P_"max"=1$], [92.61%], [10.9 M],
    [PyramidNet-110 ($alpha=150$)], [$P_"max"=0.5$], [*96.9%*], [10.9 M],
  ),
)

*+0.08%*의 성능이 개선되었다.

*WideResnet* 모델에서 사용했던 전략과 *PyramidNet* 모델과 *ShakeDrop* 기법을 적용하여 *200 Epoch* 학습 결과는 다음과 같다.

#figure(
  table(
    columns: (1.5fr, 1.5fr, 0.6fr, 0.6fr),
    align: (left, center, center, center),
    inset: 10pt,
    stroke: none,
    table.hline(y: 0, stroke: 1pt),
    table.hline(y: 1, stroke: 0.5pt),
    table.header(
      [*Model Variant*], [*Optimization & Strategy*], [*Best Acc*], [*Params*]
    ),
    [WRN-16-8], [--], [95.08%], [10.9 M],
    [WRN-16-8], [ASAM + EMA + LS + Epoch 200], [97.07%], [10.9 M],
    [PyramidNet-110 ($alpha=150$)], [--], [96.82%], [10.9 M],
    [PyramidNet-110 ($alpha=150$)], [$P_"max"=1$], [92.61%], [10.9 M],
    [PyramidNet-110 ($alpha=150$)], [$P_"max"=0.5$], [96.9%], [10.9 M],
    [*PyramidNet-110 ($alpha=150$)*], [*ASAM + EMA + LS + Epoch 200 + ShakeDrop ($P_"max"$=0.5)*], [*97.48%*], [*10.9 M*]
  ),
)

*WRN-16-8*의 기존 최고 정확도였던 *97.07%*$->$*97.48*로 *\+0.41%*의 성능이 개선되었다.

= 성능 극한으로 밀어붙이기
== Epoch 확장
기존 200 Epoch 학습 시 모델이 완전히 수렴하지 않았을 가능성을 배제하기 위해, 학습 횟수를 2배인 400 Epoch로 확장하여 추가 학습을 진행하였다. 

실험 결과, 정확도는 기존 *97.48%*에서 *97.70%*로 약 *0.22%p* 상승하였다.
#image("pyramidnet_comparsion_200_400.png")

#figure(
  table(
    columns: (1.5fr, 1.5fr, 0.6fr, 0.6fr),
    align: (left, center, center, center),
    inset: 10pt,
    stroke: none,
    table.hline(y: 0, stroke: 1pt),
    table.hline(y: 1, stroke: 0.5pt),
    table.header(
      [*Model Variant*], [*Optimization & Strategy*], [*Best Acc*], [*Params*]
    ),
    [WRN-16-8], [ASAM + EMA + LS + Epoch 200], [97.07%], [10.9 M],
    [PyramidNet-110 ($alpha=150$)], [ASAM + EMA + LS + Epoch 200 + ShakeDrop ($P_"max"$=0.5)], [97.48%], [10.9 M],
    [*PyramidNet-110 ($alpha=150$)*], [*ASAM + EMA + LS + Epoch 400 + ShakeDrop ($P_"max"$=0.5)*], [*97.70*], [*10.9 M*]
  ),
)

== TTA
모델의 가중치를 변경하지 않고 추론 단계에서 성능을 극대화하기 위해 TTA(Test Time Augmentation)@shanmugam2021betteraggregationtesttimeaugmentation 기법을 도입하였다. 구체적으로는 *AutoAugment의 CIFAR 정책*을 활용하여 테스트 이미지당 4회의 변형을 가한 후, 추론 결과의 평균을 사용하는 방식을 채택했다.

기대와 달리 성능은 최고 기록인 *97.70%*에서 *97.43%*로 오히려 *0.27%p* 하락하는 결과를 보였다.

Horizontal Flip을 사용하여 기본 테스트 이미지와 회전한 이미지에 대해서 TTA를 시행하였지만 *97.70%*에서 *97.67%*로 *0.03%* 하락하였다.



#figure(
  table(
    columns: (1fr, 1fr),
    align: (center, center),
    inset: 10pt,
    stroke: none,
    table.hline(y: 0, stroke: 1pt),
    table.hline(y: 1, stroke: 0.5pt),
    table.header(
      [TTA Variant], [*Acc*]
    ),
    [*Original*], [*97.7*],
    [Original + AutoAugment (k=4)], [97.43],
    [Original + Horizontal Flip], [97.67],
  ),
)
== Ensemble
기존의 가장 좋은 성능을 보였던 3개의 모델을 뽑아서 *Soft Voting* 방식을 적용하여 앙상블을 수행하였다.

실험 결과 *+0.01%*의 성능 향상을 얻을 수 있었다.

또한 앙상블 가중치를 단일 모델 성능이 가장 뛰어난 PyramidNet-400 epoch 모델에 0.5의 가중치를 부여하고, 나머지 두 모델은 보조적인 역할을 하도록 균등 배분하였다. 이를 통해 *+0.05%*의 추가적인 성능 향상을 얻을 수 있었다.

#figure(
  table(
    columns: (1.5fr, 1.5fr, 0.6fr, 0.6fr),
    align: (left, center, center, center),
    inset: 10pt,
    stroke: none,
    table.hline(y: 0, stroke: 1pt),
    table.hline(y: 1, stroke: 0.5pt),
    table.header(
      [*Model Variant*], [*Optimization & Strategy*], [*Best Acc*], [*Params*]
    ),
    [WRN-16-8], [ASAM + EMA + LS + Epoch 200], [97.07%], [10.9 M],
    [PyramidNet-110 ($alpha=150$)], [ASAM + EMA + LS + Epoch 200 + ShakeDrop ($P_"max"$=0.5)], [97.48%], [10.9 M],
    [PyramidNet-110 ($alpha=150$)], [ASAM + EMA + LS + Epoch 400 + ShakeDrop ($P_"max"$=0.5)], [97.70%], [10.9 M],
    [*Ensemble*], [--], [*97.71%*], [--],
    [*Ensemble (0.5, 0.25, 0.25)*], [--], [*97.75%*], [--]
  ),
)

== Stacking
여러 개의 예측 결과를 새로운 메타 모델의 입력으로 사용하여 새로운 예측 결과를 얻는 스태킹을 통해 일반화 성능을 강화하고자 하였다.

학습된 기반 모델들의 출력값인 로짓 벡터를 결합하여 메타 모델의 입력으로 사용하였다.

메타 모델로는 과적합을 방지하고 비선형적 관계를 학습할 수 있도록 3개의 은닉층을 가진 MLP를 설계하였다. 구체적인 구조는 다음과 같다.

#figure(
  rect(fill: luma(240), stroke: 0.5pt + luma(180), radius: 5pt, inset: 10pt)[
    #set text(font: "Cascadia Code", size: 8pt)
    #align(left)[
    ```python
    class MetaLearner(nn.Module):
        def __init__(self, input_dim, num_classes):
            super(MetaLearner, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                
                nn.Linear(64, num_classes)
            )
            self._initialize_weights()
    ```
    ]
  ],
  caption: [Pytorch로 구현된 Meta-Learner 모델 구조]
) <code-meta>

/ Optimizer: Adam ($lr=0.005$)을 사용하였으며, Weight Decay ($1 times 10^{-1}$)를 적용하여 L2 규제 효과를 주었다.
/ Scheduler: Cosine Annealing 기법을 적용하여 학습률을 점진적으로 감소시킴으로써, 학습 후반부의 미세 조정을 유도하였다.
/ Loss Function: Cross Entropy Loss를 사용하되, Label Smoothing을 0.1로 설정하여 정답 레이블에 대한 과도한 확신을 방지하고 일반화 성능을 높였다.

총 *20 Epoch* 동안 훈련을 진행한 결과, 최종적으로 *97.78%*의 정확도를 달성하였다. 
#link(<stacking_code>)[Code]

#figure(
  table(
    columns: (1.5fr, 1.5fr, 0.6fr, 0.6fr),
    align: (left, center, center, center),
    inset: 10pt,
    stroke: none,
    table.hline(y: 0, stroke: 1pt),
    table.hline(y: 1, stroke: 0.5pt),
    table.header(
      [*Model Variant*], [*Optimization & Strategy*], [*Best Acc*], [*Params*]
    ),
    [WRN-16-8], [ASAM + EMA + LS + Epoch 200], [97.07%], [10.9 M],
    [PyramidNet-110 ($alpha=150$)], [ASAM + EMA + LS + Epoch 200 + ShakeDrop ($P_"max"$=0.5)], [97.48%], [10.9 M],
    [PyramidNet-110 ($alpha=150$)], [ASAM + EMA + LS + Epoch 400 + ShakeDrop ($P_"max"$=0.5)], [97.70%], [10.9 M],
    [*Ensemble*], [--], [*97.71%*], [--],
    [*Ensemble (0.5, 0.25, 0.25)*], [--], [*97.75%*], [--],
    [*Ensemble (0.5, 0.25, 0.25)*], [--], [*97.75%*], [--],
    [*Stacking (seed=250)*], [--], [*97.78%*], [--]
  ),
)

= 추가 개선 아이디어 및 한계점
앞서 사용한 모델 아키텍처를 제외하고 *Residual Attention*@wang2017residualattentionnetworkimage, *ConvNext*@liu2022convnet2020s, *ConvNextV2*@woo2023convnextv2codesigningscaling, *DLA*@yu2019deeplayeraggregation, *Squeeze-And-Excitation*@hu2019squeezeandexcitationnetworks 등 여러 CNN 아키텍처를 다각도로 비교 분석하였다(@experimentals 참조). 그러나 Transformer 기반의 모델이나 하이브리드 아키텍처에 대한 광범위한 탐색은 수행하지 못했다는 점에서 아키텍처 탐색의 범위를 더욱 넓힐 여지가 있다.

실험은 약 10M 파라미터 크기의 모델로 제한하여 진행되었다. 이는 제한적인 하드웨어 자원으로 인한 선택이었으나, 데이터셋의 특징을 온전히 포착하기에는 모델의 수용력이 부족할 수 있다. 향후 연구에서는 파라미터 수를 점진적으로 늘리거나, Knowledge Distillation 기법을 도입하여 큰 모델의 성능을 경량 모델로 전이시키는 시도가 필요하다.

#let classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

#let data = (
  (975, 0, 7, 2, 1, 0, 0, 0, 12, 3),
  (1, 988, 0, 0, 0, 0, 0, 0, 1, 10),
  (3, 0, 963, 5, 9, 9, 7, 2, 2, 0),
  (2, 1, 4, 924, 7, 53, 6, 0, 2, 1),
  (0, 0, 6, 6, 979, 4, 3, 2, 0, 0),
  (1, 1, 8, 31, 8, 947, 2, 2, 0, 0),
  (3, 0, 4, 3, 1, 0, 988, 1, 0, 0),
  (2, 0, 2, 4, 6, 4, 0, 982, 0, 0),
  (6, 4, 2, 0, 0, 0, 1, 0, 984, 3),
  (1, 17, 0, 1, 0, 0, 0, 0, 4, 977),
)

// 색상 생성 함수 (값이 클수록 진한 파란색)
#let heatmap-cell(value) = {
  let max-val = 1000 // 데이터의 최대값 (row sum)
  let intensity = value / max-val * 100%
  
  // 배경색: 흰색 -> 파란색 그라데이션
  let bg-color = color.mix((white, 100% - intensity), (blue, intensity))
  
  // 글자색: 배경이 어두우면 흰색, 밝으면 검은색 (가독성 확보)
  let text-color = if value > 500 { white } else { black }
  
  // 0인 값은 너무 흐려서 안 보일 수 있으니 회색 텍스트로 처리하거나 빈칸으로 둘 수 있음
  // 여기서는 0은 흐리게 표시
  let content = if value == 0 { text(fill: gray.lighten(50%))[0] } else { text(fill: text-color)[#value] }

  table.cell(fill: bg-color)[#content]
}

학습된 모델에 대해서 *Confusion Matrix*를 보면 공통적으로 *dog*와 *cat* 라벨의 예측 정확도가 다른 라벨보다 낮은 것을 알 수 있었다. 

#grid(
    columns: (auto, auto),
    gutter: 5pt,
    align: center + horizon,
    
    // Y축 라벨 (회전)
    rotate(-90deg)[*Actual Class*],
    
    // 테이블 영역
    stack(
      dir: ltr,
      // 테이블
      table(
        columns: (auto, ) + (2.5em, ) * 10, // 첫 열(라벨) + 데이터 10열
        align: center + horizon,
        stroke: 0.5pt + gray.lighten(50%),
        inset: 6pt,
        
        // 헤더 (Predicted Class)
        table.header(
          [], // 왼쪽 위 모서리 빈칸
          table.cell(colspan: 10, stroke: none)[*Predicted Class*],
        ),
        table.header(
          [], // 모서리 빈칸
          ..classes.map(c => table.cell(fill: gray.lighten(90%))[#c]) // 클래스 헤더
        ),

        // 데이터 행 생성
        ..data.enumerate().map(((i, row)) => {
          (
            // 행 라벨 (Actual Class Name)
            table.cell(fill: gray.lighten(90%))[#strong(classes.at(i))],
            // 데이터 셀들 (Heatmap 적용)
            ..row.map(val => heatmap-cell(val))
          )
        }).flatten()
      )
    )
  )

이를 해결하기 위해 *Weighted Cross Entropy*를 적용하여 정확도가 상대적으로 낮은 dog, cat에 대해서 1.5의 가중치를 부여하였지만 유의미한 성과는 얻을 수 없었다.
#figure(
  table(
    columns: (1.5fr, 1.5fr, 0.6fr, 0.6fr),
    align: (left, center, center, center),
    inset: 10pt,
    stroke: none,
    table.hline(y: 0, stroke: 1pt),
    table.hline(y: 1, stroke: 0.5pt),
    table.header(
      [*Model Variant*], [*Optimization & Strategy*], [*Best Acc*], [*Params*]
    ),
    [*WRN-16-8*], [*ASAM + EMA + LS*], [*96.86%*], [*10.9 M*],
    [WRN-16-8], [ASAM + EMA + LS + Weighted CE], [96.56%], [10.9 M],
  ),
)

== 정성적 오류 분석
여전히 오분류되는 데이터들의 특성을 파악하기 위해, 테스트 단계에서 *손실값이 가장 높은 샘플*들을 추출하여 시각화하였다. 이들은 모델이 예측에 실패했을 뿐만 아니라, 정답 클래스와의 괴리가 가장 큰 데이터들이다.

#figure(
  grid(
    columns: (1fr),
    gutter: 1em,
    align: center,
    // 가지고 계신 이미지 파일명으로 교체하세요
    stack(dir: ttb, image("top_loss_sample.png")),
  ),
  caption: [Top-Loss 샘플 시각화. 모델은 높은 확신을 가지고 예측했으나, 정답과 불일치하여 큰 손실이 발생하였다.]
)

위 결과를 육안으로 검토하던 중 흥미로운 현상을 발견하였는데, 오분류된 이미지 중 인간의 눈으로 보았을 때 모델의 예측이 실제 정답보다 더 타당해 보인다는 점이다.

#figure(image("top_loss_sample_frog.png"), caption: [모델은 frog라고 예측하였지만 데이터셋의 정답은 cat으로 표시되어있음.])

=== 데이터셋의 내재적 오류
CIFAR-10은 저해상도 이미지를 수동으로 분류하는 과정에서 휴먼 에러가 포함된 것으로 알려져 있다. 실제로 Northcutt et al. @northcutt2021pervasivelabelerrorstest 의 연구에 따르면, CIFAR-10 테스트 세트에는 약 *0.54%*의 잘못된 레이블이 존재한다고 보고되었다.#footnote[https://labelerrors.com/ 를 통해 CIFAR-10 오분류 데이터를 확인할 수 있다.]

마지막으로 본 실험은 제한된 자원으로 인해 고정된 시드에서의 성능을 측정하였다. 0.1% 미만의 미세한 성능 차이는 초기화 및 데이터 로딩의 무작위성에 기인할 수 있으므로, 향후 다중 시드 실험을 통한 통계적 유의성 검증이 요구된다.

#pagebreak()
#bibliography("refs.bib", title: [Reference])

#import "@preview/numbly:0.1.0": numbly
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *
#show: codly-init.with()
#codly(languages: codly-languages)

#set heading(numbering: numbly(
  "Appendix {1:A}.", // use {level:format} to specify the format
  "{1:A}.{2}.", // if format is not specified, arabic numbers will be used
  "Step {3}.", // here, we only want the 3rd level
))
#pagebreak()
= Models
== Baseline <baseline_code>
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineNet(nn.Module):
    def __init__(self, init_weights=False):
        super(BaselineNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
        if init_weights:
            self._initialize_weights()
    
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

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
```

== DeepBaselineBN <deep_baseline_bn_code>
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepBaselineNetBN(nn.Module):
    def __init__(self, init_weights=False):
        super(DeepBaselineNetBN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        
        if init_weights:
            self._initialize_weights()
    
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
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.pool(x)
        
        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x
```

== DeepBaselineNetBN3Residual <deep_baseline3_bn_residual_code>
```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity

        out = F.relu(out)

        return out


class DeepBaselineNetBN3Residual(nn.Module):
    def __init__(self, init_weights=False):
        super(DeepBaselineNetBN3Residual, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.res_block1 = ResidualBlock(64, 64, stride=1)

        self.res_block2 = ResidualBlock(64, 128, stride=1)

        self.res_block3 = ResidualBlock(128, 256, stride=1)

        self.res_block4 = ResidualBlock(256, 256, stride=1)

        self.res_block5 = ResidualBlock(256, 512, stride=1)

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

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
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
        x = self.pool(x)

        x = self.res_block3(x)

        x = self.res_block4(x)
        x = self.pool(x)

        x = self.res_block5(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)

        x = self.classifier(x)

        return x
```
== WideResnet <wide_resnet_code>
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import ShakeDrop

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, shakedrop_prob=0.0, last_batch_norm=False, remove_first_relu=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        
        self.shake_drop = ShakeDrop(p_drop=shakedrop_prob) if shakedrop_prob > 0.0 else None
        self.bn3 = nn.BatchNorm2d(out_planes) if last_batch_norm else None
        
        self.remove_first_relu = remove_first_relu
        
        self.last_batch_norm = last_batch_norm

    def forward(self, x):
        if not self.equalInOut:
            x = self.bn1(x) if self.remove_first_relu else self.relu1(self.bn1(x))
        else:
            out = self.bn1(x) if self.remove_first_relu else self.relu1(self.bn1(x))
        
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0 and self.shake_drop is None:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        
        if self.last_batch_norm:
            out = self.bn3(out)
        
        if self.shake_drop is not None:
            out = self.shake_drop(out)
        
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, shakedrop_probs=None, last_batch_norm=False, remove_first_relu=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, shakedrop_probs, last_batch_norm, remove_first_relu)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, shakedrop_probs, last_batch_norm, remove_first_relu):
        layers = []
        for i in range(int(nb_layers)):
            prob = shakedrop_probs[i] if shakedrop_probs and i < len(shakedrop_probs) else 0.0
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, dropRate, shakedrop_prob=prob,
                                last_batch_norm=last_batch_norm, remove_first_relu=remove_first_relu))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth=28, num_classes=10, widen_factor=10, dropRate=0.3, shakedrop_prob=0.0, last_batch_norm=False, remove_first_relu=False):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        
        total_blocks = int(n * 3)
        if shakedrop_prob > 0 and total_blocks > 1:
            step = shakedrop_prob / (total_blocks - 1)
            probs = [i * step for i in range(total_blocks)]
        elif shakedrop_prob > 0:
            probs = [shakedrop_prob]
        else:
            probs = [0.0] * total_blocks
        
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        
        n_int = int(n)
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, shakedrop_probs=probs[0:n_int], last_batch_norm=last_batch_norm, remove_first_relu=remove_first_relu)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, shakedrop_probs=probs[n_int:2*n_int], last_batch_norm=last_batch_norm, remove_first_relu=remove_first_relu)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, shakedrop_probs=probs[2*n_int:], last_batch_norm=last_batch_norm, remove_first_relu=remove_first_relu)
        
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
    
def wideresnet28_10(shakedrop_prob=0.0, last_batch_norm=False, remove_first_relu=False):
    return WideResNet(depth=28, num_classes=10, widen_factor=10, dropRate=0.3, shakedrop_prob=shakedrop_prob, last_batch_norm=last_batch_norm, remove_first_relu=remove_first_relu)

def wideresnet16_8(shakedrop_prob=0.0, last_batch_norm=False, remove_first_relu=False):
    return WideResNet(depth=16, num_classes=10, widen_factor=8, dropRate=0.3, shakedrop_prob=shakedrop_prob, last_batch_norm=last_batch_norm, remove_first_relu=remove_first_relu)
```

== PyramidNet <pyramidnet_code>
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import ShakeDrop


class PyramidBasicBlock(nn.Module):
    """
    """
    def __init__(self, in_planes, out_planes, stride=1, dropRate=0.0, shakedrop_prob=0.0):
        super(PyramidBasicBlock, self).__init__()
        
        self.downsampled = stride == 2
        self.branch = self._make_branch(in_planes, out_planes, stride=stride)
        self.shortcut = None if not self.downsampled else nn.AvgPool2d(2)
        
        self.shake_drop = ShakeDrop(p_drop=shakedrop_prob)
        
        self.droprate = dropRate
        self.in_planes = in_planes
        self.out_planes = out_planes

    def _make_branch(self, in_ch, out_ch, stride=1):
        return nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch))

    def forward(self, x):
        h = self.branch(x)
        h = self.shake_drop(h)
        
        h0 = x if not self.downsampled else self.shortcut(x)
        
        if h.size(1) > h0.size(1):
            pad_size = h.size(1) - h0.size(1)
            pad_zero = torch.zeros(h0.size(0), pad_size, h0.size(2), h0.size(3), 
                                  dtype=h0.dtype, device=h0.device)
            h0 = torch.cat([h0, pad_zero], dim=1)
        
        return h + h0


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_chs, start_idx, block, stride, dropRate=0.0, 
                 shakedrop_probs=None):
        super(NetworkBlock, self).__init__()
        self.start_idx = start_idx
        self.in_chs = in_chs
        self.layer = self._make_layer(block, nb_layers, stride, dropRate, shakedrop_probs)

    def _make_layer(self, block, nb_layers, stride, dropRate, shakedrop_probs):
        layers = []
        u_idx = self.start_idx
        
        for i in range(int(nb_layers)):
            prob = shakedrop_probs[u_idx] if shakedrop_probs and u_idx < len(shakedrop_probs) else 0.0
            
            current_in = self.in_chs[u_idx]
            current_out = self.in_chs[u_idx + 1]
            
            current_stride = stride if i == 0 else 1
            
            layers.append(block(current_in, current_out, current_stride, dropRate, shakedrop_prob=prob))
            u_idx += 1
        
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNetPyramid(nn.Module):
    """
    
    Args:
        depth: 네트워크 깊이
        num_classes: 클래스 수
        widen_factor: width 배수
        dropRate: dropout 비율
        shakedrop_prob: 마지막 블록의 최대 ShakeDrop 확률
        use_pyramid: True이면 pyramid 스타일 채널 증가 사용
        alpha: pyramid 스타일 총 채널 증가량
    """
    def __init__(self, depth=28, num_classes=10, widen_factor=10, dropRate=0.3, 
                 shakedrop_prob=0.5, use_pyramid=False, alpha=48, use_original_depth=False):
        super(WideResNetPyramid, self).__init__()
        
        in_ch = 16
        block = PyramidBasicBlock
        
        if use_original_depth:
            n_units = (depth - 2) // 6
            assert (depth - 2) % 6 == 0, f"depth must satisfy (depth-2) % 6 == 0, got depth={depth}"
        else:
            n_units = (depth - 4) // 6
            assert (depth - 4) % 6 == 0, f"depth must satisfy (depth-4) % 6 == 0, got depth={depth}"
        
        if use_pyramid:
            in_chs = [in_ch] + [in_ch + math.ceil((alpha / (3 * n_units)) * (i + 1)) 
                                for i in range(3 * n_units)]
        else:
            nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
            in_chs = [nChannels[0]]
            for stage_idx in range(3):
                stage_ch = nChannels[stage_idx + 1]
                for _ in range(n_units):
                    in_chs.append(stage_ch)
        
        total_blocks = 3 * n_units
        if shakedrop_prob > 0 and total_blocks > 1:
            probs = [1 - (1.0 - (shakedrop_prob / total_blocks) * (i + 1)) 
                    for i in range(total_blocks)]
        else:
            probs = [0.0] * total_blocks
        
        self.c_in = nn.Conv2d(3, in_chs[0], 3, padding=1)
        self.bn_in = nn.BatchNorm2d(in_chs[0])
        
        self.block1 = NetworkBlock(n_units, in_chs, 0, block, 1, dropRate, 
                                   shakedrop_probs=probs)
        self.block2 = NetworkBlock(n_units, in_chs, n_units, block, 2, dropRate, 
                                   shakedrop_probs=probs)
        self.block3 = NetworkBlock(n_units, in_chs, 2*n_units, block, 2, dropRate, 
                                   shakedrop_probs=probs)
        
        self.bn_out = nn.BatchNorm2d(in_chs[-1])
        self.fc_out = nn.Linear(in_chs[-1], num_classes)
        self.nChannels = in_chs[-1]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        h = self.bn_in(self.c_in(x))
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = F.relu(self.bn_out(h))
        h = F.avg_pool2d(h, 8)
        h = h.view(h.size(0), -1)
        h = self.fc_out(h)
        return h

def wideresnet28_10_pyramid(shakedrop_prob=0.5, alpha=48):
    """
    """
    return WideResNetPyramid(depth=28, num_classes=10, widen_factor=10, dropRate=0.3,
                            shakedrop_prob=shakedrop_prob, use_pyramid=True, alpha=alpha)


def wideresnet16_8_pyramid(shakedrop_prob=0.5, alpha=32):
    """
    """
    return WideResNetPyramid(depth=16, num_classes=10, widen_factor=8, dropRate=0.3,
                            shakedrop_prob=shakedrop_prob, use_pyramid=True, alpha=alpha)


def pyramidnet110_270(shakedrop_prob=0.5, alpha=270):
    """
    Args:
        shakedrop_prob: 최대 ShakeDrop 확률
        alpha: pyramid 채널 증가량
    """
    return WideResNetPyramid(depth=110, num_classes=10, widen_factor=1, dropRate=0.0,
                            shakedrop_prob=shakedrop_prob, use_pyramid=True, alpha=alpha,
                            use_original_depth=True)


def pyramidnet110_150(shakedrop_prob=0.5, alpha=150):
    """
    Args:
        shakedrop_prob: 최대 ShakeDrop 확률
        alpha: pyramid 채널 증가량
    """
    return WideResNetPyramid(depth=110, num_classes=10, widen_factor=1, dropRate=0.0,
                            shakedrop_prob=shakedrop_prob, use_pyramid=True, alpha=alpha,
                            use_original_depth=True)


```

= Augmentation
== Cutmix <cutmix_code>
```python
import numpy as np
import torch
import torch.nn as nn


def _rand_bbox(size, lam):
    _, _, h, w = size
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(w * cut_ratio)
    cut_h = int(h * cut_ratio)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    x0 = np.clip(cx - cut_w // 2, 0, w)
    x1 = np.clip(cx + cut_w // 2, 0, w)
    y0 = np.clip(cy - cut_h // 2, 0, h)
    y1 = np.clip(cy + cut_h // 2, 0, h)

    return x0, y0, x1, y1


def cutmix(batch, alpha):
    data, targets = batch
    batch_size = data.size(0)

    if alpha <= 0:
        return data, targets

    lam = np.random.beta(alpha, alpha)
    indices = torch.randperm(batch_size)
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    x0, y0, x1, y1 = _rand_bbox(data.size(), lam)

    mixed_data = data.clone()
    mixed_data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]

    area = (x1 - x0) * (y1 - y0)
    lam = 1.0 - area / (data.size(2) * data.size(3))

    targets = (targets, shuffled_targets, float(lam))

    return mixed_data, targets


class CutMixCollator:
    def __init__(self, alpha, prob=1.0):
        self.alpha = alpha
        self.prob = prob

    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        if self.prob > 0.0 and np.random.rand() < self.prob:
            batch = cutmix(batch, self.alpha)
        return batch


class CutMixCriterion:
    def __init__(self, reduction='mean', label_smoothing=0.0):
        self.criterion = nn.CrossEntropyLoss(reduction=reduction, label_smoothing=label_smoothing)

    def __call__(self, preds, targets):
        targets1, targets2, lam = targets
        return lam * self.criterion(preds, targets1) + (1 - lam) * self.criterion(preds, targets2)
```
== Mixup <mixup_code>
```python
import numpy as np
import torch
import torch.nn as nn


def mixup(batch, alpha):
    data, targets = batch
    batch_size = data.size(0)

    if alpha <= 0:
        return data, targets

    lam = np.random.beta(alpha, alpha)
    
    indices = torch.randperm(batch_size)
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    mixed_data = lam * data + (1 - lam) * shuffled_data

    targets = (targets, shuffled_targets, float(lam))

    return mixed_data, targets


class MixupCollator:
    def __init__(self, alpha, prob=1.0):
        self.alpha = alpha
        self.prob = prob

    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        if self.prob > 0.0 and np.random.rand() < self.prob:
            batch = mixup(batch, self.alpha)
        return batch


class MixupCriterion:
    def __init__(self, reduction='mean', label_smoothing=0.0):
        self.criterion = nn.CrossEntropyLoss(reduction=reduction, label_smoothing=label_smoothing)

    def __call__(self, preds, targets):
        targets1, targets2, lam = targets
        return lam * self.criterion(preds, targets1) + (1 - lam) * self.criterion(preds, targets2)


```
== Autoaugment <autoaugment_code>
Cifar-10 Policy를 사용하였다.
```python
...
return [
          (("Invert", 0.1, None), ("Contrast", 0.2, 6)),
          (("Rotate", 0.7, 2), ("TranslateX", 0.3, 9)),
          (("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)),
          (("ShearY", 0.5, 8), ("TranslateY", 0.7, 9)),
          (("AutoContrast", 0.5, None), ("Equalize", 0.9, None)),
          (("ShearY", 0.2, 7), ("Posterize", 0.3, 7)),
          (("Color", 0.4, 3), ("Brightness", 0.6, 7)),
          (("Sharpness", 0.3, 9), ("Brightness", 0.7, 9)),
          (("Equalize", 0.6, None), ("Equalize", 0.5, None)),
          (("Contrast", 0.6, 7), ("Sharpness", 0.6, 5)),
          (("Color", 0.7, 7), ("TranslateX", 0.5, 8)),
          (("Equalize", 0.3, None), ("AutoContrast", 0.4, None)),
          (("TranslateY", 0.4, 3), ("Sharpness", 0.2, 6)),
          (("Brightness", 0.9, 6), ("Color", 0.2, 8)),
          (("Solarize", 0.5, 2), ("Invert", 0.0, None)),
          (("Equalize", 0.2, None), ("AutoContrast", 0.6, None)),
          (("Equalize", 0.2, None), ("Equalize", 0.6, None)),
          (("Color", 0.9, 9), ("Equalize", 0.6, None)),
          (("AutoContrast", 0.8, None), ("Solarize", 0.2, 8)),
          (("Brightness", 0.1, 3), ("Color", 0.7, 0)),
          (("Solarize", 0.4, 5), ("AutoContrast", 0.9, None)),
          (("TranslateY", 0.9, 9), ("TranslateY", 0.7, 9)),
          (("AutoContrast", 0.9, None), ("Solarize", 0.8, 3)),
          (("Equalize", 0.8, None), ("Invert", 0.1, None)),
          (("TranslateY", 0.7, 9), ("AutoContrast", 0.9, None)),
      ]
...
```

== Cutout <cutout_code>
```python
import numpy as np
import torch


class Cutout:    
    def __init__(self, n_holes=1, length=16, prob=1.0):
        self.n_holes = n_holes
        self.length = length
        self.prob = prob
    
    def __call__(self, img):
        if self.prob < 1.0 and np.random.rand() > self.prob:
            return img
        
        h = img.size(1)
        w = img.size(2)
        
        mask = np.ones((h, w), np.float32)
        
        for _ in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            mask[y1:y2, x1:x2] = 0
        
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        
        img = img * mask
        
        return img


```
= Experimentals <experimentals>

#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  inset: 5pt,
  [
    *Model Settings:*
    - *Architecture:* Various
    - *Parameters:* Various
    - *Weight Init:* Kaiming Normal
  ],
  [
    *Training Strategy:*
    - *Optimizer:* Adam (LR 3e-4)
    - *Scheduler:* Cosine Annealing (100 Epochs)
    - *Augmentation:* Standard + AutoAugment,
    - *Seed*: 42
  ]
)
== Residual Attention

#figure(
  table(
    columns: (1.5fr, 1.5fr, 0.6fr, 0.6fr),
    align: (left, center, center, center),
    inset: 10pt,
    stroke: none,
    table.hline(y: 0, stroke: 1pt),
    table.hline(y: 1, stroke: 0.5pt),
    table.header(
      [*Model Variant*], [*Optimization & Strategy*], [*Best Acc*], [*Params*]
    ),
    // 1. Standard Tiny
    [ResAttn-92 (Tiny)], [Standard], [95.43], [14.5 M],
    
    // 3. GeLU Tiny (Best Acc)
    [ResAttn-92 (Tiny)], [GeLU Activation], [95.47], [14.5 M],
    
    // 4. GeLU Medium
    [ResAttn-92 (Medium)], [GeLU Activation], [95.36], [25.5 M],
    
    // 5. GeLU Tiny + AdamW
    [ResAttn-92 (Tiny)], [GeLU, AdamW], [94.99], [14.5 M],
    
    // 6. SE Tiny
    [ResAttn-92 (Tiny)], [SE Block], [94.96], [14.6 M],
    
    // 7. Preact Tiny
    [ResAttn-92 (Tiny)], [Pre-activation], [94.82], [14.5 M],
    
    // 8. GeLU + DLA Tiny
    [ResAttn-92 (Tiny)], [GeLU, DLA (Tiny)], [94.05], [12.6 M],

    table.hline(stroke: 1pt),
  ),
  caption: [Residual Attention Network 변형 모델에 대한 Ablation Study]
)
== ConvNextV2
#figure(
  table(
    columns: (1.3fr, 2.2fr, 0.6fr, 0.6fr), // 전략 설명이 길어서 너비 조정
    align: (left, left, center, center),   // 전략 설명은 왼쪽 정렬이 깔끔함
    inset: 10pt,
    stroke: none,
    table.hline(y: 0, stroke: 1pt),
    table.hline(y: 1, stroke: 0.5pt),
    table.header(
      [*Model Variant*], [*Optimization & Strategy*], [*Best Acc*], [*Params*]
    ),
    
    // 1. Nano Base
    [ConvNeXt V2 (Nano)], [Standard (AdamW, LR 2e-3)], [94.51], [13.3 M],
    
    // 2. Nano k3 Base (Kernel size change)
    [ConvNeXt V2 (Nano, $k=3$)], [Standard (AdamW, LR 2e-3)], [95.48], [13.3 M],
    
    // 3. ASAM
    [ConvNeXt V2 (Nano, $k=3$)], [Standard + ASAM ($rho=2.0$)], [95.93], [13.3 M],
    
    // 4. ASAM + EMA
    [ConvNeXt V2 (Nano, $k=3$)], [Standard + ASAM + EMA], [95.95], [13.3 M],
    
    // 5. Heavy Augmentation (Best)
    [ConvNeXt V2 (Nano, $k=3$)], [CutMix, Mixup, LS 0.1, 200 Epochs], [96.51], [13.3 M],

    table.hline(stroke: 1pt),
  ),
  caption: [ConvNeXt V2 변형 및 최적화 전략에 대한 Ablation Study]
)
== Other
#figure(
  table(
    columns: (1.6fr, 1.8fr, 0.6fr, 0.6fr), // 모델명과 전략 설명 비율 조정
    align: (left, left, center, center),
    inset: 10pt,
    stroke: none,
    table.hline(y: 0, stroke: 1pt),
    table.hline(y: 1, stroke: 0.5pt),
    table.header(
      [*Model Variant*], [*Optimization & Strategy*], [*Best Acc*], [*Params*]
    ),
    
    // 1. Custom Filter/Pooling Variant
    [DeepBase3 (Custom)], [GAP+GMP, Custom Channels], [92.82], [7.8 M],
    
    // 2. Standard Base (Best in class)
    [DeepBase3 (Res)], [Standard], [94.65], [10.4 M],
    
    // 3. Bottleneck
    [DeepBase3 (Bottleneck)], [Standard], [89.69], [10.3 M],
    
    // 5. Res15 Base (Best in class)
    [DeepBase3 (Res-15)], [Standard], [94.84], [13.5 M],
    
    // 6. Res15 + AdamW
    [DeepBase3 (Res-15)], [AdamW], [94.39], [13.5 M],
    
    // 7. Res18 Base (Best Overall)
    [DeepBase3 (Res-18)], [Standard], [95.08], [24.5 M],
    
    // --- ConvNeXt Variants Group ---
    // 8. ConvNeXt Base
    [DeepBase3 (Res-15 Cx)], [Standard], [83.76], [11.0 M],
    
    // 10. ConvNeXt + AdamW + WD
    [DeepBase3 (Res-15 Cx)], [AdamW, Weight Decay 5e-5], [92.82], [11.0 M],
    
    // 11. ConvNeXt + AdamW + WD (Stronger)
    [DeepBase3 (Res-15 Cx)], [AdamW, Weight Decay 5e-3], [93.19], [11.0 M],
    
    // 12. ConvNeXt + LN Head
    [DeepBase3 (Res-15 Cx)], [AdamW, LN Classifier], [91.29], [11.0 M],
    
    // --- Other Variants ---
    // 14. LayerNorm Variant
    [DeepBase3 (Res-15 LN)], [Standard], [92.40], [13.5 M],
    
    // 15. Attention Variant
    [DeepBase3 (Res-15 Attn)], [Standard (Tiny)], [93.48], [13.0 M],

    table.hline(stroke: 1pt),
  ),
  caption: [Deep Baseline 3 변형(Res-15/18, ConvNeXt 적용)에 대한 Ablation Study]
)

= EMA <ema_code>
```python
import torch
import torch.nn as nn
from copy import deepcopy

class ModelEMA:
    def __init__(self, model, decay=0.9999, device=None):
        self.ema_model = deepcopy(model)
        self.ema_model.eval()
        self.decay = decay
        self.device = device
        
        for param in self.ema_model.parameters():
            param.requires_grad = False
            
        if self.device:
            self.ema_model.to(self.device)

    def update(self, model):
        needs_module = hasattr(model, 'module') and not self.is_parallel(self.ema_model)
        with torch.no_grad():
            msd = model.module.state_dict() if needs_module else model.state_dict()
            esd = self.ema_model.state_dict()
            
            for k in msd.keys():
                if msd[k].dtype.is_floating_point:
                    esd[k].mul_(self.decay).add_(msd[k], alpha=1 - self.decay)
    
    def is_parallel(self, model):
        return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

    def get_model(self):
        return self.ema_model
```

= ASAM <asam_code>
```python
import torch

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
```

= ShakeDrop <shakedrop_code>
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ShakeDropFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, training=True, p_drop=0.5, alpha_range=[-1, 1]):
        if training:
            gate = torch.zeros(1, device=x.device, dtype=torch.float32).bernoulli_(1 - p_drop)
            ctx.save_for_backward(gate)
            
            alpha = torch.empty(x.size(0), device=x.device, dtype=torch.float32).uniform_(*alpha_range)
            alpha = alpha.view(alpha.size(0), 1, 1, 1).expand_as(x)
            return gate * x + (1 - gate) * alpha * x
        else:
            return (1 - p_drop) * x

    @staticmethod
    def backward(ctx, grad_output):
        gate = ctx.saved_tensors[0]
        beta = torch.rand(grad_output.size(0), device=grad_output.device, dtype=torch.float32)
        beta = beta.view(beta.size(0), 1, 1, 1).expand_as(grad_output)
        return gate * grad_output + (1 - gate) * beta * grad_output, None, None, None


class ShakeDrop(nn.Module):
    def __init__(self, p_drop=0.5, alpha_range=[-1, 1]):
        super(ShakeDrop, self).__init__()
        self.p_drop = p_drop
        self.alpha_range = alpha_range

    def forward(self, x):
        return ShakeDropFunction.apply(x, self.training, self.p_drop, self.alpha_range)
```
= Stacking <stacking_code>
```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import json
import os
import sys
import random
import argparse
from pathlib import Path
from tqdm import tqdm
import hashlib

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from main import get_net

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS_L1 = 20
STACKING_DIR = "stacking"
STACKING_META_FILE = os.path.join(STACKING_DIR, "meta.json")
STACKING_TRAIN_FILE = os.path.join(STACKING_DIR, "S_train.npy")
STACKING_TEST_FILE = os.path.join(STACKING_DIR, "S_test.npy")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"시드 설정 완료: {seed}")

def load_model_from_history(history_path):
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"History 파일을 찾을 수 없습니다: {history_path}")
    
    with open(history_path, 'r') as f:
        history_data = json.load(f)
    
    if 'hyperparameters' in history_data and 'net' in history_data['hyperparameters']:
        model_name = history_data['hyperparameters']['net']
    else:
        raise ValueError(f"History 파일에 'hyperparameters.net' 정보가 없습니다: {history_path}")
    
    if history_path.endswith('_history.json'):
        model_path = history_path.replace('_history.json', '.pth')
    else:
        base_path = history_path.rsplit('.json', 1)[0]
        model_path = f"{base_path.rsplit('_history', 1)[0]}.pth"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
    
    normalize_mean = (0.4914, 0.4822, 0.4465)  # CIFAR-10 기본값
    normalize_std = (0.2023, 0.1994, 0.2010)  # CIFAR-10 기본값
    if 'hyperparameters' in history_data:
        hp = history_data['hyperparameters']
        if 'normalize_mean' in hp and 'normalize_std' in hp:
            normalize_mean = tuple(hp['normalize_mean'])
            normalize_std = tuple(hp['normalize_std'])
    
    shakedrop_prob = 0.0
    if 'hyperparameters' in history_data:
        hp = history_data['hyperparameters']
        if 'shakedrop_prob' in hp and hp['shakedrop_prob'] is not None:
            shakedrop_prob = hp['shakedrop_prob']
    
    return model_name, model_path, normalize_mean, normalize_std, shakedrop_prob

def get_history_paths_hash(history_paths):
    paths_str = json.dumps(sorted(history_paths), sort_keys=True)
    return hashlib.md5(paths_str.encode()).hexdigest()

def load_cached_meta_features():
    if not os.path.exists(STACKING_META_FILE):
        return None, None
    
    try:
        with open(STACKING_META_FILE, 'r') as f:
            meta_data = json.load(f)
        
        if not os.path.exists(STACKING_TRAIN_FILE) or not os.path.exists(STACKING_TEST_FILE):
            return None, None
        
        S_train = np.load(STACKING_TRAIN_FILE)
        S_test = np.load(STACKING_TEST_FILE)
        
        return meta_data, (S_train, S_test)
    except Exception as e:
        print(f"캐시 로드 중 오류 발생: {e}")
        return None, None

def save_meta_features(history_paths, S_train, S_test):
    os.makedirs(STACKING_DIR, exist_ok=True)
    
    meta_data = {
        'history_paths': history_paths,
        'history_paths_hash': get_history_paths_hash(history_paths),
        'num_models': len(history_paths),
        'num_classes': NUM_CLASSES
    }
    
    with open(STACKING_META_FILE, 'w') as f:
        json.dump(meta_data, f, indent=2)
    
    np.save(STACKING_TRAIN_FILE, S_train)
    np.save(STACKING_TEST_FILE, S_test)
    
    print(f"\n>>> Meta-features 저장 완료: {STACKING_DIR}/ <<<")

def predict(model, loader):
    model.eval()
    preds_list = []
    with torch.no_grad():
        for inputs, _ in tqdm(loader, desc="  예측 중", leave=False):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds_list.append(outputs.cpu().numpy())
    return np.vstack(preds_list)

class MetaLearner(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MetaLearner, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.net(x)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none', weight=alpha)

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss) # 예측 확률
        # 맞춘 샘플(pt가 큼)은 Loss가 0에 수렴, 틀린 샘플에 가중치 부여
        loss = (1 - pt) ** self.gamma * ce_loss 
        return loss.mean()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stacking Meta-Learner Training')
    parser.add_argument('--seed', type=int, default=42, help='랜덤 시드 (기본값: 42)')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    print(f"Running on {DEVICE}")
    
    HISTORY_PATHS = [
        "outputs/final2/pyramidnet/pyramidnet110_150_sgd_crossentropy_bs128_ep400_lr0.1_mom0.9_nesterov_schcosineannealinglr_tmax400_ls0.1_aug_autoaug_winit_ema_emad0.999_sam_samrho2.0_samadaptive_shakedrop_history.json",
        "outputs/final2/pyramidnet/pyramidnet110_150_sgd_crossentropy_bs128_ep200_lr0.1_mom0.9_nesterov_schcosineannealinglr_tmax200_ls0.1_aug_autoaug_winit_ema_emad0.999_sam_samrho2.0_samadaptive_shakedrop_history.json",
        "outputs/final2/wideresnet/wideresnet16_8_sgd_crossentropy_bs128_ep200_lr0.1_mom0.9_nesterov_schcosineannealinglr_tmax200_ls0.1_aug_autoaug_winit_ema_emad0.999_sam_samrho2.0_samadaptive_history.json"
    ]
    
    model_infos = []
    for history_path in HISTORY_PATHS:
        model_name, model_path, normalize_mean, normalize_std, shakedrop_prob = load_model_from_history(history_path)
        model_infos.append({
            'name': model_name,
            'path': model_path,
            'normalize_mean': normalize_mean,
            'normalize_std': normalize_std,
            'shakedrop_prob': shakedrop_prob
        })
        print(f"모델 로드: {model_name}")
        print(f"  경로: {model_path}")
        print(f"  Normalize: mean={normalize_mean}, std={normalize_std}")
        print(f"  ShakeDrop: {shakedrop_prob}")
        print()
    
    normalize_mean = model_infos[0]['normalize_mean']
    normalize_std = model_infos[0]['normalize_std']
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
    ])
    
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalize_mean, normalize_std),
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    
    train_targets = np.array(train_dataset.targets)
    
    current_hash = get_history_paths_hash(HISTORY_PATHS)
    cached_meta, cached_data = load_cached_meta_features()
    
    if cached_meta is not None and cached_data is not None:
        cached_hash = cached_meta.get('history_paths_hash')
        if cached_hash == current_hash:
            print(">>> 저장된 Meta-features 발견! Level 0 건너뛰기 <<<")
            S_train, S_test = cached_data
            print(f"S_train shape: {S_train.shape}")
            print(f"S_test shape: {S_test.shape}")
        else:
            print(">>> 히스토리 경로가 변경됨. Level 0 재실행 <<<")
            cached_meta = None
            cached_data = None
    
    if cached_meta is None or cached_data is None:
        num_train_samples = len(train_dataset)
        num_test_samples = len(test_dataset)
        
        S_train = np.zeros((num_train_samples, len(model_infos) * NUM_CLASSES))
        S_test = np.zeros((num_test_samples, len(model_infos) * NUM_CLASSES))
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
        print(">>> Start Level 0 Inference (Base Models) <<<")
        
        for model_idx, model_info in enumerate(tqdm(model_infos, desc="모델 처리", unit="model")):
            model_name = model_info['name']
            model_path = model_info['path']
            shakedrop_prob = model_info['shakedrop_prob']
            
            print(f"\n모델 {model_idx+1}/{len(model_infos)}: {model_name}")
            print(f"  모델 경로: {model_path}")
            
            model = get_net(model_name, init_weights=False, shakedrop_prob=shakedrop_prob)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model = model.to(DEVICE)
            model.eval()
            
            print(f"  Train set 예측 중...")
            train_preds = predict(model, train_loader)
            start_col = model_idx * NUM_CLASSES
            end_col = (model_idx + 1) * NUM_CLASSES
            S_train[:, start_col:end_col] = train_preds
            
            print(f"  Test set 예측 중...")
            test_preds = predict(model, test_loader)
            S_test[:, start_col:end_col] = test_preds
        
        print("\n>>> Level 0 Inference Complete. Meta-features generated.")
        print(f"S_train shape: {S_train.shape}") # (50000, 모델수 * 10)
        print(f"S_test shape: {S_test.shape}")   # (10000, 모델수 * 10)
        
        # Meta-features 저장
        save_meta_features(HISTORY_PATHS, S_train, S_test)
    

    X_meta_train = torch.FloatTensor(S_train)
    y_meta_train = torch.LongTensor(train_targets)
    X_meta_test = torch.FloatTensor(S_test)
    y_meta_test = torch.LongTensor(np.array(test_dataset.targets))
    
    meta_train_dataset = torch.utils.data.TensorDataset(X_meta_train, y_meta_train)
    meta_train_loader = DataLoader(meta_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    meta_test_dataset = torch.utils.data.TensorDataset(X_meta_test, y_meta_test)
    meta_test_loader = DataLoader(meta_test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    meta_model = MetaLearner(input_dim=len(model_infos)*NUM_CLASSES, num_classes=NUM_CLASSES).to(DEVICE)
    meta_optimizer = optim.Adam(meta_model.parameters(), lr=0.005, weight_decay=1e-1)
    meta_scheduler = optim.lr_scheduler.CosineAnnealingLR(meta_optimizer, T_max=EPOCHS_L1)
    meta_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    # meta_criterion = FocalLoss(gamma=2)
    
    print("\n>>> Start Level 1 Training (Stacking) <<<")
    
    os.makedirs(STACKING_DIR, exist_ok=True)
    
    best_acc = 0.0
    best_epoch = 0
    best_model_path = None
    for epoch in tqdm(range(EPOCHS_L1), desc="Meta Learner 학습", unit="epoch"):
        meta_model.train()
        total_loss = 0
        pbar_train = tqdm(meta_train_loader, desc=f"  Epoch {epoch+1}/{EPOCHS_L1} [Train]", leave=False)
        for inputs, targets in pbar_train:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            meta_optimizer.zero_grad()
            outputs = meta_model(inputs)
            loss = meta_criterion(outputs, targets)
            loss.backward()
            meta_optimizer.step()
            total_loss += loss.item()
            pbar_train.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Evaluation
        meta_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            pbar_test = tqdm(meta_test_loader, desc=f"  Epoch {epoch+1}/{EPOCHS_L1} [Eval]", leave=False)
            for inputs, targets in pbar_test:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = meta_model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                pbar_test.set_postfix({'acc': f'{100 * correct / total:.2f}%'})
                
        acc = 100 * correct / total
        current_lr = meta_scheduler.get_last_lr()[0]
        
        # 최고 성능 모델 저장
        if acc > best_acc:
            # 이전 최고 모델 파일 삭제
            if best_model_path is not None and os.path.exists(best_model_path):
                os.remove(best_model_path)
            
            best_acc = acc
            best_epoch = epoch + 1
            # 파일 이름에 test accuracy 포함
            best_model_path = os.path.join(STACKING_DIR, f"best_meta_learner_seed{args.seed}_acc{acc:.2f}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': meta_model.state_dict(),
                'optimizer_state_dict': meta_optimizer.state_dict(),
                'scheduler_state_dict': meta_scheduler.state_dict(),
                'test_acc': acc,
                'seed': args.seed
            }, best_model_path)
            print(f"Epoch {epoch+1}/{EPOCHS_L1} | Loss: {total_loss/len(meta_train_loader):.4f} | Test Acc: {acc:.2f}% | LR: {current_lr:.6f} | [BEST] 모델 저장")
        else:
            print(f"Epoch {epoch+1}/{EPOCHS_L1} | Loss: {total_loss/len(meta_train_loader):.4f} | Test Acc: {acc:.2f}% | LR: {current_lr:.6f}")
        
        meta_scheduler.step()
    
    print(f"\n>>> Stacking Complete. <<<")
    print(f">>> 최고 Test Accuracy: {best_acc:.2f}% (Epoch {best_epoch}) <<<")
    if best_model_path is not None:
        print(f">>> 최고 성능 모델 저장 경로: {best_model_path} <<<")
```