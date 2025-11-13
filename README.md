

uv run main.py --optimizer adam --epochs 100 --lr 3e-4 --scheduler cosineannealinglr --net densenet121

*다른 모델 결과는 아래에 추가될 예정입니다.*

핵심 개선 포인트
아키텍처
Conv-BN-ReLU 표준화: 각 Conv2d 뒤에 BatchNorm2d+ReLU를 붙이고, 커널은 3×3, padding=1로 공간 크기 보존.
채널 폭 확장: 3단계 스테이지로 폭을 64→128→256 등으로 점진 확대.
다운샘플 방식: MaxPool2d(2) 또는 stride=2 conv로 단계별 해상도 절반.
GAP로 FC 축소: 큰 FC 스택 제거, AdaptiveAvgPool2d(1)+Linear(num_features, 10)로 파라미터·과적합 감소.
규제: Dropout(0.1~0.3)를 분류기 앞에 사용.
선택적: 잔차 연결(ResNet BasicBlock), Squeeze-and-Excitation(채널 어텐션), SiLU 대체 등으로 추가 이득.
정규화/손실
Weight Decay: 5e-4(또는 AdamW 1e-4~5e-4).
Label Smoothing: 0.05~0.1.
마지막 로짓의 Temperature Scaling으로 캘리브레이션(선택).
옵티마이저/스케줄러
SGD + momentum 0.9(또는 Nesterov) 권장. 초기 lr 0.1(배치 128 기준), 배치에 따라 선형 스케일.
CosineAnnealingLR(+ 5~10 epoch warmup) 또는 OneCycleLR.
EMA(Exponential Moving Average) 가중치로 일반화 향상.
데이터 증강(CIFAR-10 권장)
기본: RandomCrop(32, padding=4), RandomHorizontalFlip().
추가: 약한 ColorJitter 또는 TrivialAugmentWide/AutoAugment.
Mixup 0.2~0.4 / CutMix 0.5 중 하나 또는 병행.
텐서 변환 후 RandomErasing(p≈0.25).

학습 안정화/속도
AMP 자동 혼합정밀(torch.cuda.amp.autocast, GradScaler).
Gradient Clipping(예: 1.0).
cudnn.benchmark=True(고정 해상도일 때), 재현성 필요 시 시드 고정.
최고 검증 성능 체크포인트 및 Early Stopping.
평가/모니터링
TTA(수평뒤집음 등 소규모)로 최종 한두 점 향상.
혼동행렬/클래스별 F1, 잘못 분류 샘플 점검.