import numpy as np
import torch
import torch.nn as nn


def mixup(batch, alpha):
    """Batch 단위 Mixup 적용"""
    data, targets = batch
    batch_size = data.size(0)

    if alpha <= 0:
        return data, targets

    # Beta 분포에서 lam 샘플링
    lam = np.random.beta(alpha, alpha)
    
    # 배치 내에서 랜덤하게 섞기
    indices = torch.randperm(batch_size)
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    # Mixup: 두 이미지를 가중치로 선형 결합
    mixed_data = lam * data + (1 - lam) * shuffled_data

    # targets를 (targets1, targets2, lam) 튜플 형태로 반환
    targets = (targets, shuffled_targets, float(lam))

    return mixed_data, targets


class MixupCollator:
    def __init__(self, alpha, prob=1.0):
        self.alpha = alpha
        self.prob = prob

    def __call__(self, batch):
        batch = torch.utils.data.dataloader.default_collate(batch)
        # 확률적으로 Mixup 적용
        if self.prob > 0.0 and np.random.rand() < self.prob:
            batch = mixup(batch, self.alpha)
        return batch


class MixupCriterion:
    def __init__(self, reduction='mean', label_smoothing=0.0):
        self.criterion = nn.CrossEntropyLoss(reduction=reduction, label_smoothing=label_smoothing)

    def __call__(self, preds, targets):
        targets1, targets2, lam = targets
        return lam * self.criterion(preds, targets1) + (1 - lam) * self.criterion(preds, targets2)

