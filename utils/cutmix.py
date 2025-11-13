import numpy as np
import torch
import torch.nn as nn


def _rand_bbox(size, lam):
    """CutMix에서 사용할 박스 좌표 생성"""
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
    """Batch 단위 CutMix 적용"""
    data, targets = batch
    batch_size = data.size(0)

    if alpha <= 0:
        return data, targets

    lam = np.random.beta(alpha, alpha)
    indices = torch.randperm(batch_size)
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    x0, y0, x1, y1 = _rand_bbox(data.size(), lam)

    # 데이터 복사 후 CutMix 적용 (in-place 수정 방지)
    mixed_data = data.clone()
    mixed_data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]

    # 실제로 잘린 영역 비율로 lam 업데이트
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
        # 확률적으로 CutMix 적용
        if self.prob > 0.0 and np.random.rand() < self.prob:
            batch = cutmix(batch, self.alpha)
        return batch


class CutMixCriterion:
    def __init__(self, reduction='mean', label_smoothing=0.0):
        self.criterion = nn.CrossEntropyLoss(reduction=reduction, label_smoothing=label_smoothing)

    def __call__(self, preds, targets):
        targets1, targets2, lam = targets
        return lam * self.criterion(preds, targets1) + (1 - lam) * self.criterion(preds, targets2)