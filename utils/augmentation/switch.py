import numpy as np


class SwitchCollator:
    """CutMix와 Mixup을 각 배치마다 랜덤하게 선택하는 Collator"""
    def __init__(self, cutmix_collator, mixup_collator, switch_prob=0.5):
        self.cutmix_collator = cutmix_collator
        self.mixup_collator = mixup_collator
        self.switch_prob = switch_prob
    
    def __call__(self, batch):
        # switch_prob 확률로 Mixup 선택, (1 - switch_prob) 확률로 CutMix 선택
        if np.random.rand() < self.switch_prob:
            return self.mixup_collator(batch)
        else:
            return self.cutmix_collator(batch)

