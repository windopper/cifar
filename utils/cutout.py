import numpy as np
import torch


class Cutout:
    """
    Cutout augmentation: 이미지의 일부 영역을 랜덤하게 0으로 마스킹
    
    Args:
        n_holes: 마스킹할 영역의 개수 (default: 1)
        length: 마스킹 영역의 크기 (픽셀 단위, default: 16)
        prob: Cutout을 적용할 확률 (default: 1.0)
    """
    
    def __init__(self, n_holes=1, length=16, prob=1.0):
        self.n_holes = n_holes
        self.length = length
        self.prob = prob
    
    def __call__(self, img):
        """
        Args:
            img: Tensor 이미지 (C, H, W)
        
        Returns:
            Cutout이 적용된 이미지
        """
        # 확률적으로 Cutout 적용
        if self.prob < 1.0 and np.random.rand() > self.prob:
            return img
        
        h = img.size(1)
        w = img.size(2)
        
        mask = np.ones((h, w), np.float32)
        
        for _ in range(self.n_holes):
            # 랜덤한 중심점 선택
            y = np.random.randint(h)
            x = np.random.randint(w)
            
            # 마스킹 영역 계산
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            
            # 마스킹 영역을 0으로 설정
            mask[y1:y2, x1:x2] = 0
        
        # 마스크를 텐서로 변환하고 채널 차원 추가
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        
        # 이미지에 마스크 적용
        img = img * mask
        
        return img

