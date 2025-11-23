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

