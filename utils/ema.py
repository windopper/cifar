import torch
import torch.nn as nn
from copy import deepcopy

class ModelEMA:
    """ 
    Model Exponential Moving Average V2
    학습 중인 모델(model)의 가중치를 decay 비율만큼 EMA 모델(ema_model)에 반영
    """
    def __init__(self, model, decay=0.9999, device=None):
        # 1. 모델의 구조와 초기 가중치를 깊은 복사(Deep Copy)
        self.ema_model = deepcopy(model)
        self.ema_model.eval() # EMA 모델은 학습(Backprop)하지 않으므로 평가 모드 고정
        self.decay = decay
        self.device = device  # 특정 GPU 지정이 필요할 경우 사용
        
        # 2. EMA 모델의 파라미터는 업데이트(Gradient 계산) 되지 않도록 설정
        for param in self.ema_model.parameters():
            param.requires_grad = False
            
        if self.device:
            self.ema_model.to(self.device)

    def update(self, model):
        # 모델이 DataParallel 등으로 감싸져 있을 경우 처리
        needs_module = hasattr(model, 'module') and not self.is_parallel(self.ema_model)
        with torch.no_grad():
            msd = model.module.state_dict() if needs_module else model.state_dict()
            esd = self.ema_model.state_dict()
            
            for k in msd.keys():
                # EMA 공식: New_EMA = Decay * Old_EMA + (1-Decay) * New_Weight
                if msd[k].dtype.is_floating_point: # float 타입 파라미터만 업데이트
                    esd[k].mul_(self.decay).add_(msd[k], alpha=1 - self.decay)
                    # .mul_(), .add_()는 In-place 연산으로 메모리 절약
    
    # Helper function for DataParallel check
    def is_parallel(self, model):
        return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)

    # 학습이 끝난 후 EMA 모델을 반환
    def get_model(self):
        return self.ema_model