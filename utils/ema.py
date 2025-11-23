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