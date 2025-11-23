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