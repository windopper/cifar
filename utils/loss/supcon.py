import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al., 2020)

    - 입력 feature는 [batch_size, feature_dim] 형태라고 가정합니다.
    - 내부적으로는 view 수가 1인 SupCon으로 처리합니다.
    """

    def __init__(self, temperature: float = 0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, feature_dim]
            labels: [batch_size]

        Returns:
            loss: scalar tensor
        """
        device = features.device

        # [batch, dim] -> [batch, 1, dim] -> [batch, dim]
        features = F.normalize(features, dim=1)

        batch_size = features.shape[0]

        if batch_size <= 1:
            # batch 크기가 1인 경우 contrastive term을 계산할 수 없음
            return torch.tensor(0.0, device=device, requires_grad=True)

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError("features와 labels의 batch 크기가 일치하지 않습니다.")

        # 동일 클래스 마스크 (자기 자신 포함)
        mask = torch.eq(labels, labels.T).float().to(device)

        # similarity matrix: [N, N]
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature,
        )

        # 수치 안정성을 위한 최대값 빼기
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # 자기 자신은 제외
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask

        # log_prob 계산
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # 각 anchor에 대해, 동일 클래스(양성)의 log_prob 평균
        mask_sum = mask.sum(dim=1)
        # 양성이 하나도 없는 경우(배치 내에 해당 클래스가 1개뿐인 경우)는 제외
        valid_mask = mask_sum > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask_sum + 1e-12)

        # 유효한 anchor만 사용
        loss = -mean_log_prob_pos[valid_mask].mean()
        return loss


