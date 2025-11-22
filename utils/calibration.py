"""
Temperature Scaling 캘리브레이션

References:
- Guo et al. "On Calibration of Modern Neural Networks" (2017)
- https://github.com/gpleiss/temperature_scaling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ECELoss(nn.Module):
    """
    Expected Calibration Error (ECE) 계산
    
    ECE는 모델의 예측 신뢰도와 실제 정확도 간의 차이를 측정합니다.
    """
    def __init__(self, n_bins=15):
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
    
    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # 각 bin에서 |confidence - accuracy| 계산
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece


def calibrate_temperature(net, val_loader, device, cross_validate='ece', log=True):
    """
    Temperature Scaling을 사용하여 모델 캘리브레이션
    
    Grid search 방식으로 0.1부터 10.0까지 temperature를 탐색하여
    NLL 또는 ECE를 최소화하는 최적 temperature를 찾습니다.
    
    Args:
        net: 학습된 모델
        val_loader: 검증 데이터로더
        device: 디바이스
        cross_validate: 'ece' 또는 'nll' - 최적화할 metric 선택
        log: 로그 출력 여부
    
    Returns:
        optimal_temperature: 최적의 temperature 값
        metrics_dict: 캘리브레이션 전후 metric 값 딕셔너리
    """
    net.eval()
    
    # 검증 세트의 모든 로짓과 레이블 수집
    logits_list = []
    labels_list = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            logits = net(inputs)
            logits_list.append(logits)
            labels_list.append(labels)
    
    # 전체 로짓과 레이블을 하나로 합치기
    all_logits = torch.cat(logits_list).to(device)
    all_labels = torch.cat(labels_list).to(device)
    
    # Criterion 정의
    nll_criterion = nn.CrossEntropyLoss().to(device)
    ece_criterion = ECELoss().to(device)
    
    # Temperature scaling 전 성능 측정
    with torch.no_grad():
        before_nll = nll_criterion(all_logits, all_labels).item()
        before_ece = ece_criterion(all_logits, all_labels).item()
    
    if log:
        print('캘리브레이션 전 - NLL: %.4f, ECE: %.4f' % (before_nll, before_ece))
    
    # Grid search로 최적 temperature 찾기 (0.1 ~ 10.0, step=0.1)
    best_nll = float('inf')
    best_ece = float('inf')
    T_opt_nll = 1.0
    T_opt_ece = 1.0
    
    T = 0.1
    for i in range(100):  # 0.1부터 10.0까지
        with torch.no_grad():
            scaled_logits = all_logits / T
            current_nll = nll_criterion(scaled_logits, all_labels).item()
            current_ece = ece_criterion(scaled_logits, all_labels).item()
        
        if current_nll < best_nll:
            best_nll = current_nll
            T_opt_nll = T
        
        if current_ece < best_ece:
            best_ece = current_ece
            T_opt_ece = T
        
        T += 0.1
    
    # 선택한 metric에 따라 최적 temperature 결정
    if cross_validate == 'ece':
        optimal_temperature = T_opt_ece
    else:
        optimal_temperature = T_opt_nll
    
    # Temperature scaling 후 성능 측정
    with torch.no_grad():
        scaled_logits = all_logits / optimal_temperature
        after_nll = nll_criterion(scaled_logits, all_labels).item()
        after_ece = ece_criterion(scaled_logits, all_labels).item()
    
    if log:
        print('최적 Temperature: %.4f (기준: %s)' % (optimal_temperature, cross_validate.upper()))
        print('캘리브레이션 후 - NLL: %.4f, ECE: %.4f' % (after_nll, after_ece))
    
    metrics_dict = {
        'before_nll': before_nll,
        'before_ece': before_ece,
        'after_nll': after_nll,
        'after_ece': after_ece,
        'optimal_temperature': optimal_temperature,
        'T_opt_nll': T_opt_nll,
        'T_opt_ece': T_opt_ece
    }
    
    return optimal_temperature, metrics_dict

