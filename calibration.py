import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def calibrate_temperature(net, val_loader, device, max_iter=50):
    """
    Temperature Scaling을 사용하여 모델 캘리브레이션
    
    Args:
        net: 학습된 모델
        val_loader: 검증 데이터로더
        device: 디바이스
        max_iter: 최적화 최대 반복 횟수
    
    Returns:
        최적의 temperature 값
    """
    net.eval()
    
    # 검증 세트의 모든 로짓과 레이블 수집
    logits_list = []
    labels_list = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = net(inputs)
            logits_list.append(logits.cpu())
            labels_list.append(labels.cpu())
    
    # 전체 로짓과 레이블을 하나로 합치기
    all_logits = torch.cat(logits_list, dim=0).to(device)
    all_labels = torch.cat(labels_list, dim=0).to(device)
    
    # Temperature 파라미터 초기화 (1.0에서 시작)
    temperature = nn.Parameter(torch.ones(1).to(device) * 1.0)
    
    # Temperature만 최적화
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)
    
    def eval():
        optimizer.zero_grad()
        # 로짓을 temperature로 나눔
        scaled_logits = all_logits / temperature
        loss = F.cross_entropy(scaled_logits, all_labels)
        loss.backward()
        return loss
    
    # 최적화 수행
    optimizer.step(eval)
    
    # 최종 temperature 값
    optimal_temperature = temperature.item()
    
    net.train()
    return optimal_temperature

