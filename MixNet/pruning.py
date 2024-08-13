import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from network.textnet import TextNet
from config import cfg

def load_model(model, checkpoint_path):
    print(f'모델 로딩 중: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(cfg.device))
    model.load_state_dict(checkpoint['model'])
    return model

def prune_model(model, amount=0.3):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
    return model

def evaluate_model(model, test_loader):
    # 여기에 모델 평가 로직을 구현하세요
    # 예: 정확도, F1 점수 등을 계산
    pass

def main():
    # 모델 초기화
    model = TextNet(backbone='FSNet_M', is_training=False)
    model = model.to(cfg.device)

    # 학습된 가중치 로드
    checkpoint_path = 'path/to/your/checkpoint.pth'
    model = load_model(model, checkpoint_path)

    # 프루닝 전 모델 평가
    print("프루닝 전 모델 평가:")
    evaluate_model(model, test_loader)

    # 모델 프루닝
    pruned_model = prune_model(model, amount=0.3)

    # 프루닝 후 모델 평가
    print("프루닝 후 모델 평가:")
    evaluate_model(pruned_model, test_loader)

    # 프루닝된 모델 저장
    torch.save({
        'model': pruned_model.state_dict(),
    }, 'pruned_model.pth')

if __name__ == '__main__':
    main()
