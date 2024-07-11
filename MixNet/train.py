import os
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.optim import lr_scheduler
from accelerate import Accelerator, DistributedDataParallelKwargs

import gc

from dataset.concat_datasets import AllDataset, AllDataset_mid
from dataset.my_dataset_mid import myDataset_mid
from network.loss import TextLoss
from network.textnet import TextNet
from cfglib.config import config as cfg, update_config, print_config
from util.misc import AverageMeter
from util.misc import mkdirs
from util.visualize import visualize_network_output
from cfglib.option import BaseOptions

from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from util.augmentation import Augmentation

# ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
# accelerator = Accelerator(device_placement=True ,kwargs_handlers=[ddp_kwargs])
accelerator = Accelerator(device_placement=True)
# accelerator = Accelerator(device_placement=True, mixed_precision='bf16')

def save_model(model, epoch, lr):
    save_dir = os.path.join(cfg.save_dir, cfg.exp_name)
    if not os.path.exists(save_dir):
        mkdirs(save_dir)
    
    if accelerator.state.distributed_type == "MULTI_GPU":
        model = accelerator.unwrap_model(model)
    
    save_path = os.path.join(save_dir, f'MixNet_{model.backbone_name}_{epoch}.pth')
    print(f'Saving to {save_path}')
    state_dict = {
        'lr' : lr,
        'epoch' : epoch,
        'model' : model.state_dict()
    }
    torch.save(state_dict, save_path)


# def load_model(model, model_path):
#     if accelerator.is_main_process:
#         print(f"Loading from {model_path}")
#     # state_dict = torch.load(model_path,  map_location=cfg.device)
#     state_dict = torch.load(model_path, map_location=accelerator.device)
#     try:
#         model.load_state_dict(state_dict['model'])
#     except RuntimeError as e:
#         model.load_state_dict(state_dict['model'], strict = False)

def load_model(model, model_path):
    print(f"가중치 파일 로드 중: {model_path}")
    state_dict = torch.load(model_path, map_location=accelerator.device)
    
    # 기존 모델의 state_dict와 새 모델의 state_dict 키 비교
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
    
    # 새 모델의 state_dict 업데이트
    model_dict.update(pretrained_dict)
    
    # 수정된 state_dict를 모델에 로드
    model.load_state_dict(model_dict, strict=False)
    
    print("기존 가중치 로드 완료. 새로운 레이어는 초기화된 상태로 유지됩니다.")



def _parse_data(inputs):
    input_dict = {}
    # inputs = list(map(lambda x: to_device(x), inputs))
    inputs = list(map(lambda x: accelerator.prepare(x), inputs))
    input_dict['img'] = inputs[0]
    input_dict['train_mask'] = inputs[1]
    input_dict['tr_mask'] = inputs[2]
    input_dict['distance_field'] = inputs[3]
    input_dict['direction_field'] = inputs[4]
    input_dict['weight_matrix'] = inputs[5]
    input_dict['gt_points'] = inputs[6]
    input_dict['proposal_points'] = inputs[7]
    input_dict['ignore_tags'] = inputs[8]
    if cfg.mid:
        input_dict['gt_mid_points'] = inputs[9]
        input_dict['edge_field'] = inputs[10]
    return input_dict

def train(model, train_loader, criterion, scheduler, optimizer, epoch, writer):
    global train_step

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    model.train()

    if accelerator.is_main_process:
        print(f'Epoch: {epoch} : LR = {scheduler.get_lr()}')

    accumulation_steps = 8
    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for i, inputs in enumerate(pbar):
        data_time.update(time.time() - end)
        train_step = 1
        input_dict = _parse_data(inputs)

        with accelerator.accumulate(accumulation_steps):
            output_dict = model(input_dict)
            loss_dict = criterion(input_dict, output_dict, eps=epoch+1)
            loss = loss_dict["total_loss"]
            accelerator.backward(loss)

        if (i+1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        losses.update(loss.item())

        batch_time.update(time.time()-end)
        end = time.time()

        # 학습과정 visualization
        if cfg.viz and (i % cfg.viz_freq == 0 and i > 0):
            visualize_network_output(output_dict, input_dict, mode='train')

        max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

        # 각 레이어의 GPU 메모리 사용량 출력
        # layer_memory = {}
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         layer_memory[name] = param.element_size() * param.nelement() / 1024 / 1024
        # print(layer_memory)
        ## 
        pbar.set_postfix({'Training Loss': f'{losses.avg:.2f}', 'Max Memory': f'{max_memory:.2f} MB'})
        writer.add_scalar('Loss/train', losses.avg, epoch * len(train_loader) + i)

        # 메모리 정리
        del input_dict, output_dict, loss_dict, loss
        torch.cuda.empty_cache()
        gc.collect()

    if epoch % cfg.save_freq == 0:
        save_model(model, epoch, scheduler.get_lr())

def main():
    global lr
    torch.autograd.set_detect_anomaly(True)

    if not cfg.temp:
        if not cfg.mid:
            trainset = AllDataset(config=cfg, custom_data_root="./data/kor_extended", open_data_root="./data/open_datas", is_training=True, load_memory=cfg.load_memory)

        elif cfg.mid:
            trainset = AllDataset_mid(config=cfg, custom_data_root="./data/kor_extended", open_data_root="./data/open_datas", is_training=True, load_memory=cfg.load_memory)

    elif cfg.temp:
        trainset = myDataset_mid(data_root="./data/kor_extended",
                                 is_training=True, 
                                 load_memory=cfg.load_memory,
                                 transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds))

    if cfg.server_code == 24:       ## 24 서버 Torch 버전 문제로 인해 Generator 호환 X
        train_loader = data.DataLoader(trainset, batch_size=cfg.batch_size,
                                       shuffle=True, num_workers=cfg.num_workers,
                                       pin_memory=True)
    else:
        train_loader = data.DataLoader(trainset, batch_size=cfg.batch_size,
                                       shuffle=True, num_workers=cfg.num_workers,
                                       pin_memory=True, generator=torch.Generator(device=cfg.device))
    
    model = TextNet(backbone=cfg.net, is_training=True, freeze_backbone=cfg.freeze_backbone)
    # model = model.to(cfg.device)

    criterion = TextLoss(accelerator)
    lr = cfg.lr

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=0.9)

    model, optimizer, train_loader, criterion = accelerator.prepare(model, optimizer, train_loader, criterion)
    writer = SummaryWriter(log_dir=os.path.join(cfg.save_dir, cfg.exp_name, 'logs'))

    if cfg.cuda:
        cudnn.benchmark = True
    if cfg.resume:
        load_model(model, cfg.resume)
    if cfg.freeze_backbone and not cfg.resume:
        assert "Freeze backbone is only available when resume is True"

    if accelerator.is_main_process:
        print('Start training MixNet.')
    for epoch in range(cfg.start_epoch, cfg.max_epoch+1):
        scheduler.step()
        train(model, train_loader, criterion, scheduler, optimizer, epoch, writer)

    writer.close()
    print('End.')

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    np.random.seed(2022)
    torch.manual_seed(2022)
    # parse arguments
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    
    if accelerator.is_main_process:
        print_config(cfg)

    # main
    main()
