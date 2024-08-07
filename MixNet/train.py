import os
import time
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.optim import lr_scheduler

import gc

from dataset.concat_datasets import AllDataset, AllDataset_mid
from network.loss import TextLoss
from network.textnet import TextNet
from cfglib.config import config as cfg, update_config, print_config
from util.misc import AverageMeter
from util.misc import mkdirs, to_device
from util.visualize import visualize_network_output
from cfglib.option import BaseOptions

from tqdm.auto import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs

kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs])
# accelerator = Accelerator()
train_step = 0

def save_model(model, epoch, lr):
    save_dir = os.path.join(cfg.save_dir, cfg.exp_name)
    if not os.path.exists(save_dir):
        mkdirs(save_dir)
    
    if accelerator:
        if accelerator.state.distributed_type == "MULTI_GPU":
            model = accelerator.unwrap_model(model)

    save_path = os.path.join(save_dir, f'MixNet_{model.backbone_name}_{epoch}.pth')

    if accelerator:
        if accelerator.is_main_process:
            print(f'Saving to {save_path}')
    else:
        print(f'Saving to {save_path}')

    state_dict = {
        'lr' : lr,
        'epoch' : epoch,
        'model' : model.state_dict()
    }
    if accelerator:
        if accelerator.is_main_process:
            torch.save(state_dict, save_path)
    else:
        torch.save(state_dict, save_path)

def load_model(model, model_path):
    print('Loading from {}'.format(model_path))
    if accelerator:
        state_dict = torch.load(model_path, map_location=accelerator.device)
    state_dict = torch.load(model_path, map_location=cfg.device)
    try:
        model.load_state_dict(state_dict['model'])
    except RuntimeError as e:
        print("Missing key in state_dict, try to load with strict = False")
        model.load_state_dict(state_dict['model'], strict = False)
        print(e)

def _parse_data(inputs):
    input_dict = {}
    if accelerator:
        inputs = list(map(lambda x: accelerator.prepare(x), inputs))
    else:
        inputs = list(map(lambda x: to_device(x), inputs))
    input_dict['img'] = inputs[0]
    input_dict['train_mask'] = inputs[1]
    input_dict['tr_mask'] = inputs[2]
    input_dict['distance_field'] = inputs[3]
    input_dict['direction_field'] = inputs[4]
    input_dict['weight_matrix'] = inputs[5]
    input_dict['gt_points'] = inputs[6]
    input_dict['proposal_points'] = inputs[7]
    input_dict['ignore_tags'] = inputs[8]
    if cfg.embed:
        input_dict['edge_field'] = inputs[9]
    if cfg.mid:
        input_dict['gt_mid_points'] = inputs[9]
        input_dict['edge_field'] = inputs[10]

    return input_dict

def train(model, train_loader, criterion, scheduler, optimizer, epoch):
    global train_step

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    model.train()

    if accelerator:
        if accelerator.is_main_process:
            print(f'Epoch: {epoch} : LR = {scheduler.get_lr()}')
    else:
        print(f'Epoch: {epoch} : LR = {scheduler.get_lr()}')
    

    # log 파일 경로 설정
    log_dir = os.path.join(cfg.save_dir, cfg.exp_name)
    log_path = os.path.join(log_dir, 'train_log.txt')
    if not os.path.exists(log_dir):
        mkdirs(log_dir)
    
    with open(log_path, 'a') as log_file:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for i, inputs in enumerate(pbar):
            data_time.update(time.time() - end)
            train_step += 1
            input_dict = _parse_data(inputs)

            output_dict = model(input_dict)
            loss_dict = criterion(input_dict, output_dict, eps=epoch+1)
            loss = loss_dict["total_loss"]

            optimizer.zero_grad()

            if accelerator:
                accelerator.backward(loss)
            else:
                loss.backward()

            # 그래디언트 확인
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm()
                    if torch.isnan(grad_norm):
                        print(f"NaN gradient detected in {name}")
                    elif grad_norm > 1000:
                        print(f"Large gradient detected in {name}: {grad_norm}")
            ####
                                    
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            optimizer.step()
            
            losses.update(loss.item())

            batch_time.update(time.time()-end)
            end = time.time()

            # 학습과정 visualization
            if cfg.viz and (i % cfg.viz_freq == 0 and i > 0):
                visualize_network_output(output_dict, input_dict, mode='train')

            max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024

            pbar.set_postfix({'Training Loss': f'{losses.avg:.2f}', 'Max Memory': f'{max_memory:.2f} MB'})

            # 로그 파일에 학습 정보 저장
            log_file.write(f'Epoch: {epoch}, Step: {i}, Loss: {losses.avg:.2f}, Max Memory: {max_memory:.2f} MB\n')

            # 메모리 정리
            del input_dict, output_dict, loss_dict, loss
            torch.cuda.empty_cache()
            gc.collect()

    if epoch % cfg.save_freq == 0:
        save_model(model, epoch, scheduler.get_lr())

def main():
    global lr

    dataset_params = {
        "config": cfg,
        "is_training" : True
    }

    trainset = AllDataset_mid(**dataset_params) if cfg.mid else AllDataset(**dataset_params)
    train_loader = data.DataLoader(trainset, batch_size=cfg.batch_size,
                                       shuffle=True, num_workers=cfg.num_workers,
                                       pin_memory=True, generator=torch.Generator(device=cfg.device))
    
    model = TextNet(backbone=cfg.net, is_training=True)

    if cfg.resume:
        load_model(model, cfg.resume)
        
    criterion = TextLoss()
    lr = cfg.lr

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=0.9)

    if accelerator:
        model, optimizer, train_loader, criterion, scheduler = accelerator.prepare(model, optimizer, train_loader, criterion, scheduler)
    if cfg.cuda:
        cudnn.benchmark = True

    for epoch in range(cfg.start_epoch, cfg.max_epoch+1):
        scheduler.step()
        train(model, train_loader, criterion, scheduler, optimizer, epoch)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if __name__ == "__main__":
    np.random.seed(2022)
    torch.manual_seed(2022)
    # parse arguments
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    
    if accelerator:
        if accelerator.is_main_process:
            print_config(cfg)
    else:
        print_config(cfg)

    log_dir = os.path.join(cfg.save_dir, cfg.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'config.txt'), 'w') as f:
        for key, value in vars(cfg).items():
            f.write(f'{key}: {value}\n')
    
    main()
