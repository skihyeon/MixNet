import os
import gc
import time
import torch
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.optim import lr_scheduler

from dataset.concat_datasets import AllDataset
from network.loss import TextLoss, knowledge_loss
from network.textnet import TextNet
from util.augmentation import Augmentation
from cfglib.config import config as cfg, update_config, print_config
from util.misc import AverageMeter
from util.misc import mkdirs, to_device
from util.visualize import visualize_network_output
from cfglib.option import BaseOptions

from tqdm.auto import tqdm

def save_model(model, epoch, lr):
    save_dir = os.path.join(cfg.save_dir, cfg.exp_name)
    if not os.path.exists(save_dir):
        mkdirs(save_dir)
    
    save_path = os.path.join(save_dir, f'MixNet_{model.backbone_name}_{epoch}.pth')
    print(f'Saving tp {save_path}')
    state_dict = {
        'lr' : lr,
        'epoch' : epoch,
        'model' : model.state_dict() if not cfg.mgpu else model.module.state_dict()
    }
    torch.save(state_dict, save_path)

def load_model(model, model_path):
    print(f"Loading from {model_path}")
    state_dict = torch.load(model_path)
    try:
        model.load_state_dict(state_dict['model'])
    except RuntimeError as e:
        # print("Missing key in state_dict, try to load with strict = False")
        model.load_state_dict(state_dict['model'], strict = False)
        # print(e)

def _parse_data(inputs):
    input_dict = {}
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

    print(f'Epoch: {epoch} : LR = {scheduler.get_lr()}')

    for i, inputs in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
        data_time.update(time.time() - end)
        train_step = 1
        input_dict = _parse_data(inputs)
        output_dict = model(input_dict)
        loss_dict = criterion(input_dict, output_dict, eps=epoch+1)
        loss = loss_dict["total_loss"]

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        losses.update(loss.item())

        batch_time.update(time.time()-end)
        end = time.time()

        # 학습과정 visualization
        if cfg.viz and (i % cfg.viz_freq == 0 and i > 0):
            visualize_network_output(output_dict, input_dict, mode='train')

        tqdm.write(f'Training Loss: {losses.avg}')

    if epoch % cfg.save_freq == 0:
        save_model(model, epoch, scheduler.get_lr())


def main():
    global lr

    trainset = AllDataset(config=cfg, custom_data_root="./data/kor", open_data_root="./data/open_datas", is_training=True)

    if os.name == 'nt':  # 윈도우일 경우
        train_loader = data.DataLoader(trainset, batch_size=cfg.batch_size,
                                       shuffle=True, num_workers=cfg.num_workers,
                                       pin_memory=True, generator=torch.Generator(cfg.device))
    else:
        train_loader = data.DataLoader(trainset, batch_size=cfg.batch_size,
                                       shuffle=True, num_workers=cfg.num_workers,
                                       pin_memory=True)
    
    model = TextNet(backbone=cfg.net, is_training=True)
    model = model.to(cfg.device)
    criterion = TextLoss()

    if cfg.mgpu:
        model = nn.DataParallel(model)
    if cfg.cuda:
        cudnn.benchmark = True
    if cfg.resume:
        load_model(model, cfg.resume)

    lr = cfg.lr

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    print('Start training MixNet.')
    for epoch in range(cfg.start_epoch, cfg.max_epoch+1):
        scheduler.step()
        train(model, train_loader, criterion, scheduler, optimizer, epoch)

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
    print_config(cfg)

    # main
    main()
