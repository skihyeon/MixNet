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

from util.IoU import get_metric
from util.misc import rescale_result
from tqdm.auto import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
import wandb

kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs])
# accelerator = Accelerator()
train_step = 0

# 시간 측정을 위한 클래스 정의
class TimeMeter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.data_time = AverageMeter()
        self.forward_time = AverageMeter() 
        self.backward_time = AverageMeter()
        self.batch_time = AverageMeter()

def init_wandb(cfg):
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    wandb.init(
        project="MixNet",
        name=cfg.exp_name,
        config=vars(cfg),
        entity = os.environ.get("WANDB_ENTITY"),
        save_code= True,
    )


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
    model.train()
    time_meter = TimeMeter()

    if accelerator:
        if accelerator.is_main_process:
            print(f'Epoch: {epoch} : LR = {scheduler.get_lr()}')
    else:
        print(f'Epoch: {epoch} : LR = {scheduler.get_lr()}')
    
    ## for loss
    cls_losses = AverageMeter()
    distance_losses = AverageMeter()
    direction_losses = AverageMeter()
    norm_losses = AverageMeter()
    angle_losses = AverageMeter()
    point_losses = AverageMeter()
    energy_losses = AverageMeter()

    # log 파일 경로 설정
    log_dir = os.path.join(cfg.save_dir, cfg.exp_name)
    log_path = os.path.join(log_dir, 'train_log.txt')
    if not os.path.exists(log_dir):
        mkdirs(log_dir)

    with open(log_path, 'a') as log_file:
        if accelerator.is_main_process:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        else:
            pbar = train_loader
        for i, inputs in enumerate(pbar):
            train_step += 1
            
            batch_start = time.time()
            
            # 데이터 로딩 시간 측정
            data_start = time.time()
            input_dict = _parse_data(inputs)
            data_time = time.time() - data_start
            time_meter.data_time.update(data_time)

            # Forward 시간 측정
            forward_start = time.time()
            output_dict = model(input_dict)
            loss_dict = criterion(input_dict, output_dict)
            loss = loss_dict["total_loss"]
            forward_time = time.time() - forward_start
            time_meter.forward_time.update(forward_time)

            # Backward 시간 측정
            backward_start = time.time()
            if cfg.accumulation > 0:
                loss = loss / cfg.accumulation  # gradient accumulation step 8

                # optimizer.zero_grad()는 gradient accumulation step 8마다 수행
                if train_step % cfg.accumulation == 1:
                    model.zero_grad()
            else:
                model.zero_grad()

            if accelerator:
                accelerator.backward(loss)
            else:
                loss.backward()
                                    
            if cfg.accumulation:
                if train_step % cfg.accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optimizer.step()
                    scheduler.step()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                optimizer.step()
                scheduler.step()
                
            backward_time = time.time() - backward_start
            time_meter.backward_time.update(backward_time)
            
            # 전체 배치 시간 측정
            batch_time = time.time() - batch_start
            time_meter.batch_time.update(batch_time)

            if cfg.accumulation:
                losses.update(loss.item() * cfg.accumulation)  # 원래 loss 값으로 업데이트
            else:
                losses.update(loss.item())
            
            ## for logging ##
            cls_loss = loss_dict["cls_loss"]
            dis_loss = loss_dict["distance_loss"]
            dir_loss = loss_dict["dir_loss"]
            norm_loss = loss_dict["norm_loss"]
            angle_loss = loss_dict["angle_loss"]
            point_loss = loss_dict["point_loss"]
            energy_loss = loss_dict["energy_loss"]

            cls_losses.update(cls_loss.item())
            distance_losses.update(dis_loss.item())
            direction_losses.update(dir_loss.item())
            norm_losses.update(norm_loss.item())
            angle_losses.update(angle_loss.item())
            point_losses.update(point_loss.item())
            energy_losses.update(energy_loss.item())

            # 학습과정 visualization
            if cfg.viz and (i % cfg.viz_freq == 0 and i > 0):
                visualize_network_output(output_dict, input_dict, mode='train')

            max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            # 시간 측정 결과 출력
            time_info = {
                # 'Data Time': f'{time_meter.data_time.avg:.2f}s',
                'Forward': f'{time_meter.forward_time.avg:.2f}s',
                'Back': f'{time_meter.backward_time.avg:.2f}s',
                # 'Batch Time': f'{time_meter.batch_time.avg:.2f}s'
            }
            
            if accelerator.is_main_process:
                pbar.set_postfix({'Training Loss': f'{losses.avg:.2f}, time: {time_info}', 'Max Memory': f'{max_memory:.2f} MB'})
                # pbar.set_postfix(time_info)

            # 로그 파일에 학습 정보 저장
            if accelerator.is_main_process:
                log_file.write(f'Epoch: {epoch}, Step: {i}, Loss: {losses.avg:.2f}, Max Memory: {max_memory:.2f} MB\n')
            
            if accelerator.is_main_process and cfg.wandb:
                log_dict = {
                    "epoch": epoch,
                    "train_loss": losses.avg,
                    "lr": scheduler.get_last_lr()[0],
                }
                if cfg.onlybackbone == False:
                    log_dict.update({   
                        "losses/class_loss" : cls_losses.avg,
                        "losses/distance_loss": distance_losses.avg,
                        "losses/direction_loss": direction_losses.avg,
                        "losses/norm_loss": norm_losses.avg,
                        "losses/angle_loss": angle_losses.avg,
                        "losses/point_loss": point_losses.avg,
                        "losses/energy_loss": energy_losses.avg
                    })
                wandb.log(log_dict)

            # 메모리 정리
            del input_dict, output_dict, loss_dict, loss
            torch.cuda.empty_cache()
            gc.collect()
            
    if epoch % cfg.save_freq == 0:
        save_model(model, epoch, scheduler.get_lr())


def inference(model, test_loader, criterion):
    model.eval()
    torch.cuda.reset_max_memory_allocated()
    if accelerator.is_main_process:
        pbar = tqdm(test_loader, desc="Valid")
    else:
        pbar = test_loader
    total_hit_rate = 0
    total_precision = 0
    total_recall = 0
    total_hmean = 0
    num_images = 0

    for i, (image, meta) in enumerate(pbar):
        input_dict = dict()
        H,W = meta['Height'][0].item(), meta['Width'][0].item()

        img_show = image[0].permute(1,2,0).cpu().numpy()
        image= image.to(cfg.device, non_blocking=True)
        input_dict['img'] = image

        with torch.no_grad():
            output_dict = model(input_dict, test_speed=True)
        torch.cuda.synchronize()

        gt_contours = []
        for annot, n_annot in zip(meta['annotation'][0], meta['n_annotation'][0]):
            if n_annot.item() > 0:
                gt_contours.append(annot[:n_annot].int().cpu().numpy())
        contours = output_dict["py_preds"][-1].int().cpu().numpy()
        # _, contours = rescale_result(img_show, contours, H, W)
        hit_rate = len(contours)/len(gt_contours) if gt_contours else 0
        precision, recall, hmean = get_metric(gt_contours, contours)

        total_hit_rate += hit_rate
        total_precision += precision
        total_recall += recall
        total_hmean += hmean
        num_images += 1

        max_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        if accelerator.is_main_process:
            pbar.set_postfix({'hr': f'{hit_rate}', "f1": f'{precision:.2f}/{recall:.2f}/{hmean:.2f}', 'Max Memory': f'{max_memory:.2f} MB'})

    avg_hit_rate = total_hit_rate / num_images
    avg_precision = total_precision / num_images
    avg_recall = total_recall / num_images
    avg_hmean = total_hmean / num_images
    
    if accelerator.is_main_process and cfg.wandb:
        wandb.log({
            "hit_rate": avg_hit_rate,
            "f1": {
                "precision": avg_precision,
                "recall": avg_recall,
                "hmean": avg_hmean
            }
        })


def main():
    global lr
    if accelerator.is_main_process and cfg.wandb:
        init_wandb(cfg)
    
    trainset = AllDataset_mid(config=cfg, is_training=True) if cfg.mid else AllDataset(config=cfg, is_training=True)
    train_loader = data.DataLoader(trainset, batch_size=cfg.batch_size,
                                   shuffle=True, num_workers=cfg.num_workers,
                                   pin_memory=True, generator=torch.Generator(device=cfg.device),persistent_workers=True,)  
    
    testset = AllDataset(config=cfg, is_training=False)
    test_loader = data.DataLoader(testset, batch_size=1,
                                  shuffle=False, num_workers=0)  

    model = TextNet(backbone=cfg.net, is_training=True)

    if cfg.resume:
        load_model(model, cfg.resume)
        
    criterion = TextLoss()

    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))


    optimizer = torch.optim.AdamW(filtered_parameters, lr=cfg.lr)
    total_iterations = len(train_loader) * 20

    # scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=0.9)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=total_iterations)

    if accelerator:
        model, optimizer, train_loader, criterion, scheduler = accelerator.prepare(model, optimizer, train_loader, criterion, scheduler)
    if cfg.cuda:
        cudnn.benchmark = True


    if cfg.resume :
        try:
            start_iter = int(cfg.resume.split('_')[-1].split('.')[0]) + 1
            print(f'continue to train, start_iter: {start_iter}')
            cfg.start_epoch = start_iter 
        except:
            pass
        

    for epoch in range(cfg.start_epoch, cfg.max_epoch+1):
        # if epoch == 0 :
        #     model.eval()
        #     inference(model, test_loader, criterion)
        model.train()  # 훈련 모드로 설정
        train(model, train_loader, criterion, scheduler, optimizer, epoch)
        # if epoch > 0:
        model.eval()  # 평가 모드로 설정
        inference(model, test_loader, criterion)
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    np.random.seed(2022)
    torch.manual_seed(2022)
    # parse arguments
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    
    # if accelerator:
    #     if accelerator.is_main_process:
    #         print_config(cfg)
    # else:
    #     print_config(cfg)

    log_dir = os.path.join(cfg.save_dir, cfg.exp_name)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'config.txt'), 'w') as f:
        for key, value in vars(cfg).items():
            f.write(f'{key}: {value}\n')
    
    main()
