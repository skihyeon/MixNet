import os
import time
import cv2
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from dataset.concat_datasets import AllDataset, AllDataset_mid
from network.textnet import TextNet
from cfglib.config import config as cfg, update_config

from cfglib.option import BaseOptions
from util.visualize import visualize_detection, visualize_gt
from util.misc import mkdirs,rescale_result
from util.augmentation import BaseTransform

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)


def osmkdir(out_dir):
    import shutil
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

def write_to_file(contours, file_path):
    with open(file_path, 'w') as f:
        for cont in contours:
            cont = np.stack([cont[:, 0], cont[:, 1]], 1)
            if cv2.contourArea(cont) <= 0:
                continue
            cont = cont.flatten().astype(str).tolist()
            cont = ','.join(cont)
            f.write(cont + '\n')

def inference(model, test_loader, output_dir):
    model.eval()
    total_time = 0.
    osmkdir(output_dir)
    device = cfg.device
    iou_scores = []
    hit_rates = []

    for i, (image, meta) in enumerate(test_loader):
        input_dict = dict()
        idx = 0
        H,W = meta['Height'][idx].item(), meta['Width'][idx].item()

        image= image.to(device, non_blocking=True)
        input_dict['img'] = image

        start = time.time()
        with torch.no_grad():
            output_dict = model(input_dict)

        torch.cuda.synchronize()
        end = time.time()
        if i>0:
            total_time += end-start
            fps = (i+1)/total_time
        else:
            fps = 0.0
        
        img_show = image[idx].permute(1,2,0).cpu().numpy()
        img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)

        gt_contour = []
        label_tag = meta['label_tag'][idx].int().cpu().numpy()
        for annot, n_annot in zip(meta['annotation'][idx], meta['n_annotation'][idx]):
            if n_annot.item() > 0:
                gt_contour.append(annot[:n_annot].int().cpu().numpy())

        gt_vis = visualize_gt(img_show, gt_contour, label_tag)
        show_boundary, heat_map = visualize_detection(img_show, output_dict, meta=meta)
        heat_map = cv2.resize(heat_map, (W, H))
        gt_vis = cv2.resize(gt_vis, (W, H))
        show_boundary = cv2.resize(show_boundary, (W, H))
        im_vis = np.concatenate([heat_map, gt_vis, show_boundary], axis=1)
        # im_vis = np.concatenate([gt_vis, show_boundary], axis=1)

        _, buffer = cv2.imencode('.jpg', im_vis)    # 한글 경로 깨지는 경우 대비
        path = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name), meta['image_id'][idx].split(".")[0]+".jpg")
        with open(path, 'wb') as f:
            f.write(buffer)
        cv2.imwrite(path, im_vis)

        contours = output_dict["py_preds"][-1].int().cpu().numpy()
        img_show, contours = rescale_result(img_show, contours, H, W)

        torch.cuda.empty_cache()

        tmp = np.array(contours).astype(np.int32)
        if tmp.ndim == 2:
            tmp = np.expand_dims(tmp, axis=2)

        fname = meta['image_id'][idx].replace('jpg', 'txt').replace('JPG', 'txt').replace('PNG', 'txt').replace('png', 'txt').replace('jpeg', 'txt')
        write_to_file(contours, os.path.join(output_dir, fname))

        hit_rate = len(contours)/len(gt_contour) if gt_contour else 0
        hit_rates.append(hit_rate)

        print('Index {} / {},  images: {}. / Hit!: {:.2f}%'.format(i + 1, len(test_loader), meta['image_id'][idx], hit_rate*100), end = '\r', flush = True)

    print("평균 적중률: {:.2f}".format(np.mean(hit_rates)), end = '\r', flush = True)



def main(vis_dir_path):

    dataset_params = {
        "config": cfg,
        "is_training" : False
    }

    testset = AllDataset_mid(**dataset_params) if cfg.mid else AllDataset(**dataset_params)

    test_loader = data.DataLoader(testset, batch_size=cfg.batch_size,
                                       shuffle=True, num_workers=cfg.num_workers,
                                       pin_memory=True, generator=torch.Generator(device=cfg.device))
    
    torch.cuda.set_device(cfg.device)
    cudnn.benchmark = False
    test_loader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg.num_workers)
    model = TextNet(is_training=False, backbone=cfg.net)
    model_path = os.path.join(cfg.save_dir, cfg.exp_name,
                              'MixNet_{}_{}.pth'.format(model.backbone_name, cfg.checkepoch))
    model.load_model(model_path)
    model.to(torch.device(cfg.device))
    model.eval()
    with torch.no_grad():
        print('Start testing MixNet.')
        output_dir = os.path.join(cfg.output_dir, cfg.exp_name)
        inference(model, test_loader, output_dir)

    print("{} eval finished.".format(cfg.exp_name))


if __name__ == "__main__":
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)

    vis_dir = os.path.join(cfg.vis_dir, '{}_test'.format(cfg.exp_name))
    if not os.path.exists(vis_dir):
        mkdirs(vis_dir)
    # main
    main(vis_dir)