import os
import time
import cv2
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data

from dataset import TotalText, myDataset
from network.textnet import TextNet
from cfglib.config import config as cfg, update_config

from cfglib.option import BaseOptions
from util.augmentation import BaseTransform, BaseTransformNresize
from util.visualize import visualize_detection, visualize_gt
from util.misc import mkdirs,rescale_result
import multiprocessing
from dataset.dataload import pil_load_img, TextDataset, TextInstance
multiprocessing.set_start_method("spawn", force=True)


    
def write_to_file(contours, file_path):
    with open(file_path, 'w') as f:
        for cont in contours:
            cont = np.stack([cont[:, 0], cont[:, 1]], 1)
            if cv2.contourArea(cont) <= 0:
                continue
            cont = cont.flatten().astype(str).tolist()
            cont = ','.join(cont)
            f.write(cont + '\n')
            

class myDataset(TextDataset):
    def __init__(self, image_path, transform=None):
        super().__init__(transform)
        self.image_path = image_path
        self.image_root = os.path.dirname(image_path)
        
    def load_img(self, item):
        image_path = self.image_path
        image_id = image_path.split("/")[-1]

        image = pil_load_img(image_path)
        try:
            assert image.shape[-1] == 3
        except:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image)

        data = dict()
        data["image"] = image
        data["image_id"] = image_id
        data["image_path"] = image_path

        return data

    def __getitem__(self, item):
        data = self.load_img(item)
        return self.get_test_data(data["image"], [], image_id=data["image_id"], image_path=data["image_path"])

    def __len__(self):
        return 1

def inference(model, image_path, output_dir):
    device = torch.device("cuda")

    dataset = myDataset(image_path=image_path,transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds))
    data = dataset[0]
    image, meta = data

    input_dict = dict()
    H, W = meta['Height'], meta['Width']

    input_dict['img'] = torch.tensor(image[np.newaxis, :]).to(device)
    with torch.no_grad():
        output_dict = model(input_dict)
    
    print(f'{image_path} processing...')
    img_show = image.transpose(1, 2, 0)
    img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)

    show_boundary, heat_map = visualize_detection(img_show, output_dict, meta=meta)

    # im_vis = np.concatenate([heat_map, show_boundary], axis=1)
    im_vis = np.concatenate([show_boundary], axis=1)

    _, im_buf_arr = cv2.imencode('.jpg', im_vis)    # 한글경로 인식 문제 해결
    path = os.path.join(output_dir, os.path.basename(image_path).split(".")[0] + '_infered.jpg')
    with open(path, 'wb') as f:
        f.write(im_buf_arr)

    contours = output_dict["py_preds"][-1].int().cpu().numpy()
    img_show, contours = rescale_result(img_show, contours, H, W)

    tmp = np.array(contours).astype(np.int32)
    if tmp.ndim == 2:
        tmp = np.expand_dims(tmp, axis=2)
    
    fname = path.replace('jpg', 'txt')

    write_to_file(contours, fname)
    

def inference_folder(model, folder_path, output_dir):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')):
            image_path = os.path.join(folder_path, filename)
            inference(model, image_path, output_dir)



# def main(image_path):
#     cudnn.benchmark = False
#     model = TextNet(is_training=False, backbone=cfg.net)
#     model_path = os.path.join(cfg.save_dir, cfg.exp_name,
#                               'MixNet_{}_{}.pth'.format(model.backbone_name, cfg.checkepoch))
#     model.load_model(model_path)
#     model.to(torch.device("cuda"))
#     model.eval()
#     with torch.no_grad():
#         print('Start infer MixNet.')
#         output_dir = os.path.dirname(image_path) 
#         inference(model, image_path, output_dir)


def main(image_path):
    cudnn.benchmark = False
    model = TextNet(is_training=False, backbone=cfg.net)
    model_path = os.path.join(cfg.save_dir, cfg.exp_name,
                              'MixNet_{}_{}.pth'.format(model.backbone_name, cfg.checkepoch))
    model.load_model(model_path)
    model.to(torch.device("cuda"))
    model.eval()
    with torch.no_grad():
        print('Start infer MixNet.')
        if os.path.isdir(image_path):
            output_dir = image_path + + '/' + cfg.exp_name + '_result/'
            mkdirs(output_dir)
            inference_folder(model, image_path, output_dir)
        else:
            output_dir = os.path.dirname(image_path)
            inference(model, image_path, output_dir)    


if __name__ == "__main__":
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    image_path = 'C:/Users/ys/Desktop/sgh/MixNet/infer_test_datas/images'
    main(image_path)