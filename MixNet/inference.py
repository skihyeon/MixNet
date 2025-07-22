import os
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import multiprocessing

from dataset import myDataset, myDataset_mid
from dataset.dataload import pil_load_img, TextDataset
from network.textnet import TextNet
from cfglib.config import config as cfg, update_config
from cfglib.option import BaseOptions
from util.augmentation import BaseTransform
from util.visualize import visualize_detection
from util.misc import mkdirs, rescale_result
from tqdm.auto import tqdm
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

    torch.cuda.set_device(torch.cuda.current_device())
    start_time = time.time()
    dataset = myDataset(image_path=image_path,transform=BaseTransform(size=cfg.test_size, mean=cfg.means, std=cfg.stds))
    data = dataset[0]
    image, meta = data
    print(f'데이터 전처리 시간: {time.time() - start_time:.3f}초')
    input_dict = dict()
    H, W = meta['Height'], meta['Width']


    start_time = time.time()
    input_dict['img'] = torch.tensor(image[np.newaxis, :]).to(cfg.device)
    with torch.no_grad():
        output_dict = model(input_dict, test_speed=True)
    print(f'추론 시간: {time.time() - start_time:.3f}초')
    # print(f'{image_path} processing...', end='\r', flush=True)
    img_show = image.transpose(1, 2, 0)
    img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)
    from util.visualize import visualize_detection_rect
    show_boundary, heat_map = visualize_detection_rect(img_show, output_dict, meta=meta, infer=True)

    im_vis = np.concatenate([heat_map, show_boundary], axis=1)
    # im_vis = np.concatenate([show_boundary], axis=1)

    _, im_buf_arr = cv2.imencode('.jpg', im_vis)    # 한글경로 인식 문제 해결
    base_name = os.path.splitext(os.path.basename(image_path))[0]  # 파일 이름에 '.'이 여러개 들어있을 경우 마지막 확장자만 제거하도록 수정
    path = os.path.join(output_dir, base_name + '_infered.jpg')
    with open(path, 'wb') as f:
        f.write(im_buf_arr)

    contours = output_dict["py_preds"][-1].int().cpu().numpy()
    img_show, contours = rescale_result(img_show, contours, H, W)

    tmp = np.array(contours).astype(np.int32)
    if tmp.ndim == 2:
        tmp = np.expand_dims(tmp, axis=2)
    
    txt_folder = os.path.join(output_dir, 'text/')
    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)
    base_fname = os.path.splitext(path.split('/')[-1])[0].replace('_infered', '')
    fname = os.path.join(txt_folder, base_fname + '.txt')

    write_to_file(contours, fname)
    

def inference_folder(model, folder_path, output_dir):
    for filename in tqdm(os.listdir(folder_path)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')):
            image_path = os.path.join(folder_path, filename)
            inference(model, image_path, output_dir)


import time

def main(image_path):
    cudnn.benchmark = False
    model = TextNet(is_training=False, backbone=cfg.net)
    model_path = os.path.join(cfg.save_dir, cfg.exp_name,
                              'MixNet_{}_{}.pth'.format(model.backbone_name, cfg.checkepoch))
    model.load_model(model_path)
    model.to(cfg.device)
    model.eval()
    with torch.no_grad():
        print('Start infer MixNet.')
        start_time = time.time()  # 시작 시간 기록
        if os.path.isdir(image_path): ## 폴더 입력 시 해당 폴더 내 모든 이미지 처리
            output_dir = image_path + '/' + cfg.exp_name + '_result/'
            mkdirs(output_dir)
            inference_folder(model, image_path, output_dir)
        else: ## 이미지 파일 입력 시 해당 이미지 처리
            output_dir = os.path.dirname(image_path)
            inference(model, image_path, output_dir)
        end_time = time.time()  # 종료 시간 기록
        print(f'총 inference 수행 시간: {end_time - start_time:.2f}초')


if __name__ == "__main__":
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    image_path = cfg.infer_path
    main(image_path)