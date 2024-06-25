import os
import cv2
import numpy as np
import torch
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'extract_model'))

from dataset.dataload import pil_load_img, TextDataset
from util.augmentation import BaseTransform
from network.textnet import TextNet
from util.visualize import visualize_detection
from util.misc import rescale_result
from cfglib.config import config as cfg, update_config

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
    

def find_text(model, image_path, device):
    torch.cuda.set_device(0)
    dataset = myDataset(image_path,transform=BaseTransform(size=(1280, 1920), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)))
    data = dataset[0]
    image, meta = data

    input_dict = dict()
    H, W = meta['Height'], meta['Width']

    input_dict['img'] = torch.tensor(image[np.newaxis, :]).to(device)
    with torch.no_grad():
        output_dict = model(input_dict)

    print(f'{image_path} processing...')
    img_show = image.transpose(1, 2, 0)
    img_show = ((img_show * (0.229, 0.224, 0.225) + (0.485, 0.456, 0.406)) * 255).astype(np.uint8)

    show_boundary, heat_map = visualize_detection(img_show, output_dict, meta=meta, infer=True)
    im_vis = np.concatenate([show_boundary], axis=1)

    _, im_buf_arr = cv2.imencode('.jpg', im_vis)    # 한글경로 인식 문제 해결
    path = os.path.join(os.path.basename(image_path).split(".")[0] + '_infered.jpg')
    with open(path, 'wb') as f:
        f.write(im_buf_arr)
    
    contours = output_dict["py_preds"][-1].int().cpu().numpy()
    img_show, contours = rescale_result(img_show, contours, H, W)
    # im_vis = cv2.resize(im_vis, (W, H))

    bboxes = []

    for contour in contours:
        contour = np.stack([contour[:, 0], contour[:, 1]], 1)
        if cv2.contourArea(contour) <= 0:
                continue
        contour = contour.flatten().astype(str).tolist()
        contour = ','.join(contour)
        bboxes.append(contour)

    return im_vis, bboxes



def extract_polygons(image_path, bbox_coords):
    output_dir = '../vis/extracted'
    cropped_results = []

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    for idx, line in enumerate(bbox_coords):
        coords = [list(map(int, point.split(','))) for point in line.strip().split()]
        coords = [max(0, coord) for coord in coords[0]]
        
        pts = np.array(coords, np.int32)
        pts = pts.reshape((-1, 1, 2))

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        result = cv2.bitwise_and(image, image, mask=mask)

        x, y, w, h = cv2.boundingRect(pts)
        cropped_result = result[y:y+h, x:x+w]

        if cropped_result.size == 0:
            print(f"Warning: cropped_result is empty for polygon {idx}")
            continue

        # output_path = os.path.join(output_dir, f'extracted_polygon_{idx:03d}.png')
        # output_path = os.path.abspath(output_path)  # 절대 경로로 변환
        # cv2.imwrite(output_path, cropped_result)
        
        cropped_results.append(cropped_result)

    return cropped_results

from extract_model.utils import CTCLabelConverter, AttnLabelConverter
from extract_model.dataset import AlignCollate
from extract_model.CTC_model import Model
from collections import OrderedDict
import extract_model.ocr_config as ocr_config
from torch.utils.data.dataset import Dataset
from PIL import Image
import torch.nn.functional as F


class Opt:
    pool = ocr_config.number + ocr_config.symbol_s + ocr_config.pool_EN + ocr_config.pool_ko_2350
    character: str = pool
    num_class: int = len(pool)+1
    language: str = "kor"
    ign_char_idx: [] # exclude from ocr
    canvas_size: str = "medium"
    response_type: str = "basic"
    orientation: bool = False
    # data processing
    workers: int = 4
    batch_size: int = 100 # input batch size
    batch_max_length: int = 30 # maximum-label-length
    imgH: int = 32
    imgW: int = 320
    rgb: bool = False # use rgb input
    sensitive: bool = False # for sensitive character mode
    PAD: bool = True # whether to keep ratio then pad for image resize
    # model architecture
    Transformation: str = "TPS" # None|TPS
    FeatureExtraction: str = "ResNet" # VGG|RCNN|ResNet
    SequenceModeling: str = "BiLSTM" # None|BiLSTM
    Prediction: str = "CTC" # CTC|Attn
    num_fiducial: int = 20 # number of fiducial points of TPS-STN
    input_channel: int = 1 # the number of input channel of Feature extractor
    output_channel: int = 512 # the number of output channel of Feature extractor
    hidden_size: int = 256 # the size of the LSTM hidden state
    
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.__annotations__:
                setattr(self, k, v)

def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict

def load_ocrmodel(ocr_model):
    opt = Opt()

    model = Model(opt)
    model.to(device)
    # ocr_model = os.path.join(ocr_model, "saved_models", ocr_model)
    print(f"ocr model[{ocr_model}] loading ...") 
    model.load_state_dict(copyStateDict(torch.load(ocr_model, map_location=device)))
    return model


class RawDataset(Dataset):

    def __init__(self, img_list):
        self.img_list = img_list
        self.nSamples = len(self.img_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        img_array = self.img_list[index]
        img = Image.fromarray(img_array).convert('L')
        return (img, str(index))

def img_list_prediction(img_list, model, ign, opt): # return prediction / [[pred, confidence_score], [p, cs], ...]
    decode_mode = "top3" # max or top3
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    # prepare data
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)

    demo_data = RawDataset(img_list)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size, # batch_size: 100
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    prediction = []
    validation = []
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device) # [81,81]
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device) # [2, 31]
            if 'CTC' in opt.Prediction:
                preds = model(image, text_for_pred)
                
                if decode_mode == "top3":                 
                    preds_prob = F.softmax(preds, dim=2)
                    topk_prob_values, topk_prob_indexs = torch.topk(preds_prob, k=3, dim=2)
                    topk_values, topk_indexs = torch.topk(preds, k=3, dim=2)
                    ign = torch.tensor(ign, dtype=topk_indexs.dtype, device=topk_indexs.device)
                    top3_list, preds_list = converter.top3_decode(topk_indexs, topk_prob_values, ign)
                elif decode_mode == "max":
                    # Select max probabilty (greedy decoding) then decode index to character
                    preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                    _, preds_index = preds.max(2)
                    ign = torch.tensor(ign, dtype=preds_index.dtype, device=preds_index.device)
                    for i in ign: # ign에 해당하는 값 제거
                        preds_index = torch.where(preds_index == i, torch.tensor(0, dtype=preds_index.dtype, device=preds_index.device), preds_index)
                    preds_list = converter.decode(preds_index, preds_size)
            else:
                preds = model(image, text_for_pred, is_train=False)
                # select max probabilty (greedy decoding) then decode index to character
                preds[:,:,ign] = -100 #ign
                _, preds_index = preds.max(2)
                preds_str = opt.converter.decode(preds_index, length_for_pred)

            torch.cuda.empty_cache()
            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_list, preds_max_prob):
                if 'Attn' in opt.Prediction:
                    pred_EOS = pred.find('[s]')
                    pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                    pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob)
                confidence_score = 0 if len(pred_max_prob)==0 else pred_max_prob.cumprod(dim=0)[-1]
                prediction.append([pred, float(confidence_score)])
            validation.extend(top3_list)
    return prediction, validation


if __name__ == '__main__':
    device = torch.device('cuda')
    cfg.device = torch.device('cuda')
    cfg.net = 'FSNet_H_M'
    cfg.resume = False
    cfg.num_points = 100
    cfg.mid = True
    cfg.test_size = (1280, 1920)
    cfg.dis_threshold = 0.35
    cfg.cls_threshold = 0.875
    cfg.vis_dir = './'
    cfg.exp_name = './'

    model = TextNet(is_training=False, backbone='FSNet_H_M')
    model_path = "C:/Users/ys/Desktop/sgh/MixNet/MixNet/model/only_kor_H_M_mid_extended_later/MixNet_FSNet_H_M_240.pth"

    model.load_model(model_path)
    model.to(device)
    model.eval()
    image_path = '0001.jpg'

    im, bboxes = find_text(model, image_path, device)
    cropped_images = extract_polygons(image_path, bboxes)
    # print(f"Number of cropped images: {len(cropped_images)}")
    # for idx, img in enumerate(cropped_images):
    #     print(f"Type of cropped image {idx}: {type(img)}")
    # cv2.imshow(cropped_images[0])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    opt = Opt()
    ocr_model = load_ocrmodel('default.pth')
    pred, _ = img_list_prediction(cropped_images, ocr_model,[], opt)
    # print(pred)
    with open('result.txt', 'w', encoding='utf-8') as f:
        for p in pred:
            f.write(p[0] + '\n')

    