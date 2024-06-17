# coding:utf-8
import torch
import torch.utils.data
import torch.nn.functional as F
from dataset import AlignCollate
from torch.utils.data.dataset import Dataset
from PIL import Image
from utils import CTCLabelConverter, AttnLabelConverter
device = torch.device("cuda")

import os

class RawDataset(Dataset):

    def __init__(self, img_list):
        if os.path.isdir(img_list):
            self.img_list = [os.path.join(img_list, img) for img in os.listdir(img_list) if img.endswith(('png', 'jpg', 'jpeg', 'bmp'))]
        else:
            self.img_list = img_list
        self.nSamples = len(self.img_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = Image.open(img_path).convert('L')
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

from CTC_model import Model
from collections import OrderedDict
import os

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

import ocr_config
from typing import List

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

if __name__ == "__main__":
    model = load_ocrmodel('default.pth')
    opt = Opt()
    # ign = opt.ign_char_idx
    pred, top3_pred = img_list_prediction('../vis/extracted', model, [], opt)
    for p in pred:
        print(p[0])

