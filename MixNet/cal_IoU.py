import os
import numpy as np
import cv2
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
import argparse
from multiprocessing import Pool
# RuntimeWarning 무시
warnings.filterwarnings("ignore", category=RuntimeWarning)


def read_dir(root):
    file_path_list = []
    for file_path, dirs, files in os.walk(root):
        for file in files:
            file_path_list.append(os.path.join(file_path, file).replace('\\', '/'))
    file_path_list.sort()
    return file_path_list

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file_object:
        file_content = file_object.read()
    return file_content

def write_file(file_path, file_content):
    if '/' in file_path:
        father_dir = '/'.join(file_path.split('/')[:-1])
        if not os.path.exists(father_dir):
            os.makedirs(father_dir)
    with open(file_path, 'w') as file_object:
        file_object.write(file_content)

def write_file_not_cover(file_path, file_content):
    father_dir = '/'.join(file_path.split('/')[:-1])
    if not os.path.exists(father_dir):
        os.makedirs(father_dir)
    with open(file_path, 'w') as file_object:
        file_object.write(file_content)

def get_pred(path):
    lines = read_file(path).split('\n')
    bboxes = []
    for line in lines:
        if line == '':
            continue
        bbox = line.split(',')
        x_min, y_min = int(np.min(np.array(bbox[0::2], dtype=np.int32))), int(np.min(np.array(bbox[1::2], dtype=np.int32)))
        x_max, y_max = int(np.max(np.array(bbox[0::2], dtype=np.int32))), int(np.max(np.array(bbox[1::2], dtype=np.int32)))
        if len(bbox) % 2 == 1:
            print(path)
        # bbox = [int(x) for x in bbox]
        bbox = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)] 
        bboxes.append(bbox)
    return bboxes

def get_gt(path):
    lines = read_file(path).split('\n')
    bboxes = []
    tags = []
    for line in lines:
        if line == '':
            continue
        gt = line.split(',')

        bbox = [int(coord) for coord in gt[:8]]
        bbox.append(bbox[0])
        bbox.append(bbox[1])
        
        # tag = int(gt[8]) if len(gt) > 8 else 0  # 기본값을 0으로 설정
        tag = 0

        bboxes.append(bbox)
        tags.append(tag)
    return np.array(bboxes), tags

def area_of_union(det_x, det_y, gt_x, gt_y):
    p1 = Polygon(np.column_stack((det_x, det_y)))
    p2 = Polygon(np.column_stack((gt_x, gt_y)))
    
    if not p1.is_valid:
        p1 = p1.buffer(0)
    if not p2.is_valid:  
        p2 = p2.buffer(0)
        
    if not p1.is_valid or not p2.is_valid:
        return 0.0
        
    return float(p1.union(p2).area)

def area_of_intersection(det_x, det_y, gt_x, gt_y):
    p1 = Polygon(np.column_stack((det_x, det_y)))
    p2 = Polygon(np.column_stack((gt_x, gt_y)))
    
    if not p1.is_valid:
        p1 = p1.buffer(0)
    if not p2.is_valid:
        p2 = p2.buffer(0)
    
    if not p1.is_valid or not p2.is_valid:
        return 0.0
    
    pInt = p1.intersection(p2)
    if pInt.is_empty:
        return 0.0
    elif pInt.area >= min(p1.area, p2.area):
        return 0.1234
    return float(pInt.area)

def iou(det_x, det_y, gt_x, gt_y):
    intersection_area = area_of_intersection(det_x, det_y, gt_x, gt_y)
    union_area = area_of_union(det_x, det_y, gt_x, gt_y)
    if union_area == 0:
        return 0.0
    if intersection_area == 0.1234:
        return 1.1234
    return intersection_area / union_area
                
def process_file(args):
    pred_path, gt_root, th = args
    preds = get_pred(pred_path)
    gt_path = os.path.join(gt_root, pred_path.split('/')[-1].split('.')[0] + '.txt')
    gts, tags = get_gt(gt_path)

    ta = len(preds)
    tb = len(gts)
    matched_preds = [False] * len(preds)
    matched_gts = [False] * len(gts)
    tp = 0
    
    for gt_idx, (gt, tag) in enumerate(zip(gts, tags)):
        gt = np.array(gt, dtype=np.int32).reshape(-1, 2)
        
        for pred_idx, pred in enumerate(preds):
            if matched_preds[pred_idx]:
                continue
                
            pred = np.array(pred, dtype=np.int32).reshape(-1, 2)
            iou_value = iou(pred[:, 0], pred[:, 1], gt[:, 0], gt[:, 1])

            if iou_value >= th:
                matched_preds[pred_idx] = True
                matched_gts[gt_idx] = True
                tp += 1

        #     if iou_value > max_iou:
        #         max_iou = iou_value
        #         max_iou_idx = pred_idx
        
        # if max_iou >= th:
        #     matched_preds[max_iou_idx] = True
        #     matched_gts[gt_idx] = True
        #     tp += 1
    
    fn = matched_gts.count(False)
    fp = matched_preds.count(False)
    
    return tp, fp, fn, ta, tb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('exp_name', type=str)
    parser.add_argument('--pred_root', type=str)
    parser.add_argument('--gt_root', type=str)
    args = parser.parse_args()

    pred_root = args.pred_root
    gt_root = args.gt_root
    th = 0.5
    
    pred_list = read_dir(pred_root)
    
    with Pool() as pool:
        results = list(tqdm(pool.imap(process_file, [(pred_path, gt_root, th) for pred_path in pred_list]), total=len(pred_list)))
    
    tp, fp, fn, ta, tb = map(sum, zip(*results))
    
    print(f'tp: {tp}, fp: {fp}, fn: {fn}, total pred box: {ta}, total gt box: {tb}')
    precision = float(tp) / (tp+fp)
    recall = float(tp) / (tp+fn)
    
    hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
    
    print('p: %.4f, r: %.4f, f: %.4f' % (precision, recall, hmean))