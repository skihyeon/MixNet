import numpy as np
from shapely.geometry import Polygon

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


def get_metric(gt_contours, pred_contours):
    ta = len(pred_contours)
    tb = len(gt_contours)
    matched_preds = [False] * ta
    matched_gts = [False] * tb
    tp = 0

    for gt_idx, gt in enumerate(gt_contours):
        gt = np.array(gt, dtype=np.int32).reshape(-1,2)

        for pred_idx, pred in enumerate(pred_contours):
            if matched_preds[pred_idx]:
                continue

            pred = np.array(pred, dtype=np.int32).reshape(-1, 2)
            iou_value = iou(pred[:, 0], pred[:, 1], gt[:, 0], gt[:, 1])

            if iou_value >= 0.5:
                matched_preds[pred_idx] = True
                matched_gts[gt_idx] = True
                tp += 1

    fn = matched_gts.count(False)
    fp = matched_preds.count(False)

    precision = float(tp) / (tp+fp) if tp+fp != 0 else 0
    recall = float(tp) / (tp+fn) if tp+fn !=0 else 0
    
    hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)
    
    return precision, recall, hmean


