import numpy as np
from shapely.geometry import Polygon

def log_to_file(message, file_path="debug_log.txt"):
    with open(file_path, 'a') as f:
        f.write(message + '\n')

def area_of_union(det_x, det_y, gt_x, gt_y):
    p1 = Polygon(np.column_stack((det_x, det_y))).buffer(0)
    p2 = Polygon(np.column_stack((gt_x, gt_y))).buffer(0)
    if not p1.is_valid or not p2.is_valid:
        return 0.0
    return float(p1.union(p2).area)

def area_of_intersection(det_x, det_y, gt_x, gt_y):
    p1 = Polygon(np.column_stack((det_x, det_y))).buffer(0)
    p2 = Polygon(np.column_stack((gt_x, gt_y))).buffer(0)
    if not p1.is_valid or not p2.is_valid:
        return 0.0
    return float(p1.intersection(p2).area)

def iou(det_x, det_y, gt_x, gt_y):
    intersection_area = area_of_intersection(det_x, det_y, gt_x, gt_y)
    union_area = area_of_union(det_x, det_y, gt_x, gt_y)
    log_to_file(f"Intersection: {intersection_area}, Union: {union_area}")
    if union_area == 0:
        return 0.0
    return intersection_area / union_area

def evaluate_iou(pred_contours, gt_contours):
    from scipy.optimize import linear_sum_assignment
    log_to_file(f"{len(pred_contours)}/{len(gt_contours)} , {len(pred_contours)/len(gt_contours):.2f}% matched")
    iou_matrix = np.zeros((len(gt_contours), len(pred_contours)))
    for i, gt in enumerate(gt_contours):
        for j, pred in enumerate(pred_contours):
            iou_value = iou(pred[:, 0], pred[:, 1], gt[:, 0], gt[:, 1])
            log_to_file(f"iou between pred {j} and gt {i}: {iou_value}")
            iou_matrix[i, j] = iou_value
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    iou_scores = iou_matrix[row_ind, col_ind]
    return iou_scores