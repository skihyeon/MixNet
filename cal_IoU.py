import cupy as cp
from shapely.geometry import Polygon

def area_of_union(det_x, det_y, gt_x, gt_y):
    p1 = Polygon(cp.asnumpy(cp.column_stack((det_x, det_y)))).buffer(0)
    p2 = Polygon(cp.asnumpy(cp.column_stack((gt_x, gt_y)))).buffer(0)
    return float(p1.union(p2).area)

def area_of_intersection(det_x, det_y, gt_x, gt_y):
    p1 = Polygon(cp.asnumpy(cp.column_stack((det_x, det_y)))).buffer(0)
    p2 = Polygon(cp.asnumpy(cp.column_stack((gt_x, gt_y)))).buffer(0)
    return float(p1.intersection(p2).area)

def iou(det_x, det_y, gt_x, gt_y):
    intersection_area = area_of_intersection(det_x, det_y, gt_x, gt_y)
    union_area = area_of_union(det_x, det_y, gt_x, gt_y)
    if union_area == 0:
        return 0.0
    return intersection_area / union_area

def evaluate_iou(pred_contours, gt_contours):
    from scipy.optimize import linear_sum_assignment
    print(f"{len(pred_contours)}/{len(gt_contours)} , {len(pred_contours)/len(gt_contours):.2f}% matched")
    iou_matrix = cp.zeros((len(gt_contours), len(pred_contours)))
    for i, gt in enumerate(gt_contours):
        for j, pred in enumerate(pred_contours):
            iou_matrix[i, j] = iou(cp.asarray(pred[:, 0]), cp.asarray(pred[:, 1]), cp.asarray(gt[:, 0]), cp.asarray(gt[:, 1]))
    row_ind, col_ind = linear_sum_assignment(cp.asnumpy(-iou_matrix))
    iou_scores = iou_matrix[row_ind, col_ind]
    return cp.asnumpy(iou_scores)