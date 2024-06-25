import re
import os
import numpy as np
import cv2
import mmengine
import math
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dataload import TextDataset, TextInstance, pil_load_img

class MSRA_TD500(TextDataset):
    def __init__(self, data_root, is_training=True, load_memory=False, transform=None):
        super().__init__(transform, is_training)
        self.data_root = data_root
        self.is_training = is_training
        self.load_memory = load_memory

        self.image_list = []
        self.anno_list = []
        img_check = re.compile('(.jpg|.JPG|.PNG|.JPEG)')
        gt_check = re.compile('.gt')

        if is_training:
            data_root = os.path.join(self.data_root, 'train')
            fnames = os.listdir(data_root)
            self.image_list = self.image_list + sorted([os.path.join(data_root, fname) for fname in fnames if img_check.findall(fname)])
            self.anno_list = self.anno_list + sorted([os.path.join(data_root, fname) for fname in fnames if gt_check.findall(fname)])
        else:
            data_root = os.path.join(data_root, 'test')
            fnames = os.listdir(data_root)
            self.image_list = self.image_list + sorted([os.path.join(data_root, fname) for fname in fnames if img_check.findall(fname)])
            self.anno_list = self.anno_list + sorted([os.path.join(data_root, fname) for fname in fnames if gt_check.findall(fname)])

    
    def parse_txt(self,gt_path):
        lines = mmengine.list_from_file(gt_path)
        bboxes = []
        for line in lines:
            line = line.encode('utf-8').decode('utf-8-sig')
            line = line.replace('\xef\xbb\xbf', '')

            gt = line.split(' ')

            w_ = np.float64(gt[4])
            h_ = np.float64(gt[5])
            x1 = np.float64(gt[2]) + w_ / 2.0
            y1 = np.float64(gt[3]) + h_ / 2.0
            theta = np.float64(gt[6]) / math.pi * 180

            bbox = cv2.boxPoints(((x1, y1), (w_, h_), theta))
            bbox = bbox.reshape(-1,2).astype(int)
            bboxes.append(TextInstance(bbox, 'c', "word"))

        return bboxes
    
    def load_img_gt(self, item):
        image_path = self.image_list[item]
        image_id = image_path.split("/")[-1]
        # Read image data
        image = pil_load_img(image_path)
        annotation_path = self.anno_list[item]
        polygons = self.parse_txt(annotation_path)

        data = dict()
        data["image"] = image
        data["polygons"] = polygons
        data["image_id"] = image_id
        data["image_path"] = image_path

        return data
    

    def __getitem__(self, item):
        data = self.load_img_gt(item)

        if self.is_training:
            return self.get_training_data(data["image"], data["polygons"],
                                          image_id=data["image_id"], image_path=data["image_path"])
        else:
            return self.get_test_data(data["image"], data["polygons"],
                                      image_id=data["image_id"], image_path=data["image_path"])

    def __len__(self):
        return len(self.image_list)

if __name__ == '__main__':
    import os
    import cv2
    from util.augmentation import Augmentation
    from util import canvas as cav
    import time

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(
        size=640, mean=means, std=stds
    )

    trainset = MSRA_TD500(
        data_root='../../data/open_datas/MSRA-TD500',
        is_training=True,
        transform=transform
    )

    for idx in range(0, len(trainset)):
        t0 = time.time()
        img, train_mask, tr_mask, distance_field, \
        direction_field, weight_matrix, ctrl_points, proposal_points, ignore_tags, _ = trainset[idx]
        img, train_mask, tr_mask, distance_field, \
        direction_field, weight_matrix, ctrl_points, proposal_points, ignore_tags\
            = map(lambda x: x.cpu().numpy(),
                  (img, train_mask, tr_mask, distance_field,
                   direction_field, weight_matrix, ctrl_points, proposal_points, ignore_tags))

        img = img.transpose(1, 2, 0)
        img = ((img * stds + means) * 255).astype(np.uint8)

        # distance_map = cav.heatmap(np.array(distance_field * 255 / np.max(distance_field), dtype=np.uint8))
        # cv2.imshow("distance_map", (distance_map > 0.7).astype(np.uint8))
        # cv2.waitKey(0)

        # direction_map = cav.heatmap(np.array(direction_field[0] * 255 / np.max(direction_field[0]), dtype=np.uint8))
        # cv2.imshow("direction_field", direction_map)
        # cv2.waitKey(0)
        # #
        # from util.vis_flux import vis_direction_field
        # vis_direction_field(direction_field)

        # weight_map = cav.heatmap(np.array(weight_matrix * 255 / np.max(weight_matrix), dtype=np.uint8))
        # cv2.imshow("weight_matrix", weight_map)
        # cv2.waitKey(0)


        boundary_point = ctrl_points[np.where(ignore_tags!=0)[0]]
        for i, bpts in enumerate(boundary_point):
            cv2.drawContours(img, [bpts.astype(np.int32)], -1, (0, 255, 0), 1)
            for j,  pp in enumerate(bpts):
                if j==0:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (255, 0, 255), -1)
                elif j==1:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (0, 255, 255), -1)
                else:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (0, 0, 255), -1)

            ppts = proposal_points[i]
            cv2.drawContours(img, [ppts.astype(np.int32)], -1, (0, 0, 255), 1)
            for j,  pp in enumerate(ppts):
                if j==0:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (255, 0, 255), -1)
                elif j==1:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (0, 255, 255), -1)
                else:
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, (0, 0, 255), -1)
            cv2.imshow('imgs', img)
            cv2.waitKey(0)