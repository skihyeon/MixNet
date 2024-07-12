import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from dataset.dataload import pil_load_img, TextDataset, TextInstance
import json
import cv2

def get_absolute_path(p):
    if p.startswith('~'):
        p = os.path.expanduser(p)
    return os.path.abspath(p)

def read_lines(p):
    p = get_absolute_path(p)
    f = open(p,'rU')
    return f.readlines()


class myDataset(TextDataset):
    def __init__(self, data_root, is_training=True, load_memory=False, transform=None):
        super().__init__(transform, is_training)
        self.data_root = data_root
        self.is_training = is_training
        self.load_memory = load_memory

        self.image_root = os.path.join(data_root, 'Train' if is_training else 'Test', 'images')
        self.annotation_root = os.path.join(data_root, 'Train' if is_training else 'Test', 'gt')
        self.image_list = os.listdir(self.image_root)
        self.annotation_list = [os.path.join(self.annotation_root, img + '.json') for img in self.image_list]

        if self.load_memory:
            self.datas = list()
            for item in range(len(self.image_list)):
                self.datas.append(self.load_img_gt(item))

    def parse_json(self, gt_path):
        with open(gt_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        polygons = []
        if 'images' in data and data['images']:
            for field in data['images'][0]['fields']:
                vertices = field['boundingPoly']['vertices']
                poly = np.array([[v['x'], v['y']] for v in vertices], dtype=np.int32)
                label = field['inferText']
                polygons.append(TextInstance(poly, 'c', label))
        elif 'fields' in data:
            for field in data['fields']:
                vertices = field['boundingPoly']
                poly = np.array([[v['x'], v['y']] for v in vertices], dtype=np.int32)
                label = field['text']
                polygons.append(TextInstance(poly, 'c', label))
        else:
            print(f"에러가 난 파일명: {gt_path}")
            return False

        return polygons
    
    def make_txt(self, gt_path, polygons):
        txt_path = gt_path.replace('.json', '').replace('.jpg', '').replace('.jpeg', '').replace('.png', '').replace('.PNG', '').replace('.JPG', '').replace('JPEG', '') + '.txt'
        if not os.path.exists(txt_path):
            with open(txt_path, 'w', encoding='utf-8') as f:
                for poly in polygons:
                    points = poly.points.flatten()
                    points_str = ','.join(map(str, points))
                    f.write(f"{points_str},{poly.label}\n")


    def load_img_gt(self, item):
        image_path = os.path.join(self.image_root, self.image_list[item])
        if os.name == 'nt':  # 윈도우일 경우
            image_id = image_path.split("\\")[-1]
        else:  # 리눅스일 경우
            image_id = image_path.split("/")[-1]

        image = pil_load_img(image_path)
        try:
            assert image.shape[-1] == 3
        except:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image)

        annotation_path = self.annotation_list[item]
        polygons = self.parse_json(annotation_path)
        self.make_txt(annotation_path, polygons)

        data = dict()
        data["image"] = image
        data["polygons"] = polygons
        data["image_id"] = image_id
        data["image_path"] = image_path

        return data

    def __getitem__(self, item):
        if self.load_memory:
            data = self.datas[item]
        else:
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

    import time
    from util.augmentation import BaseTransform, Augmentation
    from util import canvas as cav

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    transform = Augmentation(
        size=640, mean=means, std=stds
    )

    trainset = myDataset(
        data_root='../data/bnk',
        is_training=True,
        transform=transform,
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