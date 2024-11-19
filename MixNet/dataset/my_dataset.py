import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from dataset.dataload import pil_load_img, TextDataset, TextInstance
import json
import cv2
from functools import lru_cache

def get_absolute_path(p):
    if p.startswith('~'):
        p = os.path.expanduser(p)
    return os.path.abspath(p)

def read_lines(p):
    p = get_absolute_path(p)
    with open(p,'rU') as f:
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
        # JSON 데이터 캐싱
        self.json_cache = {}
        # 각 이미지의 annotation 수 저장
        self.annotation_counts = []
        for ann_path in self.annotation_list:
            polygons = self.parse_json(ann_path)
            if polygons:  # False가 아닌 경우에만 계산
                count = len(polygons)
            else:
                count = 0
            self.annotation_counts.append(count)
            
        # print(f"Dataset {data_root}: annotation count range {min(self.annotation_counts)} ~ {max(self.annotation_counts)}")
            
        if self.load_memory:
            self.datas = []
            for item in range(len(self.image_list)):
                self.datas.append(self.load_img_gt(item))

    @lru_cache(maxsize=128)
    def parse_json(self, gt_path):
        if gt_path in self.json_cache:
            data = self.json_cache[gt_path]
        else:
            with open(gt_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.json_cache[gt_path] = data

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
            os.makedirs(os.path.dirname(txt_path), exist_ok=True)
            with open(txt_path, 'w', encoding='utf-8') as f:
                for poly in polygons:
                    points = poly.points.flatten()
                    points_str = ','.join(map(str, points))
                    f.write(f"{points_str},{poly.label}\n")

    def load_img_gt(self, item):
        image_path = os.path.join(self.image_root, self.image_list[item])
        image_id = os.path.basename(image_path)

        # 이미지 로딩 최적화
        image = pil_load_img(image_path)
        if image.shape[-1] != 3:
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        annotation_path = self.annotation_list[item]
        polygons = self.parse_json(annotation_path)
        self.make_txt(annotation_path, polygons)

        return {
            "image": image,
            "polygons": polygons,
            "image_id": image_id,
            "image_path": image_path
        }

    def __getitem__(self, item):
        data = self.datas[item] if self.load_memory else self.load_img_gt(item)

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