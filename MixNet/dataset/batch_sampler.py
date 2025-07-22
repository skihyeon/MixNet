import torch
from torch.utils.data import Sampler, ConcatDataset
import math

class BalancedBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.annotation_counts = []
        
        # ConcatDataset인 경우 하위 데이터셋의 annotation_counts를 모두 결합
        if isinstance(dataset, ConcatDataset):
            total_counts = []
            for ds in dataset.datasets:
                if isinstance(ds, ConcatDataset):  # 중첩된 ConcatDataset 처리
                    for sub_ds in ds.datasets:
                        if hasattr(sub_ds, 'annotation_counts'):
                            total_counts.extend(sub_ds.annotation_counts)
                        else:
                            print(f"Warning: Dataset {type(sub_ds)} has no annotation_counts")
                            total_counts.extend([0] * len(sub_ds))
                elif hasattr(ds, 'annotation_counts'):
                    total_counts.extend(ds.annotation_counts)
                else:
                    print(f"Warning: Dataset {type(ds)} has no annotation_counts")
                    total_counts.extend([0] * len(ds))
            self.annotation_counts = total_counts
        elif hasattr(dataset, 'annotation_counts'):
            self.annotation_counts = dataset.annotation_counts
        else:
            raise AttributeError("Dataset must have 'annotation_counts' attribute.")
        
        # print(f"Total samples: {len(self.annotation_counts)}")
        # print(f"Annotation count range: {min(self.annotation_counts)} ~ {max(self.annotation_counts)}")
        
        # annotation_counts 기준으로 인덱스 정렬 (내림차순)
        self.sorted_indices = sorted(
            range(len(self.annotation_counts)),
            key=lambda x: self.annotation_counts[x],
            reverse=True
        )

    def __iter__(self):
        left = 0
        right = len(self.sorted_indices) - 1
        
        while left <= right:
            batch = []
            # 배치 크기만큼 반복
            for i in range(self.batch_size):
                if left > right:  # 남은 인덱스가 없으면 중단
                    break
                
                if i % 2 == 0 and left <= right:  # 짝수 인덱스: 많은 annotation
                    batch.append(self.sorted_indices[left])
                    left += 1
                elif i % 2 == 1 and left <= right:  # 홀수 인덱스: 적은 annotation
                    batch.append(self.sorted_indices[right])
                    right -= 1
            
            if batch:  # 배치가 비어있지 않은 경우만 출력
                annotation_combination = [self.annotation_counts[i] for i in batch]
                # print(f"현재 배치의 어노테이션 조합: {annotation_combination}")
                yield batch

    def __len__(self):
        return math.ceil(len(self.sorted_indices) / self.batch_size)