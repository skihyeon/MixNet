import os
import sys
from torch.utils.data import ConcatDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from open_data import (
    TotalText, 
    CTW_1500, 
    MSRA_TD500, 
    FUNSD, 
    XFUND, 
    SROIE2019, 
    FUNSD_mid, 
    XFUND_mid, 
    SROIE2019_mid, 
    TotalText_mid, 
    CTW_1500_mid, 
    MSRA_TD500_mid
)
from my_dataset import myDataset
from my_dataset_mid import myDataset_mid
from util.augmentation import Augmentation, BaseTransform
# from cfglib import config as cfg

def concat_open_datas(config, data_root: str = "data/open_datas", is_training: bool = True, load_memory: bool = False):
    datasets = [
        ("totaltext", TotalText),
        ("MSRA-TD500", MSRA_TD500),
        ("ctw1500", CTW_1500),
        ("FUNSD", FUNSD),
        ("XFUND", XFUND),
        ("SROIE2019", SROIE2019)
    ]

    transformed_datasets = []
    for name, dataset_class in datasets:
        transform = Augmentation(size=config.input_size, mean=config.means, std=config.stds) if is_training else BaseTransform(size=config.test_size, mean=config.means, std=config.stds)
        dataset = dataset_class(
            data_root=os.path.join(data_root, name),
            is_training=is_training,
            transform=transform,
            load_memory=load_memory
        )
        transformed_datasets.append(dataset)

    return ConcatDataset(transformed_datasets)


def AllDataset(config, custom_data_root: str="data/kor", open_data_root: str="data/open_datas", is_training: bool = True,  load_memory: bool = False):
    opened_datasets = concat_open_datas(config, data_root=open_data_root, is_training=is_training,  load_memory = load_memory)
    myData = myDataset(
        data_root=custom_data_root,
        is_training=is_training,
        transform=Augmentation(size=config.input_size, mean=config.means, std=config.stds) if is_training else BaseTransform(size=config.test_size, mean=config.means, std=config.stds),
        load_memory = load_memory
    )

    return ConcatDataset([opened_datasets, myData])



def concat_open_datas_mid(config, data_root: str = "data/open_datas", is_training: bool = True, load_memory: bool = False):
    datasets = [
        ("totaltext", TotalText_mid),
        ("MSRA-TD500", MSRA_TD500_mid),
        ("ctw1500", CTW_1500_mid),
        ("FUNSD", FUNSD_mid),
        ("XFUND", XFUND_mid),
        ("SROIE2019", SROIE2019_mid)
    ]

    transformed_datasets = []
    for name, dataset_class in datasets:
        transform = Augmentation(size=config.input_size, mean=config.means, std=config.stds) if is_training else BaseTransform(size=config.test_size, mean=config.means, std=config.stds)
        dataset = dataset_class(
            data_root=os.path.join(data_root, name),
            is_training=is_training,
            transform=transform,
            load_memory=load_memory
        )
        transformed_datasets.append(dataset)

    return ConcatDataset(transformed_datasets)


def AllDataset_mid(config, custom_data_root: str="data/kor", open_data_root: str="data/open_datas", is_training: bool = True,  load_memory: bool = False):
    opened_datasets = concat_open_datas_mid(config, data_root=open_data_root, is_training=is_training,  load_memory = load_memory)
    myData = myDataset_mid(
        data_root=custom_data_root,
        is_training=is_training,
        transform=Augmentation(size=config.input_size, mean=config.means, std=config.stds) if is_training else BaseTransform(size=config.test_size, mean=config.means, std=config.stds),
        load_memory = load_memory
    )

    return ConcatDataset([opened_datasets, myData])
