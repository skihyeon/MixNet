import os
import sys
from torch.utils.data import ConcatDataset
from typing import List

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

def concat_open_datas(config, open_data_root: str = "data/open_datas"):
    available_datasets = {
        "totaltext": TotalText,
        "MSRA-TD500": MSRA_TD500,
        "ctw1500": CTW_1500,
        "FUNSD": FUNSD,
        "XFUND": XFUND,
        "SROIE2019": SROIE2019
    }

    selected_datasets = config.select_open_data.split(',') if config.select_open_data else available_datasets.keys()
    print(f"dataset_root: {open_data_root}\t selected_dataset: {selected_datasets}")
    print('-'*100)
    
    transformed_datasets = []
    for name in selected_datasets:
        if name in available_datasets:
            dataset_class = available_datasets[name]
            transform = Augmentation(size=config.input_size, mean=config.means, std=config.stds) if config.is_training else BaseTransform(size=config.test_size, mean=config.means, std=config.stds)
            dataset = dataset_class(
                data_root=os.path.join(open_data_root, name),
                is_training=config.is_training,
                transform=transform,
                load_memory=config.load_memory
            )
            print(f'folder: {name:<20} num samples: {len(dataset):<6}')
            transformed_datasets.append(dataset)
    print('-'*100)
    return ConcatDataset(transformed_datasets)


def hierarchical_custom_dataset(config, custom_data_root, select_data='/'):
    dataset_list = []
    if select_data == '/':
        folders_to_process = [f for f in os.listdir(custom_data_root) if os.path.isdir(os.path.join(custom_data_root, f))]
    else:
        folders_to_process = select_data.split(',')

    dataset_log = f'dataset_root: {custom_data_root}\t selected_dataset: {folders_to_process}'
    dataset_log += '\n' + '-'*100
    print(dataset_log)

    for folder in folders_to_process:
        folder_path = os.path.join(custom_data_root, folder.strip())
        if os.path.exists(folder_path):
            dataset = myDataset(
                data_root = folder_path,
                is_training = config.is_training,
                transform = Augmentation(size=config.input_size, mean=config.means, std=config.stds) if config.is_training else BaseTransform(size=config.test_size, mean=config.means, std=config.stds),
                load_memory = config.load_memory
            )
            sub_dataset_log = f'folder: {folder.strip():<20} num samples: {len(dataset):<6}'
            print(sub_dataset_log)
            dataset_list.append(dataset)
        else:
            print(f'Warning: {folder_path} path does not exist.')
    print('-'*100)
    concatenated_dataset = ConcatDataset(dataset_list)
    return concatenated_dataset


def AllDataset(config):
    open_datasets = concat_open_datas(config, open_data_root=config.open_data_root)
    
    custom_datasets = hierarchical_custom_dataset(
        config, 
        custom_data_root=config.custom_data_root,
        select_data=config.select_custom_data
    )

    all_datasets = [open_datasets] + [custom_datasets]
    return ConcatDataset(all_datasets)


def concat_open_datas_mid(config, data_root: str = "data/open_datas"):
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
        transform = Augmentation(size=config.input_size, mean=config.means, std=config.stds) if config.is_training else BaseTransform(size=config.test_size, mean=config.means, std=config.stds)
        dataset = dataset_class(
            data_root=os.path.join(data_root, name),
            is_training=config.is_training,
            transform=transform,
            load_memory=config.load_memory
        )
        transformed_datasets.append(dataset)

    return ConcatDataset(transformed_datasets)


def AllDataset_mid(config, custom_data_root: str="kor_extended,bnk", open_data_root: str="data/open_datas"):
    datasets = concat_open_datas_mid(config, data_root=open_data_root)

    custom_data_dir = custom_data_root.split(',')
    custom_data_dirs = [os.path.join('./data/', dir) for dir in custom_data_dir]
    for data_dir in custom_data_dirs:
        myData = myDataset_mid(
            data_root=data_dir,
            is_training=config.is_training,
            transform=Augmentation(size=config.input_size, mean=config.means, std=config.stds) if config.is_training else BaseTransform(size=config.test_size, mean=config.means, std=config.stds),
            load_memory = config.load_memory
        )
        datasets = ConcatDataset([datasets, myData])

    return datasets


