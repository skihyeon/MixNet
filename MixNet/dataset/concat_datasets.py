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

def concat_open_datas(config, is_training, open_data_root: str = "data/open_datas"):
    available_datasets = {
        "totaltext": TotalText,
        "MSRA-TD500": MSRA_TD500,
        "ctw1500": CTW_1500,
        "FUNSD": FUNSD,
        "XFUND": XFUND,
        "SROIE2019": SROIE2019
    }

    if config.select_open_data == '/':
        selected_datasets = [f for f in os.listdir(open_data_root) if os.path.isdir(os.path.join(open_data_root, f))]
    elif config.select_open_data == '.':
        selected_datasets = []
    else:
        selected_datasets = config.select_open_data.split(',') if config.select_open_data else available_datasets.keys()
    
    if selected_datasets:
        print(f"dataset_root: {open_data_root}\t selected_dataset: {selected_datasets}")
        print('-'*100)
    
    transformed_datasets = []
    for name in selected_datasets:
        if name in available_datasets:
            dataset_class = available_datasets[name]
            transform = Augmentation(size=config.input_size, mean=config.means, std=config.stds) if is_training else BaseTransform(size=config.test_size, mean=config.means, std=config.stds)
            dataset = dataset_class(
                data_root=os.path.join(open_data_root, name),
                is_training=is_training,
                transform=transform,
                load_memory=config.load_memory
            )
            print(f'folder: {name:<20} num samples: {len(dataset):<6}')
            transformed_datasets.append(dataset)
    
    if transformed_datasets:
        print('-'*100)
        return ConcatDataset(transformed_datasets)
    return None


def hierarchical_custom_dataset(config, is_training, custom_data_root, select_data='/'):
    dataset_list = []
    if select_data == '/':
        folders_to_process = [f for f in os.listdir(custom_data_root) if os.path.isdir(os.path.join(custom_data_root, f))]
    else:
        folders_to_process = select_data.split(',')

    dataset_log = ""
    sub_dataset_logs = []

    if folders_to_process:
        dataset_log = f'dataset_root: {custom_data_root}\t selected_dataset: {folders_to_process}'
        dataset_log += '\n' + '-'*100

    for folder in folders_to_process:
        folder_path = os.path.join(custom_data_root, folder.strip())
        if os.path.exists(folder_path):
            dataset = myDataset(
                data_root = folder_path,
                is_training = is_training,
                transform = Augmentation(size=config.input_size, mean=config.means, std=config.stds) if is_training else BaseTransform(size=config.test_size, mean=config.means, std=config.stds),
                load_memory = config.load_memory
            )
            sub_dataset_log = f'folder: {folder.strip():<20} num samples: {len(dataset):<6}'
            sub_dataset_logs.append(sub_dataset_log)
            dataset_list.append(dataset)
        else:
            sub_dataset_logs.append(f'경고: {folder_path} 경로가 존재하지 않습니다.')
    
    if dataset_list:
        print(dataset_log)
        for log in sub_dataset_logs:
            print(log)
        print('-'*100)
        return ConcatDataset(dataset_list)
    return None


def AllDataset(config, is_training):
    datasets = []
    
    open_datasets = concat_open_datas(config, is_training, open_data_root=config.open_data_root)
    if open_datasets:
        datasets.append(open_datasets)
    
    custom_datasets = hierarchical_custom_dataset(
        config, 
        is_training=is_training,
        custom_data_root=config.custom_data_root,
        select_data=config.select_custom_data
    )
    if custom_datasets:
        datasets.append(custom_datasets)

    if not datasets:
        raise ValueError("데이터셋이 선택되지 않았습니다. open_data 또는 custom_data 중 하나 이상을 선택해주세요.")
    
    return ConcatDataset(datasets)


def concat_open_datas_mid(config, is_training, open_data_root: str = "data/open_datas"):
    available_datasets = {
        "totaltext": TotalText_mid,
        "MSRA-TD500": MSRA_TD500_mid,
        "ctw1500": CTW_1500_mid,
        "FUNSD": FUNSD_mid,
        "XFUND": XFUND_mid,
        "SROIE2019": SROIE2019_mid
    }

    if config.select_open_data == '/':
        selected_datasets = [f for f in os.listdir(open_data_root) if os.path.isdir(os.path.join(open_data_root, f))]
    elif config.select_open_data == '.':
        selected_datasets = []
    else:
        selected_datasets = config.select_open_data.split(',') if config.select_open_data else available_datasets.keys()
    
    if selected_datasets:
        print(f"dataset_root: {open_data_root}\t selected_dataset: {selected_datasets}")
        print('-'*100)
    
    transformed_datasets = []
    for name in selected_datasets:
        if name in available_datasets:
            dataset_class = available_datasets[name]
            transform = Augmentation(size=config.input_size, mean=config.means, std=config.stds) if is_training else BaseTransform(size=config.test_size, mean=config.means, std=config.stds)
            dataset = dataset_class(
                data_root=os.path.join(open_data_root, name),
                is_training=is_training,
                transform=transform,
                load_memory=config.load_memory
            )
            print(f'folder: {name:<20} num samples: {len(dataset):<6}')
            transformed_datasets.append(dataset)
    
    if transformed_datasets:
        print('-'*100)
        return ConcatDataset(transformed_datasets)
    return None


def hierarchical_custom_dataset_mid(config, is_training, custom_data_root, select_data='/'):
    dataset_list = []
    if select_data == '/':
        folders_to_process = [f for f in os.listdir(custom_data_root) if os.path.isdir(os.path.join(custom_data_root, f))]
    else:
        folders_to_process = select_data.split(',')

    dataset_log = ""
    sub_dataset_logs = []

    if folders_to_process:
        dataset_log = f'dataset_root: {custom_data_root}\t selected_dataset: {folders_to_process}'
        dataset_log += '\n' + '-'*100

    for folder in folders_to_process:
        folder_path = os.path.join(custom_data_root, folder.strip())
        if os.path.exists(folder_path):
            dataset = myDataset_mid(
                data_root = folder_path,
                is_training = is_training,
                transform = Augmentation(size=config.input_size, mean=config.means, std=config.stds) if is_training else BaseTransform(size=config.test_size, mean=config.means, std=config.stds),
                load_memory = config.load_memory
            )
            sub_dataset_log = f'folder: {folder.strip():<20} num samples: {len(dataset):<6}'
            sub_dataset_logs.append(sub_dataset_log)
            dataset_list.append(dataset)
        else:
            sub_dataset_logs.append(f'경고: {folder_path} 경로가 존재하지 않습니다.')
    
    if dataset_list:
        print(dataset_log)
        for log in sub_dataset_logs:
            print(log)
        print('-'*100)
        return ConcatDataset(dataset_list)
    return None


def AllDataset_mid(config, is_training):
    datasets = []
    
    open_datasets = concat_open_datas_mid(config, is_training, open_data_root=config.open_data_root)
    if open_datasets:
        datasets.append(open_datasets)
    
    custom_datasets = hierarchical_custom_dataset_mid(
        config, 
        is_training=is_training,
        custom_data_root=config.custom_data_root,
        select_data=config.select_custom_data
    )
    if custom_datasets:
        datasets.append(custom_datasets)

    if not datasets:
        raise ValueError("데이터셋이 선택되지 않았습니다. open_data 또는 custom_data 중 하나 이상을 선택해주세요.")
    
    return ConcatDataset(datasets)