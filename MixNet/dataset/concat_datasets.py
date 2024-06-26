import os
import sys
from torch.utils.data import ConcatDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from open_data import TotalText, CTW_1500, MSRA_TD500, FUNSD, XFUND, SROIE2019, FUNSD_mid, XFUND_mid, SROIE2019_mid
from my_dataset import myDataset
from my_dataset_mid import myDataset_mid
from util.augmentation import Augmentation, BaseTransform
# from cfglib import config as cfg

def concat_open_datas(config, data_root: str = "data/open_datas", is_training: bool = True, load_memory: bool = False):
    # tr1 = TotalText(
    #     data_root = os.path.join(data_root, "totaltext"),
    #     is_training=is_training,
    #     transform=Augmentation(size=config.input_size, mean=config.means, std=config.stds),
    #     load_memory = load_memory
    # )

    # tr2 = MSRA_TD500(
    #     data_root = os.path.join(data_root, "MSRA-TD500"),
    #     is_training=is_training,
    #     transform=Augmentation(size=config.input_size, mean=config.means, std=config.stds),
    #     load_memory = load_memory
    # )
    # tr3 = CTW_1500(
    #     data_root = os.path.join(data_root, "ctw1500"),
    #     is_training=is_training,
    #     transform=Augmentation(size=config.input_size, mean=config.means, std=config.stds),
    #     load_memory = load_memory
    # )
    tr2 = FUNSD(
        data_root = os.path.join(data_root, "FUNSD"),
        is_training=is_training,
        transform=Augmentation(size=config.input_size, mean=config.means, std=config.stds) if is_training else BaseTransform(size=config.test_size, mean=config.means, std=config.stds),
        load_memory = load_memory
    )

    tr3 = XFUND(
        data_root = os.path.join(data_root, "XFUND"),
        is_training=is_training,
        transform=Augmentation(size=config.input_size, mean=config.means, std=config.stds) if is_training else BaseTransform(size=config.test_size, mean=config.means, std=config.stds),
        load_memory = load_memory
    )

    tr4 = SROIE2019(
        data_root = os.path.join(data_root, "SROIE2019"),
        is_training=is_training,
        transform=Augmentation(size=config.input_size, mean=config.means, std=config.stds) if is_training else BaseTransform(size=config.test_size, mean=config.means, std=config.stds),
        load_memory = load_memory
    )

    return ConcatDataset([tr2, tr3, tr4])


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
    # tr1 = TotalText(
    #     data_root = os.path.join(data_root, "totaltext"),
    #     is_training=is_training,
    #     transform=Augmentation(size=config.input_size, mean=config.means, std=config.stds),
    #     load_memory = load_memory
    # )

    # tr2 = MSRA_TD500(
    #     data_root = os.path.join(data_root, "MSRA-TD500"),
    #     is_training=is_training,
    #     transform=Augmentation(size=config.input_size, mean=config.means, std=config.stds),
    #     load_memory = load_memory
    # )
    # tr3 = CTW_1500(
    #     data_root = os.path.join(data_root, "ctw1500"),
    #     is_training=is_training,
    #     transform=Augmentation(size=config.input_size, mean=config.means, std=config.stds),
    #     load_memory = load_memory
    # )
    tr2 = FUNSD_mid(
        data_root = os.path.join(data_root, "FUNSD"),
        is_training=is_training,
        transform=Augmentation(size=config.input_size, mean=config.means, std=config.stds) if is_training else BaseTransform(size=config.test_size, mean=config.means, std=config.stds),
        load_memory = load_memory
    )

    tr3 = XFUND_mid(
        data_root = os.path.join(data_root, "XFUND"),
        is_training=is_training,
        transform=Augmentation(size=config.input_size, mean=config.means, std=config.stds) if is_training else BaseTransform(size=config.test_size, mean=config.means, std=config.stds),
        load_memory = load_memory
    )

    tr4 = SROIE2019_mid(
        data_root = os.path.join(data_root, "SROIE2019"),
        is_training=is_training,
        transform=Augmentation(size=config.input_size, mean=config.means, std=config.stds) if is_training else BaseTransform(size=config.test_size, mean=config.means, std=config.stds),
        load_memory = load_memory
    )

    return ConcatDataset([tr2, tr3, tr4])
    # return ConcatDataset([tr2, tr3])


def AllDataset_mid(config, custom_data_root: str="data/kor", open_data_root: str="data/open_datas", is_training: bool = True,  load_memory: bool = False):
    opened_datasets = concat_open_datas_mid(config, data_root=open_data_root, is_training=is_training,  load_memory = load_memory)
    myData = myDataset_mid(
        data_root=custom_data_root,
        is_training=is_training,
        transform=Augmentation(size=config.input_size, mean=config.means, std=config.stds) if is_training else BaseTransform(size=config.test_size, mean=config.means, std=config.stds),
        load_memory = load_memory
    )

    return ConcatDataset([opened_datasets, myData])
