import os
import sys
from torch.utils.data import ConcatDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from open_data import TotalText, CTW_1500, MSRA_TD500
from my_dataset import myDataset
from util.augmentation import Augmentation
# from cfglib import config as cfg

def concat_open_datas(config, data_root: str = "data/open_datas", is_training: bool = True):
    tr1 = TotalText(
        data_root = os.path.join(data_root, "totaltext"),
        is_training=is_training,
        transform=Augmentation(size=config.input_size, mean=config.means, std=config.stds)
    )

    tr2 = MSRA_TD500(
        data_root = os.path.join(data_root, "MSRA-TD500"),
        is_training=is_training,
        transform=Augmentation(size=config.input_size, mean=config.means, std=config.stds)
    )
    tr3 = CTW_1500(
        data_root = os.path.join(data_root, "ctw1500"),
        is_training=is_training,
        transform=Augmentation(size=config.input_size, mean=config.means, std=config.stds)
    )

    return ConcatDataset([tr1, tr2, tr3])


def AllDataset(config, custom_data_root: str="data/kor", open_data_root: str="data/open_datas", is_training: bool = True):
    opened_datasets = concat_open_datas(config, data_root=open_data_root, is_training=is_training)
    myData = myDataset(
        data_root=custom_data_root,
        is_training=is_training,
        transform=Augmentation(size=config.input_size, mean=config.means, std=config.stds)
    )

    return ConcatDataset([opened_datasets, myData])