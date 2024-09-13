import argparse
import torch
import os
import torch.backends.cudnn as cudnn

from datetime import datetime


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def arg2str(args):
    args_dict = vars(args)
    option_str = datetime.now().strftime('%b%d_%H-%M-%S') + '\n'

    for k, v in sorted(args_dict.items()):
        option_str += ('{}: {}\n'.format(str(k), str(v)))

    return option_str


class BaseOptions(object):

    def __init__(self):

        self.parser = argparse.ArgumentParser()

        # basic opts
        self.parser.add_argument('--exp_name', default="ConcatDatas", type=str, help='Experiment name')
        self.parser.add_argument('--resume', default=None, type=str, help='Path to target resume checkpoint')
        # self.parser.add_argument('--dataset_name', default=None, type=str, help='dataset_name, under ./data/')

        self.parser.add_argument('--num_workers', default=32, type=int, help='Number of workers used in dataloading')
        self.parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
        self.parser.add_argument('--save_dir', default='./model/', help='Path to save checkpoint models')
        self.parser.add_argument('--vis_dir', default='./vis/', help='Path to save visualization images')
        self.parser.add_argument('--viz', default=False, type=str2bool, help='Whether to output debug info')

        # train opts
        self.parser.add_argument('--max_epoch', default=1000, type=int, help='Max epochs')
        self.parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
        self.parser.add_argument('--step_size', default=10, type=int, help='step size for learning rate decay')
        self.parser.add_argument('--batch_size', default=2, type=int, help='Batch size for training')
        self.parser.add_argument('--save_freq', default=5, type=int, help='save weights every # epoch')
        self.parser.add_argument('--viz_freq', default=50, type=int, help='visualize training process every # iter')

        # backbone
        self.parser.add_argument('--scale', default=1, type=int, help='prediction on 1/scale feature map')
        self.parser.add_argument('--net', default='FSNet_M', type=str,
                                 choices=["FSNet_M", "FSNet_S","FSNet_hor", "FSNet_H_M"],
                                 help='Network architecture')
        self.parser.add_argument('--mid', default=False, type=str2bool, help='midline predict to Transformer')
        self.parser.add_argument('--embed', default=False, type=str2bool, help='predict embeding value for training')
        self.parser.add_argument('--know', default=False, type=str2bool, help='Knowledge Distillation')
        self.parser.add_argument('--onlybackbone', default=False, type=str2bool, help='skip the Transformer block, only train the FSNet. ')
        
        # data args
        self.parser.add_argument('--load_memory', default=False, type=str2bool, help='Load data into memory')
        self.parser.add_argument('--rescale', type=float, default=255.0, help='rescale factor')
        self.parser.add_argument('--input_size', default=768, type=int, help='model input size')
        self.parser.add_argument('--test_size', default=[768, 1152], type=int, nargs='+', help='test size')

        # eval args00
        self.parser.add_argument('--checkepoch', default=1070, type=int, help='Load checkpoint number')
        self.parser.add_argument('--start_epoch', default=0, type=int, help='start epoch number')
        self.parser.add_argument('--cls_threshold', default=0.875, type=float, help='threshold of pse')
        self.parser.add_argument('--dis_threshold', default=0.35, type=float, help='filter the socre < score_i')

        # demo args
        self.parser.add_argument('--img_root', default=None,   type=str, help='Path to deploy images')

        self.parser.add_argument('--infer_path', default=None, type=str, help='inferene image or folder path')
        self.parser.add_argument('--gpu_num', default='0', type=str, help='GPU number')
        self.parser.add_argument('--eval_dataset', default=None, choices=['All', 'my'],type=str, help='Eval dataset')
        self.parser.add_argument('--num_points', default=20, type=int, help ='sampling points')
        self.parser.add_argument('--custom_data_root', default="data/custom_datas", type=str, help='custom data root')
        self.parser.add_argument('--open_data_root', default="data/open_datas", type=str, help='open data root')
        self.parser.add_argument('--select_open_data', default="totaltext,MSRA-TD500,ctw1500,FUNSD,XFUND,SROIE2019", type=str, help='select open data')
        self.parser.add_argument('--select_custom_data', default="kor_extended,bnk", type=str, help='select custom data')
        self.parser.add_argument('--wandb', action='store_true')
        self.parser.add_argument('--accumulation', default=0, type=int)
    def parse(self, fixed=None):

        if fixed is not None:
            args = self.parser.parse_args(fixed)
        else:
            args = self.parser.parse_args()

        return args

    def initialize(self, fixed=None):

        # Parse options
        self.args = self.parse(fixed)

        # Setting default torch Tensor type
        if self.args.cuda and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            # cudnn.benchmark = True
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        # Create weights saving directory
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        # Create weights saving directory of target model
        model_save_path = os.path.join(self.args.save_dir, self.args.exp_name)

        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)

        return self.args

    def update(self, args, extra_options):

        for k, v in extra_options.items():
            setattr(args, k, v)
