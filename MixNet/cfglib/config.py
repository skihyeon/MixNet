from easydict import EasyDict
import torch
config = EasyDict()


# Normalize image
config.means = (0.485, 0.456, 0.406)
config.stds = (0.229, 0.224, 0.225)

config.gpu = "1"

# Experiment name #
config.exp_name = "Test"

# dataloader jobs number
config.num_workers = 24

# batch_size
config.batch_size = 12

# training epoch number
config.max_epoch = 1

config.start_epoch = 0

# learning rate
config.lr = 1e-4

# using GPU
config.cuda = True

config.output_dir = 'output'

config.input_size = 640

# max polygon per image
# synText, total-text:64; CTW1500: 64; icdar: 64;  MLT: 32; TD500: 64.
config.max_annotation = 700

# adj num for graph
config.adj_num = 4

# control points number
config.num_points = 20

# use hard examples (annotated as '#')
config.use_hard = True

# Load data into memory at one time
config.load_memory = False

# prediction on 1/scale feature map
config.scale = 1

# # clip gradient of loss
config.grad_clip = 25

# demo tcl threshold
# config.dis_threshold = 0.3
#
# config.cls_threshold = 0.8

# Contour approximation factor
config.approx_factor = 0.004

# knowledge
config.know = False
# config.knownet = "craft"
config.know_resume = "./model/Test/craft.pth"


def update_config(config, extra_config):
    for k, v in vars(extra_config).items():
        config[k] = v

    if config.cuda and torch.cuda.device_count() > 1:
        config.device = torch.device('cuda:'+str(config.gpu_num))
        if config.mgpu:
            config.device_ids = [2,3]
            config.device = torch.device('cuda:'+str(config.device_ids[0]))
    else:
        config.device = torch.device('cuda:0') if config.cuda else torch.device('cpu')


def print_config(config):
    print('==========Options============')
    for k, v in config.items():
        print('{}: {}'.format(k, v))
    print('=============End=============')
