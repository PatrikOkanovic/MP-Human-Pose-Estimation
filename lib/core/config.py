import os
import yaml

import numpy as np
from easydict import EasyDict as edict

config = edict()

config.OUTPUT_DIR = ''
config.LOG_DIR = ''
config.GPUS = '0'
config.WORKERS = 8
config.PRINT_FREQ = 20
config.EXP_NAME = 'default'

# common params for NETWORK
config.MODEL = edict()
config.MODEL.NAME = 'pose3d_alexnet_reg'
config.MODEL.BACKBONE = 'resnext101_32x8d'
config.MODEL.INIT_WEIGHTS = True
config.MODEL.DOWNLOAD_PRETRAINED = True
config.MODEL.PRETRAINED = ''
config.MODEL.TIME_STR = ''
config.MODEL.RESUME = ''
config.MODEL.NUM_JOINTS = 17
config.MODEL.DEPTH_RES = 64
config.MODEL.IMAGE_SIZE = [288, 384]  # width * height, ex: 192 * 256
config.MODEL.VOLUME = {}

# Loss function params
config.LOSS = edict()
config.LOSS.FN = 'L1JointRegressionLoss'
# h36m specific training params
config.LOSS.NORM = False

# DATASET related params
config.DATASET = edict()
config.DATASET.ROOT = ''
config.DATASET.DATASET = 'h36m'
config.DATASET.TRAIN_SET = 'train'
config.DATASET.TEST_SET = 'valid'
config.DATASET.HYBRID_JOINTS_TYPE = ''
config.DATASET.OCCLUSION = False  # Assign True if you want to use occlusion augmentation proposed by Sarandi et al.
# in https://arxiv.org/abs/1808.09316
config.DATASET.VOC = '/media/muhammed/Other/RESEARCH/datasets/VOCdevkit/VOC2012'  # path to PASCAL VOC2012 dataset

# H36M related params
config.DATASET.TRAIN_FRAME = 32
config.DATASET.VAL_FRAME = 64
config.DATASET.NUM_CAMS = 4
config.DATASET.RECT_WIDTH = 1485.0 # width, height of the area around the subject in mm
config.DATASET.RECT_HEIGHT = 1980.0 # width, height of the area around the subject in mm

# training data augmentation
config.DATASET.Z_WEIGHT = 1. # weighting parameter for z axis
config.DATASET.AUGMENT = False

# train
config.TRAIN = edict()

config.TRAIN.DEBUG_MODE = False

config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = [90, 110]
config.TRAIN.LR = 0.001

config.TRAIN.OPTIMIZER = 'adam'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0001
config.TRAIN.NESTEROV = False
config.TRAIN.GAMMA1 = 0.99
config.TRAIN.GAMMA2 = 0.0

config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 140

config.TRAIN.BATCH_SIZE = 32
config.TRAIN.SHUFFLE = True

config.TRAIN.BETA1 = 0.9
config.TRAIN.BETA2 = 0.999

# testing
config.TEST = edict()

# size of images for each device
config.TEST.BATCH_SIZE = 32
# Test Model Epoch
# debug
config.DEBUG = edict()
config.DEBUG.DEBUG = False

config.CPU = False

def _update_dict(k, v):
	if k == 'MODEL':
            if 'IMAGE_SIZE' in v:
                if isinstance(v['IMAGE_SIZE'], int):
                    v['IMAGE_SIZE'] = np.array([v['IMAGE_SIZE'], v['IMAGE_SIZE']])
                else:
                    v['IMAGE_SIZE'] = np.array(v['IMAGE_SIZE'])
	for vk, vv in v.items():
            if vk in config[k]:
                config[k][vk] = vv
            else:
                raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))


def gen_config(config_file, config):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def get_model_name(cfg):
    name = cfg.MODEL.NAME
    full_name = cfg.MODEL.NAME
    volume = cfg.MODEL.VOLUME

    name = '{model}'.format(model=name)
    full_name = '{height}x{width}_{name}'.format(
                     height=cfg.MODEL.IMAGE_SIZE[1],
                     width=cfg.MODEL.IMAGE_SIZE[0],
                     name=name)

    print(name, full_name)
    return name, full_name


if __name__ == '__main__':
    import sys
    gen_config(sys.argv[1], config)
