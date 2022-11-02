import argparse
import logging
import os
import pprint
import shutil

import yaml

import _init_paths

import torch

# You can uncomment this line for reproducibility
torch.manual_seed(69)
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter

from lib.core.config import config, gen_config
from lib.core.config import update_config
from lib.core.config import get_model_name
from lib.core.function import train, validate, evaluate
from lib.utils.utils import get_optimizer
from lib.utils.utils import save_checkpoint
from lib.utils.utils import create_logger

import lib.core.loss as loss
import lib.dataset as dataset
import lib.models as models
from torch.utils.data import ConcatDataset, WeightedRandomSampler, BatchSampler

import numpy as np

# You can uncomment this line for reproducibility
np.random.seed(69)

# enable IDE debugging: set working directory to project root folder
import sys  # sys.path.append('../')


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    # takes default values from config.py and sets them based on the cfg.yaml
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int,
                        default=8)

    parser.add_argument('--model_init_weights',
                        help='load pre trained',
                        type=int,
                        default=-1)  # -1 so that we do not overwrite config if we do not pass anything
    parser.add_argument('--model_name',
                        help='file to load. Note: Use cfg instead',
                        type=str)
    parser.add_argument('--train_debug_mode',
                        help='train only few iterations to see if it works',
                        type=int,
                        default=-1)
    parser.add_argument('--model_time_str',
                        help='Preload a previously trained model. Currently only works for validation.',
                        type=str)
    parser.add_argument('--cpu',
                        help='run on cpu',
                        type=int,
                        default=-1)
    parser.add_argument('--model_backbone',
                        help='Preload a previously trained model. Currently only works for validation.',
                        type=str)
    parser.add_argument('--dataset_augment',
                        help='flip, rotate, scale etc',
                        type=int,
                        default=-1)

    args = parser.parse_args()

    return args


def reset_config(config, args, valid=False):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.model_init_weights != -1:
        config.MODEL.INIT_WEIGHTS = bool(args.model_init_weights)
    if args.model_name:
        config.MODEL.NAME = args.model_name
    if args.train_debug_mode != -1:
        config.TRAIN.DEBUG_MODE = bool(args.train_debug_mode)
    if args.cpu != -1:
        config.CPU = bool(args.cpu)
    if args.model_time_str:
        config.MODEL.TIME_STR = args.model_time_str
    elif valid and not config.MODEL.TIME_STR:
        raise ValueError('No MODEL.TIMESTR provided')
    if args.model_backbone:
        config.MODEL.BACKBONE = args.model_backbone
    if args.dataset_augment != -1:
        config.DATASET.AUGMENT = bool(args.dataset_augment)


def check_train_mode(config, logger):
    if config.TRAIN.DEBUG_MODE:
        logger.info(f'WARN: Training only for 2 epochs with batch size 2')
        config.TRAIN.BATCH_SIZE = 2
        config.TRAIN.END_EPOCH = 2
        config.TEST.BATCH_SIZE = 2


def main():
    best_perf = -10000  # store first created model (primarily for DEBUG_MODE when we only train for 1 iteration)

    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir, time_str = create_logger(config, args.cfg, config.MODEL.TIME_STR, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    check_train_mode(config, logger)

    model = models.nets[config.MODEL.NAME].get_pose_net(config, True and config.MODEL.INIT_WEIGHTS)

    # copy model file
    this_dir = os.path.dirname(__file__)

    # overwrite config with the provided arguments for dumping
    gen_config(os.path.join(final_output_dir, args.cfg.split('/')[-1]), config)

    tb_log_dir = os.path.join(final_output_dir, 'logs')

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    gpus = [int(i) for i in config.GPUS.split(',')]
    if not config.CPU:
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    loss_fn = eval('loss.' + config.LOSS.FN)
    is_2d = config.MODEL.DEPTH_RES == 1
    if is_2d:
        logging.info('=> model is trained on 2D ground truth')
    else:
        logging.info('=> model is trained on 3D ground truth')
    if not config.CPU:
        criterion = loss_fn(num_joints=config.MODEL.NUM_JOINTS, norm=config.LOSS.NORM,
                            is_2d=is_2d).cuda()
    else:
        criterion = loss_fn(num_joints=config.MODEL.NUM_JOINTS, norm=config.LOSS.NORM,
                            is_2d=is_2d)

    optimizer = get_optimizer(config, model)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR
    )

    # Resume from a trained model
    if config.MODEL.RESUME != '':
        ckpt_file = os.path.join(final_output_dir, config.MODEL.RESUME)
        if os.path.exists(ckpt_file):
            checkpoint = torch.load(ckpt_file)
            if 'epoch' in checkpoint.keys():
                config.TRAIN.BEGIN_EPOCH = checkpoint['epoch']
                config.TRAIN.END_EPOCH = config.TRAIN.BEGIN_EPOCH + config.TRAIN.END_EPOCH
                best_perf = checkpoint['perf']
                incompat_keys = model.load_state_dict(checkpoint['state_dict'], strict=False)
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                if 'train_step' in checkpoint:
                    writer_dict['train_global_steps'] = checkpoint['train_step']
                if 'valid_step' in checkpoint:
                    writer_dict['valid_global_steps'] = checkpoint['valid_step']
                logger.info('=> resume from pretrained model {}'.format(config.MODEL.RESUME))
            else:
                incompat_keys = model.load_state_dict(checkpoint, strict=False)
                logger.info('=> resume from pretrained model {}'.format(config.MODEL.RESUME))
            logger.warning('State loading, missing keys %s' % incompat_keys.missing_keys)
            logger.warning('State loading, unexpected keys %s' % incompat_keys.unexpected_keys)
        else:
            logger.warning(f'WARN: MODEL.RESUME set, but no file found {config.MODEL.RESUME}')

    if config.DATASET.DATASET == 'both':
        logger.info('=> using MPII and H36m for training')
        train_dataset1, train_dataset2 = get_datasets(logger, eval('dataset.h36m'), 'h36m', config.DATASET.TRAIN_SET,
                                                      config.DATASET.TEST_SET)
        train_dataset3, train_dataset4 = get_datasets(logger, eval('dataset.mpii'), 'mpii', config.DATASET.TRAIN_SET,
                                                      config.DATASET.TEST_SET)

        len1 = len(train_dataset1) + len(train_dataset2)
        len2 = len(train_dataset3) + len(train_dataset4)
        # Sampling from MPII and H36M with same probability s.t. each batch is 50/50
        weights = np.concatenate((np.full((len1), 1, dtype=float),
                                  np.full((len2), len1 / len2, dtype=float)), axis=0)
        sampler = WeightedRandomSampler(weights=weights, num_samples=weights.size,
                                        replacement=True)
        train_dataset = ConcatDataset([train_dataset1, train_dataset2, train_dataset3, train_dataset4])
        valid_dataset = train_dataset2
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=BatchSampler(sampler=sampler, batch_size=config.TRAIN.BATCH_SIZE * len(gpus), drop_last=True),
            num_workers=config.WORKERS,
            pin_memory=True,
        )
    else:
        train_dataset, valid_dataset = get_datasets(logger, eval('dataset.' + config.DATASET.DATASET),
                                                    config.DATASET.DATASET, config.DATASET.TRAIN_SET,
                                                    config.DATASET.TEST_SET)

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config.TRAIN.BATCH_SIZE * len(gpus),
            shuffle=config.TRAIN.SHUFFLE,
            num_workers=config.WORKERS,
            pin_memory=True,
        )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    best_model = False
    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):
        # train for one epoch
        train(config, train_loader, model, criterion, optimizer, epoch, writer_dict)
        lr_scheduler.step()

        # evaluate on validation set
        preds_in_patch_with_score = validate(config, valid_loader, model, config.TRAIN.DEBUG_MODE)
        acc = evaluate(config, epoch, preds_in_patch_with_score, valid_loader, final_output_dir,
                       config.TRAIN.DEBUG_MODE,
                       config.DEBUG.DEBUG,
                       writer_dict)

        if not config.TRAIN.DEBUG_MODE or True:
            perf_indicator = 500. - acc if config.DATASET.DATASET == 'h36m' or 'mpii_3dhp' or 'jta' else acc

            if perf_indicator > best_perf:
                best_perf = perf_indicator
                best_model = True
            else:
                best_model = False

            logger.info('=> saving checkpoint to {}'.format(final_output_dir))
            save_checkpoint({
                'train_step': writer_dict['train_global_steps'],
                'valid_step': writer_dict['valid_global_steps'],
                'epoch': epoch + 1,
                'model': get_model_name(config),
                'state_dict': model.state_dict(),
                'perf': perf_indicator,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
            }, best_model, final_output_dir)

    if not config.TRAIN.DEBUG_MODE or True:
        final_model_state_file = os.path.join(final_output_dir,
                                              'final_state.pth.tar')
        logger.info('saving final model state to {}'.format(
            final_model_state_file))
        torch.save(model.module.state_dict(), final_model_state_file)


def get_datasets(logger, ds, name, train, test):
    # Data loading code
    logger.info('=> loading the train dataset')
    train_dataset = ds(
        cfg=config,
        root=f'data/{name}/',
        image_set=train,
        is_train=True
    )
    logger.info('=> loading the valid dataset')
    valid_dataset = ds(
        cfg=config,
        root=f'data/{name}/',
        image_set=test if name == 'h36m' else 'valid',
        is_train=False
    )
    return train_dataset, valid_dataset


if __name__ == '__main__':
    main()
