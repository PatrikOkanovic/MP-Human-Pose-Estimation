import argparse
import os
import pprint

import torchvision

import _init_paths

import torch
import torch.nn.parallel
import torch.utils.data
from tensorboardX import SummaryWriter

from lib.core.config import config
from lib.core.config import update_config
from lib.core.function import validate, evaluate
from lib.utils.utils import create_logger

import lib.dataset as dataset
import lib.models as models
from scripts.train import check_train_mode, parse_args, reset_config


def main():
    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir, time_str = create_logger(config, args.cfg, config.MODEL.TIME_STR, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    tb_log_dir = os.path.join(final_output_dir, 'logs')

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'valid_global_steps': 0,
    }

    check_train_mode(config, logger)

    model = models.nets[config.MODEL.NAME].get_pose_net(config, False)

    gpus = [int(i) for i in config.GPUS.split(',')]
    if not config.CPU:
        model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # Load model
    model_file = os.path.join(final_output_dir, 'model_best.pth.tar')
    checkpoint = torch.load(model_file)
    incompat_keys = model.load_state_dict(checkpoint, strict=False)
    logger.warning('State loading, missing keys %s' % incompat_keys.missing_keys)
    logger.warning('State loading, unexpected keys %s' % incompat_keys.unexpected_keys)
    logger.info('=> resume from pretrained model {}'.format(model_file))

    ds = eval('dataset.h36m')# + config.DATASET.DATASET)

    # Data loading code
    logger.info('=> loading the valid/test dataset')
    valid_dataset = ds(
        cfg=config,
        root='data/h36m/',
        image_set=config.DATASET.TEST_SET,
        is_train=False
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True
    )

    DEBUG_MODE_BREAK = False  # for prediction, do not break but predict all test samples
    # evaluate on validation set
    preds_in_patch_with_score = validate(config, valid_loader, model, DEBUG_MODE_BREAK)
    acc = evaluate(config, 0, preds_in_patch_with_score, valid_loader, final_output_dir, DEBUG_MODE_BREAK,
                   config.DEBUG.DEBUG,
                   writer_dict, time_str)


if __name__ == '__main__':
    main()
