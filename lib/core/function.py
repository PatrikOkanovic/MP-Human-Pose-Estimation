import os
import logging
import time

import numpy as np
import torch

from lib.utils.img_utils import trans_coords_from_patch_to_org_3d
from lib.core.loss import get_result_func
from lib.utils.utils import AverageMeter
from lib.core.loss import get_joint_regression_result

DEBUG_MODE_MAX_ITER = 2

logger = logging.getLogger(__name__)


def train(config, train_loader, model, criterion, optimizer, epoch, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        batch_data, batch_label, batch_label_weight, meta = data

        optimizer.zero_grad()

        if not config.CPU:
            batch_data = batch_data.cuda()
            batch_label = batch_label.cuda()
            batch_label_weight = batch_label_weight.cuda()

        batch_size = batch_data.size(0)
        # compute output
        preds = model(batch_data)

        loss = criterion(preds, batch_label, batch_label_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record loss
        losses.update(loss.item(), batch_size)
        del loss
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0 or i == len(train_loader) - 1 \
                or config.TRAIN.DEBUG_MODE:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                speed=batch_size / batch_time.val,
                data_time=data_time, loss=losses)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

        if config.TRAIN.DEBUG_MODE and i == 0 and config.DATASET.DATASET == 'h36m':
            # Scale to transformed image size i.e. 256x256
            preds_in_patch_with_score = [
                get_joint_regression_result(config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1],
                                            preds[0].cpu().reshape([1, 17, 3]))]
            preds = dirty_solution_for_partial_batches(preds_in_patch_with_score)[0]

            gt_in_patch_with_score = [
                get_joint_regression_result(config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1],
                                            batch_label[0].cpu().reshape([1, 17, 3]))]
            gt = dirty_solution_for_partial_batches(gt_in_patch_with_score)[0]

            train_loader.dataset.plot_2d_images_with_coords(batch_data[0].cpu(), preds, gt)

        del batch_data, batch_label, batch_label_weight, preds

        if config.TRAIN.DEBUG_MODE and i == DEBUG_MODE_MAX_ITER:
            break


def validate(config, val_loader, model, DEBUG_MODE_BREAK):
    print("Validation stage")
    result_func = get_result_func()

    # switch to evaluate mode
    model.eval()

    preds_in_patch_with_score = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            batch_data, batch_label, batch_label_weight, meta = data

            if not config.CPU:
                batch_data = batch_data.cuda()

            # compute output
            preds = model(batch_data)
            del batch_data, batch_label, batch_label_weight

            # Scale to transformed image size i.e. 256x256
            preds_in_patch_with_score.append(result_func(config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1], preds))
            del preds
            if DEBUG_MODE_BREAK and i == DEBUG_MODE_MAX_ITER:
                break

        _p = dirty_solution_for_partial_batches(preds_in_patch_with_score)

        preds_in_patch_with_score = _p[0:
                                       len(val_loader.dataset) if not DEBUG_MODE_BREAK
                                       else DEBUG_MODE_MAX_ITER
                                    ]

        return preds_in_patch_with_score


def dirty_solution_for_partial_batches(preds_in_patch_with_score):
    _p = np.asarray(preds_in_patch_with_score)
    if len(_p.shape) < 2:
        tp = np.zeros(((_p.shape[0] - 1) * _p[0].shape[0] + _p[-1].shape[0], _p[0].shape[1], _p[0].shape[2]))

        start = 0
        end = _p[0].shape[0]

        for t in _p:
            tp[start:end] = t
            start = end
            end += t.shape[0]

        _p = tp
    else:
        _p = _p.reshape((_p.shape[0] * _p.shape[1], _p.shape[2], _p.shape[3]))
    return _p


def evaluate(config, epoch, preds_in_patch_with_score, val_loader, final_output_path, DEBUG_MODE_BREAK, debug=False,
             writer_dict=None,
             time_str=''):
    print("Evaluation stage")

    # From patch to original image coordinate system
    imdb_list = val_loader.dataset.db
    imdb = val_loader.dataset

    preds_in_img_with_score = []

    for n_sample in range(
            len(val_loader.dataset) if not DEBUG_MODE_BREAK
            else DEBUG_MODE_MAX_ITER
    ):
        # Scale to original image size i.e. 1000x1002
        preds_in_img_with_score.append(
            get_preds_in_patch_with_score(config, imdb_list, n_sample, preds_in_patch_with_score))

    preds_in_img_with_score = np.asarray(preds_in_img_with_score)

    # Evaluate
    if 'joints_3d' in imdb.db[0].keys():
        name_value, perf = imdb.evaluate(preds_in_img_with_score.copy(), final_output_path, debug=debug,
                                         writer_dict=writer_dict, time_str=time_str)
        for name, value in name_value:
            logger.info('Epoch[%d] Validation-%s %f', epoch, name, value)
        logger.info('Epoch[%d] Validation-Dim-%s %f', epoch, 'Average', np.mean([nv[1] for nv in name_value]))
    else:
        logger.info('Test set is used, saving results to %s', final_output_path)
        _, perf = imdb.evaluate(preds_in_img_with_score.copy(), final_output_path, debug=debug, writer_dict=writer_dict,
                                time_str=time_str)
        perf = 0.0

    return perf


def get_preds_in_patch_with_score(config, imdb_list, n_sample, preds_in_patch_with_score):
    return trans_coords_from_patch_to_org_3d(preds_in_patch_with_score[n_sample], imdb_list[n_sample]['center_x'],
                                             imdb_list[n_sample]['center_y'], imdb_list[n_sample]['width'],
                                             imdb_list[n_sample]['height'], config.MODEL.IMAGE_SIZE[0],
                                             config.MODEL.IMAGE_SIZE[1],
                                             config.DATASET.RECT_WIDTH, config.DATASET.RECT_HEIGHT)
