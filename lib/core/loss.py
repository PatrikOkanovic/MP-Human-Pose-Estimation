import torch
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
import math


def weighted_l1_loss(input, target, weights, size_average, norm=False):
    if norm:
        input = input / torch.norm(input, 1)
        target = target / torch.norm(target, 1)

    out = torch.abs(input - target)
    out = out * weights
    if size_average:
        num_valid = weights.byte().any(dim=1).float().sum()
        return out.sum() / num_valid
    else:
        return out.sum()


def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"


class L1JointRegressionLoss(nn.Module):
    def __init__(self, num_joints, size_average=True, reduce=True, norm=False, is_2d=False):
        super(L1JointRegressionLoss, self).__init__()
        self.size_average = size_average
        self.reduce = reduce
        self.num_joints = num_joints
        self.norm = norm
        self.is_2d = is_2d

    def forward(self, preds, *args):
        gt_joints = args[0]
        gt_joints_vis = args[1]
        if self.is_2d:
            gt_joints[:, 2:gt_joints.shape[1]:3] = gt_joints[:, 2:gt_joints_vis.shape[1]:3] * 0
            gt_joints_vis[:, 2:gt_joints_vis.shape[1]:3] = gt_joints_vis[:, 2:gt_joints_vis.shape[1]:3] * 0

        pred_jts = preds.reshape((preds.shape[0], self.num_joints * 3))

        _assert_no_grad(gt_joints)
        _assert_no_grad(gt_joints_vis)
        return weighted_l1_loss(pred_jts, gt_joints, gt_joints_vis, self.size_average, self.norm)


def generate_joint_location_label(patch_width, patch_height, joints, joints_vis):
    joints[:, 0] = joints[:, 0] / patch_width - 0.5
    joints[:, 1] = joints[:, 1] / patch_height - 0.5
    joints[:, 2] = joints[:, 2] / patch_width

    joints = joints.reshape((-1))
    joints_vis = joints_vis.reshape((-1))
    return joints, joints_vis


def get_joint_regression_result(patch_width, patch_height, preds):
    coords = preds.detach().cpu().numpy()
    coords = coords.astype(float)
    # project to original image size
    coords[:, :, 0] = (coords[:, :, 0] + 0.5) * patch_width
    coords[:, :, 1] = (coords[:, :, 1] + 0.5) * patch_height
    coords[:, :, 2] = coords[:, :, 2] * patch_width
    scores = np.ones((coords.shape[0], coords.shape[1], 1), dtype=float)

    # add score to last dimension
    coords = np.concatenate((coords, scores), axis=2)

    return coords


def get_label_func():
    return generate_joint_location_label


def get_result_func():
    return get_joint_regression_result
