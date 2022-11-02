import logging
import re

import torch
import torch.nn as nn

import torchvision.models
from torch.hub import load_state_dict_from_url
from torchvision.models.resnet import model_urls, BasicBlock, Bottleneck

logger = logging.getLogger(__name__)

block_layers = {"18": [BasicBlock, [2, 2, 2, 2]],
                "34": [BasicBlock, [3, 4, 6, 3]],
                "50": [Bottleneck, [3, 4, 6, 3]],
                "101": [Bottleneck, [3, 4, 23, 3]],
                "152": [Bottleneck, [3, 8, 36, 3]],
                }


class CustomResNet(torchvision.models.ResNet):
    def __init__(self, model_backbone):
        kwargs = {}
        size = [s for s in re.findall(r'\d+', model_backbone)][0]  # first number
        b_l = block_layers[size]
        kwargs['block'] = b_l[0]
        kwargs['layers'] = b_l[1]
        if model_backbone.startswith('resnext'):
            kwargs['groups'] = 32
            if model_backbone.startswith('resnext50'):
                kwargs['width_per_group'] = 4
            else:
                kwargs['width_per_group'] = 8
        if model_backbone.startswith('wide_resnet'):
            kwargs['width_per_group'] = 64 * 2
        super(CustomResNet, self).__init__(**kwargs)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class PoseResNetVol(nn.Module):

    def __init__(self, cfg, is_pretrain, n_frozen_layers=7):
        super(PoseResNetVol, self).__init__()
        self.volume_config = cfg.MODEL.VOLUME
        self.num_joints = cfg.MODEL.NUM_JOINTS
        self.depth_res = cfg.MODEL.DEPTH_RES
        self.n_frozen_layers = n_frozen_layers
        self._make_backbone(is_pretrain, cfg.MODEL.BACKBONE)
        self._make_head(self.num_joints, self.depth_res)
        self._softmax = torch.nn.Softmax(dim=2)
        self._range = torch.nn.Parameter(torch.arange(1, cfg.MODEL.DEPTH_RES + 1), requires_grad=False)  # needed for old model loading

        if cfg.MODEL.DEPTH_RES == 72:
            self._rangexz = torch.nn.Parameter(torch.arange(72), requires_grad=False)
            self._rangey = torch.nn.Parameter(torch.arange(96), requires_grad=False)
        elif cfg.MODEL.DEPTH_RES == 64:
            self._rangexz = self._range
            self._rangey = self._range
        else:
            raise NotImplementedError('Set correct depth res')

    def _make_backbone(self, is_pretrain, model_backbone):
        self._backbone = CustomResNet(model_backbone)
        # delete head of backbone, because we create our own
        del self._backbone.avgpool
        del self._backbone.fc

        if is_pretrain:
            logger.info(f'=> loading {model_backbone} weights')
            state_dict = load_state_dict_from_url(model_urls[model_backbone],
                                                  model_dir='lib/models/pretrained_data',  # /checkpoints',
                                                  file_name=f'{model_backbone}_original.pth',
                                                  progress=True)
            self._backbone.load_state_dict(state_dict, strict=False)

            logger.info(f'=> freezing {model_backbone} first {self.n_frozen_layers} layers')
            for i, child in enumerate(list(self._backbone.children())[:self.n_frozen_layers]):
                for param in child.parameters():
                    param.requires_grad = False

    def _make_head(self, num_joints, depth_res):
        layers = []
        kernel, padding, output_padding = [4, 1, 0]
        filters = self.volume_config.NUM_DECONV_FILTERS
        output_size = 2048
        for i in range(self.volume_config.NUM_DECONV_LAYERS):
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=output_size,
                    out_channels=filters,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.volume_config.DECONV_WITH_BIAS))
            layers.append(nn.BatchNorm2d(filters, momentum=self.volume_config.BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            output_size = filters

        self._head = nn.Sequential(*layers, nn.Conv2d(
            in_channels=filters,
            out_channels=num_joints * depth_res,
            kernel_size=self.volume_config.FINAL_CONV_KERNEL,
            stride=1,
            padding=0
        ))

    def forward(self, x):
        cnn_features = self._backbone.forward(x)
        heatmaps = self._head(cnn_features)
        pred_jnts = self._softmax_integral(heatmaps)
        return pred_jnts

    # FROM https://github.com/JimmySuen/integral-human-pose/blob/master/pytorch_projects/common_pytorch/common_loss/integral.py
    def _softmax_integral(self, heatmaps):
        # global softmax
        depth, height, width = heatmaps.shape[1:]
        depth = depth // self.num_joints
        heatmaps = heatmaps.reshape((heatmaps.shape[0], self.num_joints, -1))
        heatmaps = self._softmax(heatmaps)

        # integrate heatmap into joint location
        heatmaps = heatmaps.reshape((heatmaps.shape[0], self.num_joints, depth, height, width))

        x = heatmaps.sum(dim=2)
        x = x.sum(dim=2)
        x = x * self._rangexz
        x = x.sum(dim=2, keepdim=True)
        x = x / float(width) - 0.5

        y = heatmaps.sum(dim=2)
        y = y.sum(dim=3)
        y = y * self._rangey
        y = y.sum(dim=2, keepdim=True)
        y = y / float(height) - 0.5

        if self.depth_res != 1:
            z = heatmaps.sum(dim=3)
            z = z.sum(dim=3)
            z = z * self._rangexz
            z = z.sum(dim=2, keepdim=True)
            z = z / float(depth) - 0.5
        else:
            z = torch.zeros(x.shape, requires_grad=False).to(x.device)

        return torch.cat((x, y, z), dim=2)


def get_pose_net(cfg, is_pretrain, **kwargs):
    model = PoseResNetVol(cfg, is_pretrain, **kwargs)
    return model
