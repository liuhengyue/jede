import numpy as np
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, Linear, ShapeSpec, get_norm, ModulatedDeformConv
from detectron2.utils.registry import Registry
from detectron2.modeling.backbone.resnet import DeformBottleneckBlock

ROI_DIGIT_HEAD_REGISTRY = Registry("ROI_DIGIT_HEAD")

@ROI_DIGIT_HEAD_REGISTRY.register()
class Kpts2MatHead(nn.Module):
    @configurable
    def __init__(self, transform_dim: int, num_proposal: int, input_shape: ShapeSpec, *,
                 conv_dims: List[int], fc_dims: List[int], conv_norm=""):
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []

        self.num_proposal = num_proposal
        self.transform_dim = transform_dim

        for k, conv_dim in enumerate(conv_dims):
            conv = Conv2d(
                self._output_size[0],
                conv_dim,
                kernel_size=3,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, conv_dim),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            fc = Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        # add final regression layer to 3x3 matrix
        output_mat_fc = Linear(np.prod(self._output_size), num_proposal * transform_dim)
        weight_init.c2_xavier_fill(output_mat_fc)
        self.add_module("fc_perspective_pars", output_mat_fc)
        self._output_size = transform_dim
        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))
        x = self.fc_perspective_pars(x) # no activation since the values are not bounded
        x = x.view(-1, self.num_proposal, self.transform_dim)
        return x

    @classmethod
    def from_config(cls, cfg, input_shape):
        num_conv = cfg.MODEL.ROI_DIGIT_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_DIGIT_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_DIGIT_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_DIGIT_HEAD.FC_DIM

        return {
            "transform_dim": cfg.MODEL.ROI_DIGIT_HEAD.TRANSFORM_DIM,
            "input_shape": input_shape,
            "conv_dims": [conv_dim] * num_conv,
            "fc_dims": [fc_dim] * num_fc,
            "conv_norm": cfg.MODEL.ROI_DIGIT_HEAD.NORM,
            "num_proposal": cfg.MODEL.ROI_DIGIT_HEAD.NUM_PROPOSAL,
            "use_deform": cfg.MODEL.ROI_DIGIT_HEAD.DEFORMABLE
        }

    @property
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])


@ROI_DIGIT_HEAD_REGISTRY.register()
class Kpts2DigitHead(nn.Module):
    @configurable
    def __init__(self, transform_dim: int, num_proposal: int, input_shape: ShapeSpec, *,
                 conv_dims: List[int], fc_dims: List[int], use_deform, conv_norm=""):
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        self.heads = []

        self.num_proposal = num_proposal
        self.transform_dim = transform_dim

        # 1x1 conv to compress K keypoint maps
        compress = False
        if compress:
            conv1 = Conv2d(self._output_size[0], 1, kernel_size=1,
                           padding=0,
                           bias=not conv_norm,
                           norm=get_norm(conv_norm, 1),
                           activation=F.relu,
                           )

            self.add_module("conv_1x1", conv1)
            self.conv_norm_relus.append(conv1)
            self._output_size = (1, self._output_size[1], self._output_size[2])

        # normal conv or deformableConv v2
        conv_op = DeformBottleneckBlock if use_deform else Conv2d

        if use_deform:
            num_groups = 1
            width_per_group = 64
            bottleneck_channels = num_groups * width_per_group
            for k, conv_dim in enumerate(conv_dims):
                conv = DeformBottleneckBlock(
                    in_channels = self._output_size[0],
                    out_channels = conv_dim,
                    stride = 1,
                    bottleneck_channels = bottleneck_channels,
                    stride_in_1x1 = True,
                    deform_modulated = True,
                    norm = ''
                )
                self.add_module("deform_conv{}".format(k + 1), conv)
                self.conv_norm_relus.append(conv)
                self._output_size = (conv_dim, self._output_size[1], self._output_size[2])
        else:
            for k, conv_dim in enumerate(conv_dims):
                conv = Conv2d(
                    self._output_size[0],
                    conv_dim,
                    kernel_size=3,
                    padding=1,
                    bias=not conv_norm,
                    norm=get_norm(conv_norm, conv_dim),
                    activation=F.relu,
                )
                self.add_module("conv{}".format(k + 1), conv)
                self.conv_norm_relus.append(conv)
                self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k, fc_dim in enumerate(fc_dims):
            fc = Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        # add center and scale heads
        # 3 cls: left digit, center digit, right digit
        ct_head = Conv2d(
                self._output_size[0],
                3,
                kernel_size=3,
                padding=1,
                bias=not conv_norm,
                norm=get_norm(conv_norm, 3),
                activation=None,
            )
        self.add_module("ct_head", ct_head)
        self.heads.append(ct_head)

        scale_head = Conv2d(
            self._output_size[0],
            2,
            kernel_size=3,
            padding=1,
            bias=not conv_norm,
            norm=get_norm(conv_norm, 2),
            activation=F.relu,
        )
        self.add_module("scale_head", scale_head)
        self.heads.append(scale_head)

        for layer in self.conv_norm_relus:
            if type(layer).__name__ != 'DeformBottleneckBlock': #  already init in the block
                weight_init.c2_msra_fill(layer)
        for layer in self.heads:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        for layer in self.conv_norm_relus:
            x = layer(x)
        if len(self.fcs):
            if x.dim() > 2:
                x = torch.flatten(x, start_dim=1)
            for layer in self.fcs:
                x = F.relu(layer(x))
        res = [head(x) for head in self.heads]

        return res

    @classmethod
    def from_config(cls, cfg, input_shape):
        num_conv = cfg.MODEL.ROI_DIGIT_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_DIGIT_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_DIGIT_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_DIGIT_HEAD.FC_DIM

        return {
            "transform_dim": cfg.MODEL.ROI_DIGIT_HEAD.TRANSFORM_DIM,
            "input_shape": input_shape,
            "conv_dims": [conv_dim] * num_conv,
            "fc_dims": [fc_dim] * num_fc,
            "conv_norm": cfg.MODEL.ROI_DIGIT_HEAD.NORM,
            "num_proposal": cfg.MODEL.ROI_DIGIT_HEAD.NUM_PROPOSAL,
            "use_deform": cfg.MODEL.ROI_DIGIT_HEAD.DEFORMABLE
        }

    @property
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        if isinstance(o, int):
            return ShapeSpec(channels=o)
        else:
            return ShapeSpec(channels=o[0], height=o[1], width=o[2])

def build_digit_head(cfg, input_shape):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_DIGIT_HEAD.NAME
    return ROI_DIGIT_HEAD_REGISTRY.get(name)(cfg, input_shape)