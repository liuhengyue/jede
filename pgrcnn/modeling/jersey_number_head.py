import detectron2.config
import numpy as np
from typing import List, Dict
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, Linear, ShapeSpec, get_norm, ModulatedDeformConv
from detectron2.layers.wrappers import _NewEmptyTensorOp
from detectron2.utils.registry import Registry
from detectron2.modeling.backbone.resnet import DeformBottleneckBlock

from .digit_neck_branches import build_digit_neck_branch
ROI_JERSEY_NUMBER_DET_REGISTRY = Registry("ROI_JERSEY_NUMBER_DET")


@ROI_JERSEY_NUMBER_DET_REGISTRY.register()
class NumDigitClassification(nn.Module):
    def __init__(self,
                 cfg: detectron2.config.CfgNode,
                 input_shapes: ShapeSpec
                 ):
        """

        """
        super().__init__()
        in_channels = input_shapes.channels * 1 * input_shapes.width
        self.out_channels = 3  # 0 digit, 1 digit, 2 digit
        self.pool = nn.AdaptiveMaxPool2d((1, input_shapes.width))
        self.linears = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.Linear(128, 128),
            nn.Linear(128, self.out_channels)
        )
        for m in self.linears.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)



    def forward(self, x):
        if x.numel() == 0:
            # AdaptiveMaxPool2d does not take empty tensor
            return _NewEmptyTensorOp.apply(x, (0, self.out_channels))
        x = self.pool(x)
        x = self.linears(x)
        return x

@ROI_JERSEY_NUMBER_DET_REGISTRY.register()
class NumDigitClassification(nn.Module):
    def __init__(self,
                 cfg: detectron2.config.CfgNode,
                 input_shapes: ShapeSpec
                 ):
        """

        """
        super().__init__()
        in_channels = input_shapes.channels * 1 * input_shapes.width
        self.out_channels = 3  # 0 digit, 1 digit, 2 digit
        self.pool = nn.AdaptiveMaxPool2d((1, input_shapes.width))
        self.linears = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.Linear(128, 128),
            nn.Linear(128, self.out_channels)
        )
        for m in self.linears.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)



    def forward(self, x):
        if x.numel() == 0:
            # AdaptiveMaxPool2d does not take empty tensor
            return _NewEmptyTensorOp.apply(x, (0, self.out_channels))
        x = self.pool(x)
        x = self.linears(x)
        return x


def build_jersey_number_head(cfg, input_shapes):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_JERSEY_NUMBER_DET.NAME
    return ROI_JERSEY_NUMBER_DET_REGISTRY.get(name)(cfg, input_shapes)