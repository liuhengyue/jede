import detectron2.config
from typing import List, Dict, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, Linear, ShapeSpec, get_norm, ModulatedDeformConv, interpolate
from detectron2.utils.registry import Registry
from detectron2.modeling.backbone.resnet import DeformBottleneckBlock

ROI_DIGIT_NECK_BRANCHES_REGISTRY = Registry("ROI_DIGIT_NECK_BRANCHES")



@ROI_DIGIT_NECK_BRANCHES_REGISTRY.register()
class DigitNeckBranch(nn.Sequential):
    def __init__(self, cfg, input_shape):
        """

        """
        super().__init__()
        self.norm = cfg.MODEL.ROI_DIGIT_NECK_BRANCHES.NORM
        # should be changed for children class
        self.up_scale = 1


    def forward(self, x: Union[torch.Tensor, None]):
        """
        Args:
            x: input 4D region feature(s) provided by :class:`ROIHeads`.

        Returns:
            Output features.
        """
        if x is not None:
            for layer in self:
                x = layer(x)
            x = interpolate(x, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
            return x
        return None


    @property
    def output_shape(self):
        """
        Returns:
            ShapeSpec: the output feature shape
        """
        o = self._output_size
        return ShapeSpec(channels=o[0], height=o[1], width=o[2])


@ROI_DIGIT_NECK_BRANCHES_REGISTRY.register()
class PersonROIBranch(DigitNeckBranch):
    def __init__(self, cfg, input_shape):
        """


        """
        super().__init__(cfg, input_shape)
        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)
        cfg = cfg.MODEL.ROI_DIGIT_NECK_BRANCHES.PERSON_BRANCH
        self.up_scale = cfg.UP_SCALE
        deconv_kernel = 4
        conv_dims = cfg.CONV_DIMS
        in_channels = input_shape.channels
        for idx, layer_channels in enumerate(conv_dims, 1):
            module = Conv2d(in_channels, layer_channels, 3, stride=1, padding=1,
                            norm=get_norm(self.norm, layer_channels), activation=F.relu)
            self.add_module("conv_fcn{}".format(idx), module)
            in_channels = layer_channels
        self._output_size = (in_channels,
                             self._output_size[1] * self.up_scale * deconv_kernel // 2,
                             self._output_size[2] * self.up_scale * deconv_kernel // 2)
        # add a transpose conv
        module = ConvTranspose2d(
            in_channels, in_channels, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1
        )
        self.add_module("feat_upconv{}".format(1), module)

@ROI_DIGIT_NECK_BRANCHES_REGISTRY.register()
class KptsROIBranch(DigitNeckBranch):
    def __init__(self, cfg, input_shape):
        """


        """
        super().__init__(cfg, input_shape)
        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)
        cfg = cfg.MODEL.ROI_DIGIT_NECK_BRANCHES.KEYPOINTS_BRANCH
        self.up_scale = cfg.UP_SCALE
        conv_dims = cfg.CONV_DIMS
        kernel_size, stride, padding = cfg.CONV_SPECS
        in_channels = input_shape.channels
        for idx, layer_channels in enumerate(conv_dims, 1):
            module = Conv2d(in_channels, layer_channels, kernel_size, stride=stride, padding=padding,
                            norm=get_norm(self.norm, layer_channels), activation=F.relu)
            self.add_module("conv_fcn{}".format(idx), module)
            in_channels = layer_channels
        self._output_size = (in_channels, self._output_size[1] * self.up_scale, self._output_size[2] * self.up_scale)


def build_digit_neck_branch(name, cfg, input_shapes):
    """
    Build a feature transformation head defined by `cfg.MODEL.ROI_DIGIT_NECK_BRANCHES`.
    """
    if name == "":
        return None
    return ROI_DIGIT_NECK_BRANCHES_REGISTRY.get(name)(cfg, input_shapes)