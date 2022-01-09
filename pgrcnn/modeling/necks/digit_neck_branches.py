import detectron2.config
from typing import List, Dict, Union
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import (
    Conv2d,
    ConvTranspose2d,
    ShapeSpec,
    get_norm,
    ModulatedDeformConv,
    interpolate,
    DepthwiseSeparableConv2d,
)
from detectron2.utils.registry import Registry
from detectron2.modeling.backbone.resnet import DeformBottleneckBlock
from detectron2.layers.wrappers import _NewEmptyTensorOp
from ..layers import PositionalEncoder, AttentionConv, CrissCrossAttention

ROI_DIGIT_NECK_BRANCHES_REGISTRY = Registry("ROI_DIGIT_NECK_BRANCHES")



@ROI_DIGIT_NECK_BRANCHES_REGISTRY.register()
class DigitNeckBranch(nn.Sequential):
    def __init__(self, cfg, input_shape):
        """

        """
        super().__init__()
        self.norm = cfg.MODEL.ROI_NECK_BASE_BRANCHES.NORM
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
            if self.up_scale > 1:
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
        cfg = cfg.MODEL.ROI_NECK_BASE_BRANCHES.PERSON_BRANCH
        self.up_scale = cfg.UP_SCALE
        self.deconv_kernel = cfg.DECONV_KERNEL
        conv_dims = cfg.CONV_DIMS
        in_channels = input_shape.channels
        for idx, layer_channels in enumerate(conv_dims, 1):
            module = Conv2d(in_channels, layer_channels, 3, stride=1, padding=1,
                            norm=get_norm(self.norm, layer_channels), activation=F.relu)
            self.add_module("conv_fcn{}".format(idx), module)
            in_channels = layer_channels
        if self.deconv_kernel > 3:
            # add a transpose conv
            module = ConvTranspose2d(
                in_channels, in_channels, self.deconv_kernel, stride=2, padding=self.deconv_kernel // 2 - 1
            )
            self.add_module("feat_upconv{}".format(1), module)
        self._output_size = (conv_dims[-1],
                             self._output_size[1] * self.up_scale * max(1, self.deconv_kernel // 2),
                             self._output_size[2] * self.up_scale * max(1, self.deconv_kernel // 2))




@ROI_DIGIT_NECK_BRANCHES_REGISTRY.register()
class KptsROIBranch(DigitNeckBranch):
    def __init__(self, cfg, input_shape):
        """


        """
        super().__init__(cfg, input_shape)
        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)
        cfg = cfg.MODEL.ROI_NECK_BASE_BRANCHES.KEYPOINTS_BRANCH
        self.add_pe = cfg.PE
        self.up_scale = cfg.UP_SCALE
        self.deconv_kernel = cfg.DECONV_KERNEL
        conv_dims = cfg.CONV_DIMS
        kernel_size, stride, padding = cfg.CONV_SPECS
        in_channels = input_shape.channels # should be 17
        # add pe first
        if self.add_pe:
            pe_dim = 32
            self.add_module("pe_encoder", PositionalEncoder(pe_dim, self._output_size[1], self._output_size[2]))
            in_channels = in_channels + pe_dim
            self._output_size = (in_channels, self._output_size[1], self._output_size[2])

        for idx, layer_channels in enumerate(conv_dims, 1):
            module = Conv2d(in_channels, layer_channels, kernel_size, stride=stride, padding=padding,
                            norm=get_norm(self.norm, layer_channels), activation=F.relu)
            self.add_module("conv_fcn{}".format(idx), module)
            in_channels = layer_channels
            out_height = out_width = (self._output_size[1] - kernel_size + 2 * padding) // stride + 1
            self._output_size = (in_channels, out_height, out_width)
        if self.deconv_kernel > 3:
            self.deconv = ConvTranspose2d(
                conv_dims[-1], conv_dims[-1], self.deconv_kernel, stride=2, padding=self.deconv_kernel // 2 - 1
            )
        else:
            self.deconv = None

        self._output_size = (conv_dims[-1],
                             self._output_size[1] * self.up_scale * max(1, self.deconv_kernel // 2),
                             self._output_size[2] * self.up_scale * max(1, self.deconv_kernel // 2))


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
            if self.up_scale > 1:
                x = interpolate(x, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
            return x
        return None

@ROI_DIGIT_NECK_BRANCHES_REGISTRY.register()
class KptsAttentionBranch(DigitNeckBranch):

    def __init__(self, cfg, input_shape):
        super(KptsAttentionBranch, self).__init__(cfg, input_shape)
        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)
        cfg = cfg.MODEL.ROI_NECK_BASE_BRANCHES.KEYPOINTS_BRANCH
        self.up_scale = cfg.UP_SCALE
        # do not use norm
        self.norm = "BN"
        self.only_four_kpts = True
        in_channels = 4 if self.only_four_kpts else self._output_size[0]
        out_channels = 64
        down_scale = 2
        # down by 4
        self.max_pool = nn.MaxPool2d(kernel_size=7, stride=4, padding=3)
        # self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        attention_type = CrissCrossAttention
        module = attention_type(in_channels, out_channels, norm=self.norm)
        self.add_module("atten_conv1", module)
        module = attention_type(out_channels, out_channels, norm=self.norm)
        self.add_module("atten_conv2", module)
        # self._output_size = (out_channels // 2, input_shape.height // 4, input_shape.width // 4)
        # module = Conv2d(out_channels // 2, 2, kernel_size=3, stride=1, padding=1,
        #                 norm=None, activation=torch.sigmoid)
        # self.add_module("output_conv", module)
        self._output_size = (out_channels, 28, 28)
        # self._output_size = (1,
        #                      self._output_size[1] * self.up_scale,
        #                      self._output_size[2] * self.up_scale)


    def forward(self, x):
        """

        Args:
            x: Tensor, shape of (N, 17, 56, 56)

        Returns:

        """
        if x is not None:
            if x.numel() == 0:
                return _NewEmptyTensorOp.apply(x, (0, ) + self._output_size)
            # only select the keypoint annotations we have
            if self.only_four_kpts:
                x = x[:, [5, 6, 12, 11], :, :]
            # downsample (N, 17, S, S)
            x = self.max_pool(x)
            # get the keypoint heatmap in the right way
            # x = get_gaussian_map(x)
            x = self.atten_conv1(x)
            x = self.atten_conv2(x)
            # x = self.output_conv(x)
            # x = self.softmax(x)
            x = interpolate(x, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
            return x
        return None
# @ROI_DIGIT_NECK_BRANCHES_REGISTRY.register()
# class KptsAttentionBranch(DigitNeckBranch):
#
#     def __init__(self, cfg, input_shape):
#         super(KptsAttentionBranch, self).__init__(cfg, input_shape)
#         self._output_size = (input_shape.channels, input_shape.height, input_shape.width)
#         cfg = cfg.MODEL.ROI_NECK_BASE_BRANCHES.KEYPOINTS_BRANCH
#         self.up_scale = cfg.UP_SCALE
#         self.deconv_kernel = 1 # cfg.DECONV_KERNEL
#         # do not use norm
#         self.norm = ""
#         in_channels = self._output_size[0]
#         out_channels = in_channels
#         stride = 2
#         # down by 4
#         module = Conv2d(in_channels, out_channels, 5, stride=stride, padding=2,
#                             norm=get_norm(self.norm, 64), activation=F.relu)
#         self.add_module("conv1", module)
#         self._output_size = (out_channels, input_shape.height // 2, input_shape.width // 2)
#         self._init_channel_attention()
#         self._init_spatial_attention()
#         if self.deconv_kernel > 3:
#             self.deconv = ConvTranspose2d(
#                 out_channels, 1, self.deconv_kernel, stride=2, padding=self.deconv_kernel // 2 - 1
#             )
#         else:
#             self.deconv = None
#         module = Conv2d(out_channels, 1, 1, stride=1, padding=0,
#                         norm=get_norm(self.norm, 1), activation=torch.sigmoid)
#         self.add_module("conv2", module)
#         self._output_size = (1,
#                              self._output_size[1] * self.up_scale * max(1, self.deconv_kernel // 2),
#                              self._output_size[2] * self.up_scale * max(1, self.deconv_kernel // 2))
#
#
#     def _init_spatial_attention(self):
#         modules = []
#         in_channels = self._output_size[0]
#         out_channels = in_channels
#         kernel_size = 5
#         padding = 2
#         dilation = 1
#         stride  = 1
#         # 1/2 spatial
#         module = Conv2d(in_channels, out_channels, 3, stride=2, padding=1,
#                norm=get_norm(self.norm, out_channels), activation=F.relu)
#         modules.append(module)
#         self._output_size = (out_channels, self._output_size[1] // 2, self._output_size[2] // 2)
#         in_channels = out_channels
#         module = DepthwiseSeparableConv2d(
#                         in_channels,
#                         out_channels,
#                         kernel_size=kernel_size,
#                         padding=padding,
#                         dilation=dilation,
#                         norm1=self.norm,
#                         activation1=F.relu,
#                         norm2=self.norm,
#                         activation2=torch.sigmoid,
#                     )
#         modules.append(module)
#         self.add_module("spatial_attn", nn.Sequential(*modules))
#         out_height = out_width = (self._output_size[1] + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
#         self._output_size = (out_channels, out_height, out_width)
#
#     def _init_channel_attention(self):
#         modules = []
#         in_channels = self._output_size[0]
#         out_channels = in_channels
#         # changed to maxpool
#         module = nn.AdaptiveMaxPool2d((1, 1))
#         modules.append(module)
#         module = Conv2d(in_channels, out_channels, 1, stride=1, padding=0,
#                norm=get_norm(self.norm, out_channels), activation=F.relu)
#         modules.append(module)
#         in_channels = out_channels
#         module = Conv2d(in_channels, out_channels, 1, stride=1, padding=0,
#                         norm=get_norm(self.norm, out_channels), activation=torch.sigmoid)
#         modules.append(module)
#         self.add_module("channel_attn", nn.Sequential(*modules))
#
#     def forward(self, x):
#         if x is not None:
#             x = self.conv1(x)
#             y = x
#             x = self.channel_attn(x)
#             y = self.spatial_attn(y)
#             x = torch.mul(x, y)
#             x = self.conv2(x)
#             # if self.deconv:
#             #     x = self.deconv(x)
#             #     x = torch.sigmoid(x)
#             x = interpolate(x, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
#             return x
#         return None
def get_gaussian_map(heatmaps, sigma=1):
    n, c, h, w = heatmaps.size()

    heatmaps_y = F.softmax(heatmaps.sum(dim=3), dim=2).reshape(n, c, h, 1)
    heatmaps_x = F.softmax(heatmaps.sum(dim=2), dim=2).reshape(n, c, 1, w)

    coord_y = heatmaps.new_tensor(range(h)).reshape(1, 1, h, 1)
    coord_x = heatmaps.new_tensor(range(w)).reshape(1, 1, 1, w)

    joints_y = heatmaps_y * coord_y
    joints_x = heatmaps_x * coord_x

    joints_y = joints_y.sum(dim=2)
    joints_x = joints_x.sum(dim=3)

    joints_y = joints_y.reshape(n, c, 1, 1)
    joints_x = joints_x.reshape(n, c, 1, 1)

    gaussian_map = torch.exp(-((coord_y - joints_y) ** 2 + (coord_x - joints_x) ** 2) / (2 * sigma ** 2))
    return gaussian_map

def build_digit_neck_branch(name, cfg, input_shapes):
    """
    Build a feature transformation head defined by `cfg.MODEL.ROI_DIGIT_NECK_BRANCHES`.
    """
    if name == "" or name is None:
        return None
    return ROI_DIGIT_NECK_BRANCHES_REGISTRY.get(name)(cfg, input_shapes)