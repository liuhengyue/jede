import numpy as np
from typing import List, Dict
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, Linear, ShapeSpec, get_norm, ModulatedDeformConv
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
    def __init__(self,
                 transform_dim: int,
                 num_proposal: int,
                 input_shapes: Dict[str, ShapeSpec],
                 *,
                 conv_dims: List[int],
                 fc_dims: List[int],
                 use_deform: bool,
                 conv_norm="",
                 use_person_box_features=True,
                 num_interests=1,
                 focal_bias=0.01
                 ):
        """

                Args:
                    input_shape (ShapeSpec): shape of the input feature
                    conv_dims: an iterable of output channel counts for each conv in the head
                                 e.g. (512, 512, 512) for three convs outputting 512 channels.
                For this head, it performs a sequence of convs on person bbox features and kepoint features,
                to get digit proposals (bboxes).
                """
        super().__init__()
        assert len(conv_dims) + len(fc_dims) > 0
        keypoint_heatmap_shape = input_shapes["keypoint_heatmap_shape"]
        person_box_features_shape = input_shapes["person_box_features_shape"]
        self._output_size = (keypoint_heatmap_shape.channels, keypoint_heatmap_shape.height, keypoint_heatmap_shape.width)
        self.use_deform = use_deform
        # modules used for keypoint convs
        self.conv_norm_relus = []
        # modules used for person features
        self.person_feat_layers = []
        # final detection heads
        self.ct_head = []
        self.size_head = []
        self.num_proposal = num_proposal
        self.transform_dim = transform_dim
        self.use_person_box_features = use_person_box_features
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

        if self.use_person_box_features:
            conv_dims = [64] * 2
            self._init_layers_with_person_features(conv_dims, conv_norm, person_box_features_shape)
        else:
            self._init_plain_layers(conv_dims, conv_norm)
        conv_dims = [64] * 4
        self._init_output_layers(conv_norm, conv_dims, num_interests)

        # for layer in self.conv_norm_relus:
        #     if type(layer).__name__ != 'DeformBottleneckBlock': #  already init in the block
        #         weight_init.c2_msra_fill(layer)
        # for layer in self.heads:
        #     weight_init.c2_msra_fill(layer)

        # or another initialization
        for name, param in self.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                # Caffe2 implementation uses MSRAFill, which in fact
                # corresponds to kaiming_normal_ in PyTorch
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")

        # for output layer, init with CenterNet params
        self.center_heatmaps.bias.data.fill_(focal_bias)
        self.size_heatmaps.bias.data.fill_(0)
        for m in [self.center_heatmaps, self.size_heatmaps]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)


    def _init_plain_layers(self, conv_dims, conv_norm):
        if self.use_deform:
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

    def _init_layers_with_person_features(self, conv_dims, conv_norm, person_box_features_shape):
        if self.use_deform:
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
                self.add_module("kpt_deform_conv{}".format(k + 1), conv)
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
                self.add_module("kpt_conv{}".format(k + 1), conv)
                self.conv_norm_relus.append(conv)
                self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        # for box person feature transformations
        # default up_scale to 2.0 (this can be made an option)
        up_scale = 2.0
        in_channels = person_box_features_shape.channels

        for idx, conv_dim in enumerate(conv_dims, 1):
            module = Conv2d(in_channels,
                            conv_dim,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=not conv_norm,
                            norm=get_norm(conv_norm, conv_dim),
                            activation=F.relu,
                            )
            self.add_module("feat_conv{}".format(idx), module)
            in_channels = conv_dim
            self.person_feat_layers.append(module)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        deconv_kernel = 4
        module = ConvTranspose2d(
            in_channels, conv_dim, deconv_kernel, stride=2, padding=deconv_kernel // 2 - 1
        )
        self.add_module("feat_upconv{}".format(1), module)
        self.up_scale = up_scale
        self.person_feat_layers.append(module)
        # we will interpolate by 2 during forward so the spatial dim remains the same
        self._output_size = (conv_dim * 2, self._output_size[1], self._output_size[2])


    def _init_output_layers(self, conv_norm, conv_dims, num_interests=1):
        # need a few convs
        in_channels = self._output_size[0]
        for idx, conv_dim in enumerate(conv_dims, 1):
            module = Conv2d(self._output_size[0],
                            conv_dim,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=not conv_norm,
                            norm=get_norm(conv_norm, conv_dim),
                            activation=F.relu,
                            )
            self.add_module("ct_conv{}".format(idx), module)
            self.ct_head.append(module)
            self._output_size = (conv_dim, self._output_size[1], self._output_size[2])
        # add center and scale heads
        # 3 cls: left digit, center digit, right digit
        head = Conv2d(
            self._output_size[0],
            num_interests,
            kernel_size=1,
            bias=True,
            norm=None,
            activation=None, # need logits for sigmoid
        )
        self.add_module("center_heatmaps", head)
        self.ct_head.append(head)
        # size head
        for idx, conv_dim in enumerate(conv_dims, 1):
            module = Conv2d(in_channels,
                            conv_dim,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=not conv_norm,
                            norm=get_norm(conv_norm, conv_dim),
                            activation=F.relu,
                            )
            self.add_module("size_conv{}".format(idx), module)
            self.size_head.append(module)
            in_channels = conv_dim

        head = Conv2d(
            in_channels,
            2,
            kernel_size=1,
            bias=True,
            norm=None,
            activation=None,
        )
        self.add_module("size_heatmaps", head)
        self.size_head.append(head)

    def forward(self, kpts_features, box_features):
        for layer in self.conv_norm_relus:
            kpts_features = layer(kpts_features)
        if self.use_person_box_features:
            for layer in self.person_feat_layers:
                box_features = layer(box_features)
            box_features = F.interpolate(box_features, scale_factor=self.up_scale, mode="bilinear", align_corners=False)
            # merge the features
            kpts_features = torch.cat((kpts_features, box_features), dim=1)
        #  kpts_features will be feed into two heads
        x = kpts_features
        y = kpts_features
        for layer in self.ct_head:
            x = layer(x)
        for layer in self.size_head:
            y = layer(y)
        # center heatmaps, and size heatmaps
        return [x, y]

    @classmethod
    def from_config(cls, cfg, input_shapes):
        num_conv = cfg.MODEL.ROI_DIGIT_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_DIGIT_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_DIGIT_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_DIGIT_HEAD.FC_DIM


        return {
            "transform_dim": cfg.MODEL.ROI_DIGIT_HEAD.TRANSFORM_DIM,
            "input_shapes": input_shapes,
            "conv_dims": [conv_dim] * num_conv,
            "fc_dims": [fc_dim] * num_fc,
            "conv_norm": cfg.MODEL.ROI_DIGIT_HEAD.NORM,
            "num_proposal": cfg.MODEL.ROI_DIGIT_HEAD.NUM_PROPOSAL,
            "use_deform": cfg.MODEL.ROI_DIGIT_HEAD.DEFORMABLE,
            "use_person_box_features": cfg.MODEL.ROI_DIGIT_HEAD.USE_PERSON_BOX_FEATURES,
            "num_interests": cfg.DATASETS.NUM_INTERESTS,
            "focal_bias": cfg.MODEL.ROI_DIGIT_HEAD.FOCAL_BIAS
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

def build_digit_head(cfg, input_shapes):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_DIGIT_HEAD.NAME
    return ROI_DIGIT_HEAD_REGISTRY.get(name)(cfg, input_shapes)