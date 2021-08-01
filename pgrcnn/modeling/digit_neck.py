import detectron2.config
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

from .digit_neck_branches import build_digit_neck_branch
ROI_DIGIT_NECK_REGISTRY = Registry("ROI_DIGIT_NECK")


@ROI_DIGIT_NECK_REGISTRY.register()
class DigitNeck(nn.Module):
    def __init__(self,
                 cfg: detectron2.config.CfgNode,
                 input_shapes: Dict[str, ShapeSpec]
                 ):
        """

                Args:
                    input_shape (ShapeSpec): shape of the input feature
                    conv_dims: an iterable of output channel counts for each conv in the head
                                 e.g. (512, 512, 512) for three convs outputting 512 channels.
                For this head, it performs a sequence of convs on person bbox features and kepoint features,
                to get digit proposals (bboxes).

                Network flow:
                                            convsx2              convT              interpolate
                person_features: (NxKx14x14) -------> Nx64x14x14 -------> Nx64x28x28 -----------> Nx64x56x56
                   (or None)

                                            convs2
                kpts_features: (Nx17x56x56)  -------> Nx64x56x56
                   (or None)

                cat(kpts_features, person_features) ----> output_heads


                """
        super().__init__()
        cfg_roi_digit_neck = cfg.MODEL.ROI_DIGIT_NECK
        self.conv_norm = cfg_roi_digit_neck.NORM
        self.output_head_names = cfg_roi_digit_neck.OUTPUT_HEAD_NAMES
        self.output_head_channels = cfg_roi_digit_neck.OUTPUT_HEAD_CHANNELS
        conv_dims = [cfg_roi_digit_neck.CONV_DIM] * cfg_roi_digit_neck.NUM_CONV
        assert len(conv_dims) > 0
        self.use_deform = cfg_roi_digit_neck.DEFORMABLE

        self.focal_bias = cfg_roi_digit_neck.FOCAL_BIAS
        activations = {name: activation for name, activation in zip(self.output_head_names, (torch.sigmoid, F.relu, torch.sigmoid))}
        # activations = {name: activation for name, activation in
        #                zip(self.output_head_names, (torch.sigmoid, None, None))}
        cfg_roi_digit_neck_branches = cfg.MODEL.ROI_DIGIT_NECK_BRANCHES
        module = build_digit_neck_branch(cfg_roi_digit_neck_branches.PERSON_BRANCH.NAME, cfg, input_shapes["person_box_features_shape"])
        if module:
            self.add_module("person_branch", module)
        module = build_digit_neck_branch(cfg_roi_digit_neck_branches.KEYPOINTS_BRANCH.NAME, cfg, input_shapes["keypoint_heatmap_shape"])
        if module:
            self.add_module("kpts_branch", module)
        self.use_person_features = hasattr(self, "person_branch")
        self.use_kpts_features = hasattr(self, "kpts_branch")
        self.fusion_type = cfg_roi_digit_neck.FUSION_TYPE

        keypoint_heatmap_shape = input_shapes["keypoint_heatmap_shape"]
        person_box_features_shape = input_shapes["keypoint_heatmap_shape"]
        if self.use_person_features:
            person_box_features_shape = self.person_branch.output_shape
        if self.use_kpts_features:
            keypoint_heatmap_shape = self.kpts_branch.output_shape
        if self.fusion_type == "cat":
            assert person_box_features_shape.height == keypoint_heatmap_shape.height
            assert person_box_features_shape.width == keypoint_heatmap_shape.width
            in_channels = keypoint_heatmap_shape.channels + person_box_features_shape.channels
        elif self.fusion_type == "sum":
            assert person_box_features_shape == keypoint_heatmap_shape
            in_channels = keypoint_heatmap_shape.channels
        for name, out_channels in zip(self.output_head_names, self.output_head_channels):
            activation = activations[name]
            module = self._init_output_layers(conv_dims, in_channels, out_channels, activation=activation)
            self.add_module(name, module)

        self.offset_reg = hasattr(self, "offset")
        assert self.use_person_features or self.use_kpts_features, "One of them has to be True."
        self._init_weights()


    def _init_weights(self):
        def normal_init(module, mean=0, std=1, bias=0):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.normal_(module.weight, mean, std)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)
        def kaiming_normal_init(module, mode="fan_out", nonlinearity="relu", bias=0):
            if hasattr(module, 'weight') and module.weight is not None:
                nn.init.kaiming_normal_(module.weight, mode=mode, nonlinearity=nonlinearity)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, bias)

        for m in self.modules():
            if isinstance(m, Conv2d):
                kaiming_normal_init(m)

        # for output layer, init with CenterNet params
        if hasattr(self, "center"):
            # for m in self.center.modules():
            #     if isinstance(m, Conv2d):
            #         normal_init(m, std=0.001)
            nn.init.normal_(self.center[-1].weight, std=0.001)
            self.center[-1].bias.data.fill_(self.focal_bias)
        if hasattr(self, "size"):
            # for m in self.size.modules():
            #     if isinstance(m, Conv2d):
            #         normal_init(m, std=0.001)
            nn.init.normal_(self.size[-1].weight, std=0.001)
            self.size[-1].bias.data.fill_(0.)
        if hasattr(self, "offset"):
            # for m in self.offset.modules():
            #     if isinstance(m, Conv2d):
            #         normal_init(m, std=0.001)
            nn.init.normal_(self.offset[-1].weight, std=0.001)
            self.offset[-1].bias.data.fill_(0.)


    def _init_output_layers(self, conv_dims, in_channels, output_channels, activation=None):
        ### add center, scale, and offset heads ###
        modules = []
        # The conv dims currently are same
        conv_dims = [in_channels] + conv_dims
        for idx in range(1, len(conv_dims)):
            module = Conv2d(conv_dims[idx-1],
                            conv_dims[idx],
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=not self.conv_norm,
                            norm=get_norm(self.conv_norm, conv_dims[idx]),
                            activation=F.relu,
                            )
            modules.append(module)

        # 3 cls: left digit, center digit, right digit
        head = Conv2d(
            conv_dims[-1],
            output_channels,
            kernel_size=1,
            bias=True,
            norm=None,
            activation=activation, # need logits for sigmoid
        )
        modules.append(head)
        return nn.Sequential(*modules)

    def forward(self, kpts_features, person_features):
        if self.use_kpts_features:
            kpts_features = self.kpts_branch(kpts_features)
        if self.use_person_features:
            person_features = self.person_branch(person_features)
        # merge the features
        if self.fusion_type == "cat":
            kpts_features = torch.cat((kpts_features, person_features), dim=1)
        elif self.fusion_type == "sum":
            kpts_features = torch.add(kpts_features, person_features)
        #  kpts_features will be feed into two heads
        x = kpts_features
        y = kpts_features
        x = self.center(x)
        y = self.size(y)
        if self.offset_reg:
            z = kpts_features
            z = self.offset(z)
        else:
            z = None
        # center, size, offset heatmaps
        return [x, y, z]


def build_digit_neck(cfg, input_shapes):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_DIGIT_NECK.NAME
    return ROI_DIGIT_NECK_REGISTRY.get(name)(cfg, input_shapes)