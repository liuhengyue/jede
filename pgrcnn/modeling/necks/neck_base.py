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
from ..layers import CoordAtt, DualAttention, PositionalEncoder

from .digit_neck_branches import build_digit_neck_branch
ROI_NECK_BASE_REGISTRY = Registry("ROI_NECK_BASE")


@ROI_NECK_BASE_REGISTRY.register()
class NeckBase(nn.Module):
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
        cfg_roi_neck_base = cfg.MODEL.ROI_NECK_BASE
        cfg_roi_neck_base_branches = cfg.MODEL.ROI_NECK_BASE_BRANCHES
        self.use_person_features = cfg_roi_neck_base.USE_PERSON_BOX_FEATURES
        self.use_kpts_features = cfg_roi_neck_base.USE_KEYPOINTS_FEATURES
        self.add_pe = cfg_roi_neck_base.PE

        if self.use_person_features:
            module = build_digit_neck_branch(cfg_roi_neck_base_branches.PERSON_BRANCH.NAME, cfg,
                                             input_shapes["person_box_features_shape"])
            self.add_module("person_branch", module)

        if self.use_kpts_features:
            module = build_digit_neck_branch(cfg_roi_neck_base_branches.KEYPOINTS_BRANCH.NAME, cfg,
                                             input_shapes["keypoint_heatmap_shape"])
            self.add_module("kpts_branch", module)

        self.fusion_type = cfg_roi_neck_base.FUSION_TYPE

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
        elif self.fusion_type == "multiply":
            assert person_box_features_shape == keypoint_heatmap_shape
            in_channels = keypoint_heatmap_shape.channels
        else: # only single branch
            if self.use_person_features and (not self.use_kpts_features):
                assert person_box_features_shape.height == keypoint_heatmap_shape.height
                assert person_box_features_shape.width == keypoint_heatmap_shape.width
                in_channels = person_box_features_shape.channels
            elif (not self.use_person_features) and self.use_kpts_features:
                in_channels = keypoint_heatmap_shape.channels
            else:
                raise NotImplementedError("Wrong combinations of FUSION_TYPE / USE_KEYPOINTS_FEATURES / USE_PERSON_BOX_FEATURES.")
        self._output_size = (in_channels, keypoint_heatmap_shape.height, keypoint_heatmap_shape.width)
        # check if any attn is used
        attn_name = cfg_roi_neck_base.ATTN
        self.attn_on = False
        # keep the same channel dim
        if attn_name == "CoordAtt":
            self.attn = CoordAtt(in_channels, in_channels)
            self.attn_on = True
        elif attn_name == "DualAttention":
            self.attn = DualAttention(in_channels)
            self.attn_on = True
        if self.add_pe:
            # double for now
            self.pe_encoder = PositionalEncoder(in_channels, self._output_size[1], self._output_size[2])
            self._output_size = (in_channels + in_channels, self._output_size[1], self._output_size[2])

        self._init_weights()

    @property
    @torch.jit.unused
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

    def forward(self, kpts_features, person_features):
        if self.use_kpts_features:
            kpts_features = self.kpts_branch(kpts_features)
        if self.use_person_features:
            person_features = self.person_branch(person_features)
        # merge the features
        if self.fusion_type == "cat":
            x = torch.cat((kpts_features, person_features), dim=1)
        elif self.fusion_type == "sum":
            x = torch.add(kpts_features, person_features)
        elif self.fusion_type == "multiply":
            x = kpts_features * person_features
        else:
            x = kpts_features if self.use_kpts_features else person_features
        if self.attn_on:
            x = self.attn(x)
        if self.add_pe:
            x = self.pe_encoder(x)
        #  x will be feed into different prediction heads
        return x


def build_neck_base(cfg, input_shapes):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_NECK_BASE.NAME
    return ROI_NECK_BASE_REGISTRY.get(name)(cfg, input_shapes)