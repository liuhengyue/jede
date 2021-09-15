# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
from typing import List
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.utils.registry import Registry
from detectron2.modeling.roi_heads import FastRCNNConvFCHead
__all__ = ["build_digit_box_head", "build_number_box_head"]

ROI_DIGIT_BOX_HEAD_REGISTRY = Registry("ROI_DIGIT_BOX_HEAD")
ROI_NUMBER_BOX_HEAD_REGISTRY = Registry("ROI_NUMBER_BOX_HEAD")
@ROI_DIGIT_BOX_HEAD_REGISTRY.register()
class DigitConvFCHead(FastRCNNConvFCHead):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and then
    several fc layers (each followed by relu).
    """

    @configurable
    def __init__(
        self, *args, **kwargs):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature.
            conv_dims (list[int]): the output dimensions of the conv layers
            fc_dims (list[int]): the output dimensions of the fc layers
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, cfg, input_shape):
        num_conv = cfg.MODEL.ROI_DIGIT_BOX_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_DIGIT_BOX_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_DIGIT_BOX_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_DIGIT_BOX_HEAD.FC_DIM
        return {
            "input_shape": input_shape,
            "conv_dims": [conv_dim] * num_conv,
            "fc_dims": [fc_dim] * num_fc,
            "conv_norm": cfg.MODEL.ROI_DIGIT_BOX_HEAD.NORM,
        }

@ROI_NUMBER_BOX_HEAD_REGISTRY.register()
class NumberConvFCHead(FastRCNNConvFCHead):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and then
    several fc layers (each followed by relu).
    """

    @configurable
    def __init__(
        self, *args, **kwargs):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature.
            conv_dims (list[int]): the output dimensions of the conv layers
            fc_dims (list[int]): the output dimensions of the fc layers
            conv_norm (str or callable): normalization for the conv layers.
                See :func:`detectron2.layers.get_norm` for supported types.
        """
        super().__init__(*args, **kwargs)

    def forward(self, x):
        for layer in self:
            x = layer(x)
        return x

    @classmethod
    def from_config(cls, cfg, input_shape):
        num_conv = cfg.MODEL.ROI_NUMBER_BOX_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_NUMBER_BOX_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_NUMBER_BOX_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_NUMBER_BOX_HEAD.FC_DIM
        return {
            "input_shape": input_shape,
            "conv_dims": [conv_dim] * num_conv,
            "fc_dims": [fc_dim] * num_fc,
            "conv_norm": cfg.MODEL.ROI_NUMBER_BOX_HEAD.NORM,
        }


def build_digit_box_head(cfg, input_shape):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_DIGIT_BOX_HEAD.NAME
    return ROI_DIGIT_BOX_HEAD_REGISTRY.get(name)(cfg, input_shape)

def build_number_box_head(cfg, input_shape):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_NUMBER_BOX_HEAD.NAME
    return ROI_NUMBER_BOX_HEAD_REGISTRY.get(name)(cfg, input_shape)