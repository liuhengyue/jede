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
from detectron2.layers.wrappers import _NewEmptyTensorOp
from detectron2.modeling.backbone.resnet import DeformBottleneckBlock

from .digit_neck_branches import build_digit_neck_branch
from ..layers import CrissCrossAttention

from pgrcnn.modeling.layers import ConvLSTMWrapper
ROI_NECK_OUTPUT_REGISTRY = Registry("ROI_NECK_OUTPUT")

class NumDigitsClassifier(nn.Module):
    def __init__(self,
                 cfg: detectron2.config.CfgNode,
                 input_shapes: ShapeSpec
                 ):
        """

        """
        super().__init__()
        pool_stride = 1
        in_channels = \
        input_shapes.channels *\
        input_shapes.height // pool_stride * \
        input_shapes.width // pool_stride
        self.out_channels = 3  # 0 digit, 1 digit, 2 digit
        # self.pool = Conv2d(self.c, self.c, 3, stride=self.pool_stride, padding=1,
        #                     norm=None, activation=None)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=pool_stride, padding=1)
        self.linears = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, self.out_channels)
        )
        for m in self.linears.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)



    def forward(self, x):
        if x.numel() == 0:
            return _NewEmptyTensorOp.apply(x, (0, self.out_channels))
        x = self.pool(x)
        x = self.linears(x)
        return x

@ROI_NECK_OUTPUT_REGISTRY.register()
class FCNNeckOutput(nn.Module):
    def __init__(self,
                 cfg: detectron2.config.CfgNode,
                 input_shape: ShapeSpec
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
        # cfg is not the full cfg: either cfg.MODEL.ROI_NECK_OUTPUT or ROI_NUMBER_NECK_OUTPUT
        self.conv_norm = cfg.NORM
        self.output_head_names = cfg.OUTPUT_HEAD_NAMES
        self.output_head_channels = cfg.OUTPUT_HEAD_CHANNELS
        conv_dims = [cfg.CONV_DIM] * cfg.NUM_CONV
        assert len(conv_dims) > 0
        self.use_deform = cfg.DEFORMABLE
        self.attn = cfg.ATTN
        self.focal_bias = cfg.FOCAL_BIAS
        self.num_digits_classifier_on = cfg.NUM_DIGITS_CLASSIFIER_ON # 0, 1, or 2
        # moved sigmoid to loss and decode
        if len(self.output_head_names) == 4:
            activations = (None, F.relu, None, None)
            self.dynamic_conv = True
        else:
            activations = (None, F.relu, torch.sigmoid)
            self.dynamic_conv = False
        activations = {name: activation for name, activation in zip(self.output_head_names, activations)}

        in_channels = input_shape.channels
        for name, out_channels in zip(self.output_head_names, self.output_head_channels):
            activation = activations[name]
            if name == "center" and cfg.CONVLSTM:
                module = ConvLSTMWrapper(cfg, in_channels, activation=activation)
                self.add_module(name, module)
            else:
                self._init_output_layers(name, conv_dims, in_channels, out_channels, activation=activation)
        self.size_reg = hasattr(self, "size")
        self.offset_reg = hasattr(self, "offset")
        self._init_weights()
        # add number of digits classification here
        self._init_num_digit_classifier(cfg, input_shape)


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
            if hasattr(self, "center_head"):
                nn.init.normal_(self.center_head.weight, std=0.001)
                self.center_head.bias.data.fill_(self.focal_bias)
            else:
                if isinstance(self.center, nn.Sequential):
                    nn.init.normal_(self.center[-1].weight, std=0.001)
                    self.center[-1].bias.data.fill_(self.focal_bias)
        if hasattr(self, "size"):
            nn.init.normal_(self.size[-1].weight, std=0.001)
            self.size[-1].bias.data.fill_(0.)
        if hasattr(self, "offset"):
            nn.init.normal_(self.offset[-1].weight, std=0.001)
            self.offset[-1].bias.data.fill_(0.)
        if hasattr(self, "kernel"):
            nn.init.normal_(self.conv[-1].weight, std=0.001)
            self.conv[-1].bias.data.fill_(0.)

    def _init_num_digit_classifier(self, cfg, input_shape):

        if self.num_digits_classifier_on:
            # use the predicted center heatmaps as the input
            if self.num_digits_classifier_on == 1:
                conv_dim = self.output_head_channels[0]
            # use the features before the output center head
            elif self.num_digits_classifier_on == 2:
                conv_dim = cfg.CONV_DIM
            # use the fused features as input
            elif self.num_digits_classifier_on == 3:
                conv_dim = input_shape.channels
            else:
                raise NotImplementedError()
            self.num_digits_cls = NumDigitsClassifier(cfg, ShapeSpec(channels=conv_dim,
                                                                     height=input_shape.height,
                                                                     width=input_shape.width))
        else:
            self.num_digits_cls = None


    def _init_output_layers(self, name, conv_dims, in_channels, output_channels, activation=None):
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


        # add attention layers
        if self.attn and name == "center":
            attention_type = CrissCrossAttention
            module = attention_type(conv_dims[-1], conv_dims[-1], norm="")
            modules.append(module)
            module = attention_type(conv_dims[-1], conv_dims[-1], norm="")
            modules.append(module)

        # 3 cls: left digit, center digit, right digit
        head = Conv2d(
            conv_dims[-1],
            output_channels,
            kernel_size=1,
            bias=True,
            norm=None,
            activation=activation,  # need logits for sigmoid
        )
        if name == "center" and self.num_digits_classifier_on == 2:
            self.add_module("center_head", head)
        else:
            modules.append(head)
        modules = nn.Sequential(*modules)
        self.add_module(name, modules)

    def forward(self, x):
        pred_center_heatmaps = self.center(x)
        if self.num_digits_classifier_on == 1:
            num_digits_logits = self.num_digits_cls(pred_center_heatmaps)
        elif self.num_digits_classifier_on == 2:
            num_digits_logits = self.num_digits_cls(pred_center_heatmaps)
            pred_center_heatmaps = self.center_head(pred_center_heatmaps)
        elif self.num_digits_classifier_on == 3:
            num_digits_logits = self.num_digits_cls(x)
        else:
            num_digits_logits = None
        if self.size_reg:
            pred_scale_heatmaps = self.size(x)
        else:
            pred_scale_heatmaps = None
        if self.offset_reg:
            pred_offset_heatmaps = self.offset(x)
        else:
            pred_offset_heatmaps = None
        # if self.dynamic_conv:
        #     pred_kernels = self.kernel(x)
        # center, size, offset heatmaps
        return [pred_center_heatmaps, pred_scale_heatmaps, pred_offset_heatmaps, num_digits_logits]


def build_neck_output(cfg, input_shape):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.NAME
    if not name or name == '':
        return None
    return ROI_NECK_OUTPUT_REGISTRY.get(name)(cfg, input_shape)