import detectron2.config
import numpy as np
from typing import List, Dict
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from pgrcnn.structures import Boxes, inside_matched_box
from detectron2.layers import ShapeSpec, cat, Conv2d
from detectron2.utils.registry import Registry
from detectron2.modeling.backbone.resnet import DeformBottleneckBlock
from detectron2.layers.wrappers import _NewEmptyTensorOp

from .digit_neck_branches import build_digit_neck_branch
from . import build_neck_base, build_neck_output
from ..losses import pg_rcnn_loss
from ..utils import ctdet_decode

ROI_DIGIT_NECK_REGISTRY = Registry("ROI_DIGIT_NECK")

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
            nn.Linear(128, 128),
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

@ROI_DIGIT_NECK_REGISTRY.register()
class DigitNeck(nn.Module):
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
        neck_cfg = cfg.MODEL.ROI_DIGIT_NECK_OUTPUT
        self.neck_output = build_neck_output(neck_cfg, input_shape)
        self.size_target_type = cfg.MODEL.ROI_DIGIT_NECK_OUTPUT.SIZE_TARGET_TYPE
        self.size_target_scale = cfg.MODEL.ROI_DIGIT_NECK_OUTPUT.SIZE_TARGET_SCALE
        self.out_head_weights = cfg.MODEL.ROI_DIGIT_NECK_OUTPUT.OUTPUT_HEAD_WEIGHTS
        self.fg_ratio = cfg.MODEL.ROI_NECK_BASE.FG_RATIO
        self.num_proposal_train = cfg.MODEL.ROI_NECK_BASE.NUM_PROPOSAL_TRAIN
        self.num_proposal_test = cfg.MODEL.ROI_NECK_BASE.NUM_PROPOSAL_TEST
        self.target_name = "digit"
        self.num_digits_classifier_on = cfg.MODEL.ROI_DIGIT_NECK_OUTPUT.NUM_DIGITS_CLASSIFIER_ON
        if self.num_digits_classifier_on:
            self.num_digits_cls = NumDigitsClassifier(cfg, ShapeSpec(channels=self.neck_output.output_head_channels[0],
                                                                     height=input_shape.height,
                                                                     width=input_shape.width))
        else:
            self.num_digits_cls = None
        self.add_box_constraints = cfg.MODEL.ROI_DIGIT_NECK_OUTPUT.ADD_BOX_CONSTRAINTS


    def forward(self, x):
        pred_center_heatmaps, pred_scale_heatmaps, pred_offset_heatmaps = self.neck_output(x)
        if self.num_digits_classifier_on:
            num_digits_logits = self.num_digits_cls(pred_center_heatmaps)
        else:
            num_digits_logits = None
        return (pred_center_heatmaps, pred_scale_heatmaps, pred_offset_heatmaps, num_digits_logits)

    def decode(self, instances, outputs):
        center_heatmaps, scale_heatmaps, offset_heatmaps, _ = outputs
        if self.training:
            # deal with svhn since the instances will not contain proposal_boxes
            num_instances = [len(instance) if instance.has("proposal_boxes") else 0 for instance in instances]
            person_boxes = cat([b.proposal_boxes.tensor if b.has("proposal_boxes")
                                else torch.empty((0, 4), dtype=torch.float32, device=center_heatmaps.device)
                                for b in instances], dim=0)
            num_proposals_keep = self.num_proposal_train
        else:
            num_instances = [len(instance) if instance.has("pred_boxes") else 0 for instance in instances]
            person_boxes = cat([b.pred_boxes.tensor for b in instances], dim=0)
            num_proposals_keep = self.num_proposal_test
        # (N, num of candidates, (x1, y1, x2, y2, score, class)
        detections = ctdet_decode(center_heatmaps, scale_heatmaps, offset_heatmaps,
                                 person_boxes,
                                 K=num_proposals_keep,
                                 size_target_type=self.size_target_type,
                                 size_target_scale=self.size_target_scale,
                                 training=self.training,
                                 offset=0
                                 )
        detections = list(detections.split(num_instances))
        return detections

    def inference(self, instances, outputs):
        detections = self.decode(instances, outputs)
        _, _, _, pred_num_digits_logits = outputs
        if pred_num_digits_logits is not None:
            pred_num_digits = F.softmax(pred_num_digits_logits, dim=-1)
            pred_num_digits = pred_num_digits.argmax(dim=-1)
            pred_num_digits = torch.split(pred_num_digits, [len(x) for x in instances])
        detection_boxes = [detection[..., :4] for detection in detections]
        detection_center_classes = [detection[..., 5].long() for detection in detections]
        detection_center_scores = [detection[..., 4] for detection in detections]
        # detection_ct_classes = list(detection[..., -1].split(num_instances))
        # assign new fields to instances
        for i, (boxes, center_classes, center_scores) in enumerate(zip(detection_boxes, detection_center_classes, detection_center_scores)):
            # perform a person roi clip or not
            boxes = [Boxes(b) for b in boxes]  # List of N `Boxes'
            center_classes = [ct_cls for ct_cls in center_classes] # split into a per-instance
            center_scores = [ct_score for ct_score in center_scores]
            pred_person_boxes = instances[i].pred_boxes  # [N, 4]
            keep_mask = [inside_matched_box(boxes_per_ins, pred_person_boxes[j]) for j, boxes_per_ins
                     in enumerate(boxes)]
            boxes = [boxes_per_ins[keep] for boxes_per_ins, keep in zip(boxes, keep_mask)]
            center_classes = [ct_cls_per_ins[keep] for ct_cls_per_ins, keep in zip(center_classes, keep_mask)]
            center_scores = [ct_score_per_ins[keep] for ct_score_per_ins, keep in zip(center_scores, keep_mask)]
            instances[i].proposal_digit_boxes = boxes
            instances[i].pred_digit_center_classes = center_classes
            instances[i].pred_digit_center_scores = center_scores
            if pred_num_digits_logits is not None:
                instances[i].pred_num_digits = pred_num_digits[i]
        return instances

    def loss(self, instances, outputs):
        center_heatmaps, scale_heatmaps, offset_heatmaps, num_digits_logits = outputs
        return pg_rcnn_loss(center_heatmaps, scale_heatmaps, offset_heatmaps,
                            num_digits_logits,
                            instances,
                            size_target_type=self.size_target_type,
                            size_target_scale=self.size_target_scale,
                            output_head_weights=self.out_head_weights,
                            target_name=self.target_name,
                            add_box_constraints=self.add_box_constraints)



def build_digit_neck_output(cfg, input_shapes):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_DIGIT_NECK.NAME
    if not name or name == '':
        return None
    return ROI_DIGIT_NECK_REGISTRY.get(name)(cfg, input_shapes)