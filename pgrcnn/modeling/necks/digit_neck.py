import detectron2.config
import numpy as np
from typing import List, Dict
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from pgrcnn.structures import Boxes, inside_matched_box
from detectron2.layers import ShapeSpec, cat
from detectron2.utils.registry import Registry
from detectron2.modeling.backbone.resnet import DeformBottleneckBlock

from .digit_neck_branches import build_digit_neck_branch
from . import build_neck_base, build_neck_output
from ..losses import pg_rcnn_loss
from ..utils import ctdet_decode

ROI_DIGIT_NECK_REGISTRY = Registry("ROI_DIGIT_NECK")


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


    def forward(self, x):
        return self.neck_output(x)


    def decode(self, instances, center_heatmaps, scale_heatmaps, offset_heatmaps):

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

    def inference(self, instances, center_heatmaps, scale_heatmaps, offset_heatmaps):
        detections = self.decode(instances, center_heatmaps, scale_heatmaps, offset_heatmaps)

        detection_boxes = [detection[..., :4] for detection in detections]
        # detection_ct_classes = list(detection[..., -1].split(num_instances))
        # assign new fields to instances
        for i, boxes in enumerate(detection_boxes):
            # perform a person roi clip or not
            boxes = [Boxes(b) for b in boxes]  # List of N `Boxes'
            pred_person_boxes = instances[i].pred_boxes  # [N, 4]
            boxes = [boxes_per_ins[inside_matched_box(boxes_per_ins, pred_person_boxes[j])] for j, boxes_per_ins
                     in enumerate(boxes)]
            instances[i].proposal_digit_boxes = boxes
        return instances

    def loss(self, instances, center_heatmaps, scale_heatmaps, offset_heatmaps):
        return pg_rcnn_loss(center_heatmaps, scale_heatmaps, offset_heatmaps,
                                    instances,
                                    size_target_type=self.size_target_type,
                                    size_target_scale=self.size_target_scale,
                                    output_head_weights=self.out_head_weights,
                                    target_name=self.target_name)


def build_digit_neck_output(cfg, input_shapes):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_DIGIT_NECK.NAME
    if not name or name == '':
        return None
    return ROI_DIGIT_NECK_REGISTRY.get(name)(cfg, input_shapes)