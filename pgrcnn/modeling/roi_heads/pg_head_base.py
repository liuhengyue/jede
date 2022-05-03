import logging
from typing import Dict, List, Optional, Tuple
# from kornia import morphology as morph

import torch
from torch import nn
from detectron2.layers import ShapeSpec
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import build_box_head
from detectron2.structures import ImageList
from detectron2.layers import cat
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY
from detectron2.config import configurable

from pgrcnn.modeling.losses import pg_rcnn_loss

from pgrcnn.modeling.necks.digit_neck import build_digit_neck_output
from pgrcnn.modeling.utils.decode_utils import ctdet_decode
from pgrcnn.modeling.digit_head import DigitOutputLayers
from pgrcnn.structures import Boxes, Players, inside_matched_box
from pgrcnn.modeling.roi_heads import BaseROIHeads, build_number_box_head, build_digit_box_head
from pgrcnn.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from pgrcnn.modeling.jersey_number_head import JerseyNumberOutputLayers
from pgrcnn.modeling.necks.number_neck import build_number_neck_output
from pgrcnn.modeling.necks import build_neck_base

_TOTAL_SKIPPED = 0

logger = logging.getLogger(__name__)

__all__ = ["BasePGROIHeads"]

@ROI_HEADS_REGISTRY.register()
class BasePGROIHeads(BaseROIHeads):

    @configurable
    def __init__(self,
                 *,
                 neck_base: Optional[nn.Module] = None,
                 player_box_pooler: Optional[ROIPooler] = None,
                 neck_digit_output: Optional[nn.Module] = None,
                 digit_box_pooler: Optional[ROIPooler] = None,
                 digit_box_head: Optional[nn.Module] = None,
                 digit_box_predictor: Optional[nn.Module] = None,
                 neck_number_output: Optional[nn.Module] = None,
                 number_box_pooler: Optional[ROIPooler] = None,
                 number_box_head: Optional[nn.Module] = None,
                 number_box_predictor: Optional[nn.Module] = None,
                 **kwargs,
                 ):
        super().__init__(**kwargs)

        self.neck_base = neck_base
        if self.neck_base is not None:
            self.use_person_features = self.neck_base.use_person_features
            self.use_kpts_features = self.neck_base.use_kpts_features
        self.player_box_pooler = player_box_pooler
        self.neck_digit_output = neck_digit_output
        self.digit_box_pooler = digit_box_pooler
        self.digit_box_head = digit_box_head
        self.digit_box_predictor = digit_box_predictor
        self.digit_box_head_on = digit_box_head is not None
        self.digit_neck_on = neck_digit_output is not None

        self.neck_number_output = neck_number_output
        self.number_box_pooler = number_box_pooler
        self.number_box_head = number_box_head
        self.number_box_predictor = number_box_predictor
        self.number_box_head_on = number_box_head is not None
        self.number_neck_on = neck_number_output is not None

        self.pose_guide_on = self.digit_neck_on or self.number_neck_on


        # @classmethod
    # def _init_heads(self, cfg, input_shape):
    #     raise NotImplementedError()

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        # we use our FastRCNNOutputLayers since we have mixed dataset
        ret.update({"box_predictor": FastRCNNOutputLayers(cfg, ret["box_head"].output_shape)})
        ret.update(cls._init_neck_base(cfg, input_shape))
        if ret["neck_base"] is None:
            neck_base_output_shape = None
        else:
            neck_base_output_shape = ret["neck_base"].output_shape
        ret.update(cls._init_neck_digit_output(cfg, neck_base_output_shape))
        if ret["neck_digit_output"] is None:
            digit_head_in_shape = None
        else:
            digit_head_in_shape = input_shape
        # modify if pretrain
        if not cfg.MODEL.ROI_HEADS.ENABLE_POSE_GUIDE:
            digit_head_in_shape = input_shape
        ret.update(cls._init_digit_head(cfg, digit_head_in_shape))
        return ret

    @classmethod
    def _init_neck_base(cls, cfg, input_shape):
        """
        After the feature pooling, each ROI feature is fed into
        a cls head, bbox reg head and kpts head
        """
        ret = {}
        if not cfg.MODEL.ROI_NECK_BASE.ON or (not cfg.MODEL.ROI_HEADS.ENABLE_POSE_GUIDE):
            ret["neck_base"] = None
            return ret
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        # add a player pooler, default is 14x14
        pooler_resolution = cfg.MODEL.ROI_NECK_BASE_BRANCHES.PERSON_BRANCH.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_DIGIT_BOX_HEAD.POOLER_TYPE
        player_box_pooler = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        ret["player_box_pooler"] = player_box_pooler
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        # digit_neck is used for predicting the initial digit bboxes
        # digit neck takes in the features,
        # and keypoint heatmaps of K x 56 x 56,
        # where K could be 4 or 17 depending on
        K = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS if cfg.DATASETS.PAD_TO_FULL else cfg.DATASETS.NUM_KEYPOINTS
        kernel_size, stride, padding = cfg.MODEL.ROI_NECK_BASE_BRANCHES.KEYPOINTS_BRANCH.CONV_SPECS
        kpts_size = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION * 4
        conv_out_size = (kpts_size - kernel_size + 2 * padding) // stride + 1
        up_scale = max(1, cfg.MODEL.ROI_NECK_BASE_BRANCHES.KEYPOINTS_BRANCH.DECONV_KERNEL // 2) * cfg.MODEL.ROI_NECK_BASE_BRANCHES.KEYPOINTS_BRANCH.UP_SCALE
        out_size = conv_out_size * up_scale
        # construct the input shapes, in the order of keypoint heatmap, then person box features
        input_shapes = {
            "keypoint_heatmap_shape": ShapeSpec(channels=K, height=out_size, width=out_size),
            "person_box_features_shape": ShapeSpec(channels=in_channels,
                                                   height=pooler_resolution,
                                                   width=pooler_resolution)
        }

        ret["neck_base"] = build_neck_base(cfg, input_shapes)
        return ret

    @classmethod
    def _init_neck_digit_output(cls, cfg, input_shape):
        ret = {}
        if input_shape is None:
            ret["neck_digit_output"] = None
        else:
            ret["neck_digit_output"] = build_digit_neck_output(cfg, input_shape)
        return ret

    @classmethod
    def _init_digit_head(cls, cfg, input_shape):
        """
        box_in_features are the same from backbone
        After the feature pooling, each ROI feature is fed into
        a cls head, bbox reg head and kpts head
        """
        if input_shape is None:
            return {
            "digit_box_pooler": None,
            "digit_box_head": None,
            "digit_box_predictor": None,
        }
        # fmt: off
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_DIGIT_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_DIGIT_BOX_HEAD.POOLER_TYPE
        # self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # fmt: on
        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        digit_box_head = build_digit_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        # these are used for digit classification/regression after we get the digit ROI
        digit_box_pooler = (
                ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )

        digit_box_predictor = DigitOutputLayers(cfg, digit_box_head.output_shape)
        return {
            "digit_box_pooler": digit_box_pooler,
            "digit_box_head": digit_box_head,
            "digit_box_predictor": digit_box_predictor,
        }

    @classmethod
    def _init_neck_number_output(cls, cfg, input_shape):
        ret = {}
        if input_shape is None:
            ret["neck_number_output"] = None
        else:
            ret["neck_number_output"] = build_number_neck_output(cfg, input_shape)
        return ret

    @classmethod
    def _init_number_head(self, cfg, input_shape):
        """
        After the feature pooling, each ROI feature is fed into
        a cls head, bbox reg head and kpts head
        """
        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        if input_shape is None:
            return {
            "number_box_pooler": None,
            "number_box_head": None,
            "number_box_predictor": None,
        }
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]
        pooler_resolution = cfg.MODEL.ROI_NUMBER_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = list(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_NUMBER_BOX_HEAD.POOLER_TYPE

        box_feat_shape = ShapeSpec(channels=in_channels,
                               height=pooler_resolution[0],
                               width=pooler_resolution[1])
        # if it has something
        box_head_on = cfg.MODEL.ROI_NUMBER_BOX_HEAD.NUM_FC + cfg.MODEL.ROI_NUMBER_BOX_HEAD.NUM_CONV > 0
        if box_head_on:
            number_box_head = build_number_box_head(cfg, box_feat_shape)
            number_box_pooler = (
                ROIPooler(
                    output_size=pooler_resolution,
                    scales=pooler_scales,
                    sampling_ratio=sampling_ratio,
                    pooler_type=pooler_type,
                )
                if pooler_type
                else None
            )
        else:
            number_box_head = None
            number_box_pooler = None

        number_box_predictor = JerseyNumberOutputLayers(cfg, box_feat_shape)

        return {
            "number_box_pooler": number_box_pooler,
            "number_box_head": number_box_head,
            "number_box_predictor": number_box_predictor,
        }



    def _forward_neck_base(self, features, instances):
        if self.use_person_features:
            # we pool the features again for convenience
            # 14 x 14 pooler
            if self.training:
                person_features = self.player_box_pooler(features, [x.proposal_boxes if x.has("proposal_boxes")
                                                                  else Boxes([]).to(features[0].device) for x in instances])
            else:
                person_features = self.player_box_pooler(features, [x.pred_boxes for x in instances])
        else:
            person_features = None

        if self.use_kpts_features:
            kpts_features = cat([b.pred_keypoints_logits if b.has("pred_keypoints_logits")
                                 else torch.empty((0, self.keypoint_head.num_keypoints,
                                                   int(2 * self.keypoint_head.up_scale *
                                                       self.keypoint_pooler.output_size[0]),
                                                   int(2 * self.keypoint_head.up_scale *
                                                       self.keypoint_pooler.output_size[1])),
                                                  dtype=torch.float32, device=features[0].device)
                                 for b in instances], dim=0).detach()
        else:
            kpts_features = None

        # go through the neck base
        fused_features = self.neck_base(kpts_features, person_features)
        return fused_features

    def _forward_pose_guided(self, features, instances):
        """
        Forward logic from kpts heatmaps to digit centers and scales (centerNet)

        Arguments:
        """


        features = [features[f] for f in self.box_in_features]
        # when disabled
        if not self.pose_guide_on:
            detections = [None for _ in instances]
            # assign new fields to instances
            self.label_and_sample_digit_proposals(detections, instances)
            if not self.number_neck_on:
                detections = [None for _ in instances]
                self.label_and_sample_jerseynumber_proposals(detections, instances)
            return {}


        # go through the neck base
        fused_features = self._forward_neck_base(features, instances)

        if self.training:
            losses = {}
            if self.digit_neck_on:  # do not use digit prediction
                # shape (N, 3, 56, 56) (N, 2, 56, 56)
                outputs = self.neck_digit_output(fused_features)
                loss, _ = self.neck_digit_output.loss(instances, outputs)
                losses.update(loss)
                with torch.no_grad():
                    detections = self.neck_digit_output.decode(instances, outputs)
                    # assign new fields to instances
                    self.label_and_sample_digit_proposals(detections, instances)

            # forward with number features
            if self.number_neck_on:
                center_heatmaps, scale_heatmaps, offset_heatmaps = self.neck_number_output(fused_features)
                loss, _ = self.neck_number_output.loss(instances, center_heatmaps, scale_heatmaps, offset_heatmaps)
                losses.update(loss)
                with torch.no_grad():
                    detections = self.neck_number_output.decode(instances, center_heatmaps, scale_heatmaps, offset_heatmaps)
                    # assign new fields to instances
                    self.label_and_sample_jerseynumber_proposals(detections, instances)

            return losses

        else:
            if self.digit_neck_on:  # do not use digit prediction
                # shape (N, 3, 56, 56) (N, 2, 56, 56)
                outputs = self.neck_digit_output(fused_features)
                instances = self.neck_digit_output.inference(instances, outputs)
            if self.number_neck_on:
                center_heatmaps, scale_heatmaps, offset_heatmaps = self.neck_number_output(fused_features)
                instances = self.neck_number_output.inference(instances, center_heatmaps, scale_heatmaps, offset_heatmaps)

            return instances

    def _forward_digit_box(self, features, proposals):
        if not self.digit_box_head_on:
            if self.training:
                return {}
            else:
                return proposals
        features = [features[f] for f in self.in_features]
        # most likely have empty boxes, Boxes.cat([]) will return Boxes on cpu
        detection_boxes = [Boxes.cat(x.proposal_digit_boxes).to(features[0].device)
                           if x.has('proposal_digit_boxes') else Boxes.cat([]).to(features[0].device)
                           for x in proposals]
        # digit box features across all batches
        box_features = self.digit_box_pooler(features, detection_boxes)
        # we split and assign to each proposal
        # num_proposal_digit_boxes = [[len(instance) for instance in p.proposal_digit_boxes] for p in proposals]
        # box_features_all = torch.split(box_features, [sum(num_per_image) for num_per_image in num_proposal_digit_boxes])
        # for proposals_per_image, pred_box_features_per_image, num_proposal_digit_boxes_per_image in zip(proposals, box_features_all, num_proposal_digit_boxes):
        #     proposal_digit_box_features = torch.split(pred_box_features_per_image, num_proposal_digit_boxes_per_image)
        #     proposals_per_image.proposal_digit_box_features = list(proposal_digit_box_features)
        fc_box_features = self.digit_box_head(box_features)
        predictions = self.digit_box_predictor(fc_box_features)

        if self.training:
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.digit_box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_digit_boxes = Boxes(pred_boxes_per_image)
            return self.digit_box_predictor.losses(predictions, proposals)
        else:
            pred_instances, _ = self.digit_box_predictor.inference(predictions, proposals)
            return pred_instances

    def _forward_number_box(self, features, proposals):
        if not self.number_box_head_on:
            if self.training:
                return {}
            else:
                return proposals
        features = [features[f] for f in self.in_features]
        # most likely have empty boxes, Boxes.cat([]) will return Boxes on cpu
        if self.training:
            # self.label_and_sample_jerseynumber_proposals(proposals)
            detection_boxes = [Boxes.cat(x.proposal_number_boxes).to(features[0].device)
                               if x.has('proposal_number_boxes') else Boxes.cat([]).to(features[0].device)
                               for x in proposals]
            if sum([len(b) for b in detection_boxes]) == 0:
                # here we need to have at least something to feed the head
                # currently, pytorch does not support empty input to LSTM or adaptivepooling
                # which will cause mutliprocssing deadlocking for allgather
                dummy_boxes = [Boxes(torch.as_tensor([[p.image_size[1] / 2 - 1,
                                       p.image_size[1] / 2 - 1,
                                       p.image_size[1] / 2 + 1,
                                       p.image_size[1] / 2 + 1]], device=features[0].device))
                               for p in proposals]
                detection_boxes = dummy_boxes
        else:
            detection_boxes = [Boxes.cat(p.pred_number_boxes) for p in proposals]
            # detection_boxes = [Boxes.cat(p.pred_digit_boxes).union() for p in proposals]
            # for number_boxes, p in zip(detection_boxes, proposals):
            #     p.proposal_number_boxes = number_boxes
        box_features = self.number_box_pooler(features, detection_boxes)
        if self.number_box_head_on:
            fc_features = self.number_box_head(box_features)
        else:
            fc_features = None
        predictions = self.number_box_predictor(box_features, fc_features)

        if self.training:
            return self.number_box_predictor.losses(predictions, proposals)
        else:
            pred_instances = self.number_box_predictor.inference(predictions, proposals)
            return pred_instances


    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Players]
    ) -> List[Players]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (list[Instances]):
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        instances = self._forward_pose_guided(features, instances)
        instances = self._forward_digit_box(features, instances)
        instances = self._forward_number_box(features, instances)

        return instances

    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            proposals: List[Players],
            targets: Optional[List[Players]] = None,
    ) -> Tuple[List[Players], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            # proposals will be modified in-place
            losses = self._forward_box(features, proposals)
            kpt_loss, proposals = self._forward_keypoint(features, proposals)
            losses.update(kpt_loss)
            losses.update(self._forward_pose_guided(features, proposals))
            losses.update(self._forward_digit_box(features, proposals))
            losses.update(self._forward_number_box(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}


