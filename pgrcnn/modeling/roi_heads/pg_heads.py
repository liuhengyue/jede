import logging
from typing import Dict, List, Optional, Tuple
# from kornia import morphology as morph

import torch
import torch.nn.functional as F
from detectron2.layers import ShapeSpec
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import build_box_head
from detectron2.structures import ImageList
from detectron2.layers import cat
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY
from detectron2.config import configurable


from pgrcnn.modeling.necks.digit_neck import build_digit_neck_output
from pgrcnn.modeling.utils.decode_utils import ctdet_decode
from pgrcnn.modeling.digit_head import DigitOutputLayers
from pgrcnn.structures import Boxes, Players, inside_matched_box
from pgrcnn.modeling.roi_heads import BaseROIHeads, build_number_box_head, build_digit_box_head
from pgrcnn.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from pgrcnn.modeling.utils import compute_targets, compute_number_targets
from pgrcnn.modeling.jersey_number_head import JerseyNumberOutputLayers
from pgrcnn.modeling.necks.number_neck import build_number_neck_output
from pgrcnn.modeling.necks import build_neck_base, build_neck_output
from pgrcnn.modeling.losses import pg_rcnn_loss

_TOTAL_SKIPPED = 0

logger = logging.getLogger(__name__)

__all__ = ["PGMaskingROIHeads"]

from .pg_head_base import BasePGROIHeads

@ROI_HEADS_REGISTRY.register()
class PGROIHeads(BasePGROIHeads):
    @configurable
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        if ret["neck_base"] is None:
            neck_base_output_shape = None
        else:
            neck_base_output_shape = ret["neck_base"].output_shape
        ret.update(cls._init_neck_number_output(cfg, neck_base_output_shape))
        if ret["neck_number_output"] is None:
            number_head_in_shape = None
        else:
            number_head_in_shape = input_shape
        ret.update(cls._init_number_head(cfg, number_head_in_shape))
        return ret



@ROI_HEADS_REGISTRY.register()
class PGMaskingROIHeads(BasePGROIHeads):
    @configurable
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.update(cls._init_number_head(cfg, input_shape))
        return ret

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
                center_heatmaps, scale_heatmaps, offset_heatmaps = self.neck_digit_output(fused_features)
                loss, gt_center_heatmaps = self.neck_digit_output.loss(instances, center_heatmaps, scale_heatmaps,
                                                                       offset_heatmaps)
                losses.update(loss)
                with torch.no_grad():
                    detections = self.neck_digit_output.decode(instances, center_heatmaps, scale_heatmaps,
                                                               offset_heatmaps)
                    # assign new fields to instances
                    self.label_and_sample_digit_proposals(detections, instances)
                return losses, gt_center_heatmaps

        else:
            if self.digit_neck_on:  # do not use digit prediction
                # shape (N, 3, 56, 56) (N, 2, 56, 56)
                center_heatmaps, scale_heatmaps, offset_heatmaps = self.neck_digit_output(fused_features)
                instances = self.neck_digit_output.inference(instances, center_heatmaps, scale_heatmaps,
                                                             offset_heatmaps)
                return instances, center_heatmaps
            else:
                return instances, None

    def _forward_number_box(self, features, proposals, mask_maps):
        features = [features[f] for f in self.in_features]
        # we use the person roi to get the features, then perform masking
        if self.training:
            boxes = [x.proposal_boxes if x.has("proposal_boxes") else Boxes([]).to(mask_maps.device) for x in proposals]
        else:
            boxes = [x.pred_boxes for x in proposals]

        box_features = self.number_box_pooler(features, boxes)
        # masking
        mask_maps = morph.dilation(mask_maps.detach(), torch.ones((7, 7), device=mask_maps.device))
        box_features = box_features * mask_maps
        predictions = self.number_box_predictor(box_features, None)

        if self.training:
            return self.number_box_predictor.losses(predictions, proposals)
        else:
            pred_instances = self.number_box_predictor.inference(predictions, proposals)
            return pred_instances

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
            # proposals or sampled_instances will be modified in-place
            losses = self._forward_box(features, proposals)
            kpt_loss, proposals = self._forward_keypoint(features, proposals)
            losses.update(kpt_loss)
            loss, mask_maps = self._forward_pose_guided(features, proposals)
            losses.update(loss)
            losses.update(self._forward_digit_box(features, proposals))
            losses.update(self._forward_number_box(features, proposals, mask_maps))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

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
        instances, mask_maps = self._forward_pose_guided(features, instances)
        instances = self._forward_digit_box(features, instances)
        instances = self._forward_number_box(features, instances, mask_maps)

        return instances