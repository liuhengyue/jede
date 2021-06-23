import logging
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from detectron2.layers import ShapeSpec
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import build_box_head
from detectron2.structures import ImageList
from detectron2.layers import cat
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY


from pgrcnn.modeling.kpts2digit_head import build_digit_head
from pgrcnn.modeling.utils.decode_utils import ctdet_decode
from pgrcnn.modeling.digit_head import DigitOutputLayers
from pgrcnn.structures import Boxes, Players
from pgrcnn.modeling.roi_heads import BaseROIHeads
from pgrcnn.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from pgrcnn.modeling.utils import compute_targets

_TOTAL_SKIPPED = 0

logger = logging.getLogger(__name__)

__all__ = ["PGROIHeads"]

@ROI_HEADS_REGISTRY.register()
class PGROIHeads(BaseROIHeads):
    def __init__(self, cfg, input_shape):
        super(PGROIHeads, self).__init__(cfg, input_shape)
        self.num_digit_classes = cfg.MODEL.ROI_DIGIT_HEAD.NUM_DIGIT_CLASSES
        self.use_person_box_features = cfg.MODEL.ROI_DIGIT_HEAD.USE_PERSON_BOX_FEATURES
        self.num_ctdet_proposal = cfg.MODEL.ROI_DIGIT_HEAD.NUM_PROPOSAL
        self.num_interests = cfg.DATASETS.NUM_INTERESTS
        self.batch_digit_size_per_image = cfg.MODEL.ROI_DIGIT_HEAD.BATCH_DIGIT_SIZE_PER_IMAGE
        self._init_digit_head(cfg, input_shape)



    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    def _init_digit_head(self, cfg, input_shape):
        """
        After the feature pooling, each ROI feature is fed into
        a cls head, bbox reg head and kpts head
        """
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # fmt: on
        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]


        # these are used for digit classification/regression after we get the digit ROI
        self.digit_box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.digit_box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )

        self.digit_box_predictor = DigitOutputLayers(cfg, self.box_head.output_shape)

        # digit_head is used for predicting the initial digit bboxes
        # digit head takes in the features,
        # and keypoint heatmaps of K x 56 x 56,
        # where K could be 4 or 17 depending on
        K = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS if cfg.DATASETS.PAD_TO_FULL else cfg.DATASETS.NUM_KEYPOINTS
        out_size = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION * 4
        # construct the input shapes, in the order of keypoint heatmap, then person box features
        input_shapes = {
            "keypoint_heatmap_shape": ShapeSpec(channels=K, height=out_size, width=out_size),
            "person_box_features_shape": ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        }
        self.digit_head = build_digit_head(
            cfg, input_shapes
        )



    def _forward_pose_guided(self, features, instances):
        """
        Forward logic from kpts heatmaps to digit centers and scales (centerNet)

        Arguments:
        """
        if self.use_person_box_features:
            # we pool the features again for convenience
            features = [features[f] for f in self.box_in_features]
            # 14 x 14 pooler
            if self.training:
                box_features = self.keypoint_pooler(features, [x.proposal_boxes for x in instances])
            else:
                box_features = self.keypoint_pooler(features, [x.pred_boxes for x in instances])
        else:
            box_features = None
        kpts_logits = cat([b.pred_keypoints_logits for b in instances], dim=0)
        # shape (N, 3, 56, 56) (N, 2, 56, 56)
        center_heatmaps, scale_heatmaps = self.digit_head(kpts_logits, box_features)
        if self.training:
            loss = pg_rcnn_loss(center_heatmaps, scale_heatmaps, instances, normalizer=None)
            with torch.no_grad():
                bboxes_flat = cat([b.proposal_boxes.tensor for b in instances], dim=0)
                # detection boxes (N, num of candidates, (x1, y1, x2, y2, score, center cls))
                detections = ctdet_decode(center_heatmaps, scale_heatmaps, bboxes_flat,
                                         K=self.num_ctdet_proposal, feature_scale="feature")
                # todo: has duplicate boxes
                len_instances = [len(instance) for instance in instances]
                detections = list(detections.split(len_instances))
                # assign new fields to instances
                # per image
                self.label_and_sample_digit_proposals(detections, instances)

            return loss

        else:
            bboxes_flat = cat([b.pred_boxes.tensor for b in instances], dim=0)
            # (N, num of candidates, (x1, y1, x2, y2, score, center 0 /left 1/right 2)
            detection = ctdet_decode(center_heatmaps, scale_heatmaps, bboxes_flat,
                                     K=self.num_ctdet_proposal, feature_scale="feature", training=False)
            detection_boxes = list(detection[..., :4].split([len(instance) for instance in instances]))
            detection_ct_classes = list(detection[..., -1].split([len(instance) for instance in instances]))
            # assign new fields to instances
            for i, (boxes, detection_ct_cls) in enumerate(zip(detection_boxes, detection_ct_classes)):
                # could be empty list
                instances[i].proposal_digit_boxes = [Boxes(b) for b in boxes]
                # instances[i].proposal_digit_ct_classes = [c for c in detection_ct_cls]
            return instances

    def _forward_digit_box(self, features, proposals):
        features = [features[f] for f in self.in_features]
        # most likely have empty boxes, Boxes.cat([]) will return Boxes on cpu
        detection_boxes = [Boxes.cat(x.proposal_digit_boxes).to(features[0].device) for x in proposals]
        box_features = self.digit_box_pooler(features, detection_boxes)
        box_features = self.digit_box_head(box_features)
        predictions = self.digit_box_predictor(box_features)

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
        # keypoints_logits = cat([instance.pred_keypoints_logits for instance in instances], dim=0)
        instances = self._forward_pose_guided(features, instances)
        instances = self._forward_digit_box(features, instances)
        # remove proposal boxes
        for instance in instances:
            instance.remove('proposal_digit_boxes')

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
            # proposals or sampled_instances will be modified in-place
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            kpt_loss, sampled_instances = self._forward_keypoint(features, proposals)
            losses.update(kpt_loss)
            # we may not have the instances for further training
            if len(sampled_instances) and sum([len(ins) for ins in sampled_instances]):
                losses.update(self._forward_pose_guided(features, sampled_instances))
                losses.update(self._forward_digit_box(features, sampled_instances))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}


def gaussian_focal_loss(pred, gaussian_target, alpha=2.0, gamma=4.0, eps=1e-12):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.
    Args:
        pred (torch.Tensor): The prediction.
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
    """
    pred = torch.sigmoid(pred)
    pos_weights = gaussian_target.eq(1)
    neg_weights = (1 - gaussian_target).pow(gamma)
    pos_loss = (-(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights).sum()
    neg_loss = (-(1 - pred + eps).log() * pred.pow(alpha) * neg_weights).sum()
    return pos_loss + neg_loss


def pg_rcnn_loss(pred_keypoint_logits, pred_scale_logits, instances, normalizer, size_weight=0.1,
                 ):
    """
    Wrap center and size loss here.
    We treat the predicted centers as keypoints.
    Arguments:
        pred_keypoint_logits (Tensor): A tensor of shape (N, K, S, S) where N is the total number
            of instances in the batch, K is the number of keypoints, and S is the side length
            of the keypoint heatmap. The values are spatial logits.
        instances (list[Instances]): A list of M Instances, where M is the batch size.
            These instances are predictions from the model
            that are in 1:1 correspondence with pred_keypoint_logits.
            Each Instances should contain a `gt_keypoints` field containing a `structures.Keypoint`
            instance.
        normalizer (float): Normalize the loss by this amount.
            If not specified, we normalize by the number of visible keypoints in the minibatch.

    Returns a scalar tensor containing the loss.
    """
    heatmaps = []
    valid = []
    scale_targets = []
    # keypoint_side_len = pred_keypoint_logits.shape[2]
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        heatmaps_per_image, scales_per_image, valid_per_image = compute_targets(
            instances_per_image.gt_digit_centers,
            instances_per_image.gt_digit_scales,
            instances_per_image.proposal_boxes.tensor,
            instances_per_image.pred_keypoints_logits
        )
        heatmaps.append(heatmaps_per_image)
        valid.append(valid_per_image)
        scale_targets.append(scales_per_image)
    if len(heatmaps):
        keypoint_targets = cat(heatmaps, dim=0)
        valid = cat(valid, dim=0)
        scale_targets = cat(scale_targets, dim=0)
    else:
        valid = pred_keypoint_logits.new_empty((0))
    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if len(heatmaps) == 0 or valid.numel() == 0:
        global _TOTAL_SKIPPED
        _TOTAL_SKIPPED += 1
        storage = get_event_storage()
        storage.put_scalar("kpts_num_skipped_batches", _TOTAL_SKIPPED, smoothing_hint=False)
        return {'ct_loss': pred_keypoint_logits.sum() * 0,
                'wh_loss': pred_keypoint_logits.sum() * 0}



    ct_loss = gaussian_focal_loss(
        pred_keypoint_logits,
        keypoint_targets
    )

    # If a normalizer isn't specified, normalize by the number of visible keypoints in the minibatch
    if normalizer is None:
        normalizer = valid.size(0)
    ct_loss /= normalizer

    # size loss
    # pred_scale_logits = pred_scale_logits.view(N, 2, H * W)
    pred_scale_logits = pred_scale_logits[valid[:, 0], :, valid[:, 1], valid[:, 2]]
    # we predict the scale wrt. feature box
    wh_loss = size_weight * F.l1_loss(pred_scale_logits, scale_targets, reduction='sum') / normalizer

    return {'ct_loss': ct_loss, 'wh_loss': wh_loss}