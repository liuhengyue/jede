import logging
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from detectron2.utils import comm
from detectron2.layers import ShapeSpec, cross_entropy
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import build_box_head
from detectron2.structures import ImageList
from detectron2.layers import cat
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY


from pgrcnn.modeling.digit_neck import build_digit_neck
from pgrcnn.modeling.utils.decode_utils import ctdet_decode
from pgrcnn.modeling.digit_head import DigitOutputLayers
from pgrcnn.structures import Boxes, Players, inside_matched_box
from pgrcnn.modeling.roi_heads import BaseROIHeads
from pgrcnn.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from pgrcnn.modeling.utils import compute_targets, compute_number_targets
from .jersey_number_head import build_jersey_number_head
from .jersey_number_neck import build_jersey_number_neck

_TOTAL_SKIPPED = 0

logger = logging.getLogger(__name__)

__all__ = ["PGROIHeads"]

@ROI_HEADS_REGISTRY.register()
class PGROIHeads(BaseROIHeads):
    def __init__(self, cfg, input_shape):
        super(PGROIHeads, self).__init__(cfg, input_shape)
        self.enable_pose_guide = cfg.MODEL.ROI_HEADS.ENABLE_POSE_GUIDE
        self.num_digit_classes = cfg.MODEL.ROI_DIGIT_NECK.NUM_DIGIT_CLASSES
        self.use_person_box_features = cfg.MODEL.ROI_DIGIT_NECK.USE_PERSON_BOX_FEATURES
        self.use_kpts_features = cfg.MODEL.ROI_DIGIT_NECK.USE_KEYPOINTS_FEATURES
        self.num_interests = cfg.DATASETS.NUM_INTERESTS
        self.batch_digit_size_per_image = cfg.MODEL.ROI_DIGIT_NECK.BATCH_DIGIT_SIZE_PER_IMAGE
        self.offset_test = cfg.MODEL.ROI_HEADS.OFFSET_TEST
        self.size_target_type = cfg.MODEL.ROI_DIGIT_NECK.SIZE_TARGET_TYPE
        self.size_target_scale = cfg.MODEL.ROI_DIGIT_NECK.SIZE_TARGET_SCALE
        self.out_head_weights = cfg.MODEL.ROI_DIGIT_NECK.OUTPUT_HEAD_WEIGHTS
        self.enable_jersey_number_det = cfg.MODEL.ROI_JERSEY_NUMBER_DET.NAME is not None
        self.enable_jersey_number_neck = cfg.MODEL.ROI_JERSEY_NUMBER_NECK.NAME is not None
        self._init_digit_head(cfg, input_shape)
        # add classification of number of jersey number
        if self.enable_jersey_number_det:
            self._init_number_head(cfg, input_shape)
            if self.enable_jersey_number_neck:
                feat_shape = self.digit_neck.intermediate_shape()
                self.jersey_number_neck = build_jersey_number_neck(cfg, feat_shape)



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
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.DIGIT_POOLER_RESOLUTION
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

        # digit_neck is used for predicting the initial digit bboxes
        # digit neck takes in the features,
        # and keypoint heatmaps of K x 56 x 56,
        # where K could be 4 or 17 depending on
        K = cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS if cfg.DATASETS.PAD_TO_FULL else cfg.DATASETS.NUM_KEYPOINTS
        out_size = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION * 4
        # construct the input shapes, in the order of keypoint heatmap, then person box features
        input_shapes = {
            "keypoint_heatmap_shape": ShapeSpec(channels=K, height=out_size, width=out_size),
            "person_box_features_shape": ShapeSpec(channels=in_channels,
                                                   height=self.keypoint_pooler.output_size[0],
                                                   width=self.keypoint_pooler.output_size[1])
        }

        if self.enable_pose_guide:
            self.digit_neck = build_digit_neck(
                cfg, input_shapes
            )

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


    def _init_number_head(self, cfg, input_shape):
        """
        After the feature pooling, each ROI feature is fed into
        a cls head, bbox reg head and kpts head
        """
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_JERSEY_NUMBER_DET.NUMBER_POOLER_RESOLUTION
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
        # do we need a neck

        # these are used for digit classification/regression after we get the digit ROI
        self.number_box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        feat_shape = ShapeSpec(channels=in_channels,
                               height=pooler_resolution[0],
                               width=pooler_resolution[1])
        self.jersey_number_head = build_jersey_number_head(cfg, feat_shape)



    def _forward_pose_guided(self, features, instances):
        """
        Forward logic from kpts heatmaps to digit centers and scales (centerNet)

        Arguments:
        """


        features = [features[f] for f in self.box_in_features]
        device = features[0].device
        # when disabled
        if not self.enable_pose_guide:
            detections = [None for _ in instances]
            # assign new fields to instances
            self.label_and_sample_digit_proposals(detections, instances)
            return {}
        if self.use_person_box_features:
            # we pool the features again for convenience
            # 14 x 14 pooler
            if self.training:
                person_features = self.keypoint_pooler(features, [x.proposal_boxes if x.has("proposal_boxes")
                                                               else Boxes([]).to(device) for x in instances])
            else:
                person_features = self.keypoint_pooler(features, [x.pred_boxes for x in instances])
        else:
            person_features = None

        if self.use_kpts_features:
            kpts_features = cat([b.pred_keypoints_logits if b.has("pred_keypoints_logits")
                               else torch.empty((0, self.keypoint_head.num_keypoints,
                                                 int(2 * self.keypoint_head.up_scale * self.keypoint_pooler.output_size[0]),
                                                 int(2 * self.keypoint_head.up_scale * self.keypoint_pooler.output_size[1])),
                                                 dtype=torch.float32, device=device)
                               for b in instances], dim=0).detach()
        else:
            kpts_features = None
        # shape (N, 3, 56, 56) (N, 2, 56, 56)
        center_heatmaps, scale_heatmaps, offset_heatmaps, fused_features = self.digit_neck(kpts_features, person_features)
        if self.training:
            loss = pg_rcnn_loss(center_heatmaps, scale_heatmaps, offset_heatmaps,
                                instances,
                                size_target_type=self.size_target_type,
                                size_target_scale=self.size_target_scale,
                                output_head_weights=self.out_head_weights,
                                target_name="digit")
            with torch.no_grad():
                bboxes_flat = cat([b.proposal_boxes.tensor if b.has("proposal_boxes")
                                   else torch.empty((0, 4), dtype=torch.float32, device=device)
                                   for b in instances], dim=0)
                # detection boxes (N, num of candidates, (x1, y1, x2, y2, score, center cls))
                detections = ctdet_decode(center_heatmaps, scale_heatmaps, bboxes_flat,
                                          reg=offset_heatmaps,
                                          K=self.num_proposal_train,
                                          fg_ratio=self.fg_ratio,
                                          size_target_type=self.size_target_type,
                                          size_target_scale=self.size_target_scale)
                # deal with svhn since the instances will not contain proposal_boxes
                len_instances = [len(instance) if instance.has("proposal_boxes") else 0 for instance in instances]
                detections = list(detections.split(len_instances))
                # assign new fields to instances
                self.label_and_sample_digit_proposals(detections, instances)

            # forward with number features
            if self.enable_jersey_number_neck:
                center_heatmaps, scale_heatmaps, offset_heatmaps = self.jersey_number_neck(fused_features)
                number_loss = pg_rcnn_loss(center_heatmaps, scale_heatmaps, offset_heatmaps,
                                            instances,
                                            size_target_type=self.size_target_type,
                                            size_target_scale=self.size_target_scale,
                                            output_head_weights=self.out_head_weights,
                                            target_name="number")
                loss.update(number_loss)
                with torch.no_grad():
                    bboxes_flat = cat([b.proposal_boxes.tensor if b.has("proposal_boxes")
                                       else torch.empty((0, 4), dtype=torch.float32, device=device)
                                       for b in instances], dim=0)
                    # detection boxes (N, num of candidates, (x1, y1, x2, y2, score, center cls))
                    detections = ctdet_decode(center_heatmaps, scale_heatmaps, bboxes_flat,
                                              reg=offset_heatmaps,
                                              K=self.num_proposal_train,
                                              fg_ratio=self.fg_ratio,
                                              size_target_type=self.size_target_type,
                                              size_target_scale=self.size_target_scale)
                    # deal with svhn since the instances will not contain proposal_boxes
                    len_instances = [len(instance) if instance.has("proposal_boxes") else 0 for instance in instances]
                    detections = list(detections.split(len_instances))
                    # assign new fields to instances
                    self.label_and_sample_jerseynumber_proposals(detections, instances)


            return loss

        else:
            bboxes_flat = cat([b.pred_boxes.tensor for b in instances], dim=0)
            # (N, num of candidates, (x1, y1, x2, y2, score, center 0 /left 1/right 2)
            detection = ctdet_decode(center_heatmaps, scale_heatmaps, bboxes_flat,
                                     reg=offset_heatmaps,
                                     K=self.num_proposal_test,
                                     size_target_type=self.size_target_type,
                                     size_target_scale=self.size_target_scale,
                                     training=False,
                                     offset = self.offset_test
            )
            num_instances = [len(instance) for instance in instances]
            detection_boxes = list(detection[..., :4].split(num_instances))
            detection_ct_classes = list(detection[..., -1].split(num_instances))
            if self.enable_jersey_number_neck:
                center_heatmaps, scale_heatmaps, offset_heatmaps = self.jersey_number_neck(fused_features)
                detection = ctdet_decode(center_heatmaps, scale_heatmaps, bboxes_flat,
                                         reg=offset_heatmaps,
                                         K=self.num_proposal_test,
                                         size_target_type=self.size_target_type,
                                         size_target_scale=self.size_target_scale,
                                         training=False,
                                         offset=self.offset_test
                                         )
                detection_number_boxes = list(detection[..., :4].split(num_instances))
            # assign new fields to instances
            for i, (boxes, detection_ct_cls) in enumerate(zip(detection_boxes, detection_ct_classes)):
                # perform a person roi clip or not
                boxes = [Boxes(b) for b in boxes] # List of N `Boxes'
                pred_person_boxes = instances[i].pred_boxes # [N, 4]
                boxes = [boxes_per_ins[inside_matched_box(boxes_per_ins, pred_person_boxes[j])] for j, boxes_per_ins in enumerate(boxes)]
                instances[i].proposal_digit_boxes = boxes
                if self.enable_jersey_number_neck:
                    number_boxes = [Boxes(b) for b in detection_number_boxes[i]]
                    number_boxes = [boxes_per_ins[inside_matched_box(boxes_per_ins, pred_person_boxes[j])] for j, boxes_per_ins
                             in enumerate(number_boxes)]
                    instances[i].pred_number_boxes = number_boxes

            return instances

    def _forward_digit_box(self, features, proposals):
        features = [features[f] for f in self.in_features]
        # most likely have empty boxes, Boxes.cat([]) will return Boxes on cpu
        detection_boxes = [Boxes.cat(x.proposal_digit_boxes).to(features[0].device)
                           if x.has('proposal_digit_boxes') else Boxes.cat([]).to(features[0].device)
                           for x in proposals]
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

    def _forward_number_box(self, features, proposals):
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
        box_features = self.jersey_number_head(box_features)

        if self.training:
            return self.jersey_number_head.losses(box_features, proposals)
        else:
            pred_instances = self.jersey_number_head.inference(box_features, proposals)
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
        if self.enable_jersey_number_det:
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
            # proposals or sampled_instances will be modified in-place
            losses = self._forward_box(features, proposals)
            kpt_loss, sampled_instances = self._forward_keypoint(features, proposals)
            losses.update(kpt_loss)
            losses.update(self._forward_pose_guided(features, sampled_instances))
            losses.update(self._forward_digit_box(features, sampled_instances))
            if self.enable_jersey_number_det:
                losses.update(self._forward_number_box(features, sampled_instances))
            return sampled_instances, losses
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
    # pred = torch.sigmoid(pred)
    pos_weights = gaussian_target.eq(1)
    neg_weights = (1 - gaussian_target).pow(gamma)
    pos_loss = (-(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights).sum()
    neg_loss = (-(1 - pred + eps).log() * pred.pow(alpha) * neg_weights).sum()
    return pos_loss + neg_loss

def ltrb_giou_loss(pred, target, ct_weights = None, eps: float = 1e-7, reduction='sum'):
    """
    Args:
        pred: Nx4 predicted bounding boxes
        target: Nx4 target bounding boxes
        Both are in the form of FCOS prediction (l, t, r, b)
    """
    # pred.clamp_(min=0)
    pred_left = pred[:, 0]
    pred_top = pred[:, 1]
    pred_right = pred[:, 2]
    pred_bottom = pred[:, 3]

    target_left = target[:, 0]
    target_top = target[:, 1]
    target_right = target[:, 2]
    target_bottom = target[:, 3]

    target_aera = (target_left + target_right) * \
                  (target_top + target_bottom)
    pred_aera = (pred_left + pred_right) * \
                (pred_top + pred_bottom)

    w_intersect = torch.min(pred_left, target_left) + \
                  torch.min(pred_right, target_right)
    h_intersect = torch.min(pred_bottom, target_bottom) + \
                  torch.min(pred_top, target_top)

    g_w_intersect = torch.max(pred_left, target_left) + \
                    torch.max(pred_right, target_right)
    g_h_intersect = torch.max(pred_bottom, target_bottom) + \
                    torch.max(pred_top, target_top)
    ac_uion = g_w_intersect * g_h_intersect

    area_intersect = w_intersect * h_intersect
    area_union = target_aera + pred_aera - area_intersect

    ious = area_intersect / (area_union + eps)
    gious = ious - (ac_uion - area_union) / (ac_uion + eps)
    loss = 1 - gious
    if ct_weights is not None:
        loss = loss * ct_weights
    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def pg_rcnn_loss(
        pred_keypoint_logits,
        pred_scale_logits,
        pred_offset_logits,
        instances,
        normalizer=None,
        output_head_weights=(1,0, 1.0, 1.0),
        size_target_type="ltrb",
        size_target_scale="feature",
        target_name="digit"
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
        normalizer (Union[float, None]): Normalize the loss by this amount.
            If not specified, we normalize by the number of visible keypoints in the minibatch.
        size_weight (float): Weight for the size regression loss.

    Returns a scalar tensor containing the loss.
    """
    has_offset_reg = pred_offset_logits is not None
    heatmaps = []
    valid = []
    scale_targets = []
    offset_targets =[]
    # keypoint_side_len = pred_keypoint_logits.shape[2]
    for instances_per_image in instances:
        heatmaps_per_image, scales_per_image, offsets_per_image, valid_per_image = \
            compute_targets(instances_per_image, pred_keypoint_logits.shape[1:],
                            offset_reg=has_offset_reg,
                            size_target_type=size_target_type,
                            size_target_scale=size_target_scale,
                            target_name=target_name)
        heatmaps.append(heatmaps_per_image)
        valid.append(valid_per_image)
        scale_targets.append(scales_per_image)
        offset_targets.append(offsets_per_image)
    # should be safe since we return empty tensors from `compute_targets'
    keypoint_targets = cat(heatmaps, dim=0)
    valid = cat(valid, dim=0)
    scale_targets = cat(scale_targets, dim=0)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if keypoint_targets.numel() == 0 or valid.numel() == 0:
        loss = {target_name + '_ct_loss': pred_keypoint_logits.sum() * 0,
                target_name + '_wh_loss': pred_scale_logits.sum() * 0}
        if has_offset_reg:
            loss.update({target_name + "_os_loss": pred_offset_logits.sum() * 0})
        return loss



    ct_loss = gaussian_focal_loss(
        pred_keypoint_logits,
        keypoint_targets
    ) * output_head_weights[0]

    # If a normalizer isn't specified, normalize by the number of visible keypoints in the minibatch
    if normalizer is None:
        normalizer = valid.size(0)
    ct_loss /= normalizer

    # size loss
    if size_target_type == 'wh':
        pred_scale_logits = pred_scale_logits[valid[:, 0], :, valid[:, 1], valid[:, 2]]
        # we predict the scale wrt. feature box
        wh_loss = output_head_weights[1] * F.smooth_l1_loss(pred_scale_logits, scale_targets,
                                                            reduction='sum') / normalizer
    elif size_target_type == 'ltrb':
        # valid_loc = (keypoint_targets > 0.).squeeze(1)
        # giou loss
        # wh_loss = output_head_weights[1] * ltrb_giou_loss(pred_scale_logits.permute(0, 2, 3, 1)[valid_loc],
        #                                                   scale_targets.permute(0, 2, 3, 1)[valid_loc],
        #                                                   None,
        #                                                   reduction='sum') / valid_loc.sum()

        # smooth_l1
        # wh_loss = output_head_weights[1] * F.smooth_l1_loss(pred_scale_logits.permute(0, 2, 3, 1)[valid_loc],
        #                                                     scale_targets.permute(0, 2, 3, 1)[valid_loc],
        #                                                     reduction='sum') / normalizer

        # weighted by center
        wh_loss = output_head_weights[1] * (F.smooth_l1_loss(pred_scale_logits,
                                                          scale_targets) * torch.square(keypoint_targets)).sum()

    loss = {target_name + '_ct_loss': ct_loss, target_name + '_wh_loss': wh_loss}
    if has_offset_reg:
        offset_targets = cat(offset_targets, dim=0)
        pred_offset_logits = pred_offset_logits[valid[:, 0], :, valid[:, 1], valid[:, 2]]
        os_loss = output_head_weights[2] * F.smooth_l1_loss(pred_offset_logits, offset_targets, reduction='sum') / normalizer
        loss.update({target_name + "_os_loss": os_loss})

    # if has_num_digits_logits:
    #     gt_num_digits = torch.bincount(valid[:, 0], minlength=keypoint_targets.shape[0])
    #     num_digit_cls_loss = cross_entropy(num_digits_logits, gt_num_digits)
    #     loss.update({"num_digit_cls_loss": num_digit_cls_loss})

    return loss

def pg_rcnn_number_loss(
        pred_keypoint_logits,
        pred_scale_logits,
        pred_offset_logits,
        instances,
        normalizer=None,
        output_head_weights=(1,0, 1.0, 1.0),
        size_target_type="ltrb",
        size_target_scale="feature"
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
        normalizer (Union[float, None]): Normalize the loss by this amount.
            If not specified, we normalize by the number of visible keypoints in the minibatch.
        size_weight (float): Weight for the size regression loss.

    Returns a scalar tensor containing the loss.
    """
    has_offset_reg = pred_offset_logits is not None
    heatmaps = []
    valid = []
    scale_targets = []
    offset_targets =[]
    # keypoint_side_len = pred_keypoint_logits.shape[2]
    for instances_per_image in instances:
        heatmaps_per_image, scales_per_image, offsets_per_image, valid_per_image = \
            compute_number_targets(instances_per_image, pred_keypoint_logits.shape[1:],
                                    offset_reg=has_offset_reg,
                                    size_target_type=size_target_type,
                                    size_target_scale=size_target_scale)
        heatmaps.append(heatmaps_per_image)
        valid.append(valid_per_image)
        scale_targets.append(scales_per_image)
        offset_targets.append(offsets_per_image)
    # should be safe since we return empty tensors from `compute_targets'
    keypoint_targets = cat(heatmaps, dim=0)
    valid = cat(valid, dim=0)
    scale_targets = cat(scale_targets, dim=0)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if keypoint_targets.numel() == 0 or valid.numel() == 0:
        loss = {'ct_loss': pred_keypoint_logits.sum() * 0,
                'wh_loss': pred_scale_logits.sum() * 0}
        if has_offset_reg:
            loss.update({"os_loss": pred_offset_logits.sum() * 0})
        return loss



    ct_loss = gaussian_focal_loss(
        pred_keypoint_logits,
        keypoint_targets
    ) * output_head_weights[0]

    # If a normalizer isn't specified, normalize by the number of visible keypoints in the minibatch
    if normalizer is None:
        normalizer = valid.size(0)
    ct_loss /= normalizer

    # size loss
    if size_target_type == 'wh':
        pred_scale_logits = pred_scale_logits[valid[:, 0], :, valid[:, 1], valid[:, 2]]
        # we predict the scale wrt. feature box
        wh_loss = output_head_weights[1] * F.smooth_l1_loss(pred_scale_logits, scale_targets,
                                                            reduction='sum') / normalizer
    elif size_target_type == 'ltrb':
        # valid_loc = (keypoint_targets > 0.).squeeze(1)
        # giou loss
        # wh_loss = output_head_weights[1] * ltrb_giou_loss(pred_scale_logits.permute(0, 2, 3, 1)[valid_loc],
        #                                                   scale_targets.permute(0, 2, 3, 1)[valid_loc],
        #                                                   None,
        #                                                   reduction='sum') / valid_loc.sum()

        # smooth_l1
        # wh_loss = output_head_weights[1] * F.smooth_l1_loss(pred_scale_logits.permute(0, 2, 3, 1)[valid_loc],
        #                                                     scale_targets.permute(0, 2, 3, 1)[valid_loc],
        #                                                     reduction='sum') / normalizer

        # weighted by center
        wh_loss = output_head_weights[1] * (F.smooth_l1_loss(pred_scale_logits,
                                                          scale_targets) * torch.square(keypoint_targets)).sum()

    loss = {'ct_loss': ct_loss, 'wh_loss': wh_loss}
    if has_offset_reg:
        offset_targets = cat(offset_targets, dim=0)
        pred_offset_logits = pred_offset_logits[valid[:, 0], :, valid[:, 1], valid[:, 2]]
        os_loss = output_head_weights[2] * F.smooth_l1_loss(pred_offset_logits, offset_targets, reduction='sum') / normalizer
        loss.update({"os_loss": os_loss})

    # if has_num_digits_logits:
    #     gt_num_digits = torch.bincount(valid[:, 0], minlength=keypoint_targets.shape[0])
    #     num_digit_cls_loss = cross_entropy(num_digits_logits, gt_num_digits)
    #     loss.update({"num_digit_cls_loss": num_digit_cls_loss})

    return loss