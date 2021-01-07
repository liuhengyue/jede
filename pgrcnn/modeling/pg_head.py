import torch
from torch.nn import functional as F
from typing import Dict, List, Optional, Tuple, Union
from detectron2.layers import ShapeSpec
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import build_box_head
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs
from detectron2.structures import Boxes, ImageList, pairwise_iou, heatmaps_to_keypoints
from detectron2.layers import cat
from detectron2.utils.events import get_event_storage
from detectron2.structures.boxes import Boxes
from pgrcnn.modeling.kpts2digit_head import build_digit_head
from pgrcnn.utils.ctnet_utils import ctdet_decode
from pgrcnn.structures.digitboxes import DigitBoxes
from pgrcnn.modeling.digit_head import DigitOutputLayers
from pgrcnn.structures.instances import CustomizedInstances as Instances
from pgrcnn.modeling.digit_head import paired_iou
from pgrcnn.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.sampling import subsample_labels
from detectron2.structures.keypoints import Keypoints
_TOTAL_SKIPPED = 0



@ROI_HEADS_REGISTRY.register()
class PGROIHeads(StandardROIHeads):
    def __init__(self, cfg, input_shape):
        super(PGROIHeads, self).__init__(cfg, input_shape)
        self.num_digit_classes = cfg.MODEL.ROI_DIGIT_HEAD.NUM_DIGIT_CLASSES
        self._init_digit_head(cfg, input_shape)

    def _sample_digit_proposals(
        self, matched_idxs: torch.Tensor, matched_labels: torch.Tensor, gt_classes: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            gt_classes (Tensor): a vector of length M.

        Returns:
            Tensor: a vector of indices of sampled proposals. Each is in [0, N).
            Tensor: a vector of the same length, the classification label for
                each sampled proposal. Each sample is labeled as either a category in
                [0, num_classes) or the background (num_classes).
        """
        has_gt = gt_classes.numel() > 0
        # Get the corresponding GT for each proposal
        if has_gt:
            gt_classes = gt_classes[matched_idxs]
            # Label unmatched proposals (0 label from matcher) as background (for digit is 0)
            gt_classes[matched_labels == 0] = 0
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1


        else:
            gt_classes = torch.zeros_like(matched_idxs) + self.num_classes

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_size_per_image, self.positive_sample_fraction, 0
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

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

        self.num_ctdet_proposal = cfg.MODEL.ROI_DIGIT_HEAD.NUM_PROPOSAL

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


        self.digit_head = build_digit_head(
            cfg, ShapeSpec(channels=4, height=56, width=56)
        )

    def _process_single_instance(self, detection, instance):
        """
        detecion_boxes: (3, 4)
        instance: single instance of one image
        """
        boxes = Boxes(detection[..., :4])
        boxes.clip(instance.image_size)
        keep = boxes.nonempty()
        boxes = boxes[keep]
        detection_ct_classes = detection[..., -1]
        # (1, 2, 4) -> (2, 4)
        gt_digit_boxes = instance.gt_digit_boxes.flat()
        # keep = gt_digit_boxes.nonempty()
        # gt_digit_boxes = gt_digit_boxes[keep]
        # gt_digit_ct_classes = instance.gt_digit_ct_classes
        # (1, 2, 1) -> (2)
        gt_digit_classes = instance.gt_digit_classes.view(-1)
        # Keypoints (1, 3, 3)
        # gt_digit_centers = instance.gt_digit_centers
        # (1, 2, 2) -> (2, 2)
        # gt_digit_scales = instance.gt_digit_scales.view(-1, 2)

        # get ground-truth match based on iou
        match_quality_matrix = pairwise_iou(
            gt_digit_boxes, boxes
        )
        matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
        # if gt_digit_classes is 0, then it is background
        sampled_idxs, gt_digit_classes = \
            self._sample_digit_proposals(matched_idxs, matched_labels, gt_digit_classes)

        sampled_targets = matched_idxs[sampled_idxs]
        # need centers as Keypoints of shape (N, 3, 3)
        instance.proposal_digit_boxes = boxes[sampled_idxs]
        instance.proposal_digit_ct_classes = detection_ct_classes[keep]
        instance.gt_digit_boxes = gt_digit_boxes[sampled_targets]
        instance.gt_digit_classes = gt_digit_classes
        # instance.gt_digit_ct_classes = gt_digit_ct_classes[sampled_targets]
        # instance.gt_digit_centers = Keypoints(gt_digit_centers.tensor[sampled_targets])
        # instance.gt_digit_scales = gt_digit_scales[sampled_targets]
        # instance.gt_num_digit_scales =  [sampled_targets.shape[0]]
        # instance.gt_digit_centers = Keypoints(gt_digit_centers[None, ...].repeat_interleave( \
        #     len_instances[i], 0))
        # instances[i].gt_digit_scales = gt_digit_scales[None, ...].repeat_interleave( \
        #     len_instances[i], 0)
        return instance


    def _forward_ctdet(self, kpts_heatmaps, instances):
        """
        Forward logic from kpts heatmaps to digit centers and scales (centerNet)

        Arguments:
            kpts_heatmaps:
                A tensor of shape (N, K, S, S) where N is the total number
            of instances in the batch, K is the number of keypoints, and S is the side length
            of the keypoint heatmap. The values are spatial logits.
        """
        # shape (N, 3, 56, 56) (N, 2, 56, 56)
        center_heatmaps, scale_heatmaps = self.digit_head(kpts_heatmaps)
        # todo: check if center_heatmaps activations on center is not correct
        if self.training:
            with torch.no_grad():
                bboxes_flat = cat([b.proposal_boxes.tensor for b in instances], dim=0)
                # (N, num of candidates, (x1, y1, x2, y2, score, center 0 /left 1/right 2)
                detections = ctdet_decode(center_heatmaps, scale_heatmaps, bboxes_flat,
                                         K=self.num_ctdet_proposal, feature_scale=True)
                # todo: has duplicate boxes
                len_instances = [len(instance) for instance in instances]
                detections = list(detections.split(len_instances))
                # assign new fields to instances
                # per image
                for i, (detection, instance) in enumerate(zip(detections, instances)):
                    if len(instance):
                        processed_instances = []
                        for j in range(len(instance)):
                            processed_instance = self._process_single_instance(detection[j], instance[j])
                            processed_instances.append(processed_instance)
                        instances[i] = Instances.cat(processed_instances)
                    else:
                        device = instance.proposal_boxes.device
                        instances[i].proposal_digit_boxes = Boxes(torch.zeros(0, 4, device=device))
                        instances[i].proposal_digit_ct_classes = torch.zeros(0, device=device).long()
                        instances[i].gt_digit_boxes = Boxes(torch.zeros(0, 4, device=device))
                        instances[i].gt_digit_classes = torch.zeros(0, device=device).long()




            loss = ctdet_loss(center_heatmaps, scale_heatmaps, instances, None)
            # center_loss = ct_loss(center_heatmaps, instances, None)
            # scale_loss = hw_loss(scale_heatmaps, instances, feature_scale=True)
            # return {'ct_loss': center_loss,
            #         'wh_loss': scale_loss}
            return loss

        else:
            bboxes_flat = cat([b.pred_boxes.tensor for b in instances], dim=0)
            # (N, num of candidates, (x1, y1, x2, y2, score, center 0 /left 1/right 2)
            detection = ctdet_decode(center_heatmaps, scale_heatmaps, bboxes_flat,
                                     K=self.num_ctdet_proposal, feature_scale=True)
            detection_boxes = list(detection[..., :4].split([len(instance) for instance in instances]))
            detection_ct_classes = list(detection[..., -1].split([len(instance) for instance in instances]))
            # assign new fields to instances
            for i, boxes in enumerate(detection_boxes):
                instances[i].proposal_digit_boxes = Boxes(boxes.view(-1, 4))
                instances[i].proposal_digit_ct_classes = detection_ct_classes[i]
            return instances

    def _forward_digit_box(self, features, proposals):
        features = [features[f] for f in self.in_features]
        # most likely have empty boxes
        detection_boxes = [x.proposal_digit_boxes for x in proposals]
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
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
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
        keypoints_logits = cat([instance.pred_keypoints_logits for instance in instances], dim=0)
        instances = self._forward_ctdet(keypoints_logits, instances)
        instances = self._forward_digit_box(features, instances)
        # remove proposal boxes
        for instance in instances:
            instance.remove('proposal_digit_boxes')

        return instances

    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, proposals))
            kpt_loss, sampled_keypoints_logits, sampled_instances = self._forward_keypoint(features, proposals)
            losses.update(kpt_loss)
            losses.update(self._forward_ctdet(sampled_keypoints_logits, sampled_instances))
            losses.update(self._forward_digit_box(features, sampled_instances))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

def ct_loss(pred_keypoint_logits, instances, normalizer):
    """
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

    keypoint_side_len = pred_keypoint_logits.shape[2]
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        keypoints = instances_per_image.gt_digit_centers
        proposal_boxes = instances_per_image.proposal_boxes.tensor
        heatmaps_per_image, valid_per_image = keypoints.to_heatmap(
            proposal_boxes, keypoint_side_len
        )
        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))

    if len(heatmaps):
        keypoint_targets = cat(heatmaps, dim=0)
        valid = cat(valid, dim=0).to(dtype=torch.uint8)
        # use for suppress other ct classes
        # not_valid = (valid == 0).nonzero().squeeze(1)
        # valid = torch.nonzero(valid).squeeze(1)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if len(heatmaps) == 0 or valid.numel() == 0:
        global _TOTAL_SKIPPED
        _TOTAL_SKIPPED += 1
        storage = get_event_storage()
        storage.put_scalar("kpts_num_skipped_batches", _TOTAL_SKIPPED, smoothing_hint=False)
        return pred_keypoint_logits.sum() * 0

    N, K, H, W = pred_keypoint_logits.shape
    pred_keypoint_logits = pred_keypoint_logits.view(N * K, H * W)

    keypoint_loss = F.cross_entropy(
        pred_keypoint_logits[valid], keypoint_targets[valid], reduction="sum"
    ) / valid.numel()

    # keypoint_loss += F.cross_entropy(
    #     pred_keypoint_logits[not_valid], keypoint_targets[not_valid], reduction="sum"
    # ) / not_valid.numel()

    # If a normalizer isn't specified, normalize by the number of visible keypoints in the minibatch
    # if normalizer is None:
    #     normalizer = valid.numel()
    # keypoint_loss /= normalizer

    return keypoint_loss

def hw_loss(pred_scale, instances, hw_weight=1.0, feature_scale=True):
    """
    instances (list[Instances]): A list of M Instances, where M is the batch size.
        These instances are predictions from the model
        that are in 1:1 correspondence with pred_keypoint_logits.
        Each Instances should contain a `gt_keypoints` field containing a `structures.Keypoint`
        instance.
    """

    # get gt scale masks, the shape should be (N, 2, 56, 56)
    N = pred_scale.shape[0]
    ft_side_len = pred_scale.shape[2]
    gt_scale_maps = to_scale_mask(instances, ft_side_len, feature_scale=feature_scale)
    # this will only select one point per map
    valid = gt_scale_maps > 0
    loss = hw_weight * F.l1_loss(pred_scale[valid], gt_scale_maps[valid], reduction='sum') / N
    # normalize by the number of instances, this does not work
    # loss = hw_weight * F.l1_loss(pred_scale, gt_scale_maps, reduction='sum') / N
    return loss

def ctdet_loss(pred_keypoint_logits, pred_scale_logits, instances, normalizer):
    """
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
    scales = []
    x_s = []
    y_s = []

    keypoint_side_len = pred_keypoint_logits.shape[2]
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        keypoints = instances_per_image.gt_digit_centers.tensor
        scales_per_image = instances_per_image.gt_digit_scales
        proposal_boxes = instances_per_image.proposal_boxes.tensor
        # gt_num_digit_scales = instances_per_image.gt_num_digit_scales
        heatmaps_per_image, scales_per_image, valid_per_image, x, y = keypoints_to_heatmap(\
            keypoints, scales_per_image, proposal_boxes, keypoint_side_len)

        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))
        scales.append(scales_per_image.view(-1, 2))
        # scales.append(scales_per_image)
        x_s.append(x.view(-1))
        y_s.append(y.view(-1))

    if len(heatmaps):
        keypoint_targets = cat(heatmaps, dim=0)
        valid = cat(valid, dim=0).to(dtype=torch.uint8)
        valid = torch.nonzero(valid).squeeze(1)

        scale_targets = cat(scales, dim=0)
        x_s = cat(x_s, dim=0)
        y_s = cat(y_s, dim=0)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if len(heatmaps) == 0 or valid.numel() == 0:
        global _TOTAL_SKIPPED
        _TOTAL_SKIPPED += 1
        storage = get_event_storage()
        storage.put_scalar("kpts_num_skipped_batches", _TOTAL_SKIPPED, smoothing_hint=False)
        return {'ct_loss': pred_keypoint_logits.sum() * 0, 'wh_loss': pred_scale_logits.sum() * 0}
    # pred_scale_logits[torch.stack((x_s, y_s), dim=-1).unsqueeze(1).repeat(1, 2, 1, 1)]
    N, K, H, W = pred_keypoint_logits.shape
    pred_keypoint_logits = pred_keypoint_logits.view(N * K, H * W)
    # pred_scale_logits = pred_scale_logits.repeat_interleave(3, dim=0)
    valid_scale_logits = pred_scale_logits[valid // 3, :, x_s[valid], y_s[valid]]
    ct_loss = F.cross_entropy(
        pred_keypoint_logits[valid], keypoint_targets[valid], reduction="sum"
    ) / valid.numel()

    wh_loss = F.l1_loss(valid_scale_logits, scale_targets[valid], reduction='sum') / valid.numel()


    return {'ct_loss': ct_loss, 'wh_loss': wh_loss}

def ct_hm_loss(pred_keypoint_logits, instances, normalizer):
    """
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

    keypoint_side_len = pred_keypoint_logits.shape[2]
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        # shape (N, 3, 3)
        keypoints = instances_per_image.gt_digit_centers
        heatmap = draw_msra_gaussian(keypoints, instances_per_image.proposal_boxes.tensor, keypoint_side_len)
        heatmaps.append(heatmap)

    if len(heatmaps):
        keypoint_targets = cat(heatmaps, dim=0)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if len(heatmaps) == 0:
        global _TOTAL_SKIPPED
        _TOTAL_SKIPPED += 1
        storage = get_event_storage()
        storage.put_scalar("kpts_num_skipped_batches", _TOTAL_SKIPPED, smoothing_hint=False)
        return pred_keypoint_logits.sum() * 0


    keypoint_loss = F.mse_loss(
        pred_keypoint_logits, keypoint_targets, reduction="mean"
    )

    # If a normalizer isn't specified, normalize by the number of visible keypoints in the minibatch
    # if normalizer is None:
    #     normalizer = valid.numel()
    # keypoint_loss /= normalizer

    return keypoint_loss

def keypoints_to_heatmap(
    keypoints: torch.Tensor, scales: torch.Tensor, rois: torch.Tensor, heatmap_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode keypoint locations into a target heatmap for use in SoftmaxWithLoss across space.

    Maps keypoints from the half-open interval [x1, x2) on continuous image coordinates to the
    closed interval [0, heatmap_size - 1] on discrete image coordinates. We use the
    continuous-discrete conversion from Heckbert 1990 ("What is the coordinate of a pixel?"):
    d = floor(c) and c = d + 0.5, where d is a discrete coordinate and c is a continuous coordinate.

    Arguments:
        keypoints: tensor of keypoint locations in of shape (N, K, 3).
        rois: Nx4 tensor of rois in xyxy format
        heatmap_size: integer side length of square heatmap.

    Returns:
        heatmaps: A tensor of shape (N, K) containing an integer spatial label
            in the range [0, heatmap_size**2 - 1] for each keypoint in the input.
        valid: A tensor of shape (N, K) containing whether each keypoint is in
            the roi or not.
    """

    if rois.numel() == 0:
        return rois.new().long(), rois.new().long()
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]

    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()

    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1

    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    vis = keypoints[..., 2] > 0
    valid = (valid_loc & vis).long()

    lin_ind = y * heatmap_size + x
    heatmaps = lin_ind * valid

    # (N, 1, 2) or (N, 2, 2)
    # N, k, d = scales.shape
    # if k == 1:
    #     scales = cat((scales, torch.zeros((N, 2, d), device=scales.device)), dim=1)
    # else: # k == 2
    #     scales = cat((torch.zeros((N, 1, d), device=scales.device), scales), dim=1)
    #
    # scales[..., 0] *= scale_x
    # scales[..., 1] *= scale_y

    # (N, 1, 2) or (N, 2, 2)
    # scales = torch.split(scales, gt_num_digit_scales)
    # for i, scale in enumerate(scales):
    #     scale[..., 0] *= scale_x[i]
    #     scale[..., 1] *= scale_y[i]
    # scales = cat(scales, dim=0)
    scales[..., 0] *= scale_x
    scales[..., 1] *= scale_y

    return heatmaps, scales, valid, x, y

def to_scale_mask(instances, ft_side_len, feature_scale=True):
    dense_wh_maps = []

    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        # (N, 3, 3)
        cts = instances_per_image.gt_digit_centers.tensor
        N = cts.shape[0]
        # (N, 2 [two digit placeholders], 2[w, h])
        gt_wh = instances_per_image.gt_digit_scales
        # this cause the memory vary
        dense_wh = torch.zeros((N, 2, ft_side_len, ft_side_len), dtype=torch.float64, device=cts.device)
        rois = instances_per_image.proposal_boxes.tensor

        offset_x = rois[:, 0]
        offset_y = rois[:, 1]
        scale_x = ft_side_len / (rois[:, 2] - rois[:, 0])
        scale_y = ft_side_len / (rois[:, 3] - rois[:, 1])

        offset_x = offset_x[:, None]
        offset_y = offset_y[:, None]
        scale_x = scale_x[:, None]
        scale_y = scale_y[:, None]

        x = cts[..., 0]
        y = cts[..., 1]

        x_boundary_inds = x == rois[:, 2][:, None]
        y_boundary_inds = y == rois[:, 3][:, None]

        x = (x - offset_x) * scale_x
        x = x.floor().long()
        y = (y - offset_y) * scale_y
        y = y.floor().long()

        x[x_boundary_inds] = ft_side_len - 1
        y[y_boundary_inds] = ft_side_len - 1

        if feature_scale:
            gt_wh[..., 0] *= scale_x
            gt_wh[..., 1] *= scale_y

        valid_loc = (x >= 0) & (y >= 0) & (x < ft_side_len) & (y < ft_side_len)
        vis = cts[..., 2] > 0
        valid = (valid_loc & vis).long()
        # valid_digit_loc = torch.nonzero(valid, as_tuple=True)[1]

        for i in range(N):
            gt_wh_i = gt_wh[i,...]
            x_i = x[i,...]
            y_i = y[i,...]
            valid_i = valid[i,...].nonzero().squeeze_(1)
            x_i = x_i[valid_i]
            y_i = y_i[valid_i]
            # if valid_i is 0 -> 0
            # if valid_i is 1 and 2 -> 0, 1 (two boxes are all valid)
            valid_i.add_(-1).clamp_(0, 2) # map the index to center left right
            dense_wh[i,:, x_i, y_i] = gt_wh_i[valid_i].permute(1,0)

        dense_wh_maps.append(dense_wh)

    dense_wh_maps = cat(dense_wh_maps, dim=0)

    return dense_wh_maps

def draw_msra_gaussian(keypoints, rois, heatmap_size):
    keypoints = keypoints.tensor
    device = keypoints.device
    # convert into feature level keypoints
    if rois.numel() == 0:
        return rois.new().long(), rois.new().long()
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size / (rois[:, 3] - rois[:, 1])

    offset_x = offset_x[:, None]
    offset_y = offset_y[:, None]
    scale_x = scale_x[:, None]
    scale_y = scale_y[:, None]

    x = keypoints[..., 0]
    y = keypoints[..., 1]

    x_boundary_inds = x == rois[:, 2][:, None]
    y_boundary_inds = y == rois[:, 3][:, None]

    x = (x - offset_x) * scale_x
    x = x.floor().long()
    y = (y - offset_y) * scale_y
    y = y.floor().long()

    x[x_boundary_inds] = heatmap_size - 1
    y[y_boundary_inds] = heatmap_size - 1

    valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size) & (y < heatmap_size)
    vis = keypoints[..., 2] > 0
    valid = (valid_loc & vis).long()
    # get the centers
    x = x * valid
    y = y * valid
    # for each roi
    N, k = valid.shape
    heatmaps = []
    sigma = 1
    tmp_size = sigma * 3
    w, h = heatmap_size, heatmap_size
    size = 2 * tmp_size + 1
    x_g_map = torch.arange(0, size, 1, dtype=torch.float32, device=device)
    y_g_map = x_g_map[:, None]
    x0 = y0 = size // 2
    g = torch.exp(- ((x_g_map - x0) ** 2 + (y_g_map - y0) ** 2) / (2 * sigma ** 2))
    # scatter the Gaussian points
    for i in range(N):
        # create gt heatmap for the center points
        heatmap = torch.zeros((k, heatmap_size, heatmap_size), dtype=torch.float32, device=device)
        mu_x = x[i]
        mu_y = y[i]
        # it can be empty tensor
        valid_ct_idx = ((mu_y > 0) * (mu_x > 0)).nonzero(as_tuple=True)[0]
        if valid_ct_idx.numel() == 0:
            heatmaps.append(heatmap)
            continue
        mu_x = mu_x[mu_x > 0]
        mu_y = mu_y[mu_y > 0]
        ul = [(mu_x - tmp_size).clamp_(0, heatmap_size).long(), (mu_y - tmp_size).clamp_(0, heatmap_size).long()]
        br = [(mu_x + tmp_size + 1).clamp_(0, heatmap_size).long(), (mu_y + tmp_size + 1).clamp_(0, heatmap_size).long()]

        g_x = (-ul[0]).clamp_(min=0), (br[0]).clamp_(max=h) - ul[0]
        g_y = (-ul[1]).clamp_(min=0), (br[1]).clamp_(max=w) - ul[1]
        img_x = (ul[0]).clamp_(min=0), (br[0]).clamp_(max=h)
        img_y = (ul[1]).clamp_(min=0), (br[1]).clamp_(max=w)
        for i, ct_idx in enumerate(valid_ct_idx):
            try:
                heatmap[ct_idx, img_y[0][i]:img_y[1][i], img_x[0][i]:img_x[1][i]] = torch.max(
                    heatmap[ct_idx, img_y[0][i]:img_y[1][i], img_x[0][i]:img_x[1][i]],
                    g[g_y[0][i]:g_y[1][i], g_x[0][i]:g_x[1][i]])
            except:
                raise Exception('wtf')
        heatmaps.append(heatmap)
    return torch.stack(heatmaps, dim=0)