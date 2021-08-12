# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import inspect
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import nn

from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.structures import pairwise_iou
from detectron2.structures.boxes import matched_boxlist_iou
from detectron2.utils.events import get_event_storage
from detectron2.config import configurable
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.roi_heads import StandardROIHeads
from detectron2.modeling.roi_heads import ROI_HEADS_REGISTRY

from pgrcnn.structures import Players, Boxes
from pgrcnn.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals

logger = logging.getLogger(__name__)

__all__ = ["BaseROIHeads"]

def select_foreground_proposals(
    proposals: List[Players], bg_label: int
) -> Tuple[List[Players], List[torch.Tensor]]:
    """
    Given a list of N Instances (for N images), each containing a `gt_classes` field,
    return a list of Instances that contain only instances with `gt_classes != -1 &&
    gt_classes != bg_label`.

    Args:
        proposals (list[Instances]): A list of N Instances, where N is the number of
            images in the batch.
        bg_label: label index of background class.

    Returns:
        list[Instances]: N Instances, each contains only the selected foreground instances.
        list[Tensor]: N boolean vector, correspond to the selection mask of
            each Instances object. True for selected instances.
    """
    assert isinstance(proposals, (list, tuple))
    assert isinstance(proposals[0], Players)
    # assert proposals[0].has("gt_classes")
    fg_proposals = []
    fg_selection_masks = []
    for proposals_per_image in proposals:
        if not proposals_per_image.has("gt_classes"):
            # add support for svhn images
            fg_proposals.append(proposals_per_image)
            continue
        gt_classes = proposals_per_image.gt_classes
        fg_selection_mask = (gt_classes != -1) & (gt_classes != bg_label)
        fg_idxs = fg_selection_mask.nonzero().squeeze(1)
        fg_proposals.append(proposals_per_image[fg_idxs])
        fg_selection_masks.append(fg_selection_mask)

    return fg_proposals, fg_selection_masks


def select_proposals_with_visible_keypoints(proposals: List[Players]) -> List[Players]:
    """
    Args:
        proposals (list[Instances]): a list of N Instances, where N is the
            number of images.

    Returns:
        proposals: only contains proposals with at least one visible keypoint.

    Note that this is still slightly different from Detectron.
    In Detectron, proposals for training keypoint head are re-sampled from
    all the proposals with IOU>threshold & >=1 visible keypoint.

    Here, the proposals are first sampled from all proposals with
    IOU>threshold, then proposals with no visible keypoint are filtered out.
    This strategy seems to make no difference on Detectron and is easier to implement.
    """
    ret = []
    all_num_fg = []
    for proposals_per_image in proposals:
        # If empty/unannotated image (hard negatives), skip filtering for train
        # add support for svhn images
        if len(proposals_per_image) == 0 or (not proposals_per_image.has("gt_keypoints")):
            ret.append(proposals_per_image)
            continue
        gt_keypoints = proposals_per_image.gt_keypoints.tensor
        # #fg x K x 3
        vis_mask = gt_keypoints[:, :, 2] >= 1
        xs, ys = gt_keypoints[:, :, 0], gt_keypoints[:, :, 1]
        proposal_boxes = proposals_per_image.proposal_boxes.tensor.unsqueeze(dim=1)  # #fg x 1 x 4
        kp_in_box = (
            (xs >= proposal_boxes[:, :, 0])
            & (xs <= proposal_boxes[:, :, 2])
            & (ys >= proposal_boxes[:, :, 1])
            & (ys <= proposal_boxes[:, :, 3])
        )
        selection = (kp_in_box & vis_mask).any(dim=1)
        selection_idxs = nonzero_tuple(selection)[0]
        all_num_fg.append(selection_idxs.numel())
        ret.append(proposals_per_image[selection_idxs])

    storage = get_event_storage()
    storage.put_scalar("keypoint_head/num_fg_samples", np.mean(all_num_fg) if len(all_num_fg) else 0)
    return ret

@ROI_HEADS_REGISTRY.register()
class BaseROIHeads(StandardROIHeads):
    """
    A wrapper for the detectron2 roi heads.
    """

    @configurable
    def __init__(self, *args, **kwargs):
        self.fg_ratio = kwargs.pop("fg_ratio")
        self.num_proposal_train = kwargs.pop("num_proposal_train")
        self.num_proposal_test = kwargs.pop("num_proposal_test")
        self.top_k_digits = int(self.fg_ratio * self.num_proposal_train)
        super().__init__(*args, **kwargs)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = super().from_config(cfg, input_shape)
        ret.update({
        "fg_ratio": cfg.MODEL.ROI_DIGIT_NECK.FG_RATIO,
        "num_proposal_train": cfg.MODEL.ROI_DIGIT_NECK.NUM_PROPOSAL_TRAIN,
        "num_proposal_test": cfg.MODEL.ROI_DIGIT_NECK.NUM_PROPOSAL_TEST,
        })
        return ret

    @torch.no_grad()
    def label_and_sample_proposals(
            self, proposals: List[Players], targets: List[Players]
    ) -> List[Players]:
        """
        Prepare some proposals to be used to train the ROI heads.
        It performs box matching between `proposals` and `targets`, and assigns
        training labels to the proposals.
        It returns ``self.batch_size_per_image`` random samples from proposals and groundtruth
        boxes, with a fraction of positives that is no larger than
        ``self.positive_fraction``.

        Args:
            See :meth:`ROIHeads.forward`

        Returns:
            list[Instances]:
                length `N` list of `Instances`s containing the proposals
                sampled for training. Each `Instances` has the following fields:

                - proposal_boxes: the proposal boxes
                - gt_boxes: the ground-truth box that the proposal is assigned to
                  (this is only meaningful if the proposal has a label > 0; if label = 0
                  then the ground-truth box is random)

                Other fields such as "gt_classes", "gt_masks", that's included in `targets`.
        """
        # Augment proposals with ground-truth boxes.
        # In the case of learned proposals (e.g., RPN), when training starts
        # the proposals will be low quality due to random initialization.
        # It's possible that none of these initial
        # proposals have high enough overlap with the gt objects to be used
        # as positive examples for the second stage components (box head,
        # cls head, mask head). Adding the gt boxes to the set of proposals
        # ensures that the second stage components will have some positive
        # examples from the start of training. For RPN, this augmentation improves
        # convergence and empirically improves box AP on COCO by about 0.5
        # points (under one tested configuration).
        if self.proposal_append_gt:
            proposals = add_ground_truth_to_proposals(targets, proposals)

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            has_gt = len(targets_per_image) > 0 and targets_per_image.has("gt_boxes")
            if has_gt:
                match_quality_matrix = pairwise_iou(
                    targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
                )
                matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
                sampled_idxs, gt_classes = self._sample_proposals(
                    matched_idxs, matched_labels, targets_per_image.gt_classes
                )

                # Set target attributes of the sampled proposals:
                proposals_per_image = proposals_per_image[sampled_idxs]
                proposals_per_image.gt_classes = gt_classes


                sampled_targets = matched_idxs[sampled_idxs]
                # We index all the attributes of targets that start with "gt_"
                # and have not been added to proposals yet (="gt_classes").
                # NOTE: here the indexing waste some compute, because heads
                # like masks, keypoints, etc, will filter the proposals again,
                # (by foreground/background, or number of keypoints in the image, etc)
                # so we essentially index the data twice.
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(trg_name):
                        if isinstance(trg_value, list):
                            proposals_per_image.set(trg_name, [trg_value[i] for i in sampled_targets.tolist()])
                        else:
                            proposals_per_image.set(trg_name, trg_value[sampled_targets])
                num_bg_samples.append((gt_classes == self.num_classes).sum().item())
                num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            # If no GT is given in the image, we don't know what a dummy gt value can be.
            # Therefore the returned proposals won't have any gt_* fields, except for a
            # gt_classes full of background label.
            else:
                # no gt for person, copy the gt as the target
                proposals_per_image = targets_per_image

            proposals_with_gt.append(proposals_per_image)

        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("roi_head/num_fg_samples", np.mean(num_fg_samples) if len(num_fg_samples) else 0)
        storage.put_scalar("roi_head/num_bg_samples", np.mean(num_bg_samples) if len(num_bg_samples) else 0)

        return proposals_with_gt

    def _sample_digit_proposals(
            self,
            matched_idxs: torch.Tensor,
            matched_labels: torch.Tensor,
            gt_classes: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Based on the matching between N proposals and M groundtruth,
        sample the proposals and set their classification labels.

        We use label 0 as the background class.

        Args:
            matched_idxs (Tensor): a vector of length N, each is the best-matched
                gt index in [0, M) for each proposal.
            matched_labels (Tensor): a vector of length N, the matcher's label
                (one of cfg.MODEL.ROI_HEADS.IOU_LABELS) for each proposal.
            detection_ct_classes (Tensor): a vector of length N, contains the detection's
                center class (0, 1, 2)
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
            # Label unmatched proposals (0 label from matcher) as background (0)
            gt_classes[matched_labels == 0] = 0
            # Label ignore proposals (-1 label)
            gt_classes[matched_labels == -1] = -1
            # optional:
            # compare detection_ct_classes and gt_ct_classes
            # we want the prediction to be precise, such that
            # for a prediction from center heatmap, it generates a proposal
            # which is only 'focused' in the center. Label cross prediction as negative 0
            # non_valid = gt_ct_classes[matched_idxs] != detection_ct_classes
            # gt_classes[non_valid] = 0


        else:
            # all as background
            gt_classes = torch.zeros_like(matched_idxs)

        sampled_fg_idxs, sampled_bg_idxs = subsample_labels(
            gt_classes, self.batch_digit_size_per_image, self.positive_fraction, 0
        )

        sampled_idxs = torch.cat([sampled_fg_idxs, sampled_bg_idxs], dim=0)
        return sampled_idxs, gt_classes[sampled_idxs]

    @torch.no_grad()
    def label_and_sample_digit_proposals(self,
                                         detections: List[torch.tensor],
                                         targets: List[Players]):
        """
        detections: (N, K, 6) where K = self.num_ctdet_proposal
        instance: single instance of one image
        """
        num_fg_samples = []
        num_bg_samples = []
        for detection_per_image, targets_per_image in zip(detections, targets):
            if not targets_per_image.has('gt_digit_boxes'):
                continue
            if not targets_per_image.has("proposal_boxes"):
                targets_per_image.proposal_digit_boxes = targets_per_image.gt_digit_boxes
                continue
            N = len(targets_per_image)
            # create a instance index to match with person proposal_box
            inds = torch.arange(N).repeat_interleave(detection_per_image.size(1), dim=0).to(detection_per_image.device)
            # shape of (N, K, 6) -> (N * K, 6)
            detection_per_image = detection_per_image.view(-1, detection_per_image.size(-1))
            boxes = detection_per_image[:, :4]
            detection_ct_scores = detection_per_image[:, 4]
            # detection_ct_classes = detection_per_image[:, 5].to(torch.int8)
            boxes = Boxes(boxes)
            # we have empty boxes at the beginning of the training
            keep = boxes.nonempty()
            boxes = boxes[keep]
            # detection_ct_classes = detection_ct_classes[keep]
            detection_ct_scores = detection_ct_scores[keep]
            # we clip by the image, probably clipping based on the person ROI works better?
            boxes.clip(targets_per_image.image_size)
            # now we match the digit boxes with detections
            gt_digit_boxes = targets_per_image.gt_digit_boxes
            gt_digit_classes = targets_per_image.gt_digit_classes
            gt_digit_classes = torch.cat(gt_digit_classes) if len(gt_digit_classes) \
                else torch.empty(0, device=detection_per_image.device)
            # (N boxes)
            if len(boxes) and gt_digit_classes.numel():
                gt_digit_boxes = Boxes.cat(gt_digit_boxes)
                gt_digit_classes = gt_digit_boxes.remove_duplicates(gt_digit_classes)

                # get ground-truth match based on iou MxN
                match_quality_matrix = pairwise_iou(
                    gt_digit_boxes, boxes
                )
                # (N,) idx in [0, M); (N,) label of either 1, 0, or -1
                matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
                # if gt_digit_classes is 0, then it is background
                # returns vectors of length N', idx in [0, N); same length N', cls in [1, 10] or background 0
                sampled_idxs, gt_digit_classes = \
                    self._sample_digit_proposals(matched_idxs, matched_labels, gt_digit_classes)
                # the indices of which person each digit proposal is associated with
                inds = inds[sampled_idxs]  # digit -> person
                boxes = boxes[sampled_idxs]
                detection_ct_scores = detection_ct_scores[sampled_idxs]
                # detection_ct_classes = detection_ct_classes[sampled_idxs]
                # get the reverse ids for person -> digit
                reverse_inds = [torch.where(inds == i)[0] for i in range(N)]
                # init a list of Boxes to store digit proposals
                targets_per_image.proposal_digit_boxes = [
                    boxes[i] for i in reverse_inds
                ]
                # store the corresponding digit center scores
                targets_per_image.proposal_digit_scores = [
                    detection_ct_scores[i] for i in reverse_inds
                ]
                # gt_digit_classes is returned by '_sample_digit_proposals'
                targets_per_image.gt_digit_classes = [
                    gt_digit_classes[i] for i in reverse_inds]

                # the gt index for each digit proposal
                gt_digit_boxes = gt_digit_boxes[matched_idxs[sampled_idxs]]
                targets_per_image.gt_digit_boxes = [
                    gt_digit_boxes[i] for i in reverse_inds]
            else:
                # or add gt to the training
                # remove the gt fields (which will be taken care in Faster_rcnn output
                targets_per_image.remove("gt_digit_boxes")
                targets_per_image.remove("gt_digit_classes")
                device = detection_per_image.device
                targets_per_image.proposal_digit_boxes = [Boxes(torch.empty(0, 4, device=device)) for _ in range(N)]
                targets_per_image.proposal_digit_scores = [torch.empty((0, ), device=device) for _ in range(N)]
                # targets_per_image.proposal_digit_ct_classes = [torch.empty(0, device=device).long() for _ in range(N)]
                # targets_per_image.gt_digit_boxes = [Boxes(torch.empty(0, 4, device=device)) for _ in range(N)]
                # targets_per_image.gt_digit_classes = [torch.empty(0, device=device).long() for _ in range(N)]
            num_bg_samples.append((gt_digit_classes == 0).sum().item())
            num_fg_samples.append(gt_digit_classes.numel() - num_bg_samples[-1])
        # Log the number of fg/bg samples that are selected for training ROI heads
        storage = get_event_storage()
        storage.put_scalar("pg_head/num_fg_digit_samples", np.mean(num_fg_samples) if len(num_fg_samples) else 0.)
        storage.put_scalar("pg_head/num_bg_digit_samples", np.mean(num_bg_samples) if len(num_bg_samples) else 0.)

    @torch.no_grad()
    def label_and_sample_jerseynumber_proposals(self,
                                                targets: List[Players],
                                                add_gt=True):
        """
        targets: list (Players): a list of instances.
        """
        for targets_per_image in targets: # image level
            proposal_digit_boxes = targets_per_image.proposal_digit_boxes
            # get the top k boxes based on cfg
            # proposal_number_boxes = [b[:self.top_k_digits].union() for b in proposal_digit_boxes]
            # or based on threshold
            proposal_scores = targets_per_image.proposal_digit_scores
            proposal_number_boxes = [b[s > 0.5].union() for b, s in zip(proposal_digit_boxes, proposal_scores)]
            gt_number_boxes = targets_per_image.gt_number_boxes
            device = targets_per_image.proposal_boxes.device
            # gt_number_classes = targets_per_image.gt_number_classes.copy()
            gt_number_sequences = targets_per_image.gt_number_sequences.clone()
            gt_number_lengths = targets_per_image.gt_number_lengths.clone()
            # we get empty tensor for empty boxes
            valid = [len(b) > 0 and b.nonempty()[0].tolist() for b in proposal_number_boxes]
            sampled_inds = [i for i, v in enumerate(valid) if v]
            sampled_proposal_number_boxes = Boxes.cat([p for v, p in zip(valid, proposal_number_boxes) if v]).to(device)
            sampled_gt_number_boxes = Boxes.cat([p for v, p in zip(valid, gt_number_boxes) if v]).to(device)
            # perform matching [N,] scores of matching
            match_quality_matrix = matched_boxlist_iou(sampled_gt_number_boxes, sampled_proposal_number_boxes)
            # only set empty boxes or boxes of iou < 0.7 be negative
            neg_inds = [i for i in range(len(valid)) if ( (not valid[i]) or match_quality_matrix[sampled_inds.index(i)] < 0.7 )]
            # set negative to empty
            for neg_ind in neg_inds:
                gt_number_lengths[neg_ind] = 0
                # gt_number_classes[neg_ind] = torch.empty((0,), dtype=torch.long, device=match_quality_matrix.device)
                proposal_number_boxes[neg_ind] = Boxes([]).to(match_quality_matrix.device)
            if add_gt:
                proposal_number_boxes = [ Boxes.cat((gt_b, p_b)) for gt_b, p_b in zip(targets_per_image.gt_number_boxes, proposal_number_boxes)]
                # gt_number_classes = [[gt_c, p_c] for gt_c, p_c in zip(targets_per_image.gt_number_classes, gt_number_classes)]
                gt_number_sequences = torch.stack((targets_per_image.gt_number_sequences, gt_number_sequences), dim=1)
                gt_number_lengths = torch.stack((targets_per_image.gt_number_lengths, gt_number_lengths), dim=1)
            targets_per_image.proposal_number_boxes = proposal_number_boxes
            # targets_per_image.gt_number_classes = gt_number_classes
            targets_per_image.gt_number_sequences = gt_number_sequences
            targets_per_image.gt_number_lengths = gt_number_lengths




    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Players]):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        person_proposal_boxes = [x.proposal_boxes if x.has('proposal_boxes') else Boxes([]).to(features[0].device) for x in proposals]
        box_features = self.box_pooler(features, person_proposal_boxes)
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            # proposals is modified in-place below, so losses must be computed first.
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances

    def _forward_keypoint(self, features: Dict[str, torch.Tensor], instances: List[Players]):
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        if self.training:
            instances, _ = select_foreground_proposals(instances, self.num_classes)
            # head is only trained on positive proposals with >=1 visible keypoints,
            # may not needed
            instances = select_proposals_with_visible_keypoints(instances)

        if self.keypoint_pooler is not None:
            features = [features[f] for f in self.keypoint_in_features]
            if self.training:
                boxes = [x.proposal_boxes if x.has("proposal_boxes")
                         else Boxes.cat([]).to(features[0].device)
                        for x in instances]
            else:
                boxes = [x.pred_boxes for x in instances]
            # if boxes is an empty list, we get a zero tensor on cpu!
            features = self.keypoint_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.keypoint_in_features}
        return self.keypoint_head(features, instances)