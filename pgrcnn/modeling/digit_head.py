# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from typing import Dict, List, Tuple, Union
import torch
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.utils.events import get_event_storage
from pgrcnn.structures import Boxes, Players

__all__ = ["fast_rcnn_inference", "DigitOutputLayers"]


logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""

def _log_classification_stats(pred_logits, gt_classes, prefix="pg_rcnn"):
    """
    Log the classification metrics to EventStorage.

    Args:
        pred_logits: Rx(K+1) logits. The first column is for background class.
        gt_classes: R labels
    """
    num_instances = gt_classes.numel()
    if num_instances == 0:
        return
    pred_classes = pred_logits.argmax(dim=1)
    # we use 0 as the bg class
    bg_class_ind = 0

    fg_inds = (gt_classes > bg_class_ind)
    num_fg = fg_inds.nonzero().numel()
    fg_gt_classes = gt_classes[fg_inds]
    fg_pred_classes = pred_classes[fg_inds]

    num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
    num_accurate = (pred_classes == gt_classes).nonzero().numel()
    fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

    storage = get_event_storage()
    storage.put_scalar(f"{prefix}/digit_cls_accuracy", num_accurate / num_instances)
    if num_fg > 0:
        storage.put_scalar(f"{prefix}/digit_fg_cls_accuracy", fg_num_accurate / num_fg)
        storage.put_scalar(f"{prefix}/digit_false_negative", num_false_negative / num_fg)

def fast_rcnn_inference(boxes, scores, pred_digit_center_classes,
                        pred_digit_center_scores,
                        num_digits_scores, num_instances, image_shapes, digit_score_thresh,
                        number_score_thresh, nms_thresh, topk_per_image
                        ):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        num_instances (list[list[int]]): A list of integers indicating the number of person instances
            for each image.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image,
            pred_digit_center_classes_per_image, pred_digit_center_scores_per_image,
            num_digits_scores_per_image, num_instance, image_shape,
            digit_score_thresh, number_score_thresh, nms_thresh, topk_per_image
        )
        for boxes_per_image, scores_per_image, pred_digit_center_classes_per_image,
            pred_digit_center_scores_per_image, num_digits_scores_per_image, num_instance, image_shape in \
        zip(boxes, scores, pred_digit_center_classes, pred_digit_center_scores,
            num_digits_scores, num_instances, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def fast_rcnn_inference_single_image(
        boxes, scores,
        pred_digit_center_classes, pred_digit_center_scores,
        pred_num_digits, num_instance, image_shape, digit_score_thresh,
        number_score_thresh, nms_thresh, topk_per_image,
    per_class_nms=True
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.
        nms_method (int): 0 (default, nms all boxes, 1 nms per instance)

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    N = len(num_instance) # the number of detected players
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        raise Exception("not all boxes/scores are valid.")
        # boxes = boxes[valid_mask]
        # scores = scores[valid_mask]
    # bg id is 0, so we get the last 10 classes
    scores = scores[:, 1:]
    if pred_digit_center_classes is not None:
        if len(pred_digit_center_classes) == 0:
            pred_digit_center_classes = torch.as_tensor(pred_digit_center_classes, dtype=torch.long, device=scores.device)
            pred_digit_center_scores = torch.as_tensor(pred_digit_center_scores, dtype=torch.double,
                                                        device=scores.device)
        else:
            pred_digit_center_classes = torch.cat(pred_digit_center_classes)
            pred_digit_center_scores = torch.cat(pred_digit_center_scores)
        scores = scores * pred_digit_center_scores.unsqueeze(1)
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # Filter results based on detection scores
    filter_mask = scores > digit_score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes in terms of 0 - 9 class id.
    filter_inds = filter_mask.nonzero()
    # find the indices of each digit bbox in each person instance
    instance_idx = torch.as_tensor([x for i, num_digits in enumerate(num_instance) for x in [i] * num_digits],
                                   dtype=torch.long, device=scores.device)

    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    pred_digit_center_classes = pred_digit_center_classes[filter_inds[:, 0]]
    instance_idx = instance_idx[filter_inds[:, 0]] # the mask will maintain the instance_idx sorted from 0 to 1
    # Apply per-class NMS
    cls_ids = filter_inds[:, 1] + 1
    # get the jersey number detection first before nms
    if pred_num_digits is not None:
        pred_number_boxes, pred_number_classes, pred_number_scores = jersey_number_inference(boxes, scores, cls_ids,
                                                                                             pred_digit_center_classes, instance_idx, N, pred_num_digits,
                                                                                             number_score_thresh=number_score_thresh)
    nms_method = nms_per_image # nms_per_player nms_per_image
    boxes, scores, cls_ids, pred_digit_center_classes, instance_idx = \
        nms_method(boxes, scores, cls_ids, pred_digit_center_classes, instance_idx, N, nms_thresh, topk_per_image, per_class_nms)

    # count number of digit object for each instance
    counts = torch.bincount(instance_idx, minlength=N).tolist()
    boxes = torch.split(boxes, counts)
    boxes = [Boxes(x) for x in boxes]
    scores = list(torch.split(scores, counts))
    cls_ids = list(torch.split(cls_ids, counts))
    # pred_digit_center_classes = list(torch.split(pred_digit_center_classes, counts))
    # boxes, scores, cls_ids, number_preds, number_scores = jersey_number_inference(boxes, scores, cls_ids, number_score_thresh)
    # assign fields for players
    result = Players(image_shape)
    result.pred_digit_boxes = boxes
    result.digit_scores = scores
    result.pred_digit_classes = cls_ids
    if pred_num_digits is not None:
        result.pred_number_boxes = pred_number_boxes
        result.pred_number_classes = pred_number_classes
        result.pred_number_scores = pred_number_scores
    # result.proposal_number_boxes = [b[s > number_score_thresh].union()
    #                                 for b, s in zip(boxes, scores)]



    return result, filter_inds[:, 0]

def jersey_number_inference(boxes, scores, cls_ids, pred_digit_center_classes, instance_inds, num_instances, pred_num_digits,
                            number_score_thresh, max_length=2):
    counts = torch.bincount(instance_inds, minlength=num_instances).tolist()
    boxes = torch.split(boxes, counts)
    boxes = [Boxes(x) for x in boxes]
    scores = list(torch.split(scores, counts))
    cls_ids = list(torch.split(cls_ids, counts))
    pred_digit_center_classes = list(torch.split(pred_digit_center_classes, counts))
    pred_number_classes = []
    pred_number_scores = []
    pred_number_boxes = []
    for i in range(len(scores)):
        box = boxes[i]
        score = scores[i]
        cls_id = cls_ids[i]
        pred_digit_center_cls = pred_digit_center_classes[i]
        pred_num_digits_per_ins = pred_num_digits[i]
        if pred_num_digits_per_ins == 0:
            pred_number_boxes.append(Boxes([]).to(pred_num_digits.device))
            pred_number_classes.append(torch.zeros((0, max_length), dtype=torch.long, device=pred_num_digits.device))
            pred_number_scores.append(torch.zeros((0,), dtype=torch.double, device=score.device))
       # single-digit case
        elif pred_num_digits_per_ins == 1:
            valid = torch.logical_and(pred_digit_center_cls == 0, score > number_score_thresh)
            box = box[valid]
            score = score[valid]
            cls_id = cls_id[valid]
            jersey_number = cls_id.view(-1, 1)
            jersey_number = F.pad(jersey_number, (0, 1))
            pred_number_boxes.append(box)
            pred_number_classes.append(jersey_number)
            pred_number_scores.append(score)
        elif pred_num_digits_per_ins == 2:
            first_valid = pred_digit_center_cls == 0
            second_valid = pred_digit_center_cls == 1
            first_digit_boxes = box[first_valid].tensor
            second_digit_boxes = box[second_valid].tensor
            first_digit_classes = cls_id[first_valid]
            second_digit_classes = cls_id[second_valid]
            first_digit_scores = score[first_valid]
            second_digit_scores = score[second_valid]
            num_first, num_second = first_digit_classes.numel(), second_digit_classes.numel()
            first_digit_boxes = first_digit_boxes.repeat(num_second, 1)
            second_digit_boxes = second_digit_boxes.repeat_interleave(num_first, 0)
            digit_boxes = torch.stack((first_digit_boxes, second_digit_boxes), dim=1)
            number_boxes = Boxes.cat([Boxes(b).union() for b in digit_boxes])
            first_digit_classes = first_digit_classes.repeat(num_second)
            second_digit_classes = second_digit_classes.repeat_interleave(num_first)
            jersey_number = torch.stack((first_digit_classes, second_digit_classes), dim=1)
            # jersey_number = F.pad(jersey_number, (0, 1))
            number_score = first_digit_scores.repeat(num_second) * second_digit_scores.repeat_interleave(num_first)
            keep = number_score > number_score_thresh
            pred_number_classes.append(jersey_number[keep])
            pred_number_scores.append(number_score[keep])
            pred_number_boxes.append(number_boxes[keep])
    return pred_number_boxes, pred_number_classes, pred_number_scores

def nms_per_image(boxes, scores, cls_ids, pred_digit_center_classes, instance_inds, num_instances, nms_thresh, topk_per_image, per_class_nms=True):
    # proxy class ids
    proxy_cls_ids = cls_ids if per_class_nms else torch.zeros_like(cls_ids)
    keep = batched_nms(boxes, scores, proxy_cls_ids, nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    # recover digit class ids by adding one
    boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]
    pred_digit_center_classes = pred_digit_center_classes[keep]
    instance_inds = instance_inds[keep]
    instance_inds, sorted_inds = torch.sort(instance_inds)
    boxes, scores, cls_ids = boxes[sorted_inds], scores[sorted_inds], cls_ids[sorted_inds]
    pred_digit_center_classes = pred_digit_center_classes[sorted_inds]
    return boxes, scores, cls_ids, pred_digit_center_classes, instance_inds

def nms_per_player(boxes, scores, cls_ids, instance_inds, num_instances, nms_thresh, topk_per_image, per_class_nms=True):
    topk_per_instance = topk_per_image // num_instances
    # assign each box to its player first, then nms
    counts = torch.bincount(instance_inds, minlength=num_instances).tolist()
    boxes = list(torch.split(boxes, counts))
    scores = list(torch.split(scores, counts))
    proxy_cls_ids = cls_ids if per_class_nms else torch.zeros_like(cls_ids)
    cls_ids = list(torch.split(cls_ids, counts))
    proxy_cls_ids = list(torch.split(proxy_cls_ids, counts))
    keeps = [batched_nms(boxes_per_instance, scores_per_instance, cls_ids_per_instance, nms_thresh)
             for boxes_per_instance, scores_per_instance, cls_ids_per_instance in zip(boxes, scores, proxy_cls_ids)]
    if topk_per_instance >= 0:
        keeps = [keep[:topk_per_instance] for keep in keeps]
    for i, (keep, box, score, cls_id) in enumerate(zip(keeps, boxes, scores, cls_ids)):
        boxes[i] = box[keep]
        scores[i] = score[keep]
        cls_ids[i] = cls_id[keep]
    boxes = [Boxes(x) for x in boxes]
    return boxes, scores, cls_ids

""" Deprecated way to get number

def jersey_number_inference(boxes, scores, cls_ids, number_score_thresh):
    xs = [x.get_centers()[:, 0] for x in boxes]
    # sort the digit box location left -> right
    sorted_inds = [torch.sort(x)[1] for x in xs]
    boxes = [bbox[inds] for bbox, inds in zip(boxes, sorted_inds)]
    scores = [score[inds] for score, inds in zip(scores, sorted_inds)]
    cls_ids = [pred_digit_cls[inds] for pred_digit_cls, inds in zip(cls_ids, sorted_inds)]
    # these fields should have length len(num_instance)

    # # get the jersey number todo: maybe better way to get the number
    keep_for_number = [torch.nonzero(s > number_score_thresh, as_tuple=True)[0] for s in scores]
    # only get the first and last
    keep_for_number = [inds[[0, -1]] if inds.numel() > 1 else inds for inds in keep_for_number]
    number_scores = [s[keep].mean() for keep, s in zip(keep_for_number, scores)]
    number_preds = [pred[keep] for keep, pred in zip(keep_for_number, cls_ids)]

    return boxes, scores, cls_ids, number_preds, number_scores
"""

class DigitOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
    """

    @configurable
    def __init__(
            self,
            input_shape: ShapeSpec,
            *,
            box2box_transform,
            num_classes: int,
            test_digit_score_thresh: float = 0.0,
            test_number_score_thresh: float = 0.0,
            test_nms_thresh: float = 0.5,
            test_topk_per_image: int = 100,
            cls_agnostic_bbox_reg: bool = False,
            smooth_l1_beta: float = 0.0,
            box_reg_loss_type: str = "smooth_l1",
            loss_weight: Union[float, Dict[str, float]] = 1.0,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all losses, or a dict of individual weightings. Valid dict keys are:
                    * "loss_cls": applied to classification loss
                    * "loss_box_reg": applied to box regression loss
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        self.num_classes = num_classes
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        # prediction layer for num_classes foreground classes and one background class (hence + 1)
        self.cls_score = nn.Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.bbox_pred = nn.Linear(input_size, num_bbox_reg_classes * box_dim)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_digit_score_thresh = test_digit_score_thresh
        self.test_number_score_thresh = test_number_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        if isinstance(loss_weight, float):
            loss_weight = {"loss_cls": loss_weight, "loss_box_reg": loss_weight}
        self.loss_weight = loss_weight
        # we set the bg class as the first class, so the digit id is consistent
        self.bg_class_ind = 0

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes": cfg.MODEL.ROI_DIGIT_BOX_HEAD.NUM_DIGIT_CLASSES, # this is the number of digits
            "cls_agnostic_bbox_reg": cfg.MODEL.ROI_DIGIT_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta": cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_digit_score_thresh": cfg.MODEL.ROI_DIGIT_BOX_HEAD.DIGIT_SCORE_THRESH_TEST,
            "test_number_score_thresh": cfg.MODEL.ROI_DIGIT_BOX_HEAD.NUMBER_SCORE_THRESH_TEST,
            "test_nms_thresh": cfg.MODEL.ROI_DIGIT_BOX_HEAD.NMS_THRESH_TEST,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type": cfg.MODEL.ROI_DIGIT_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight": {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},
            # fmt: on
        }

    def forward(self, x):
        """
        Returns:
            Tensor: Nx(K+1) scores for each box
            Tensor: Nx4 or Nx(Kx4) bounding box regression deltas.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        return scores, proposal_deltas

    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        scores, proposal_deltas = predictions

        # todo: forgot why we need this
        # we split and assign to each proposal
        # num_proposal_digit_boxes = [[len(instance) for instance in p.proposal_digit_boxes] for p in proposals]
        # scores_all = torch.split(scores, [sum(num_per_image) for num_per_image in num_proposal_digit_boxes])
        # for proposals_per_image, scores_per_image, num_proposal_digit_boxes_per_image in zip(proposals,
        #                                                                                     scores_all,
        #                                                                                     num_proposal_digit_boxes):
        #     proposal_scores_per_image = torch.split(scores_per_image, num_proposal_digit_boxes_per_image)
        #     proposals_per_image.proposal_digit_box_scores = list(proposal_scores_per_image)
        # parse box regression outputs
        if len(proposals) and any([p.has("proposal_digit_boxes") for p in proposals]):
            proposal_boxes = cat([Boxes.cat(p.proposal_digit_boxes).tensor.to(proposal_deltas.device)
                                  if p.has('proposal_digit_boxes') else torch.empty((0, 4), device=proposal_deltas.device, dtype=torch.float)
                                  for p in proposals], dim=0)  # Nx4
            assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
            # If "gt_boxes" does not exist, the proposals must be all negative and
            # should not be included in regression loss computation.
            # Here we just use proposal_boxes as an arbitrary placeholder because its
            # value won't be used in self.box_reg_loss().
            gt_boxes = cat(
                [(Boxes.cat(p.gt_digit_boxes) if p.has("gt_digit_boxes")
                  else Boxes([])).tensor.to(proposal_deltas.device) for p in proposals],
                dim=0,
            )
            # parse classification outputs
            gt_classes = [cat(p.gt_digit_classes, dim=0) if p.has("gt_digit_classes")
                          else torch.empty(0, device=proposal_deltas.device, dtype=torch.long) for p in proposals]

            gt_classes = cat(gt_classes, dim=0)
        else:
            proposal_boxes = gt_boxes = torch.empty((0, 4), device=proposal_deltas.device)
            gt_classes = torch.empty(0, dtype=torch.long, device=proposal_deltas.device)

        _log_classification_stats(scores, gt_classes)

        losses = {
            "loss_digit_cls": cross_entropy(scores, gt_classes, reduction="mean"),
            "loss_digit_box_reg": self.box_reg_loss(
                proposal_boxes, gt_boxes, proposal_deltas, gt_classes
            ),
        }
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def box_reg_loss(self, proposal_boxes, gt_boxes, pred_deltas, gt_classes):
        """
        Args:
            All boxes are tensors with the same shape Rx(4 or 5).
            gt_classes is a long tensor of shape R, the gt class label of each proposal.
            R shall be the number of proposals.
        """
        if pred_deltas.numel() == 0:
            return pred_deltas.sum() * 0.0
        box_dim = proposal_boxes.shape[1]  # 4 or 5
        # Regression loss is only computed for foreground proposals (those matched to a GT)
        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        # bg class is 0, so the fg inds is changed
        fg_inds = nonzero_tuple(gt_classes > self.bg_class_ind)[0]
        if pred_deltas.shape[1] == box_dim:  # cls-agnostic regression
            fg_pred_deltas = pred_deltas[fg_inds]
        else:
            # we minus 1 since bg class is at the first place
            # gt_classes.add_(-1)
            fg_pred_deltas = pred_deltas.view(-1, self.num_classes, box_dim)[
                fg_inds, gt_classes[fg_inds] - 1
            ]

        if self.box_reg_loss_type == "smooth_l1":
            gt_pred_deltas = self.box2box_transform.get_deltas(
                proposal_boxes[fg_inds],
                gt_boxes[fg_inds],
            )
            loss_box_reg = smooth_l1_loss(
                fg_pred_deltas, gt_pred_deltas, self.smooth_l1_beta, reduction="sum"
            )
        elif self.box_reg_loss_type == "giou":
            fg_pred_boxes = self.box2box_transform.apply_deltas(
                fg_pred_deltas, proposal_boxes[fg_inds]
            )
            loss_box_reg = giou_loss(fg_pred_boxes, gt_boxes[fg_inds], reduction="sum")
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")
        # The reg loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        return loss_box_reg / max(gt_classes.numel(), 1.0)  # return 0 if empty

    def inference(self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Players]):
        """
        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        num_instances = [[len(p_digit_boxes) for p_digit_boxes in p.get("proposal_digit_boxes")] for p in proposals]
        pred_num_digits = [p.get("pred_num_digits") if p.has("pred_num_digits") else None for p in proposals]
        pred_digit_center_classes = [p.get("pred_digit_center_classes") if p.has("pred_digit_center_classes") else None for p in proposals]
        pred_digit_center_scores = [p.get("pred_digit_center_scores") if p.has("pred_digit_center_scores") else None
                                     for p in proposals]
        detections, kept_idx = fast_rcnn_inference(
            boxes,
            scores,
            pred_digit_center_classes,
            pred_digit_center_scores,
            pred_num_digits,
            num_instances,
            image_shapes,
            self.test_digit_score_thresh,
            self.test_number_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image
        )
        # merge detection results
        for proposal, detection in zip(proposals, detections):
            for k, v in detection.get_fields().items():
                proposal.set(k, v)
            proposal.remove('proposal_digit_boxes')
        return proposals, kept_idx

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        scores, proposal_deltas = predictions
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device), gt_classes
            ]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(
            self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Players]
    ):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, proposal_deltas = predictions
        # flat all boxes
        proposal_boxes = Boxes.cat([b for p in proposals for b in p.proposal_digit_boxes]).tensor.to(proposal_deltas.device)
        # num of digit boxes per image
        num_prop_per_image = [sum([len(p_digit_boxes) for p_digit_boxes in p.get("proposal_digit_boxes")]) for p in proposals]
        # proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        predict_boxes = self.box2box_transform.apply_deltas(
            # ensure fp32 for decoding precision
            proposal_deltas,
            proposal_boxes,
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image, dim=0)

    def predict_probs(
            self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Players]
    ):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        scores, _ = predictions
        num_prop_per_image = [sum([len(p_digit_boxes) for p_digit_boxes in p.get("proposal_digit_boxes")]) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        return probs.split(num_prop_per_image, dim=0)
