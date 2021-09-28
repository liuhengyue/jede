import detectron2.config
import math
import numpy as np
from typing import List, Dict, Union, Tuple
import fvcore.nn.weight_init as weight_init
from fvcore.nn import giou_loss, smooth_l1_loss
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers.wrappers import _NewEmptyTensorOp
from detectron2.utils.registry import Registry
from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.data import DatasetCatalog, MetadataCatalog

from pgrcnn.structures import Boxes, Players
from pgrcnn.modeling.utils import beam_search_decode

ROI_JERSEY_NUMBER_DET_REGISTRY = Registry("ROI_JERSEY_NUMBER_DET")


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        here T is the feature width since we will read column by column
        """
        self.rnn.flatten_parameters()
        # if input.numel():
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        # else:
        #     recurrent = _NewEmptyTensorOp.apply(input, (input.size(0), input.size(1), 2 * self.hidden_size))
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output



@ROI_JERSEY_NUMBER_DET_REGISTRY.register()
class SequenceModel(nn.Module):
    def __init__(self,
                 cfg: detectron2.config.CfgNode,
                 input_shape: ShapeSpec
                 ):
        """

        """
        super().__init__()
        self.max_length = cfg.MODEL.ROI_NUMBER_BOX_HEAD.SEQ_MAX_LENGTH
        input_resolution = (input_shape.height, input_shape.width) # the pooler size
        self.seq_resolution = cfg.MODEL.ROI_NUMBER_BOX_HEAD.SEQUENCE_RESOLUTION
        in_channels = input_shape.channels
        # in_channels = input_shape.channels * input_shape.height
        hidden_size = 256
        self.num_class = cfg.MODEL.ROI_DIGIT_BOX_HEAD.NUM_DIGIT_CLASSES + 1
        self.context = nn.Sequential(
            BidirectionalLSTM(in_channels, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size)
        )
        if input_resolution == self.seq_resolution:
            self.pool = None
        else:
            self.pool = nn.AdaptiveAvgPool2d(self.seq_resolution)
        self.output = nn.Linear(hidden_size, self.num_class)
        # (w, c)
        self._output_size = (self.seq_resolution[1], self.num_class)
        # indicate whether the input needs to be pooled first
        self._requires_roi_pool = True
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.normal_(self.output.weight, std=0.001)

    def forward_no_pool(self, x):
        if not x.numel():
            # should not happen
            return _NewEmptyTensorOp.apply(x, (0,) + self._output_size)
        # [n, w, c, h]
        x = x.permute(0, 3, 1, 2)
        n, w, c, h = x.size()
        # [n, w, cxh]
        x = x.reshape(n, w, -1)
        x = self.context(x)
        # [n, w, num_cls]
        x = self.output(x)
        return x

    def forward(self, x):
        if not x.numel():
            # should not happen
            return _NewEmptyTensorOp.apply(x, (0,) + self._output_size)
        # [n, c, h, w] -> [n, c, 1, w]
        if self.pool is not None:
            x = self.pool(x)
        # [n, w, c, 1]
        x = x.permute(0, 3, 1, 2)
        # [n, w, c]
        x = x.squeeze(3)
        x = self.context(x)
        # [n, w, num_cls]
        x = self.output(x)
        return x

# @ROI_JERSEY_NUMBER_DET_REGISTRY.register()
# class JerseyNumberHead(nn.Module):
#     def __init__(self,
#                  cfg: detectron2.config.CfgNode,
#                  input_shape: ShapeSpec
#                  ):
#         """
#
#         """
#         super().__init__()
#         self.max_length = cfg.MODEL.ROI_JERSEY_NUMBER_DET.SEQ_MAX_LENGTH
#         self.seq_resolution = cfg.MODEL.ROI_JERSEY_NUMBER_DET.SEQUENCE_RESOLUTION
#         in_channels = input_shape.channels * self.seq_resolution[1] * self.seq_resolution[2]
#         hidden_size = 256
#         self.num_class = cfg.MODEL.ROI_DIGIT_NECK.NUM_DIGIT_CLASSES + 1
#         self.context = nn.Sequential(
#             BidirectionalLSTM(in_channels, hidden_size, hidden_size),
#             BidirectionalLSTM(hidden_size, hidden_size, hidden_size)
#         )
#         self.pool = nn.AdaptiveAvgPool3d(self.seq_resolution) # AdaptiveMaxPool3d
#         self.output = nn.Linear(hidden_size, self.num_class)
#         # (h, w, c)
#         self._output_size = (self.seq_resolution[0], self.num_class)
#
#
#     def _forward_per_instance(self, box_features, cls_scores):
#         """
#
#         Args:
#             box_features: Tensor of shape (N, C, H, W),  (N, 256, 7, 7) by default
#             cls_scores: Tensor of shape (N, 11)
#
#         Returns:
#
#         """
#         # (N, C, H, W) -> (C, N, H, W)
#         box_features = box_features.permute(1, 0, 2, 3)
#         # (C, N', H', W')
#         box_features = self.pool(box_features)
#         # (N', C, H', W')
#         box_features = box_features.permute(1, 0, 2, 3)
#         # (1, N', H'xW'xC)
#         box_features = box_features.reshape(1, box_features.size(0), -1)
#         box_features = self.context(box_features)
#         # [1, N', num_cls]
#         box_features = self.output(box_features)
#
#         return box_features
#
#     def forward(self, proposals):
#         preds_all = []
#         for p in proposals: # per image
#             proposal_digit_box_features_per_image = p.proposal_digit_box_features
#             proposal_digit_box_scores_per_image = p.proposal_digit_box_scores
#             for proposal_digit_box_features, proposal_digit_box_scores in \
#                     zip(proposal_digit_box_features_per_image, proposal_digit_box_scores_per_image):
#                 # forward per instance
#                 if not proposal_digit_box_features.numel():
#                     preds = _NewEmptyTensorOp.apply(proposal_digit_box_features, (0,) + self._output_size)
#                 else:
#                     preds = self._forward_per_instance(proposal_digit_box_features, proposal_digit_box_scores)
#                 preds_all.append(preds)
#         preds_all = torch.cat(preds_all)
#         return preds_all
#
#     def losses(self, predictions, proposals):
#         predictions = predictions.log_softmax(2).permute(1, 0, 2) # [T, N, C]
#         pred_lenghs = torch.as_tensor([predictions.size(0)] * predictions.size(1))
#         # gt_numbers = [gt_num for p in proposals for x in p.gt_number_classes for gt_num in x if gt_num.numel()]
#         # gt_lengths = torch.as_tensor([x.numel() for x in gt_numbers], dtype=torch.long, device=predictions.device)
#         # if we have no instance
#         gt_number_classes = torch.cat([torch.cat(p.gt_number_classes)
#                                                if p.has("gt_number_classes") and len(p.gt_number_classes)
#                                                else torch.empty((0,), dtype=torch.long, device=predictions.device)
#                                                 for p in proposals]) # 0 is bg
#         gt_lengths = torch.cat([torch.cat(p.gt_number_lengths)
#                                 if p.has("gt_number_lengths") and len(p.gt_number_lengths)
#                                 else torch.empty((0,), dtype=torch.long, device=predictions.device)
#                                 for p in proposals])
#
#         gt_seq = torch.cat([torch.cat(p.gt_number_sequences)
#                             if p.has("gt_number_sequences") and len(p.gt_number_sequences)
#                             else torch.empty((0, self.max_length), dtype=torch.long, device=predictions.device)
#                             for p in proposals])
#         valid = gt_number_classes > 0
#         if valid.sum() == 0: # no valid or no gt
#             return {"number_loss": predictions.sum() * 0}
#         gt_lengths = gt_lengths[valid]
#         gt_seq = gt_seq[valid]
#         pred_lenghs = pred_lenghs[valid]
#         predictions = predictions[:, valid, :]
#         loss = F.ctc_loss(predictions, gt_seq, pred_lenghs, gt_lengths, zero_infinity=True)
#         return {"number_loss": loss}

@ROI_JERSEY_NUMBER_DET_REGISTRY.register()
class NumDigitClassification(nn.Module):
    def __init__(self,
                 cfg: detectron2.config.CfgNode,
                 input_shapes: ShapeSpec
                 ):
        """

        """
        super().__init__()
        in_channels = input_shapes.channels * 1 * input_shapes.width
        self.out_channels = 3  # 0 digit, 1 digit, 2 digit
        self.pool = nn.AdaptiveMaxPool2d((1, input_shapes.width))
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
            # AdaptiveMaxPool2d does not take empty tensor
            return _NewEmptyTensorOp.apply(x, (0, self.out_channels))
        x = self.pool(x)
        x = self.linears(x)
        return x

class JerseyNumberOutputLayers(nn.Module):
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
            char_names: List[str],
            test_score_thresh: float = 0.0,
            test_nms_thresh: float = 0.5,
            test_topk_per_image: int = 100,
            cls_agnostic_bbox_reg: bool = True, # we reg all jersey boxes cls agnostically
            smooth_l1_beta: float = 0.0,
            box_reg_loss_type: str = "smooth_l1",
            loss_weight: Union[float, Dict[str, float]] = 1.0,
            cls_predictor,
            number_box_head_channel: int = 1024,
            max_length: int = 2
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
        self.char_names = char_names
        self.cls_score = cls_predictor
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.enable_box_reg = number_box_head_channel > 0
        if self.enable_box_reg: # we may not use box reg
            # input_shape is the box feature shape, we need get the shape from box head
            self.bbox_pred = nn.Linear(number_box_head_channel, num_bbox_reg_classes * box_dim)
            for l in [self.bbox_pred]:
                nn.init.constant_(l.bias, 0)
            # set classification loss
            self.class_loss = self.box_class_loss
        else:
            self.class_loss = self.instance_class_loss


        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        if isinstance(loss_weight, float):
            loss_weight = {"loss_number_cls": loss_weight, "loss_number_box_reg": loss_weight}
        self.loss_weight = loss_weight
        self.bg_class_ind = 0
        self.max_length = max_length



    @classmethod
    def from_config(cls, cfg, input_shape):
        number_box_head_channel = cfg.MODEL.ROI_NUMBER_BOX_HEAD.FC_DIM if cfg.MODEL.ROI_NUMBER_BOX_HEAD.NUM_FC else 0
        box2box_transform = Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS) if number_box_head_channel else None
        char_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("char_names")
        return {
            "input_shape": input_shape,
            "box2box_transform": box2box_transform,
            # fmt: off
            "num_classes": cfg.MODEL.ROI_DIGIT_BOX_HEAD.NUM_DIGIT_CLASSES, # this is the number of digits
            "char_names": char_names,
            "cls_agnostic_bbox_reg": cfg.MODEL.ROI_NUMBER_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta": cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh": cfg.MODEL.ROI_NUMBER_BOX_HEAD.SCORE_THRESH_TEST,
            "test_nms_thresh": cfg.MODEL.ROI_NUMBER_BOX_HEAD.NMS_THRESH_TEST,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type": cfg.MODEL.ROI_NUMBER_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight": 1.0,
            "cls_predictor": SequenceModel(cfg, input_shape), # box feature shape
            "max_length": cfg.MODEL.ROI_NUMBER_BOX_HEAD.SEQ_MAX_LENGTH,
            "number_box_head_channel": number_box_head_channel, # if use box reg

            # fmt: on
        }

    def forward(self, x, y):
        """
        Args:
            x: box features NXCXHXW
            y: box fc features after head NXC' or None
        Returns:
            Tensor: NxCxHxW scores for each box
            Tensor: Nx4 or Nx(Kx4) bounding box regression deltas.
        """
        scores = self.cls_score(x)
        if self.enable_box_reg and y is not None:
            proposal_deltas = self.bbox_pred(y)
        else:
            proposal_deltas = None
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

        # parse box regression outputs
        if sum([len(p) for p in proposals]) and any([p.has("proposal_number_boxes") for p in proposals]):
            # parse classification outputs
            gt_classes = [cat(p.gt_number_classes, dim=0) if p.has("gt_number_classes")
                          else torch.empty(0, device=scores.device, dtype=torch.long) for p in proposals]

            gt_classes = cat(gt_classes, dim=0)
            if self.enable_box_reg:
                proposal_boxes = cat([Boxes.cat(p.proposal_number_boxes).tensor.to(scores.device)
                                      if p.has('proposal_number_boxes') else torch.empty((0, 4), device=scores.device, dtype=torch.float)
                                      for p in proposals], dim=0)  # Nx4
                assert not proposal_boxes.requires_grad, "Proposals should not require gradients!"
                # If "gt_boxes" does not exist, the proposals must be all negative and
                # should not be included in regression loss computation.
                # Here we just use proposal_boxes as an arbitrary placeholder because its
                # value won't be used in self.box_reg_loss().
                gt_boxes = cat(
                    [(Boxes.cat(p.gt_number_boxes) if p.has("gt_number_boxes")
                      else Boxes([])).tensor.to(scores.device) for p in proposals],
                    dim=0,
                )

        else:
            gt_classes = torch.empty(0, dtype=torch.long, device=scores.device)
            if self.enable_box_reg:
                proposal_boxes = gt_boxes = torch.empty((0, 4), device=scores.device)


        losses = {
            "loss_number_cls": self.class_loss(scores, proposals),
        }
        if self.enable_box_reg:
            losses.update({
                "loss_number_box_reg": self.box_reg_loss(
                    proposal_boxes, gt_boxes, proposal_deltas, gt_classes
                ),
            })
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def box_class_loss(self, scores, proposals):
        scores = scores.log_softmax(2).permute(1, 0, 2)  # [T, N, C]
        pred_lenghs = torch.as_tensor([scores.size(0)] * scores.size(1))
        # gt_numbers = [gt_num for p in proposals for x in p.gt_number_classes for gt_num in x if gt_num.numel()]
        # gt_lengths = torch.as_tensor([x.numel() for x in gt_numbers], dtype=torch.long, device=predictions.device)
        # if we have no instance
        gt_number_classes = torch.cat([torch.cat(p.gt_number_classes)
                                       if p.has("gt_number_classes") and len(p.gt_number_classes)
                                       else torch.empty((0,), dtype=torch.long, device=scores.device)
                                       for p in proposals])  # 0 is bg
        gt_lengths = torch.cat([torch.cat(p.gt_number_lengths)
                                if p.has("gt_number_lengths") and len(p.gt_number_lengths)
                                else torch.empty((0,), dtype=torch.long, device=scores.device)
                                for p in proposals])

        gt_seq = torch.cat([torch.cat(p.gt_number_sequences)
                            if p.has("gt_number_sequences") and len(p.gt_number_sequences)
                            else torch.empty((0, self.max_length), dtype=torch.long, device=scores.device)
                            for p in proposals])
        valid = gt_number_classes > 0
        if valid.sum() == 0:  # no valid or no gt
            return scores.sum() * 0
        gt_lengths = gt_lengths[valid]
        gt_seq = gt_seq[valid]
        pred_lenghs = pred_lenghs[valid]
        scores = scores[:, valid, :]
        loss = F.ctc_loss(scores, gt_seq, pred_lenghs, gt_lengths, zero_infinity=True)
        return loss

    def instance_class_loss(self, scores, proposals):
        scores = scores.log_softmax(2).permute(1, 0, 2)  # [T, N, C]
        pred_lenghs = torch.as_tensor([scores.size(0)] * scores.size(1))
        gt_lengths = torch.cat([p.gt_number_lengths for p in proposals])
        valid = gt_lengths > 0 # all proposals here are positive person
        if valid.sum() == 0:  # no valid or no gt
            return scores.sum() * 0
        gt_seq = torch.cat([p.gt_number_sequences for p in proposals])
        gt_lengths = gt_lengths[valid]
        gt_seq = gt_seq[valid]
        pred_lenghs = pred_lenghs[valid]
        scores = scores[:, valid, :]
        loss = F.ctc_loss(scores, gt_seq, pred_lenghs, gt_lengths, zero_infinity=True)
        return loss

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
        if self.enable_box_reg:
            self.predict_boxes(predictions, proposals)
            self.predict_probs(predictions, proposals)
            # self.predict_probs_beam_search(predictions, proposals)
        else:
            self.predict_probs_without_box(predictions, proposals)
        # todo: possibly add nms
        return proposals
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
            return proposals
        _, proposal_deltas = predictions
        # flat all boxes
        proposal_boxes = Boxes.cat([b for p in proposals for b in p.get("pred_number_boxes")]).tensor.to(proposal_deltas.device)
        # num of digit boxes per image
        num_prop_per_image = [sum([len(p_number_boxes) for p_number_boxes in p.get("pred_number_boxes")]) for p in proposals]
        # proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        predict_boxes = self.box2box_transform.apply_deltas(
            # ensure fp32 for decoding precision
            proposal_deltas,
            proposal_boxes,
        )  # Nx(KxB)

        # list of lists
        num_proposals_all = [[len(b) for b in p.pred_number_boxes] for p in proposals]
        num_proposals = [sum(p) for p in num_proposals_all]
        predict_boxes = torch.split(predict_boxes, num_proposals)

        for i, (num_proposals_per_image, preds_boxes_per_image) in \
                enumerate(zip(num_proposals_all, predict_boxes)):
            pred_box_per_instance = torch.split(preds_boxes_per_image, num_proposals_per_image)
            proposals[i].pred_number_boxes = [Boxes(b) for b in pred_box_per_instance]
        return proposals

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
        # select max probability (greedy decoding) then decode index to character
        preds_prob = F.softmax(scores, dim=2)
        preds_max_prob, preds_index = preds_prob.max(dim=2)
        confidence_scores = preds_max_prob.cumprod(dim=1)[:, -1]
        num_proposals_all = [[len(b) for b in p.pred_number_boxes] for p in proposals]
        num_proposals = [sum(p) for p in num_proposals_all]
        preds_index = torch.split(preds_index, num_proposals)
        confidence_scores = torch.split(confidence_scores, num_proposals)

        for i, (num_proposals_per_image, preds_index_per_image, scores_per_image) in \
                enumerate(zip(num_proposals_all, preds_index, confidence_scores)):
            labels_per_ins = self.ctc_decode(preds_index_per_image)
            labels_per_ins = torch.split(labels_per_ins, num_proposals_per_image)
            scores_per_ins = torch.split(scores_per_image, num_proposals_per_image)
            proposals[i].pred_number_classes = list(labels_per_ins)
            proposals[i].pred_number_scores = list(scores_per_ins)

        return proposals

    def predict_probs_beam_search(
            self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Players]
    ):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        # (N, T, C)
        scores, _ = predictions
        scores = F.softmax(scores, dim=2)
        preds_index, confidence_scores = [], []
        for score in scores:
            pred_index, pred_score = beam_search_decode(score, self.char_names)
            preds_index.append(pred_index)
            confidence_scores.append(pred_score)
        preds_index = torch.stack(preds_index)
        confidence_scores = torch.cat(confidence_scores)
        # select max probability (greedy decoding) then decode index to character
        # preds_prob = F.softmax(scores, dim=2)
        # preds_max_prob, preds_index = preds_prob.max(dim=2)
        # confidence_scores = preds_max_prob.cumprod(dim=1)[:, -1]
        num_proposals_all = [[len(b) for b in p.pred_number_boxes] for p in proposals]
        num_proposals = [sum(p) for p in num_proposals_all]
        preds_index = torch.split(preds_index, num_proposals)
        confidence_scores = torch.split(confidence_scores, num_proposals)
        for i, (num_proposals_per_image, preds_index_per_image, scores_per_image) in \
                enumerate(zip(num_proposals_all, preds_index, confidence_scores)):
            labels_per_ins = torch.split(preds_index_per_image, num_proposals_per_image)
            scores_per_ins = torch.split(scores_per_image, num_proposals_per_image)
            proposals[i].pred_number_classes = list(labels_per_ins)
            proposals[i].pred_number_scores = list(scores_per_ins)

        return proposals

    def predict_probs_without_box(
            self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Players]
    ):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        scores, _ = predictions
        # select max probability (greedy decoding) then decode index to character
        preds_prob = F.softmax(scores, dim=2)
        preds_max_prob, preds_index = preds_prob.max(dim=2)
        confidence_scores = preds_max_prob.cumprod(dim=1)[:, -1]
        num_proposals = [len(p) for p in proposals]
        preds_index = torch.split(preds_index, num_proposals)
        confidence_scores = torch.split(confidence_scores, num_proposals)

        for i, (num_proposals_per_image, preds_index_per_image, scores_per_image) in \
                enumerate(zip(num_proposals, preds_index, confidence_scores)):
            # labels = []
            # could contain empty tensors
            # pred_per_instance = torch.split(preds_index_per_image, preds_index_per_image.size(0))
            # for pred in pred_per_instance:
            #     labels.append(self.ctc_decode(pred))
            labels_per_ins = self.ctc_decode(preds_index_per_image)
            labels_per_ins = torch.split(labels_per_ins, [1] * num_proposals_per_image)
            scores_per_ins = torch.split(scores_per_image, [1] * num_proposals_per_image)
            proposals[i].pred_number_classes = list(labels_per_ins)
            proposals[i].pred_number_scores = list(scores_per_ins)

        return proposals


    def ctc_decode(self, pred_classes):
        """

        Args:
            pred_classes (torch.Tensor): shape (N x T) contains tensor of predicted class ids

        Returns:
            list of torch.Tensor of size N
        """
        def _decode(pred_seq, max_length):
            """

            Args:
                pred_seq: 1d tensor of shape (T,)
                max_length: the maximum length of prediction

            Returns:
                char_list: tensor padded to max_length, a predicted sequence of a single instance
            """
            t = pred_seq.numel()
            char_list = []
            for i in range(t):
                # removing repeated characters and blank.
                if pred_seq[i] != 0 \
                        and (not (i > 0 and pred_seq[i - 1] == pred_seq[i]))\
                        and len(char_list) < max_length: # should have a better way
                    char_list.append(pred_seq[i])
            pad = max_length - len(char_list)
            char_list += [torch.as_tensor(0, dtype=torch.long, device=pred_seq.device)] * pad
            char_list = torch.stack(char_list)
            return char_list

        decoded_preds = []
        N, T = pred_classes.size()
        if not N:
            return torch.empty((0,), device=pred_classes.device)
        for i in range(N):
            decoded_preds.append(_decode(pred_classes[i], self.max_length))
        decoded_preds = torch.stack(decoded_preds)
        return decoded_preds



def build_jersey_number_head(cfg, input_shapes):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_JERSEY_NUMBER_DET.NAME
    return ROI_JERSEY_NUMBER_DET_REGISTRY.get(name)(cfg, input_shapes)