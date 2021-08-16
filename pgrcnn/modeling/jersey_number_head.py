import detectron2.config
import numpy as np
from typing import List, Dict
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
from detectron2.utils import comm
from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, Linear, ShapeSpec, get_norm, ModulatedDeformConv
from detectron2.layers.wrappers import _NewEmptyTensorOp
from detectron2.utils.registry import Registry
from detectron2.modeling.backbone.resnet import DeformBottleneckBlock

from .digit_neck_branches import build_digit_neck_branch
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
                 input_shapes: ShapeSpec
                 ):
        """

        """
        super().__init__()
        self.max_length = cfg.MODEL.ROI_JERSEY_NUMBER_DET.SEQ_MAX_LENGTH
        input_resolution = (input_shapes.height, input_shapes.width)
        self.seq_resolution = cfg.MODEL.ROI_JERSEY_NUMBER_DET.SEQUENCE_RESOLUTION
        in_channels = input_shapes.channels
        hidden_size = 256
        self.num_class = 10 + 1
        self.context = nn.Sequential(
            BidirectionalLSTM(in_channels, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size)
        )

        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d(self.seq_resolution)
        self.output = nn.Linear(hidden_size, self.num_class)
        # (h, w, c)
        self._output_size = (self.seq_resolution[1], self.num_class)



    def forward(self, x):
        if not x.numel():
            return _NewEmptyTensorOp.apply(x, (0,) + self._output_size)
            # x = _NewEmptyTensorOp.apply(x, (x.size(0), x.size(1)) + self.seq_resolution)
        # else:
        # [n, c, h, w] -> [n, c, 1, w]
        x = self.AdaptiveAvgPool(x)
        # [n, w, c, 1]
        x = x.permute(0, 3, 1, 2)
        # [n, w, c]
        x = x.squeeze(3)
        x = self.context(x)
        # [n, w, num_cls]
        x = self.output(x)
        return x

    def losses(self, predictions, proposals):
        predictions = predictions.log_softmax(2).permute(1, 0, 2) # [T, N, C]
        pred_lenghs = torch.as_tensor([predictions.size(0)] * predictions.size(1))
        # gt_numbers = [gt_num for p in proposals for x in p.gt_number_classes for gt_num in x if gt_num.numel()]
        # gt_lengths = torch.as_tensor([x.numel() for x in gt_numbers], dtype=torch.long, device=predictions.device)
        # if we have no instance
        gt_number_classes = torch.cat([torch.cat(p.gt_number_classes)
                                               if len(p.gt_number_classes)
                                               else torch.empty((0,), dtype=torch.long, device=predictions.device)
                                                for p in proposals]) # 0 is bg
        try:
            gt_lengths = torch.cat([torch.cat(p.gt_number_lengths)
                                    if len(p.gt_number_lengths)
                                    else torch.empty((0,), dtype=torch.long, device=predictions.device)
                                    for p in proposals])
        except:
            pass
        gt_seq = torch.cat([torch.cat(p.gt_number_sequences)
                            if len(p.gt_number_sequences)
                            else torch.empty((0, self.max_length), dtype=torch.long, device=predictions.device)
                            for p in proposals])
        valid = gt_number_classes > 0
        if valid.sum() == 0: # no valid or no gt
            return {"number_loss": predictions.sum() * 0}
        gt_lengths = gt_lengths[valid]
        gt_seq = gt_seq[valid]
        pred_lenghs = pred_lenghs[valid]
        predictions = predictions[:, valid, :]
        loss = F.ctc_loss(predictions, gt_seq, pred_lenghs, gt_lengths, zero_infinity=True)
        return {"number_loss": loss}

    def ctc_decode(self, pred_classes):
        """

        Args:
            pred_classes (torch.Tensor): shape (N x T) contains tensor of predicted class ids

        Returns:

        """
        def _decode(pred_seq, max_length):
            """

            Args:
                pred_seq: 1d tensor of shape (T,)
                max_length: the maximum length of prediction

            Returns:
                char_list: a predicted sequence of
            """
            t = pred_seq.numel()
            char_list = []
            for i in range(t):
                # removing repeated characters and blank.
                if pred_seq[i] != 0 \
                        and (not (i > 0 and pred_seq[i - 1] == pred_seq[i]))\
                        and len(char_list) < max_length: # should have a better way
                    char_list.append(pred_seq[i])
            if len(char_list):
                char_list = torch.stack(char_list)
            else:
                char_list = torch.empty((0,), device=pred_seq.device)
            return char_list

        N, T = pred_classes.size()
        if not N:
            return torch.empty((0,), device=pred_classes.device)
        assert N == 1
        return _decode(pred_classes[0], self.max_length)
        # decoded_preds = []
        # for i in range(N):
        #     decoded_preds.append(_decode(pred_classes[i], self.max_length))
        # return decoded_preds

    def inference(self, predictions, proposals):
        # select max probability (greedy decoding) then decode index to character
        preds_prob = F.softmax(predictions, dim=2)
        preds_max_prob, preds_index = preds_prob.max(dim=2)
        confidence_scores = preds_max_prob.cumprod(dim=1)[:, -1]
        # num_proposals = [len(p) for p in proposals]
        # list of lists
        num_proposals_all = [[len(b) for b in p.proposal_number_boxes] for p in proposals]
        num_proposals = [sum(p) for p in num_proposals_all]
        preds_index = torch.split(preds_index, num_proposals)
        confidence_scores = [list(torch.split(score, num_p)) for score, num_p in zip(torch.split(confidence_scores, num_proposals), num_proposals_all)]
        labels_all = [] # list of lists

        for num_proposals_per_image, preds_index_per_image in \
            zip(num_proposals_all, preds_index):
            labels = []
            # could contain empty tensors
            pred_per_instance = torch.split(preds_index_per_image, num_proposals_per_image)
            for pred in pred_per_instance:
                labels.append(self.ctc_decode(pred))
            labels_all.append(labels)

        # for preds_index_per_image in preds_index:
        #     texts.append(self.ctc_decode(preds_index_per_image))
        try:
            for i in range(len(proposals)):
                proposals[i].pred_jersey_numbers = labels_all[i]
                proposals[i].pred_jersey_numbers_scores = confidence_scores[i]
        except:
            pass
        return proposals


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


def build_jersey_number_head(cfg, input_shapes):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_JERSEY_NUMBER_DET.NAME
    return ROI_JERSEY_NUMBER_DET_REGISTRY.get(name)(cfg, input_shapes)