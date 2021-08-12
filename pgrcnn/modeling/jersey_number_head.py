import detectron2.config
import numpy as np
from typing import List, Dict
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

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
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        here T is the feature width since we will read column by column
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
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
        self.max_length = 3
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
            return x.view((0,) + self._output_size)
            # return _NewEmptyTensorOp.apply(x, (0,) + self._output_size)
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
        if not predictions.numel():
            return {"number_loss": predictions.sum() * 0}
        predictions = predictions.log_softmax(2).permute(1, 0, 2) # [T, N, C]
        pred_lenghs = torch.as_tensor([predictions.size(0)] * predictions.size(1))
        # gt_numbers = [gt_num for p in proposals for x in p.gt_number_classes for gt_num in x if gt_num.numel()]
        # gt_lengths = torch.as_tensor([x.numel() for x in gt_numbers], dtype=torch.long, device=predictions.device)
        gt_lengths = torch.cat([p.gt_number_lengths.view(-1) for p in proposals])
        gt_seq = torch.cat([p.gt_number_sequences.view(-1, self.max_length) for p in proposals])
        valid = gt_lengths > 0
        gt_lengths = gt_lengths[valid]
        gt_seq  = gt_seq[valid]
        # N = gt_lengths.numel()
        # gt_seq = torch.zeros((N, self.max_length), dtype=torch.long, device=predictions.device)
        # for i in range(N):
        #     gt_seq[i, :gt_lengths[i]] = gt_numbers[i]
        loss = F.ctc_loss(predictions, gt_seq, pred_lenghs, gt_lengths, zero_infinity=True)
        return {"number_loss": loss}

    def inference(self, predictions, proposals):
        # select max probability (greedy decoding) then decode index to character
        preds_prob = F.softmax(predictions, dim=2)
        preds_max_prob, preds_index = preds_prob.max(dim=2)
        confidence_scores = preds_max_prob.cumprod(dim=1)[:, -1]
        pred_lenghs = torch.as_tensor([predictions.size(1)] * predictions.size(0))
        texts = []
        # texts (N,): a list of tensors
        for index, l in enumerate(pred_lenghs):
            t = preds_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(t[i])
            if len(char_list):
                char_list = torch.stack(char_list)
            else:
                char_list = torch.empty((0,), device=predictions.device)

            texts.append(char_list)
        # assign instances to each image
        num_number_boxes = [len(p.proposal_number_boxes) for p in proposals]
        for i in range(len(proposals)):
            num = num_number_boxes[i] # could be any value 0 ->
            first_few_pred, texts = texts[:num], texts[num:]
            first_few_scores, confidence_scores = confidence_scores[:num], confidence_scores[num:]
            proposals[i].pred_jersey_numbers = first_few_pred
            proposals[i].pred_jersey_numbers_scores = first_few_scores
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