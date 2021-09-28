from typing import Any, List, Tuple, Union, Optional

import torch
import torch.nn.functional as F
import numpy as np
from CTCDecoder.ctc_decoder import best_path, beam_search

def get_local_maximum(heat, kernel=3):
    """Extract local maximum pixel with given kernal.

    Args:
        heat (Tensor): Target heatmap.
        kernel (int): Kernel size of max pooling. Default: 3.

    Returns:
        heat (Tensor): A heatmap where local maximum pixels maintain its
            own value and other positions are 0.
    """
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def get_topk_from_heatmap(scores, k=20, largest=True):
    """Get top k positions from heatmap.

    Args:
        scores (Tensor): Target heatmap with shape
            [batch, num_classes, height, width].
        k (int): Target number. Default: 20.

    Returns:
        tuple[torch.Tensor]: Scores, indexes, categories and coords of
            topk keypoint. Containing following Tensors:

        - topk_scores (Tensor): Max scores of each topk keypoint.
        - topk_inds (Tensor): Indexes of each topk keypoint.
        - topk_clses (Tensor): Categories of each topk keypoint.
        - topk_ys (Tensor): Y-coord of each topk keypoint.
        - topk_xs (Tensor): X-coord of each topk keypoint.
    """
    batch, _, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), k, largest=largest)
    topk_clses = torch.div(topk_inds, (height * width), rounding_mode='floor')
    topk_inds = topk_inds % (height * width)
    topk_ys = torch.div(topk_inds, width, rounding_mode='floor')
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

def get_topk_random_from_heatmap(scores, k=20, ratio=1/4):
    """Get top k positions from heatmap.

    Args:
        scores (Tensor): Target heatmap with shape
            [batch, num_classes, height, width].
        k (int): Target total number of points to return. Default: 20.
        ratio (float): Target fraction of the top scores.

    Returns:
        tuple[torch.Tensor]: Scores, indexes, categories and coords of
            topk keypoint. Containing following Tensors:

        - topk_scores (Tensor): Max scores of each topk keypoint.
        - topk_inds (Tensor): Indexes of each topk keypoint.
        - topk_clses (Tensor): Categories of each topk keypoint.
        - topk_ys (Tensor): Y-coord of each topk keypoint.
        - topk_xs (Tensor): X-coord of each topk keypoint.
    """
    num_topk = round(k * ratio)
    num_rand = k - num_topk
    batch, _, height, width = scores.size()
    sorted_scores, sorted_inds = torch.sort(scores.view(batch, -1), descending=True)
    topk_scores = sorted_scores[:, :num_topk]
    topk_inds = sorted_inds[:, :num_topk]
    # add random selection
    rest_scores = sorted_scores[:, num_topk:]
    rest_inds = sorted_inds[:, num_topk:]
    perm = torch.randperm(rest_scores.size(1), device=scores.device)[:num_rand]
    rest_scores = rest_scores[:, perm]
    rest_inds = rest_inds[:, perm]
    topk_scores = torch.cat((topk_scores, rest_scores), dim=1)
    topk_inds = torch.cat((topk_inds, rest_inds), dim=1)
    topk_clses = torch.div(topk_inds, (height * width), rounding_mode='floor')
    topk_inds = topk_inds % (height * width)
    topk_ys = torch.div(topk_inds, width, rounding_mode='floor')
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

def _gather_feat(feat, ind, mask=None):
    """Gather feature according to index.

    Args:
        feat (Tensor): Target feature map.
        ind (Tensor): Target coord index.
        mask (Tensor | None): Mask of feature map. Default: None.

    Returns:
        feat (Tensor): Gathered feature.
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).repeat(1, 1, dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    """Transpose and gather feature according to index.

    Args:
        feat (Tensor): Target feature map.
        ind (Tensor): Target coord index.

    Returns:
        feat (Tensor): Transposed and gathered feature.
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat



def ctdet_decode(heat, wh, reg, rois,
                 cat_spec_wh=False,
                 K=100,
                 size_target_type="wh",
                 size_target_scale="feature",
                 training=True,
                 fg_ratio=1/2,
                 offset=0.0):
    batch, cat, height, width = heat.size()

    if batch == 0:
        return torch.zeros(0, 0, 6, device=heat.device)

    # heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = get_local_maximum(heat)
    if training:
        scores, inds, clses, ys, xs = get_topk_random_from_heatmap(heat, k=K, ratio=fg_ratio)
    else:
        scores, inds, clses, ys, xs = get_topk_from_heatmap(heat, k=K)
    if reg is not None:
        reg = _transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
    else:
        xs = xs.view(batch, K, 1) + 0.5
        ys = ys.view(batch, K, 1) + 0.5

    # convert heatmap coordinates to real image coordinates
    # shape (N)
    offset_x = rois[:, 0].view(-1, 1, 1)
    offset_y = rois[:, 1].view(-1, 1, 1)
    roi_widths = (rois[:, 2] - rois[:, 0]).clamp(min=1).view(-1, 1, 1)
    roi_heights = (rois[:, 3] - rois[:, 1]).clamp(min=1).view(-1, 1, 1)
    # (N, 100, 1)
    xs = (xs / width) * roi_widths + offset_x
    ys = (ys / height) * roi_heights + offset_y
    wh = _transpose_and_gather_feat(wh, inds)
    # we could have negative width or height (solved by using relu)
    # we may have values larger than the feature box
    wh.clamp_(max=height)
    # we can add a offset manually to enlarge the detected bounding boxes
    # if not training:
    #     wh[..., 1].add_(offset)
        # wh += offset
    if cat_spec_wh:
        wh = wh.view(batch, K, cat, -1)
        clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, wh.shape[-1]).long()
        wh = wh.gather(2, clses_ind).view(batch, K, -1)
    else:
        wh = wh.view(batch, K, -1)

    # get image level size
    if size_target_scale == "ratio":
        wh[..., 0::2] = wh[..., 0::2] * roi_widths
        wh[..., 1::2] = wh[..., 1::2] * roi_heights
    elif size_target_scale == "feature":
        wh[..., 0::2] = (wh[..., 0::2] / width) * roi_widths
        wh[..., 1::2] = (wh[..., 1::2] / height) * roi_heights
    if training:
        # we have many empty boxes which decrease the number bg samples, add some random boxes
        # in image scale
        num_fg = round(K * fg_ratio)
        # priors: height 0.13 0.03, width 0.17 0.05
        if size_target_type == "ltrb":
            roi_scale = torch.cat([roi_widths, roi_heights, roi_widths, roi_heights], dim=2) / 2
            h1_ratio = torch.normal(0.13, 0.03, (batch, K - num_fg, 1), device=roi_scale.device)
            w1_ratio = torch.normal(0.17, 0.05, (batch, K - num_fg, 1), device=roi_scale.device)
            h2_ratio = torch.normal(0.13, 0.03, (batch, K - num_fg, 1), device=roi_scale.device)
            w2_ratio = torch.normal(0.17, 0.05, (batch, K - num_fg, 1), device=roi_scale.device)
            scale_ratios = torch.cat([w1_ratio, h1_ratio, w2_ratio, h2_ratio], dim=2)
        elif size_target_type == "wh":
            roi_scale = torch.cat([roi_widths, roi_heights], dim=2)
            h1_ratio = torch.normal(0.13, 0.03, (batch, K - num_fg, 1), device=roi_scale.device)
            w1_ratio = torch.normal(0.17, 0.05, (batch, K - num_fg, 1), device=roi_scale.device)
            scale_ratios = torch.cat([w1_ratio, h1_ratio], dim=2)
        # += (add random number) or = (replace with random boxes)
        wh[:, num_fg:, ...] = scale_ratios * roi_scale

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    if size_target_type == "wh":
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)
    elif size_target_type == "ltrb":
        bboxes = torch.cat([xs - wh[..., 0:1],
                            ys - wh[..., 1:2],
                            xs + wh[..., 2:3],
                            ys + wh[..., 3:4]], dim=2)

    detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections


def beam_search_decode(scores: torch.Tensor, chars: str, max_length: int = 2):
    """

    Args:
        scores: Tensor of TxC shape
        chars: The whole set of chars for decoding

    Returns: text: tensor of shape (max_length,)
             score: tensor of shape (1,)

    """
    device = scores.device
    scores = scores.detach().cpu().numpy()
    # need to move the first class to the end
    scores = np.roll(scores, -1, axis=1)
    text, score = beam_search(scores, chars)
    if len(text) > max_length:
        text = text[:max_length]
    pad = max_length - len(text)
    text += [0] * pad
    text = torch.as_tensor(text, dtype=torch.long, device=device)
    score = torch.as_tensor([score], device=device)
    return text, score


