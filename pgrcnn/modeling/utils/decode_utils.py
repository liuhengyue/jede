from typing import Any, List, Tuple, Union

import torch
import torch.nn.functional as F

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
    topk_clses = topk_inds // (height * width)
    topk_inds = topk_inds % (height * width)
    topk_ys = topk_inds // width
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
    num_rand = round(k * (1 - ratio))
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
    topk_clses = topk_inds // (height * width)
    topk_inds = topk_inds % (height * width)
    topk_ys = topk_inds // width
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs

def gather_feat(feat, ind, mask=None):
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


def transpose_and_gather_feat(feat, ind):
    """Transpose and gather feature according to index.

    Args:
        feat (Tensor): Target feature map.
        ind (Tensor): Target coord index.

    Returns:
        feat (Tensor): Transposed and gathered feature.
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat


def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _topk(scores, K=40, largest=True):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K, largest=largest)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def _sample_top_n_bttm_k(scores, k=10, ratio=1/4):
    topk = get_topk_from_heatmap(scores, k=round(k * ratio))
    bttm2k = get_topk_from_heatmap(scores, k=round(k * (1 - ratio)), largest=False)
    return [torch.cat((t, b), dim=-1) for t, b in zip(topk, bttm2k)]

def ctdet_decode(heat, wh, rois, reg=None, cat_spec_wh=False, K=100, feature_scale="feature", training=True):
    batch, cat, height, width = heat.size()

    if batch == 0:
        return torch.zeros(0, 0, 6, device=heat.device)

    heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = get_local_maximum(heat)
    if training:
        scores, inds, clses, ys, xs = get_topk_random_from_heatmap(heat, k=K)
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
    # we could have negative width or height
    wh.clamp_(min=0)
    if cat_spec_wh:
        wh = wh.view(batch, K, cat, 2)
        clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
        wh = wh.gather(2, clses_ind).view(batch, K, 2)
    else:
        wh = wh.view(batch, K, 2)
    if feature_scale == "ratio":
        wh[..., 0:1] = wh[..., 0:1] * roi_widths
        wh[..., 1:2] = wh[..., 1:2] * roi_heights
    elif feature_scale == "feature":
        wh[..., 0:1] = (wh[..., 0:1] / width) * roi_widths
        wh[..., 1:2] = (wh[..., 1:2] / height) * roi_heights
    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                        ys - wh[..., 1:2] / 2,
                        xs + wh[..., 0:1] / 2,
                        ys + wh[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)

    return detections



