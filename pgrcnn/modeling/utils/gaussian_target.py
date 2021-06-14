# Copied in part from https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/utils/gaussian_target.py
from typing import Any, List, Tuple, Union
from math import sqrt

import torch
import torch.nn.functional as F

def compute_targets(
        keypoints: List[torch.Tensor],
        scales: List[torch.Tensor],
        rois: torch.Tensor,
        pred_keypoint_logits: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
        Encode keypoint locations into a target heatmap for use in Gaussian Focal loss.

        Arguments:
            keypoints: list of N tensors of keypoint locations in of shape (-1, 2).
            rois: Nx4 tensor of rois in xyxy format
            heatmap_size: represents (N, K, H, W)

        Returns:
            heatmaps: A tensor of shape (N, K, H, W)
    """
    # we define a zero tensor as the output heatmaps
    heatmaps = torch.zeros_like(pred_keypoint_logits)
    # record the positive locations
    valid = []
    # gt scale targets
    scale_targets = []
    if rois.numel() == 0:
        return heatmaps
    heatmap_size = heatmaps.shape
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    scale_x = heatmap_size[3] / (rois[:, 2] - rois[:, 0])
    scale_y = heatmap_size[2] / (rois[:, 3] - rois[:, 1])

    # process per roi
    for i, (kpts, scale, dx, dy, dw, dh, roi) in \
            enumerate(zip(keypoints, scales, offset_x, offset_y, scale_x, scale_y, rois)):
        x = kpts[..., 0]
        y = kpts[..., 1]

        x_boundary_inds = x == roi[2]
        y_boundary_inds = y == roi[3]

        x = (x - dx) * dw
        x = x.floor().long()
        y = (y - dy) * dh
        y = y.floor().long()

        x[x_boundary_inds] = heatmap_size[3] - 1
        y[y_boundary_inds] = heatmap_size[2] - 1

        valid_loc = (x >= 0) & (y >= 0) & (x < heatmap_size[3]) & (y < heatmap_size[2])
        # mark the positive targets
        y = y[valid_loc]
        x = x[valid_loc]
        # digit box size in feature size (w, h)
        scale = scale[valid_loc] * torch.stack((dw, dh))[None, ...]
        radius = gaussian_radius(scale, min_overlap=0.3)
        gen_gaussian_target(heatmaps[i],
                            [x, y],
                            radius)
        # add the roi index for easy selection for scale regression
        valid.append(
            torch.stack(((torch.ones_like(x) * i).long(), y, x), dim=1)
        )
        scale_targets.append(scale)
    # we may have different number of valid points within each roi, so cat
    return heatmaps, torch.cat(scale_targets, dim=0), torch.cat(valid, dim=0)



def gaussian2D(radius, sigma=1):
    """Generate 2D gaussian kernel.

    Args:
        radius (int): Radius of gaussian kernel.
        sigma (int): Sigma of gaussian function. Default: 1.
        dtype (torch.dtype): Dtype of gaussian tensor. Default: torch.float32.
        device (str): Device of gaussian tensor. Default: 'cpu'.

    Returns:
        h (Tensor): Gaussian kernel with a
            ``(2 * radius + 1) * (2 * radius + 1)`` shape.
    """
    num_instances = radius.size(0)
    starts, ends = -radius, radius + 1
    guassian_masks = []
    # different gauss kernels have different range, had to use loop
    for i in range(num_instances):
        y, x = torch.meshgrid(torch.arange(starts[i][1], ends[i][1], device=radius.device),
                              torch.arange(starts[i][0], ends[i][0], device=radius.device),
                              )
        # range (0, 1]
        h = torch.exp(-(x ** 2 / (2 * sigma[i, 0] ** 2) +
                        y ** 2 / (2 * sigma[i, 1] ** 2)))
        h[h < torch.finfo(h.dtype).eps * h.max()] = 0
        guassian_masks.append(h)
    return guassian_masks




def gen_gaussian_target(heatmap, center, radius, k=1):
    """Generate 2D gaussian heatmap.

    Args:
        heatmap (Tensor): Input heatmap, the gaussian kernel will cover on
            it and maintain the max value.
        center (list[int]): Coord of gaussian kernel's center.
        radius (Tensor): x-axis and y-axis Radius of gaussian kernel.
        k (int): Coefficient of gaussian kernel. Default: 1.

    Returns:
        out_heatmap (Tensor): Updated heatmap covered by gaussian kernel.
    """
    if radius.numel() == 0:
        return heatmap
    diameter = 2 * radius + 1
    # list of gaussian kernels
    gaussian_kernels = gaussian2D(
        radius, sigma=diameter / 6)

    x, y = center

    height, width = heatmap.shape[-2:]

    left, right = torch.min(x, radius[:, 0]), torch.min(width - x, radius[:, 0] + 1)
    top, bottom = torch.min(y, radius[:, 1]), torch.min(height - y, radius[:, 1] + 1)

    for i, gaussian_kernel in enumerate(gaussian_kernels):
        masked_heatmap = heatmap[:,
                         y[i] - top[i]:y[i] + bottom[i],
                         x[i] - left[i]:x[i] + right[i]]
        masked_gaussian = gaussian_kernel[radius[i, 1] - top[i]:radius[i, 1] + bottom[i],
                                          radius[i, 0] - left[i]:radius[i, 0] + right[i]]
        out_heatmap = heatmap
        torch.max(
            masked_heatmap,
            masked_gaussian * k,
            out=out_heatmap[:,
                         y[i] - top[i]:y[i] + bottom[i],
                         x[i] - left[i]:x[i] + right[i]])

    return out_heatmap


def gaussian_radius(det_size, min_overlap=0.3):
    r"""Generate 2D gaussian radius.

    Args:
        det_size (Tensor): Shape of object.
        min_overlap (float): Min IoU with ground truth for boxes generated by
            keypoints inside the gaussian kernel.

    Returns:
        radius (int): Radius of gaussian kernel.
    """
    factor = (1 - sqrt(min_overlap)) / sqrt(2)  # > 0
    radius_a_b = det_size * factor + 1
    return radius_a_b.int()



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


def get_topk_from_heatmap(scores, k=20):
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
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), k)
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