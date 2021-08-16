# Copied in part from https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/utils/gaussian_target.py
from typing import Any, List, Tuple, Union
from math import sqrt

import torch
import torch.nn.functional as F
from pgrcnn.structures import Players
def compute_targets(
        instances_per_image: Players,
        heatmap_size: Tuple[int, int, int],
        offset_reg: bool = True,
        size_target_type: str = Union["wh", "ltrb"],
        size_target_scale: str = Union["feature", "ratio"],
        min_overlap: float = 0.3,
        target_name: str = "digit"
) -> Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, None], torch.Tensor]:
    """
        Encode keypoint locations into a target heatmap for use in Gaussian Focal loss.

        Arguments:
            instances_per_image: contain fields used
                keypoints: list of N tensors of keypoint locations in of shape (-1, 2).
                rois: Nx4 tensor of rois in xyxy format
            heatmap_size: represents (K, H, W)

        Returns:
            heatmaps: A tensor of shape (N, K, H, W)
    """
    K, H, W = heatmap_size
    if not instances_per_image.has("proposal_boxes"):
        device = instances_per_image.gt_digit_boxes[0].device
        return torch.empty((0, K, H, W), device=device), \
               torch.empty((0, 2), dtype=torch.float, device=device), \
               torch.empty((0, 2), dtype=torch.float, device=device) if offset_reg else None, \
               torch.empty((0, 3), dtype=torch.long, device=device)
    pred_keypoint_logits = instances_per_image.pred_keypoints_logits
    # we define a zero tensor as the output heatmaps
    N = pred_keypoint_logits.shape[0]
    rois = instances_per_image.proposal_boxes.tensor
    heatmaps = torch.zeros((N, K, H, W), device=pred_keypoint_logits.device)
    target_center_name = "gt_" + target_name + "_centers"
    target_scale_name = "gt_" + target_name + "_scales"
    if (not instances_per_image.has(target_center_name)) or rois.numel() == 0:
        return heatmaps, \
               torch.zeros((0, 2), dtype=torch.float, device=pred_keypoint_logits.device), \
               torch.empty((0, 2), dtype=torch.float, device=pred_keypoint_logits.device) if offset_reg else None, \
               torch.zeros((0, 3), dtype=torch.long, device=pred_keypoint_logits.device)
    keypoints = instances_per_image.get(target_center_name)
    scales = instances_per_image.get(target_scale_name)

    # record the positive locations
    valid = []
    # gt scale targets
    if size_target_type == "wh":
        scale_targets = []
    elif size_target_type == "ltrb":
        scale_targets = torch.zeros((N, 4, H, W), device=pred_keypoint_logits.device)
    else:
        raise NotImplementedError()
    offset_targets = []
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    if size_target_scale == "feature":
        scale_x = W / (rois[:, 2] - rois[:, 0])
        scale_y = H / (rois[:, 3] - rois[:, 1])
    elif size_target_scale == "ratio": # wrt to person roi
        scale_x = 1. / (rois[:, 2] - rois[:, 0])
        scale_y = 1. / (rois[:, 3] - rois[:, 1])
    else:
        raise NotImplementedError()

    # process per roi
    for i, (kpts, scale, dx, dy, dw, dh, roi) in \
            enumerate(zip(keypoints, scales, offset_x, offset_y, scale_x, scale_y, rois)):
        x = kpts[..., 0]
        y = kpts[..., 1]

        x_boundary_inds = x == roi[2]
        y_boundary_inds = y == roi[3]

        x = (x - dx) * dw
        y = (y - dy) * dh
        if offset_reg:
            # also compute a offset shift for prediction
            x_offset_reg = x - x.floor()
            y_offset_reg = y - y.floor()
        x = x.floor().long()
        y = y.floor().long()

        x[x_boundary_inds] = W - 1
        y[y_boundary_inds] = H - 1

        valid_loc = (x >= 0) & (y >= 0) & (x < W) & (y < H)
        # mark the positive targets
        y = y[valid_loc]
        x = x[valid_loc]
        # digit box size in feature size (w, h)
        scale = scale[valid_loc] * torch.stack((dw, dh))[None, ...] # (N, 2)
        radius = gaussian_radius(scale, min_overlap=min_overlap).int()
        radius = torch.maximum(torch.zeros_like(radius), radius)
        gen_gaussian_target(heatmaps[i],
                            [x, y],
                            radius)
        # add the roi index for easy selection for scale regression
        valid.append(
            torch.stack(((torch.ones_like(x) * i).long(), y, x), dim=1)
        )
        if size_target_type == "wh":
            scale_targets.append(scale)
        elif size_target_type == "ltrb":
            gen_ltrb_target(scale_targets[i],
                            [x, y],
                            radius,
                            scale)

        if offset_reg:
            y_offset_reg = y_offset_reg[valid_loc]
            x_offset_reg = x_offset_reg[valid_loc]
            offset_targets.append(torch.stack((x_offset_reg, y_offset_reg), dim=1))

    # we may have different number of valid points within each roi, so cat
    return heatmaps, \
           torch.cat(scale_targets, dim=0) if size_target_type == "wh" else scale_targets, \
           torch.cat(offset_targets, dim=0) if len(offset_targets) else None, \
           torch.cat(valid, dim=0)


def compute_number_targets(
        instances_per_image: Players,
        heatmap_size: Tuple[int, int, int],
        offset_reg: bool = True,
        size_target_type: str = Union["wh", "ltrb"],
        size_target_scale: str = Union["feature", "ratio"],
        min_overlap: float = 0.3
) -> Tuple[torch.Tensor, torch.Tensor, Union[torch.Tensor, None], torch.Tensor]:
    """
        Encode keypoint locations into a target heatmap for use in Gaussian Focal loss.

        Arguments:
            instances_per_image: contain fields used
                keypoints: list of N tensors of keypoint locations in of shape (-1, 2).
                rois: Nx4 tensor of rois in xyxy format
            heatmap_size: represents (K, H, W)

        Returns:
            heatmaps: A tensor of shape (N, K, H, W)
    """
    K, H, W = heatmap_size
    if not instances_per_image.has("proposal_boxes"):
        device = instances_per_image.gt_digit_boxes[0].device
        return torch.empty((0, K, H, W), device=device), \
               torch.empty((0, 2), dtype=torch.float, device=device), \
               torch.empty((0, 2), dtype=torch.float, device=device) if offset_reg else None, \
               torch.empty((0, 3), dtype=torch.long, device=device)
    pred_keypoint_logits = instances_per_image.pred_keypoints_logits
    # we define a zero tensor as the output heatmaps
    N = pred_keypoint_logits.shape[0]
    rois = instances_per_image.proposal_boxes.tensor
    heatmaps = torch.zeros((N, K, H, W), device=pred_keypoint_logits.device)
    if (not instances_per_image.has("gt_digit_centers")) or rois.numel() == 0:
        return heatmaps, \
               torch.zeros((0, 2), dtype=torch.float, device=pred_keypoint_logits.device), \
               torch.empty((0, 2), dtype=torch.float, device=pred_keypoint_logits.device) if offset_reg else None, \
               torch.zeros((0, 3), dtype=torch.long, device=pred_keypoint_logits.device)
    keypoints = instances_per_image.gt_digit_centers
    scales = instances_per_image.gt_digit_scales

    # record the positive locations
    valid = []
    # gt scale targets
    if size_target_type == "wh":
        scale_targets = []
    elif size_target_type == "ltrb":
        scale_targets = torch.zeros((N, 4, H, W), device=pred_keypoint_logits.device)
    else:
        raise NotImplementedError()
    offset_targets = []
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    if size_target_scale == "feature":
        scale_x = W / (rois[:, 2] - rois[:, 0])
        scale_y = H / (rois[:, 3] - rois[:, 1])
    elif size_target_scale == "ratio": # wrt to person roi
        scale_x = 1. / (rois[:, 2] - rois[:, 0])
        scale_y = 1. / (rois[:, 3] - rois[:, 1])
    else:
        raise NotImplementedError()

    # process per roi
    for i, (kpts, scale, dx, dy, dw, dh, roi) in \
            enumerate(zip(keypoints, scales, offset_x, offset_y, scale_x, scale_y, rois)):
        x = kpts[..., 0]
        y = kpts[..., 1]

        x_boundary_inds = x == roi[2]
        y_boundary_inds = y == roi[3]

        x = (x - dx) * dw
        y = (y - dy) * dh
        if offset_reg:
            # also compute a offset shift for prediction
            x_offset_reg = x - x.floor()
            y_offset_reg = y - y.floor()
        x = x.floor().long()
        y = y.floor().long()

        x[x_boundary_inds] = W - 1
        y[y_boundary_inds] = H - 1

        valid_loc = (x >= 0) & (y >= 0) & (x < W) & (y < H)
        # mark the positive targets
        y = y[valid_loc]
        x = x[valid_loc]
        # digit box size in feature size (w, h)
        scale = scale[valid_loc] * torch.stack((dw, dh))[None, ...] # (N, 2)
        radius = gaussian_radius(scale, min_overlap=min_overlap).int()
        radius = torch.maximum(torch.zeros_like(radius), radius)
        gen_gaussian_target(heatmaps[i],
                            [x, y],
                            radius)
        # add the roi index for easy selection for scale regression
        valid.append(
            torch.stack(((torch.ones_like(x) * i).long(), y, x), dim=1)
        )
        if size_target_type == "wh":
            scale_targets.append(scale)
        elif size_target_type == "ltrb":
            gen_ltrb_target(scale_targets[i],
                            [x, y],
                            radius,
                            scale)

        if offset_reg:
            y_offset_reg = y_offset_reg[valid_loc]
            x_offset_reg = x_offset_reg[valid_loc]
            offset_targets.append(torch.stack((x_offset_reg, y_offset_reg), dim=1))

    # we may have different number of valid points within each roi, so cat
    return heatmaps, \
           torch.cat(scale_targets, dim=0) if size_target_type == "wh" else scale_targets, \
           torch.cat(offset_targets, dim=0) if len(offset_targets) else None, \
           torch.cat(valid, dim=0)

def gaussian2D(radius, sigma=1):
    """Generate 2D gaussian kernel.

    Args:
        radius (tensor): Radius of gaussian kernel.
        sigma (int): Sigma of gaussian function. Default: 1.

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

def meshgrid2D(radius):
    """Generate 2D gaussian kernel.

    Args:
        radius (tensor): Radius of gaussian kernel.
        sigma (int): Sigma of gaussian function. Default: 1.

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
        guassian_masks.append((y, x))
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
        # out_heatmap = heatmap
        torch.max(
            masked_heatmap,
            masked_gaussian * k,
            out=heatmap[:,
                         y[i] - top[i]:y[i] + bottom[i],
                         x[i] - left[i]:x[i] + right[i]])

    return heatmap


def gaussian_radius(det_size, min_overlap=0.1):
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
    return radius_a_b


def compute_locations(size, device):
    h, w = size
    shifts_x = torch.arange(
        -w/2, w/2, dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        -h/2, h/2, dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)

    return shift_y, shift_x

def gen_ltrb_target(heatmap, center, radius, wh, stride=1):
    """Generate 2D gaussian heatmap.

    Args:
        heatmap (Tensor): Input heatmap, the gaussian kernel will cover on
            it and maintain the max value.
        center (list[Tensor]): Coord of gaussian kernel's center.
        radius (Tensor): x-axis and y-axis Radius of gaussian kernel.
        wh (Tensor).

    Returns:
        out_heatmap (Tensor): Updated heatmap covered by gaussian kernel.
    """
    if heatmap.numel() == 0 or wh.numel() == 0:
        return heatmap

    grids = meshgrid2D(radius)
    x, y = center

    height, width = heatmap.shape[-2:]
    left, right = torch.min(x, radius[:, 0]), torch.min(width - x, radius[:, 0] + 1)
    top, bottom = torch.min(y, radius[:, 1]), torch.min(height - y, radius[:, 1] + 1)
    for i, (y_grid, x_grid) in enumerate(grids):
        shift_x = x_grid[radius[i, 1] - top[i]:radius[i, 1] + bottom[i],
                               radius[i, 0] - left[i]:radius[i, 0] + right[i]]
        shift_y = y_grid[radius[i, 1] - top[i]:radius[i, 1] + bottom[i],
                               radius[i, 0] - left[i]:radius[i, 0] + right[i]]
        box_width, box_height = wh[i]
        l = shift_x + box_width / 2
        t = shift_y + box_height / 2
        r = box_width / 2 - shift_x
        b = box_height / 2 - shift_y
        reg_ltrb = torch.stack([l, t, r, b])
        heatmap[:,
        y[i] - top[i]:y[i] + bottom[i],
        x[i] - left[i]:x[i] + right[i]] = reg_ltrb
    return heatmap