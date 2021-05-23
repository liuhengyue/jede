import torch
import torch.nn.functional as F

from detectron2.layers import cat
from detectron2.utils.events import get_event_storage

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = F.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

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

def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

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

def ctdet_decode(heat, wh, rois, reg=None, cat_spec_wh=False, K=100, feature_scale="feature"):
    batch, cat, height, width = heat.size()

    if batch == 0:
        return torch.zeros(0, 0, 6, device=heat.device)

    heat = torch.sigmoid(heat)
    # perform nms on heatmaps
    heat = _nms(heat)

    scores, inds, clses, ys, xs = _topk(heat, K=K)
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

def pg_rcnn_loss(pred_keypoint_logits, pred_scale_logits, instances, normalizer):
    """
    Wrap center and size loss here.
    We treat the predicted centers as keypoints.
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
    scale_targets = []
    keypoint_side_len = pred_keypoint_logits.shape[2]
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        keypoints = instances_per_image.gt_digit_centers
        heatmaps_per_image, valid_per_image = keypoints.to_heatmap(
            instances_per_image.proposal_boxes.tensor, keypoint_side_len
        )
        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))
        scale_targets.append(instances_per_image.gt_digit_scales[valid_per_image == 1])

    if len(heatmaps):
        keypoint_targets = cat(heatmaps, dim=0)
        valid = cat(valid, dim=0).to(dtype=torch.uint8)
        valid = torch.nonzero(valid).squeeze(1)
        scale_targets = cat(scale_targets, dim=0)

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
    keypoint_targets = keypoint_targets[valid]
    ct_loss = F.cross_entropy(
        pred_keypoint_logits[valid], keypoint_targets, reduction="sum"
    )

    # If a normalizer isn't specified, normalize by the number of visible keypoints in the minibatch
    if normalizer is None:
        normalizer = valid.numel()
    ct_loss /= normalizer

    # size loss
    pred_scale_logits = pred_scale_logits.view(N, 2, H * W)
    # (N, 2)
    valid = valid // K # K-agnostic
    pred_scale_logits = pred_scale_logits[valid, :, keypoint_targets]
    # we predict the scale wrt. S
    proposal_boxes = cat([x.proposal_boxes.tensor for x in instances], dim=0)[valid]
    proposal_boxes_w = proposal_boxes[:, 2] - proposal_boxes[:, 0]
    proposal_boxes_h = proposal_boxes[:, 3] - proposal_boxes[:, 1]
    proposal_boxes_size = torch.stack((proposal_boxes_w, proposal_boxes_h), dim=1)
    # actual digit size = ratio * person bbox size
    # pred_scale_logits = pred_scale_logits * proposal_boxes_size / W
    scale_targets = W * scale_targets / proposal_boxes_size
    wh_loss = F.smooth_l1_loss(pred_scale_logits, scale_targets, reduction='sum') / normalizer

    return {'ct_loss': ct_loss, 'wh_loss': wh_loss}