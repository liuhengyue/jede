import torch
from torch.nn import functional as F

from detectron2.layers import cat
from .utils import compute_targets, compute_number_targets


def gaussian_focal_loss(pred, gaussian_target, alpha=2.0, gamma=4.0, eps=1e-12):
    """`Focal Loss <https://arxiv.org/abs/1708.02002>`_ for targets in gaussian
    distribution.
    Args:
        pred (torch.Tensor): The prediction.
        gaussian_target (torch.Tensor): The learning target of the prediction
            in gaussian distribution.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 2.0.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 4.0.
    """
    # pred = torch.sigmoid(pred)
    pos_weights = gaussian_target.eq(1)
    neg_weights = (1 - gaussian_target).pow(gamma)
    pos_loss = (-(pred + eps).log() * (1 - pred).pow(alpha) * pos_weights).sum()
    neg_loss = (-(1 - pred + eps).log() * pred.pow(alpha) * neg_weights).sum()
    return pos_loss + neg_loss


def ltrb_giou_loss(pred, target, ct_weights = None, eps: float = 1e-7, reduction='sum'):
    """
    Args:
        pred: Nx4 predicted bounding boxes
        target: Nx4 target bounding boxes
        Both are in the form of FCOS prediction (l, t, r, b)
    """
    # pred.clamp_(min=0)
    pred_left = pred[:, 0]
    pred_top = pred[:, 1]
    pred_right = pred[:, 2]
    pred_bottom = pred[:, 3]

    target_left = target[:, 0]
    target_top = target[:, 1]
    target_right = target[:, 2]
    target_bottom = target[:, 3]

    target_aera = (target_left + target_right) * \
                  (target_top + target_bottom)
    pred_aera = (pred_left + pred_right) * \
                (pred_top + pred_bottom)

    w_intersect = torch.min(pred_left, target_left) + \
                  torch.min(pred_right, target_right)
    h_intersect = torch.min(pred_bottom, target_bottom) + \
                  torch.min(pred_top, target_top)

    g_w_intersect = torch.max(pred_left, target_left) + \
                    torch.max(pred_right, target_right)
    g_h_intersect = torch.max(pred_bottom, target_bottom) + \
                    torch.max(pred_top, target_top)
    ac_uion = g_w_intersect * g_h_intersect

    area_intersect = w_intersect * h_intersect
    area_union = target_aera + pred_aera - area_intersect

    ious = area_intersect / (area_union + eps)
    gious = ious - (ac_uion - area_union) / (ac_uion + eps)
    loss = 1 - gious
    if ct_weights is not None:
        loss = loss * ct_weights
    if reduction == "mean":
        loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def pg_rcnn_loss(
        pred_keypoint_logits,
        pred_scale_logits,
        pred_offset_logits,
        instances,
        normalizer=None,
        output_head_weights=(1,0, 1.0, 1.0),
        size_target_type="ltrb",
        size_target_scale="feature",
        target_name="digit",
        add_box_constraints=False
        ):
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
        normalizer (Union[float, None]): Normalize the loss by this amount.
            If not specified, we normalize by the number of visible keypoints in the minibatch.
        size_weight (float): Weight for the size regression loss.

    Returns a scalar tensor containing the loss.
    """
    has_offset_reg = pred_offset_logits is not None
    has_size_reg = pred_scale_logits is not None
    heatmaps = []
    valid = []
    scale_targets = []
    offset_targets =[]
    # keypoint_side_len = pred_keypoint_logits.shape[2]
    for instances_per_image in instances:
        heatmaps_per_image, scales_per_image, offsets_per_image, valid_per_image = \
            compute_targets(instances_per_image, pred_keypoint_logits.shape[1:],
                            offset_reg=has_offset_reg,
                            size_target_type=size_target_type,
                            size_target_scale=size_target_scale,
                            target_name=target_name)
        heatmaps.append(heatmaps_per_image)
        valid.append(valid_per_image)
        scale_targets.append(scales_per_image)
        offset_targets.append(offsets_per_image)
    # should be safe since we return empty tensors from `compute_targets'
    keypoint_targets = cat(heatmaps, dim=0)
    valid = cat(valid, dim=0)
    scale_targets = cat(scale_targets, dim=0)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if keypoint_targets.numel() == 0 or valid.numel() == 0:
        loss = {target_name + '_ct_loss': pred_keypoint_logits.sum() * 0}
        if has_size_reg:
            loss.update({target_name + '_wh_loss': pred_scale_logits.sum() * 0})
            if add_box_constraints:
                loss.update({target_name + '_constraint_loss': pred_scale_logits.sum() * 0})
        if has_offset_reg:
            loss.update({target_name + "_os_loss": pred_offset_logits.sum() * 0})
        return loss, keypoint_targets



    ct_loss = gaussian_focal_loss(
        pred_keypoint_logits,
        keypoint_targets
    ) * output_head_weights[0]

    # If a normalizer isn't specified, normalize by the number of visible keypoints in the minibatch
    if normalizer is None:
        normalizer = valid.size(0)
    ct_loss /= normalizer
    loss = {target_name + '_ct_loss': ct_loss}
    if has_size_reg:
        # size loss
        if size_target_type == 'wh':
            pred_scale_logits = pred_scale_logits[valid[:, 0], :, valid[:, 1], valid[:, 2]]
            # we predict the scale wrt. feature box
            wh_loss = output_head_weights[1] * F.smooth_l1_loss(pred_scale_logits, scale_targets,
                                                                reduction='sum') / normalizer

        elif size_target_type == 'ltrb':
            # valid_loc = (keypoint_targets > 0.).squeeze(1)
            # giou loss
            # wh_loss = output_head_weights[1] * ltrb_giou_loss(pred_scale_logits.permute(0, 2, 3, 1)[valid_loc],
            #                                                   scale_targets.permute(0, 2, 3, 1)[valid_loc],
            #                                                   None,
            #                                                   reduction='sum') / valid_loc.sum()

            # smooth_l1
            # wh_loss = output_head_weights[1] * F.smooth_l1_loss(pred_scale_logits.permute(0, 2, 3, 1)[valid_loc],
            #                                                     scale_targets.permute(0, 2, 3, 1)[valid_loc],
            #                                                     reduction='sum') / normalizer

            # weighted by center
            wh_loss = output_head_weights[1] * (F.smooth_l1_loss(pred_scale_logits,
                                                              scale_targets) * torch.square(keypoint_targets)).sum()

        loss.update({target_name + '_wh_loss': wh_loss})
        if add_box_constraints:
            # we want the predictions cover the gt
            diff_preds_gts = pred_scale_logits - scale_targets
            constrain_loss = -torch.clip(diff_preds_gts, max=0).sum() / normalizer
            loss.update({target_name + '_constraint_loss': constrain_loss})


    if has_offset_reg:
        offset_targets = cat(offset_targets, dim=0)
        pred_offset_logits = pred_offset_logits[valid[:, 0], :, valid[:, 1], valid[:, 2]]
        os_loss = output_head_weights[2] * F.smooth_l1_loss(pred_offset_logits, offset_targets, reduction='sum') / normalizer
        loss.update({target_name + "_os_loss": os_loss})

    # if has_num_digits_logits:
    #     gt_num_digits = torch.bincount(valid[:, 0], minlength=keypoint_targets.shape[0])
    #     num_digit_cls_loss = cross_entropy(num_digits_logits, gt_num_digits)
    #     loss.update({"num_digit_cls_loss": num_digit_cls_loss})

    return loss, keypoint_targets


def pg_rcnn_number_loss(
        pred_keypoint_logits,
        pred_scale_logits,
        pred_offset_logits,
        instances,
        normalizer=None,
        output_head_weights=(1,0, 1.0, 1.0),
        size_target_type="ltrb",
        size_target_scale="feature"
        ):
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
        normalizer (Union[float, None]): Normalize the loss by this amount.
            If not specified, we normalize by the number of visible keypoints in the minibatch.
        size_weight (float): Weight for the size regression loss.

    Returns a scalar tensor containing the loss.
    """
    has_offset_reg = pred_offset_logits is not None
    heatmaps = []
    valid = []
    scale_targets = []
    offset_targets =[]
    # keypoint_side_len = pred_keypoint_logits.shape[2]
    for instances_per_image in instances:
        heatmaps_per_image, scales_per_image, offsets_per_image, valid_per_image = \
            compute_number_targets(instances_per_image, pred_keypoint_logits.shape[1:],
                                    offset_reg=has_offset_reg,
                                    size_target_type=size_target_type,
                                    size_target_scale=size_target_scale)
        heatmaps.append(heatmaps_per_image)
        valid.append(valid_per_image)
        scale_targets.append(scales_per_image)
        offset_targets.append(offsets_per_image)
    # should be safe since we return empty tensors from `compute_targets'
    keypoint_targets = cat(heatmaps, dim=0)
    valid = cat(valid, dim=0)
    scale_targets = cat(scale_targets, dim=0)

    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if keypoint_targets.numel() == 0 or valid.numel() == 0:
        loss = {'ct_loss': pred_keypoint_logits.sum() * 0,
                'wh_loss': pred_scale_logits.sum() * 0}
        if has_offset_reg:
            loss.update({"os_loss": pred_offset_logits.sum() * 0})
        return loss



    ct_loss = gaussian_focal_loss(
        pred_keypoint_logits,
        keypoint_targets
    ) * output_head_weights[0]

    # If a normalizer isn't specified, normalize by the number of visible keypoints in the minibatch
    if normalizer is None:
        normalizer = valid.size(0)
    ct_loss /= normalizer

    # size loss
    if size_target_type == 'wh':
        pred_scale_logits = pred_scale_logits[valid[:, 0], :, valid[:, 1], valid[:, 2]]
        # we predict the scale wrt. feature box
        wh_loss = output_head_weights[1] * F.smooth_l1_loss(pred_scale_logits, scale_targets,
                                                            reduction='sum') / normalizer
    elif size_target_type == 'ltrb':
        # valid_loc = (keypoint_targets > 0.).squeeze(1)
        # giou loss
        # wh_loss = output_head_weights[1] * ltrb_giou_loss(pred_scale_logits.permute(0, 2, 3, 1)[valid_loc],
        #                                                   scale_targets.permute(0, 2, 3, 1)[valid_loc],
        #                                                   None,
        #                                                   reduction='sum') / valid_loc.sum()

        # smooth_l1
        # wh_loss = output_head_weights[1] * F.smooth_l1_loss(pred_scale_logits.permute(0, 2, 3, 1)[valid_loc],
        #                                                     scale_targets.permute(0, 2, 3, 1)[valid_loc],
        #                                                     reduction='sum') / normalizer

        # weighted by center
        wh_loss = output_head_weights[1] * (F.smooth_l1_loss(pred_scale_logits,
                                                          scale_targets) * torch.square(keypoint_targets)).sum()

    loss = {'ct_loss': ct_loss, 'wh_loss': wh_loss}
    if has_offset_reg:
        offset_targets = cat(offset_targets, dim=0)
        pred_offset_logits = pred_offset_logits[valid[:, 0], :, valid[:, 1], valid[:, 2]]
        os_loss = output_head_weights[2] * F.smooth_l1_loss(pred_offset_logits, offset_targets, reduction='sum') / normalizer
        loss.update({"os_loss": os_loss})

    # if has_num_digits_logits:
    #     gt_num_digits = torch.bincount(valid[:, 0], minlength=keypoint_targets.shape[0])
    #     num_digit_cls_loss = cross_entropy(num_digits_logits, gt_num_digits)
    #     loss.update({"num_digit_cls_loss": num_digit_cls_loss})

    return loss