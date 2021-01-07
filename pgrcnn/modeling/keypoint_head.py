from typing import List
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ConvTranspose2d, cat, interpolate
from detectron2.structures import heatmaps_to_keypoints
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from detectron2.modeling.roi_heads.keypoint_head import ROI_KEYPOINT_HEAD_REGISTRY
from detectron2.modeling.roi_heads.keypoint_head import KRCNNConvDeconvUpsampleHead
from pgrcnn.structures.instances import CustomizedInstances as Instances


_TOTAL_SKIPPED = 0

def keypoint_rcnn_loss(pred_keypoint_logits, instances, normalizer):
    """
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

    Returns:
         a scalar tensor containing the loss.
         valid pred_keypoins_logits with at least 3 keypoints inside rois
         sampled instances based on valid keypoints
    """
    heatmaps = []
    valid = []
    # list of valid 0/1 values for each instances
    # if at least 3 keypoints are valid for one roi, we consider it as a roi to train
    # the transformation matrix regression
    valid_rois = []
    sampled_instances = []
    keypoint_side_len = pred_keypoint_logits.shape[2]
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue
        keypoints = instances_per_image.gt_keypoints
        heatmaps_per_image, valid_per_image = keypoints.to_heatmap(
            instances_per_image.proposal_boxes.tensor, keypoint_side_len
        )
        heatmaps.append(heatmaps_per_image.view(-1))
        valid.append(valid_per_image.view(-1))
        valid_roi_per_image = torch.sum(valid_per_image == 1, dim=1) > 2
        valid_rois.append(valid_roi_per_image)
        # sample the instances for this image
        valid_roi_per_image = torch.nonzero(valid_roi_per_image).squeeze(1)
        sampled_instances.append(instances_per_image[valid_roi_per_image])
    # experiment on apply valid per-image
    # valid_copy = copy.deepcopy(valid)
    # pred_keypoint_logits_valid = []
    # for valid_per_image in valid_copy:
    #     valid_per_image = torch.nonzero(valid_per_image).squeeze(1)
    #     pred_keypoint_logits_valid.append(pred_keypoint_logits[valid_per_image])
    # pred_keypoint_logits_valid = cat(pred_keypoint_logits_valid, dim=0)
    if len(heatmaps):

        keypoint_targets = cat(heatmaps, dim=0)
        valid = cat(valid, dim=0).to(dtype=torch.uint8)
        valid = torch.nonzero(valid).squeeze(1)
        valid_rois = cat(valid_rois, dim=0).to(dtype=torch.uint8)
        valid_rois = torch.nonzero(valid_rois).squeeze(1)

    N, K, H, W = pred_keypoint_logits.shape
    # torch.mean (in binary_cross_entropy_with_logits) doesn't
    # accept empty tensors, so handle it separately
    if len(heatmaps) == 0 or valid.numel() == 0:
        global _TOTAL_SKIPPED
        _TOTAL_SKIPPED += 1
        storage = get_event_storage()
        storage.put_scalar("kpts_num_skipped_batches", _TOTAL_SKIPPED, smoothing_hint=False)
        return pred_keypoint_logits.sum() * 0, torch.empty((0, K, H, W)), sampled_instances

    # shape (N', 4, 56, 56)
    pred_keypoint_logits_valid = pred_keypoint_logits[valid_rois]


    pred_keypoint_logits = pred_keypoint_logits.view(N * K, H * W)



    keypoint_loss = F.cross_entropy(
        pred_keypoint_logits[valid], keypoint_targets[valid], reduction="sum"
    )

    # If a normalizer isn't specified, normalize by the number of visible keypoints in the minibatch
    if normalizer is None:
        normalizer = valid.numel()
    keypoint_loss /= normalizer

    return keypoint_loss, pred_keypoint_logits_valid, sampled_instances

def keypoint_rcnn_inference(pred_keypoint_logits, pred_instances):
    """
    Post process each predicted keypoint heatmap in `pred_keypoint_logits` into (x, y, score)
        and add it to the `pred_instances` as a `pred_keypoints` field.

    Args:
        pred_keypoint_logits (Tensor): A tensor of shape (R, K, S, S) where R is the total number
           of instances in the batch, K is the number of keypoints, and S is the side length of
           the keypoint heatmap. The values are spatial logits.
        pred_instances (list[Instances]): A list of N Instances, where N is the number of images.

    Returns:
        None. Each element in pred_instances will contain an extra "pred_keypoints" field.
            The field is a tensor of shape (#instance, K, 3) where the last
            dimension corresponds to (x, y, score).
            The scores are larger than 0.
    """
    # flatten all bboxes from all images together (list[Boxes] -> Rx4 tensor)
    bboxes_flat = cat([b.pred_boxes.tensor for b in pred_instances], dim=0)

    keypoint_results = heatmaps_to_keypoints(pred_keypoint_logits.detach(), bboxes_flat.detach())
    num_instances_per_image = [len(i) for i in pred_instances]
    keypoint_results = keypoint_results[:, :, [0, 1, 3]].split(num_instances_per_image, dim=0)
    keypoint_logits = pred_keypoint_logits.split(num_instances_per_image, dim=0)

    for keypoint_results_per_image, kpt_logits_per_image, instances_per_image in \
            zip(keypoint_results, keypoint_logits, pred_instances):
        # keypoint_results_per_image is (num instances)x(num keypoints)x(x, y, score)
        instances_per_image.pred_keypoints = keypoint_results_per_image
        instances_per_image.pred_keypoints_logits = kpt_logits_per_image


@ROI_KEYPOINT_HEAD_REGISTRY.register()
class KPGRCNNHead(KRCNNConvDeconvUpsampleHead):
    """
        A standard keypoint head containing a series of 3x3 convs, followed by
        a transpose convolution and bilinear interpolation for upsampling.

        With extra output of keypoints heatmaps for kpts to digit box usage.
        """

    @configurable
    def __init__(self, input_shape, *, num_keypoints, conv_dims, **kwargs):
        super().__init__(input_shape, num_keypoints=num_keypoints, conv_dims=conv_dims, **kwargs)


    def forward(self, x, instances: List[Instances]):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            (mainly modify this part since we want the heatmaps as return also)
            A dict of losses if in training. The predicted "instances" if in inference.
        """
        x = self.layers(x)
        if self.training:
            num_images = len(instances)
            normalizer = (
                None if self.loss_normalizer == "visible" else num_images * self.loss_normalizer
            )
            kpt_loss, sampled_keypoints_logits, sampled_instances = keypoint_rcnn_loss(x, instances, normalizer=normalizer)
            return {
                "loss_keypoint": kpt_loss * self.loss_weight
            }, sampled_keypoints_logits, sampled_instances
        else:
            keypoint_rcnn_inference(x, instances)
            return instances