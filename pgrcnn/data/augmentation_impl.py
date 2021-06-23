from typing import List
import numpy as np
from PIL import Image
from fvcore.transforms.transform import (
    Transform,
    BlendTransform,
    CropTransform,
    HFlipTransform,
    NoOpTransform,
    PadTransform,
    Transform,
    TransformList,
    VFlipTransform,
)
from detectron2.data import detection_utils as utils
from detectron2.data.transforms import Augmentation, ExtentTransform, ResizeTransform, RotationTransform

__all__ = ["ConvertGrayscale", "copy_paste_mix_images"]
class ConvertGrayscale(Augmentation):
    """
    ConvertGrayscale
    """

    def __init__(self):
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        assert img.shape[-1] == 3, "ConvertGrayscale only works on RGB images"
        grayscale = img.dot([0.299, 0.587, 0.114])[:, :, np.newaxis]
        return BlendTransform(src_image=grayscale, src_weight=1, dst_weight=0)


def paste_image(target_image, img, alpha=0.5):
    """

    Args:
        target_image:
        img:
        x: Top-left corner x-axis coordinate to place the image
        y: Top-left corner y-axis coordinate to place the image

    Returns: If the operation succeed True/False, The shift coordinates (x, y)

    """
    retry_count = 0
    h, w = img.shape[:2]
    H, W = target_image.shape[:2]
    # generate a paste location
    x_max, x_min = W - w, 0
    y_max, y_min = H - h, 0
    # find a good place to paste
    while True:
        x = np.random.randint(x_min, x_max)
        y = np.random.randint(y_min, y_max)
        roi = target_image[y:y + h, x:x + w]
        # compute where to place
        iou = roi > 0
        if iou.mean() <= 0.5:
            target_image[y:y+h, x:x+w] = np.where(iou, roi * (1 - alpha) + img * alpha, img).astype(np.uint8)
            return (True, x, y)
        retry_count += 1
        if retry_count >= 5:
            return (False, 0, 0)



def copy_paste_mix_images(dataset_dicts,
                          format=None,
                          max_size=800,
                          min_scale=0.5,
                          max_scale=1.0,
                          interp=Image.BILINEAR
                          ):
    """

    Args:
        dataset_dicts:
        format:
        target_height: the output image height.
        target_width: the output image width.

    Returns:

    """
    num_images = len(dataset_dicts)
    target_width = max_size
    target_height = int(max_size * 9/16)
    target_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    target_dataset_dict = {
        "file_name": "synthesis_" + "_".join([str(x["image_id"]) for x in dataset_dicts]),
        "width": target_width,
        "height": target_height,
        "annotations": []
    }
    per_image_height = target_height // np.sqrt(num_images)
    per_image_width = target_width // np.sqrt(num_images)

    for dataset_dict in dataset_dicts:
        img = utils.read_image(dataset_dict["file_name"], format=format)
        utils.check_image_size(dataset_dict, img)

        # Compute the image scale and scaled size.
        input_size = img.shape[:2]
        output_size = (per_image_height, per_image_width)
        random_scale = np.random.uniform(min_scale, max_scale)
        random_scale_size = np.multiply(output_size, random_scale)
        scale = np.minimum(
            random_scale_size[0] / input_size[0], random_scale_size[1] / input_size[1]
        )
        scaled_size = np.round(np.multiply(input_size, scale)).astype(int)
        # perform augmentation
        tfm = ResizeTransform(
            input_size[0], input_size[1], scaled_size[0], scaled_size[1], interp
        )
        img = tfm.apply_image(img)

        succeeded, x, y = paste_image(target_image, img)
        if not succeeded:
            continue
        for anno in dataset_dict["annotations"]:
            person_bbox = tfm.apply_box(np.array(anno["person_bbox"]))[0]
            digit_bboxes = tfm.apply_box(np.array(anno["digit_bboxes"]))
            keypoints = np.array(anno["keypoints"]).reshape(-1, 3)
            keypoints_xy = tfm.apply_coords(keypoints[:, :2])
            keypoints_xy[:, 0] += x
            keypoints_xy[:, 1] += y
            keypoints = np.concatenate((keypoints_xy, keypoints[:, 2:]), axis=1).reshape(-1).tolist()
            person_bbox[0::2] += x
            person_bbox[1::2] += y
            digit_bboxes[:, 0::2] += x
            digit_bboxes[:, 1::2] += y
            person_bbox= person_bbox.tolist()
            digit_bboxes = digit_bboxes.tolist()
            target_dataset_dict["annotations"].append({
                "person_bbox": person_bbox,
                "digit_bboxes": digit_bboxes,
                "keypoints": keypoints,
                "category_id": 0,
                "bbox_mode": anno["bbox_mode"],
                "digit_labels": anno["digit_labels"],
                "digit_ids": anno["digit_ids"]
            })
    target_dataset_dict["image"] = target_image
    return target_dataset_dict
