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
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.data.transforms import Augmentation, ExtentTransform, ResizeTransform, RotationTransform

__all__ = ["ConvertGrayscale", "copy_paste_mix_images", "RandColor"]
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

class RandColor(Augmentation):
    """
    ConvertGrayscale
    """

    def __init__(self):
        super().__init__()
        self._init(locals())

    def get_transform(self, img):
        w = np.random.rand(3)
        mix_w = 0.5
        shift = (np.random.rand(3) - 0.5) * 20
        # img.mean()[np.newaxis] * w
        return BlendTransform(src_image=img * w + shift, src_weight=1 - mix_w, dst_weight=mix_w)


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
        x = np.random.randint(x_min, x_max + 1)
        y = np.random.randint(y_min, y_max + 1)
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
                          min_scale=0.2,
                          # max_scale=1.0,
                          interp=Image.BILINEAR,
                          augmentations=None
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
    target_height = int(max_size * np.random.uniform(0.5, 1.5))
    target_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    target_dataset_dict = {
        "file_name": "synthesis_" + "_".join([str(x["image_id"]) for x in dataset_dicts]),
        "width": target_width,
        "height": target_height,
        "annotations": []
    }
    per_image_height = target_height // np.sqrt(num_images)
    per_image_width = target_width // np.sqrt(num_images)
    output_size = (per_image_height, per_image_width)
    max_scale = np.sqrt(num_images) - 0.1
    # create a list images of different size
    imgs = []
    tfms = []
    for dataset_dict in dataset_dicts:
        img = utils.read_image(dataset_dict["file_name"], format=format)
        utils.check_image_size(dataset_dict, img)
        # apply the per image augmentation
        aug_input = T.AugInput(img, sem_seg=None)
        transforms = augmentations(aug_input)
        img = aug_input.image

        # Compute the image scale and scaled size.
        input_size = img.shape[:2]
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
        imgs.append(img)
        tfms.append(tfm)

    # sort the images from largest to smallest, so it will not overlap too much
    img_sizes = [-img.size for img in imgs]
    sorted_inds = np.argsort(img_sizes)
    imgs = [imgs[i] for i in sorted_inds]
    dataset_dicts = [dataset_dicts[i] for i in sorted_inds]
    tfms = [tfms[i] for i in sorted_inds]
    for dataset_dict, img, tfm in zip(dataset_dicts, imgs, tfms):
        succeeded, x, y = paste_image(target_image, img)
        if not succeeded:
            continue
        for anno in dataset_dict["annotations"]:
            instance_anno = {"bbox_mode": anno["bbox_mode"]}
            if "person_bbox" in anno:
                person_bbox = tfm.apply_box(np.array(anno["person_bbox"], dtype=np.float32))[0]
                person_bbox[0::2] += x
                person_bbox[1::2] += y
                person_bbox = person_bbox.tolist()
                instance_anno.update({"person_bbox": person_bbox, "category_id": 0})
            if "keypoints" in anno:
                keypoints = np.array(anno["keypoints"], dtype=np.float64).reshape(-1, 3)
                keypoints_xy = tfm.apply_coords(keypoints[:, :2])
                keypoints_xy[:, 0] += x
                keypoints_xy[:, 1] += y
                keypoints = np.concatenate((keypoints_xy, keypoints[:, 2:]), axis=1).reshape(-1).tolist()
                instance_anno.update({"keypoints": keypoints})
            if "digit_bboxes" in anno:
                digit_bboxes = tfm.apply_box(np.array(anno["digit_bboxes"], dtype=np.float32))
                digit_bboxes[:, 0::2] += x
                digit_bboxes[:, 1::2] += y
                digit_bboxes = digit_bboxes.tolist()
                instance_anno.update({"digit_bboxes": digit_bboxes,
                                      "digit_ids": anno["digit_ids"]})
            if "number_bbox" in anno:
                number_bbox = tfm.apply_box(np.array(anno["number_bbox"], dtype=np.float32))[0]
                number_bbox[0::2] += x
                number_bbox[1::2] += y
                number_bbox = number_bbox.tolist()
                instance_anno.update({"number_bbox": number_bbox,
                                      "number_sequence": anno["number_sequence"],
                                      "number_id": anno["number_id"]
                                      })
            target_dataset_dict["annotations"].append(instance_anno)
    target_dataset_dict["image"] = target_image
    return target_dataset_dict
