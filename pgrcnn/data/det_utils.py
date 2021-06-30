import logging
import numpy as np
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager
from PIL import Image

from detectron2.structures import (
    Instances,
    BitMasks,
    # Boxes,
    BoxMode,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)
from detectron2.data import transforms as T
from detectron2.data.detection_utils import transform_keypoint_annotations
from detectron2.data.catalog import MetadataCatalog

from pgrcnn.structures.boxes import Boxes
from pgrcnn.structures.players import Players
from pgrcnn.structures.digitboxes import DigitBoxes
from pgrcnn.data import augmentation_impl as custom_T

# each person will only have at most 2 digits which we pad to
MAX_DIGIT_PER_INSTANCE = 2

def transform_instance_annotations(
        annotation, transforms, image_size, *,
        keypoint_hflip_indices=None,
        num_interests=3,
        pad_to_full=True,
        keypoints_inds=[5, 6, 12, 11]
):
    """
    Apply transforms to box, segmentation and keypoints annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place. It will only contain the annotation of a single player.
            Suppose it has a n-digit jersey number, then the keys/vals of the dict:
            ['digit_bboxes', list of lists [n, 4]
            'keypoints', list of ints [4 x 3]
            'person_bbox', list of ints [4]
            'category_id', int [1]
            'bbox_mode', XYXY_ABS
            'digit_ids', list of ints [n,]
            ]
        transforms (TransformList):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.
        digit_only (bool): if true, only gt digit is used.

        We pad everthing here, for digit ids, we pad with -1.

    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.



    """
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    # if we have coco loaded
    if "bbox" in annotation:
        annotation['person_bbox'] = annotation.pop('bbox')
    if "person_bbox" in annotation:
        bbox = BoxMode.convert(annotation["person_bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
        # Note that bbox is 1d (per-instance bounding box)
        annotation["person_bbox"] = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
        annotation["bbox_mode"] = BoxMode.XYXY_ABS
    if "digit_bboxes" in annotation:
        bbox = BoxMode.convert(annotation["digit_bboxes"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
        # (n, 4) if given an empty list, it will return (0, 4)
        bbox = transforms.apply_box(np.array(bbox)).clip(min=0)
        annotation["digit_bboxes"] = bbox

    # we have annotated 4 keypoints, but we can still maintain the 17 keypoints format
    # _C.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 17 use COCO dataset
    # indice: "left_shoulder" 5, "right_shoulder", 6, "left_hip", 11 "right_hip", 12
    num_keypoints = 17 if pad_to_full else 4
    full_keypoints = np.zeros((num_keypoints, 3))
    if "keypoints" in annotation and len(annotation["keypoints"]) > 0:
        keypoints = transform_keypoint_annotations(
            annotation["keypoints"], transforms, image_size, keypoint_hflip_indices
        )
        if keypoints.shape[0] == 4:
            # the keypoints order is left_sholder, right_shoulder, right_hip, left_hip
            full_keypoints[keypoints_inds, :] = keypoints
        elif keypoints.shape[0] == 17:
            full_keypoints = keypoints
        else:
            raise NotImplementedError("wrong number of keypoints {}".format(keypoints.shape[0]))
    annotation["keypoints"] = full_keypoints


    return annotation




def annotations_to_instances(annos,
                             image_size,
                             digit_only=False,
                             num_keypoints=17
                             ):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes", "gt_digits", "gt_digit_boxes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
            This is the format that builtin models expect.

            Instances:
                'fields':
                    gt_boxes       (N, 4)
                    gt_classes     (N)
                    gt_keypoints   (N, 4, 3)
                    gt_digit_boxes (N, 2, 4)
                    gt_digits      (N, 2, 1)
                    gt_digit_centers (N, 3, 3) order: center, right, left
                    gt_digit_scales (N, 2, 2)



    """
    if digit_only:
        target = Instances(image_size)
        boxes = [BoxMode.convert(obj["digit_bboxes"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
        # remove padded boxes
        boxes = np.concatenate([box[np.any(box > 0, axis=1)] for box in boxes])
        boxes = target.gt_boxes = Boxes(boxes)
        boxes.clip(image_size)
        classes = np.concatenate([obj["digit_ids"][np.where(obj["digit_ids"] > -1)] for obj in annos])
        # ids are solved by cfg in datatset
        classes = torch.tensor(classes, dtype=torch.int64).view(-1)
        target.gt_classes = classes
        return target

    target = Players(image_size)
    # person bboxes
    # check if we have person_bbox
    has_person_bbox = annos[0].get("person_bbox", 0)
    if has_person_bbox:
        boxes = [BoxMode.convert(obj["person_bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS)
                 for obj in annos if obj.get("person_bbox", 0)]
        boxes = Boxes(boxes)
        boxes.clip(image_size)
        # we may have empty after cropping
        keep = boxes.nonempty()
        target.gt_boxes = boxes[keep]
        keep_inds = keep.nonzero(as_tuple=True)[0]
        # better to just select the kept annos
        annos = [annos[keep_idx] for keep_idx in keep_inds]
        classes = [obj["category_id"] for obj in annos if obj.get("category_id", 0)]
        classes = torch.tensor(classes, dtype=torch.int64)
        target.gt_classes = classes
    if "digit_bboxes" in annos[0]:
        boxes = [BoxMode.convert(obj["digit_bboxes"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
        # if the person is filtered out, then its digit boxes should be fitlered out
        boxes = target.gt_digit_boxes = [Boxes(box) for box in boxes]
        classes = [obj["digit_ids"] for obj in annos]
        classes = [torch.tensor(cls, dtype=torch.int64) for cls in classes]
        for i, (box, label) in enumerate(zip(boxes, classes)):
            boxes[i].clip(image_size)
            keep = box.nonempty()
            boxes[i] = box[keep]
            classes[i] = label[keep]

        target.gt_digit_classes = classes
        # add centers and scales
        target.gt_digit_centers = [box.get_centers() for box in boxes]
        target.gt_digit_scales = [box.get_scales() for box in boxes]

    # not every instance has the keypoints annotation, so we pad it
    kpts = [obj.get("keypoints", np.zeros((num_keypoints, 3))) for obj in annos]
    target.gt_keypoints = Keypoints(kpts)
    return target



def filter_empty_instances(instances, box_threshold=1e-5):
    """
    Filter out empty instances in an `Instances` object (by bounding box)

    Args:
        instances (Instances):
        box_threshold (float): minimum width and height to be considered non-empty

    Returns:
        Instances: the filtered instances.
    """

    return instances[instances.gt_boxes.nonempty(threshold=box_threshold)]


def gen_crop_transform_with_instance(crop_size, image_size, instance):
    """
    Generate a CropTransform so that the cropping region contains
    the center of the given instance.

    Args:
        crop_size (tuple): h, w in pixels
        image_size (tuple): h, w
        instance (dict): an annotation dict of one instance, in Detectron2's
            dataset format.
    """
    crop_size = np.asarray(crop_size, dtype=np.int32)
    bbox = BoxMode.convert(instance["person_bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS)
    center_yx = (bbox[1] + bbox[3]) * 0.5, (bbox[0] + bbox[2]) * 0.5

    min_yx = np.maximum(np.floor(center_yx).astype(np.int32) - crop_size, 0)
    max_yx = np.maximum(np.asarray(image_size, dtype=np.int32) - crop_size, 0)
    max_yx = np.minimum(max_yx, np.ceil(center_yx).astype(np.int32))

    y0 = np.random.randint(min_yx[0], max_yx[0] + 1)
    x0 = np.random.randint(min_yx[1], max_yx[1] + 1)
    return T.CropTransform(x0, y0, crop_size[1], crop_size[0])


def check_metadata_consistency(key, dataset_names):
    """
    Check that the datasets have consistent metadata.

    Args:
        key (str): a metadata key
        dataset_names (list[str]): a list of dataset names

    Raises:
        AttributeError: if the key does not exist in the metadata
        ValueError: if the given datasets do not have the same metadata values defined by key
    """
    if len(dataset_names) == 0:
        return
    logger = logging.getLogger(__name__)
    entries_per_dataset = [getattr(MetadataCatalog.get(d), key) for d in dataset_names]
    for idx, entry in enumerate(entries_per_dataset):
        if entry != entries_per_dataset[0]:
            logger.error(
                "Metadata '{}' for dataset '{}' is '{}'".format(key, dataset_names[idx], str(entry))
            )
            logger.error(
                "Metadata '{}' for dataset '{}' is '{}'".format(
                    key, dataset_names[0], str(entries_per_dataset[0])
                )
            )
            raise ValueError("Datasets have different metadata '{}'!".format(key))



def build_augmentation(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Now it includes resizing and flipping.

    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(
            len(min_size)
        )

    augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
    # augmentation = []
    assert cfg.INPUT.RANDOM_FLIP == "none", "For jersey number, it does not make sense to do flipping."
    # we could use grayscale images for both training and testing
    if cfg.INPUT.AUG.GRAYSCALE:
        augmentation.append(custom_T.ConvertGrayscale())
    if is_train:
        if cfg.INPUT.AUG.COLOR:
            # tfm_gens.append(T.RandomLighting(scale=10.0))
            augmentation.append(T.RandomBrightness(0.5, 1.5))
            augmentation.append(T.RandomSaturation(0.5, 1.5))
            augmentation.append(T.RandomContrast(0.5, 1.5))

        if cfg.INPUT.AUG.EXTEND:
            augmentation.append(T.RandomExtent((1.2, 2.0), (0.4, 0.4)))

    return augmentation
