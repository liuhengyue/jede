import copy
import logging
import os.path

import numpy as np
import cv2
import random
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.catalog import DatasetCatalog, MetadataCatalog
from . import det_utils
from . import copy_paste_mix_images

"""
This file contains the default mapping that's applied to "dataset dicts".
With customization to the Jersey Numbers in the Wild Dataset.
"""

__all__ = ["JerseyNumberDatasetMapper"]


class JerseyNumberDatasetMapper(DatasetMapper):
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic.

    The callable currently does the following:

    1. Read the image from "filename"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(self, *args, **kwargs):
        # fmt: off
        self.digit_only        = kwargs.pop("digit_only")
        self.num_interests     = kwargs.pop("num_interests")
        self.pad_to_full       = kwargs.pop("pad_to_full")
        self.keypoints_inds    = kwargs.pop("keypoints_inds")
        self.copy_paste_mix    = kwargs.pop("copy_paste_mix")
        self.max_size_train    = kwargs.pop("max_size_train")
        self.seq_max_length    = kwargs.pop("seq_max_length")
        self.per_image_augmentations = kwargs.pop("per_image_augmentations")
        self.helper_dataset_name = kwargs.pop("helper_dataset_name")
        self.swap_digit = True if self.helper_dataset_name is not None else False

        # fmt: on
        super().__init__(*args, **kwargs)
        self.helper_dataset = self.process_helper_dataset()

    def process_helper_dataset(self, reset_cache=False):
        if not self.swap_digit:
            return None
        metadata = MetadataCatalog.get(self.helper_dataset_name)
        helper_annos_path = os.path.join(metadata.get("dataset_root"), "annotations/helper_annos.pt")
        if os.path.exists(helper_annos_path) and (not reset_cache):
            annos = torch.load(helper_annos_path)
            # some are empty or too small
            mean_size = np.median([x[0].size for x in annos])
            annos = [x for x in annos if x[0].size > mean_size]
            return annos
        dataset_dicts = DatasetCatalog.get(self.helper_dataset_name)
        # dataset_dicts = dataset_dicts[:10]
        img_patches = []
        labels = []
        for dataset_dict in dataset_dicts:
            img = utils.read_image(dataset_dict["file_name"])
            for anno in dataset_dict["annotations"]:
                digit_bboxes = anno["digit_bboxes"]
                digit_ids = anno["digit_ids"]
                labels += digit_ids
                for box in digit_bboxes:
                    img_patch = img[box[1]:box[3]+1, box[0]:box[2]+1, :]
                    img_patches.append(img_patch)
        results = list(zip(img_patches, labels))
        torch.save(results, helper_annos_path)
        return results



    @classmethod
    def from_config(cls, cfg, is_train: bool = True):
        ret = super().from_config(cfg, is_train)
        # our customizations
        augs = det_utils.build_augmentation(cfg, is_train)
        per_image_augmentations = det_utils.build_per_image_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
        ret.update(
            {
                "digit_only": cfg.DATASETS.DIGIT_ONLY,
                "num_interests": cfg.DATASETS.NUM_INTERESTS,
                "pad_to_full": cfg.DATASETS.PAD_TO_FULL,
                "keypoints_inds": cfg.DATASETS.KEYPOINTS_INDS,
                # update augmentations
                "augmentations": augs,
                "per_image_augmentations": T.AugmentationList(per_image_augmentations),
                "copy_paste_mix": cfg.INPUT.AUG.COPY_PASTE_MIX,
                "max_size_train": cfg.INPUT.MAX_SIZE_TRAIN,
                "helper_dataset_name": cfg.INPUT.AUG.HELPER_DATASET_NAME,
                # seq model
                "seq_max_length": cfg.MODEL.ROI_NUMBER_BOX_HEAD.SEQ_MAX_LENGTH,
            }
        )
        return ret

    def __call__(self, dataset_dict):
        if isinstance(dataset_dict, dict):
            return self._process_single_dataset_dict(dataset_dict)
        elif isinstance(dataset_dict, list):
            return self._process_multiple_dataset_dicts(dataset_dict)

    def apply_helper_annos(self, img, dataset_dict):
        # we randomly apply
        if np.random.rand(1) > 0.5:
            return img, dataset_dict
        img = img.copy()
        annos = dataset_dict["annotations"]
        for i, anno in enumerate(annos):
            digit_bboxes = anno["digit_bboxes"]
            digit_ids = anno["digit_ids"]
            for j, (box, label) in enumerate(zip(digit_bboxes, digit_ids)):
                # we could do for each digit
                if np.random.rand(1) > 0.5:
                    continue
                patch, helper_label = self.helper_dataset[np.random.randint(len(self.helper_dataset))]
                box = [int(coord + 0.5) for coord in box]
                x1, y1, x2, y2 = box
                h, w, _ = img.shape
                x1 = max(x1, 0)
                y1 = max(y1, 0)
                x2 = min(x2, w-1)
                y2 = min(y2, h-1)
                patch = cv2.resize(patch, dsize=(x2-x1+1, y2-y1+1))
                # simple random color + contrast
                patch = (1.2 * patch - 0.2 * patch.mean()) * (np.random.uniform(0.8, 1.2, 3))
                patch = np.clip(patch, 0, 255).astype(np.uint8)
                img[y1:y2+1, x1:x2+1, :] = patch
                dataset_dict["annotations"][i]["digit_ids"][j] = helper_label
                # todo: we did not modify the number_id
        return img, dataset_dict


    def _process_single_dataset_dict(self, dataset_dict):
        """
            Args:
                dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

            Returns:
                dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)
        if self.swap_digit:
            image, dataset_dict = self.apply_helper_annos(image, dataset_dict)
        aug_input = T.AugInput(image, sem_seg=None)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                anno.pop("digit_labels", None)
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                det_utils.transform_instance_annotations(
                    obj, transforms, image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices,
                    num_interests=self.num_interests,
                    pad_to_full=self.pad_to_full,
                    keypoints_inds=self.keypoints_inds
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = det_utils.annotations_to_instances(
                annos, image_shape, digit_only=self.digit_only,
                seq_max_length=self.seq_max_length
            )
            dataset_dict["instances"] = instances
            # dataset_dict["instances"] = det_utils.filter_empty_instances(instances)
        return dataset_dict

    def _process_multiple_dataset_dicts(self, dataset_dicts):
        """
            Args:
                dataset_dicts (List[dict]): Metadata of multiple images, in Detectron2 Dataset format.

            Returns:
                dict: a format that builtin models in detectron2 accept
        """
        if self.copy_paste_mix:
            dataset_dicts = copy.deepcopy(dataset_dicts)  # it will be modified by code below
            dataset_dict = copy_paste_mix_images(dataset_dicts,
                                                 format=self.image_format,
                                                 max_size=self.max_size_train,
                                                 augmentations=self.per_image_augmentations)
            image = dataset_dict["image"]
            if self.swap_digit:
                image, dataset_dict = self.apply_helper_annos(image, dataset_dict)
            aug_input = T.AugInput(image, sem_seg=None)
            transforms = self.augmentations(aug_input)
            image = aug_input.image
            image_shape = image.shape[:2]  # h, w
            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

            if not self.is_train:
                dataset_dict.pop("annotations", None)
                dataset_dict.pop("sem_seg_file_name", None)
                return dataset_dict

            if "annotations" in dataset_dict:
                # USER: Modify this if you want to keep them for some reason.
                for anno in dataset_dict["annotations"]:
                    anno.pop("digit_labels", None)
                    if not self.use_instance_mask:
                        anno.pop("segmentation", None)
                    if not self.use_keypoint:
                        anno.pop("keypoints", None)

                # USER: Implement additional transformations if you have other types of data
                annos = [
                    det_utils.transform_instance_annotations(
                        obj, transforms, image_shape,
                        keypoint_hflip_indices=self.keypoint_hflip_indices,
                        num_interests=self.num_interests,
                        pad_to_full=self.pad_to_full,
                        keypoints_inds=self.keypoints_inds
                    )
                    for obj in dataset_dict.pop("annotations")
                    if obj.get("iscrowd", 0) == 0
                ]
                instances = det_utils.annotations_to_instances(
                    annos, image_shape, digit_only=self.digit_only,
                    seq_max_length=self.seq_max_length
                )
                dataset_dict["instances"] = instances
                # dataset_dict["instances"] = det_utils.filter_empty_instances(instances)
            return dataset_dict
        else:
           raise NotImplementedError("Reached a wrong place.")