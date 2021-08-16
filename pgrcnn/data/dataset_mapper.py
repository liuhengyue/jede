import copy
import logging
import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
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

        # fmt: on
        super().__init__(*args, **kwargs)

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
                # seq model
                "seq_max_length": cfg.MODEL.ROI_JERSEY_NUMBER_DET.SEQ_MAX_LENGTH
            }
        )
        return ret

    def __call__(self, dataset_dict):
        if isinstance(dataset_dict, dict):
            return self._process_single_dataset_dict(dataset_dict)
        elif isinstance(dataset_dict, list):
            return self._process_multiple_dataset_dicts(dataset_dict)

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