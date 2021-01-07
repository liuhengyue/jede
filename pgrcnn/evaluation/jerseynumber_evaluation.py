# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import datetime
import pickle
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from fvcore.common.file_io import PathManager
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.utils.logger import create_small_table
from fvcore.common.file_io import PathManager, file_lock
from fvcore.common.timer import Timer
from detectron2.structures import BoxMode, PolygonMasks, Boxes

from detectron2.evaluation.evaluator import DatasetEvaluator
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.evaluation.coco_evaluation import COCOEvaluator
logger = logging.getLogger(__name__)


class JerseyNumberEvaluator(COCOEvaluator):
    """
    Evaluate object proposal, instance detection/segmentation, keypoint detection
    outputs using COCO's metrics and APIs.

    Inherit from COCOEvaluator with modifications.
    """

    def __init__(self, dataset_name, cfg, distributed, output_dir=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
                It must have either the following corresponding metadata:

                    "json_file": the path to the COCO format annotation

                Or it must be in detectron2's standard dataset format
                so it can be converted to COCO format automatically.
            cfg (CfgNode): config instance
            distributed (True): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): optional, an output directory to dump results.
        """
        self._tasks = self._tasks_from_config(cfg)
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")
        self._logger = logging.getLogger(__name__)

        self._metadata = MetadataCatalog.get(dataset_name)
        if not hasattr(self._metadata, "json_file"):
            self._logger.warning(f"json_file was not found in MetaDataCatalog for '{dataset_name}'")

            cache_path = os.path.join(output_dir, f"{dataset_name}_coco_format.json")
            self._metadata.json_file = cache_path
            # this function is modified
            convert_to_coco_json(dataset_name, cache_path, allow_cached=False) # delete previous converted coco json

        json_file = PathManager.get_local_path(self._metadata.json_file)
        with contextlib.redirect_stdout(io.StringIO()):
            self._coco_api = COCO(json_file)

        self._kpt_oks_sigmas = cfg.TEST.KEYPOINT_OKS_SIGMAS
        # Test set json files do not contain annotations (evaluation must be
        # performed using the COCO evaluation server).
        self._do_evaluation = "annotations" in self._coco_api.dataset
        self.digit_only = cfg.DATASETS.DIGIT_ONLY


    def _eval_predictions(self, tasks, predictions):
        """
        Evaluate predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in coco_results:
                category_id = result["category_id"]
                assert (
                    category_id in reverse_id_mapping
                ), "A prediction has category_id={}, which is not available in the dataset.".format(
                    category_id
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        for task in sorted(tasks):

            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api, coco_results, task, kpt_oks_sigmas=self._kpt_oks_sigmas
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results[task] = res

    def _derive_coco_results(self, coco_eval, iou_type, class_names=None):
        """
        Derive the desired score numbers from summarized COCOeval.

        Args:
            coco_eval (None or COCOEval): None represents no predictions from model.
            iou_type (str):
            class_names (None or list[str]): if provided, will use it to predict
                per-category AP.

        Returns:
            a dict of {metric name: score}
        """

        metrics = {
            "digit_bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "person_bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
            "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        }[iou_type]

        if coco_eval is None:
            self._logger.warn("No predictions from the model!")
            return {metric: float("nan") for metric in metrics}

        # the standard metrics
        results = {
            metric: float(coco_eval.stats[idx] * 100 if coco_eval.stats[idx] >= 0 else "nan")
            for idx, metric in enumerate(metrics)
        }
        self._logger.info(
            "Evaluation results for {}: \n".format(iou_type) + create_small_table(results)
        )
        if not np.isfinite(sum(results.values())):
            self._logger.info("Note that some metrics cannot be computed.")

        if class_names is None or len(class_names) <= 1:
            return results
        # Compute per-category AP
        # from https://github.com/facebookresearch/Detectron/blob/a6a835f5b8208c45d0dce217ce9bbda915f44df7/detectron/datasets/json_dataset_evaluator.py#L222-L252 # noqa
        precisions = coco_eval.eval["precision"]
        # precision has dims (iou, recall, cls, area range, max dets)
        assert len(class_names) == precisions.shape[2]

        results_per_category = []
        for idx, name in enumerate(class_names):
            # area range index 0: all area ranges
            # max dets index -1: typically 100 per image
            precision = precisions[:, :, idx, 0, -1]
            precision = precision[precision > -1]
            ap = np.mean(precision) if precision.size else float("nan")
            results_per_category.append(("{}".format(name), float(ap * 100)))

        # tabulate it
        N_COLS = min(6, len(results_per_category) * 2)
        results_flatten = list(itertools.chain(*results_per_category))
        results_2d = itertools.zip_longest(*[results_flatten[i::N_COLS] for i in range(N_COLS)])
        table = tabulate(
            results_2d,
            tablefmt="pipe",
            floatfmt=".3f",
            headers=["category", "AP"] * (N_COLS // 2),
            numalign="left",
        )
        self._logger.info("Per-category {} AP: \n".format(iou_type) + table)

        results.update({"AP-" + name: ap for name, ap in results_per_category})
        return results

    def _tasks_from_config(self, cfg):
        """
        Returns:
            tuple[str]: tasks that can be evaluated under the given configuration.
        """
        tasks = ("digit_bbox",) # this is for sure
        if cfg.DATASETS.DIGIT_ONLY:
            return tasks
        if cfg.MODEL.MASK_ON:
            tasks = tasks + ("segm",)
        if cfg.MODEL.KEYPOINT_ON:
            tasks = tasks + ("person_bbox", "keypoints")
        return tasks

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            # TODO this is ugly
            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"], self.digit_only)
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            self._predictions.append(prediction)

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[JerseyNumberEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "proposals" in predictions[0]:
            self._eval_box_proposals(predictions)
        if "instances" in predictions[0]:
            self._eval_predictions(set(self._tasks), predictions)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)


def convert_to_coco_json(dataset_name, output_file, allow_cached=True):
    """
    Converts dataset into COCO format and saves it to a json file.
    dataset_name must be registered in DatasetCatalog and in detectron2's standard format.

    Args:
        dataset_name:
            reference from the config file to the catalogs
            must be registered in DatasetCatalog and in detectron2's standard format
        output_file: path of json file that will be saved to
        allow_cached: if json file is already present then skip conversion
    """

    # TODO: The dataset or the conversion script *may* change,
    # a checksum would be useful for validating the cached data

    PathManager.mkdirs(os.path.dirname(output_file))
    with file_lock(output_file):
        if os.path.exists(output_file) and allow_cached:
            logger.info(f"Cached annotations in COCO format already exist: {output_file}")
        else:
            logger.info(f"Converting dataset annotations in '{dataset_name}' to COCO format ...)")
            coco_dict = convert_to_coco_dict(dataset_name)

            with PathManager.open(output_file, "w") as json_file:
                logger.info(f"Caching annotations in COCO format: {output_file}")
                json.dump(coco_dict, json_file)

def convert_to_coco_dict(dataset_name):
    """
    Convert a dataset in detectron2's standard format into COCO json format

    Generic dataset description can be found here:
    https://detectron2.readthedocs.io/tutorials/datasets.html#register-a-dataset

    COCO data format description can be found here:
    http://cocodataset.org/#format-data

    Args:
        dataset_name:
            name of the source dataset
            must be registered in DatastCatalog and in detectron2's standard format
    Returns:
        coco_dict: serializable dict in COCO json format
    """

    dataset_dicts = DatasetCatalog.get(dataset_name)
    categories = [
        {"id": id, "name": name}
        for id, name in enumerate(MetadataCatalog.get(dataset_name).thing_classes)
    ]
    # check if it is digit only
    digit_only = 'person' not in [cat['name'] for cat in categories]

    logger.info("Converting dataset dicts into COCO format")
    coco_images = []
    coco_annotations = []

    for image_id, image_dict in enumerate(dataset_dicts):
        coco_image = {
            "id": image_dict.get("image_id", image_id),
            "width": image_dict["width"],
            "height": image_dict["height"],
            "file_name": image_dict["file_name"],
        }
        coco_images.append(coco_image)

        anns_per_image = image_dict["annotations"]
        for annotation in anns_per_image:
            bbox_mode = annotation["bbox_mode"]
            if not digit_only:
                # add person annotation
                coco_annotation = {}
                bbox = annotation["person_bbox"]
                bbox = BoxMode.convert(bbox, bbox_mode, BoxMode.XYWH_ABS)
                if "keypoints" in annotation:
                    keypoints = annotation["keypoints"]  # list[int]
                    for idx, v in enumerate(keypoints):
                        if idx % 3 != 2:
                            # COCO's segmentation coordinates are floating points in [0, H or W],
                            # but keypoint coordinates are integers in [0, H-1 or W-1]
                            # For COCO format consistency we substract 0.5
                            # https://github.com/facebookresearch/detectron2/pull/175#issuecomment-551202163
                            keypoints[idx] = v - 0.5
                    if "num_keypoints" in annotation:
                        num_keypoints = annotation["num_keypoints"]
                    else:
                        num_keypoints = sum(kp > 0 for kp in keypoints[2::3])
                # COCO requirement: instance area
                if "segmentation" in annotation:
                    # Computing areas for instances by counting the pixels
                    segmentation = annotation["segmentation"]
                    # TODO: check segmentation type: RLE, BinaryMask or Polygon
                    polygons = PolygonMasks([segmentation])
                    area = polygons.area()[0].item()
                else:
                    # Computing areas using bounding boxes
                    bbox_xy = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
                    area = Boxes([bbox_xy]).area()[0].item()
                # Add optional fields
                if "keypoints" in annotation:
                    coco_annotation["keypoints"] = keypoints
                    coco_annotation["num_keypoints"] = num_keypoints

                if "segmentation" in annotation:
                    coco_annotation["segmentation"] = annotation["segmentation"]

                coco_annotation["id"] = len(coco_annotations) + 1
                coco_annotation["image_id"] = coco_image["id"]
                coco_annotation["bbox"] = [round(float(x), 3) for x in bbox]
                coco_annotation["area"] = area
                coco_annotation["category_id"] = annotation["category_id"]
                coco_annotation["iscrowd"] = annotation.get("iscrowd", 0)

                coco_annotations.append(coco_annotation)

            for bbox_idx, bbox in enumerate(annotation["digit_bboxes"]):
                # create a new dict with only COCO fields
                # for each annotation
                coco_annotation = {}
                # bbox = np.array(annotation["digit_bboxes"])
                # COCO requirement: XYWH box format
                bbox = BoxMode.convert(bbox, bbox_mode, BoxMode.XYWH_ABS)

                # COCO requirement: instance area
                if "segmentation" in annotation:
                    # Computing areas for instances by counting the pixels
                    segmentation = annotation["segmentation"]
                    # TODO: check segmentation type: RLE, BinaryMask or Polygon
                    polygons = PolygonMasks([segmentation])
                    area = polygons.area()[0].item()
                else:
                    # Computing areas using bounding boxes
                    bbox_xy = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
                    area = Boxes([bbox_xy]).area()[0].item()



                # COCO requirement:
                #   linking annotations to images
                #   "id" field must start with 1
                coco_annotation["id"] = len(coco_annotations) + 1
                coco_annotation["image_id"] = coco_image["id"]
                coco_annotation["bbox"] = [round(float(x), 3) for x in bbox]
                coco_annotation["area"] = area
                coco_annotation["category_id"] = annotation["digit_ids"][bbox_idx]
                coco_annotation["iscrowd"] = annotation.get("iscrowd", 0)



                coco_annotations.append(coco_annotation)

    logger.info(
        "Conversion finished, "
        f"num images: {len(coco_images)}, num annotations: {len(coco_annotations)}"
    )

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated COCO json file for Detectron2.",
    }
    coco_dict = {
        "info": info,
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
        "licenses": None,
    }
    return coco_dict

def instances_to_coco_json(instances, img_id, digit_only=True):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()
    if not digit_only:
        # convert digit related fields
        digit_boxes = [digit_boxes.tensor.numpy() for digit_boxes in instances.pred_digit_boxes if len(digit_boxes) > 0]
        if len(digit_boxes) > 0:
            digit_boxes = np.concatenate([digit_boxes.tensor.numpy() for digit_boxes in instances.pred_digit_boxes if len(digit_boxes) > 0])
            digit_boxes = BoxMode.convert(digit_boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
            digit_boxes = digit_boxes.tolist()
            digit_scores = [score for digit_scores in instances.digit_scores if len(digit_scores) > 0 \
                            for score in digit_scores.tolist()]
            digit_classes = [cls for digit_classes in instances.pred_digit_classes if len(digit_classes) > 0 \
                             for cls in digit_classes.tolist()]
        else:
            digit_scores, digit_classes = [], []
        # todo: append the results for now
        num_digit_instance = len(digit_boxes)
        boxes = boxes + digit_boxes
        scores = scores + digit_scores
        classes = classes + digit_classes
        assert len(boxes) == len(scores) == len(classes)
        # todo: we have override the variable 'num_instance" here!
        num_person_instance = num_instance
        num_instance = num_instance + num_digit_instance

    has_mask = instances.has("pred_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints and k < num_person_instance:
            # In COCO annotations,
            # keypoints coordinates are pixel indices.
            # However our predictions are floating point coordinates.
            # Therefore we subtract 0.5 to be consistent with the annotation format.
            # This is the inverse of data loading logic in `datasets/coco.py`.
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        results.append(result)
    return results

class OutputLogger:
    def __init__(self, name="root", level="INFO"):
        self.logger = logging.getLogger(name)
        self.name = self.logger.name
        self.level = getattr(logging, level)

    def write(self, msg):
        if msg and not msg.isspace():
            self.logger.log(self.level, msg)

    def flush(self): pass

def _evaluate_predictions_on_coco(coco_gt, coco_results, iou_type, kpt_oks_sigmas=None):
    """
    Evaluate the coco results using COCOEval API.
    """
    assert len(coco_results) > 0

    # faster rcnn: digit_bbox
    # pg rcnn iou_type options: [digit_bbox, person_bbox, keypoints]
    task_type = iou_type
    iou_type = 'bbox' if 'bbox' in iou_type else iou_type

    if iou_type == "segm":
        coco_results = copy.deepcopy(coco_results)
        # When evaluating mask AP, if the results contain bbox, cocoapi will
        # use the box area as the area of the instance, instead of the mask area.
        # This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for c in coco_results:
            c.pop("bbox", None)

    # filter results based on task_type
    coco_gt, coco_results = _filter_coco_results(coco_gt, coco_results, task_type)

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    # Use the COCO default keypoint OKS sigmas unless overrides are specified
    if kpt_oks_sigmas:
        coco_eval.params.kpt_oks_sigmas = np.array(kpt_oks_sigmas)

    if iou_type == "keypoints":
        num_keypoints = len(coco_results[0]["keypoints"]) // 3
        assert len(coco_eval.params.kpt_oks_sigmas) == num_keypoints, (
            "[COCOEvaluator] The length of cfg.TEST.KEYPOINT_OKS_SIGMAS (default: 17) "
            "must be equal to the number of keypoints. However the prediction has {} "
            "keypoints! For more information please refer to "
            "http://cocodataset.org/#keypoints-eval.".format(num_keypoints)
        )
    # direct print output to the logger
    with contextlib.redirect_stdout(OutputLogger(__name__, "INFO")):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    return coco_eval

def _filter_coco_results(coco_gt, coco_results, task_type):
    # determine pg rcnn or other models
    digit_gt_only = True
    # determine the eval type
    person_only = (task_type in ["person_bbox", "keypoints"])

    # hack COCO object
    coco_gt_copy = copy.deepcopy(coco_gt)
    # modify 'cats'
    for k, v in coco_gt.cats.items():
        if v['name'] == "person":
            person_id = k
            digit_gt_only = False
        if v['name'] == "person" and (not person_only):
            del coco_gt_copy.cats[k] # remove person cls
        if v['name'] != "person" and person_only:
            del coco_gt_copy.cats[k]
    # if only find digit classes, do not modify
    if digit_gt_only:
        return coco_gt, coco_results

    # modify 'anns'
    for k, v in coco_gt.anns.items():
        if v['category_id'] == person_id and (not person_only):
            del coco_gt_copy.anns[k]
        if v['category_id'] != person_id and (person_only):
            del coco_gt_copy.anns[k]
    # 'catToImgs
    if person_only:
        for k in coco_gt.catToImgs.keys():
            if k != 'default_factory':
                if k == person_id and (not person_only):
                    del coco_gt_copy.catToImgs[k]
                if k != person_id and (person_only):
                    del coco_gt_copy.catToImgs[k]
    # 'imgToAnns'
    for k, v in coco_gt.imgToAnns.items():
        i_to_keep = []
        for i, ann in enumerate(v):
            if ann['category_id'] != person_id and (not person_only):
                i_to_keep.append(i)
            if ann['category_id'] == person_id and (person_only):
                i_to_keep.append(i)
        coco_gt_copy.imgToAnns[k] = [coco_gt_copy.imgToAnns[k][i] for i in i_to_keep]

    if person_only:
        return coco_gt_copy, [coco_result for coco_result in coco_results if coco_result["category_id"] == person_id]
    elif task_type == "digit_bbox":
        return coco_gt_copy, [coco_result for coco_result in coco_results if coco_result["category_id"] != person_id]
    else:
        raise Exception("Unrecognized task_type.")