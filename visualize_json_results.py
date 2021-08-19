#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import torch
import os
from collections import defaultdict
import cv2
import tqdm
from fvcore.common.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode, Keypoints, Instances
from pgrcnn.utils.custom_visualizer import JerseyNumberVisualizer
from pgrcnn.utils.launch_utils import setup
from pgrcnn.structures import Players, Boxes


def create_instances(predictions, image_size, p_conf_threshold=0, d_conf_threshold=0, digit_only=False):
    """

    Args:
        predictions: the predictions for a single image, in the order of person, digit, jersey
        image_size:
        p_conf_threshold:
        d_conf_threshold:
        digit_only:

    Returns:

    """
    if digit_only:
        ret = Instances(image_size)
        if not len(predictions):
            ret.scores = torch.empty((0,), dtype=torch.float32)
            ret.pred_boxes = Boxes([])
            ret.pred_classes = torch.empty((0,), dtype=torch.long)
            return ret
        boxes = BoxMode.convert(torch.as_tensor([p["bbox"] for p in predictions]), BoxMode.XYWH_ABS,
                                BoxMode.XYXY_ABS)
        scores = torch.as_tensor([p["score"] for p in predictions])
        labels = torch.as_tensor([p["category_id"] for p in predictions], dtype=torch.long)
        ret.scores = scores
        ret.pred_boxes = Boxes(boxes)
        ret.pred_classes = labels
        ret = ret[ret.scores > p_conf_threshold]
        return ret
    # add fields
    person_inds = [i for i, p in enumerate(predictions) if "person_bbox" in p["tasks"]] + [len(predictions)]
    predictions = [predictions[person_inds[i-1]:person_inds[i]] for i in range(1, len(person_inds))]
    instances = []
    for pred in predictions:
        single_instance = create_single_instance(pred, image_size)
        instances.append(single_instance)
    instances = Players.cat(instances)
    # filter low confident person detection
    instances = instances[instances.scores > p_conf_threshold]
    return instances

def create_single_instance(predictions, image_size):
    """

    Args:
        predictions: list of coco results for a single instance
        image_size:

    Returns:

    """
    ret = Players(image_size)
    person_predictions = [p for p in predictions if "person_bbox" in p['tasks']]
    boxes = BoxMode.convert(torch.as_tensor([p["bbox"] for p in person_predictions]), BoxMode.XYWH_ABS,
                            BoxMode.XYXY_ABS)
    scores = torch.as_tensor([p["score"] for p in person_predictions])
    keypoints = torch.as_tensor([p["full_keypoints"] for p in person_predictions]).view(-1, 17, 3)
    labels = torch.as_tensor([p["category_id"] for p in person_predictions], dtype=torch.long)
    ret.scores = scores
    ret.pred_boxes = Boxes(boxes)
    ret.pred_classes = labels
    ret.pred_keypoints = Keypoints(keypoints)
    # process digit predictions
    digit_predictions = [p for p in predictions if "digit_bbox" in p['tasks']]
    boxes = np.asarray([p['bbox'] for p in digit_predictions]).reshape(-1, 4)
    boxes = BoxMode.convert(boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
    labels = np.asarray([p['category_id'] for p in digit_predictions])
    scores = np.asarray([p['score'] for p in digit_predictions])
    ret.digit_scores = [torch.as_tensor(scores)]
    ret.pred_digit_boxes = [Boxes(boxes)]
    ret.pred_digit_classes = [torch.as_tensor(labels)]
    # process jersey number predictions
    number_predictions = [p for p in predictions if "jersey_number" in p['tasks']]
    boxes = np.asarray([p['bbox'] for p in number_predictions]).reshape(-1, 4)
    boxes = BoxMode.convert(boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
    labels = np.asarray([p['category_id'] for p in number_predictions])
    scores = np.asarray([p['score'] for p in number_predictions])
    ret.pred_number_boxes = [Boxes(boxes)]
    ret.pred_number_classes = [torch.as_tensor(labels)]
    ret.pred_number_scores = [torch.as_tensor(scores)]
    return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or jerseynumbers_val dataset."
    )
    parser.add_argument("--config-file", help="config file path", default="configs/pg_rcnn/pg_rcnn_test.yaml")
    parser.add_argument("--dataset", help="name of the dataset", default="jerseynumbers_val")
    parser.add_argument("--p-conf-threshold", default=0.5, type=float, help="person confidence threshold")
    parser.add_argument("--d-conf-threshold", default=0.5, type=float, help="digit confidence threshold")
    args = parser.parse_args()
    # lazy add config file
    # args.config_file = ""
    cfg = setup(args)
    # modify args from cfg
    args.input = os.path.join(cfg.OUTPUT_DIR, "inference/coco_instances_results.json")
    args.output = os.path.join(cfg.OUTPUT_DIR, "inference/vis")
    args.dataset = cfg.DATASETS.TEST[0]

    with PathManager.open(args.input, "r") as f:
        predictions = json.load(f)

    pred_by_image = defaultdict(list)
    for p in predictions:
        pred_by_image[p["image_id"]].append(p)

    dicts = list(DatasetCatalog.get(args.dataset))
    metadata = MetadataCatalog.get(args.dataset)
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):

        def dataset_id_map(ds_id):
            return metadata.thing_dataset_id_to_contiguous_id[ds_id]

    elif "lvis" in args.dataset:
        # LVIS results are in the same format as COCO results, but have a different
        # mapping from dataset category id to contiguous category id in [0, #categories - 1]
        def dataset_id_map(ds_id):
            return ds_id - 1

    elif "jerseynumber" in args.dataset:
        def dataset_id_map(ds_id):
            return ds_id

    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))

    os.makedirs(args.output, exist_ok=True)
    for dic in tqdm.tqdm(dicts):
        img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        basename = os.path.basename(dic["file_name"])
        basename_wo_extension = os.path.splitext(basename)[0]

        preds = create_instances(pred_by_image[dic["image_id"]], img.shape[:2],
                                       p_conf_threshold=args.p_conf_threshold,
                                       d_conf_threshold=args.d_conf_threshold,
                                       digit_only=cfg.DATASETS.DIGIT_ONLY)
        # vis_pred = JerseyNumberVisualizer(img, metadata, montage=False)
        # vis_pred.draw_instance_predictions(predictions)
        # vis_pred.get_output().save(os.path.join(args.output, "{}_pred.pdf".format(basename_wo_extension)))

        # vis_gt = JerseyNumberVisualizer(img, metadata, montage=False)
        # vis_gt.draw_dataset_dict(dic)
        # vis_gt.get_output().save(os.path.join(args.output, "{}_gt.pdf".format(basename_wo_extension)))

        vis = JerseyNumberVisualizer(img, metadata, montage=True, nrows=1, ncols=2)
        vis.draw_montage(preds, dic)
        vis.get_output().save(os.path.join(args.output, "{}_montage.pdf".format(basename_wo_extension)))

