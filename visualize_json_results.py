#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import json
import numpy as np
import os
from collections import defaultdict
import cv2
import tqdm
from fvcore.common.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode
from pgrcnn.utils.custom_visualizer import JerseyNumberVisualizer
from pgrcnn.utils.launch_utils import setup
from pgrcnn.structures.instances import CustomizedInstances as Instances


def create_instances(predictions, image_size):
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])
    labels = [dataset_id_map(p["category_id"]) for p in predictions]
    thresholds = np.asarray([args.p_conf_threshold if label == 0 else args.d_conf_threshold for label in labels])
    chosen = (score > thresholds).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen])
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS) if bbox.size > 0 else bbox

    labels = np.asarray([labels[i] for i in chosen])
    keypoints = np.asarray([x["keypoints"] for x in predictions if "keypoints" in x]).reshape((-1, 4, 3))
    # todo - customize for jersey number data
    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels

    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or jerseynumbers_val dataset."
    )
    # parser.add_argument("--input", required=True, help="JSON file produced by the model")
    # parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--config-file", help="config file path", default="configs/pg_rcnn/pg_rcnn_base.yaml")
    parser.add_argument("--dataset", help="name of the dataset", default="jerseynumbers_val")
    parser.add_argument("--p-conf-threshold", default=0.5, type=float, help="person confidence threshold")
    parser.add_argument("--d-conf-threshold", default=0.5, type=float, help="digit confidence threshold")
    args = parser.parse_args()
    # lazy add config file
    # args.config_file = "../../configs/pg_rcnn_R_50_FPN_1x_test_2.yaml"
    args.config_file = "projects/PGRcnn/configs/pg_rcnn/pg_rcnn_base.yaml"
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

        predictions = create_instances(pred_by_image[dic["image_id"]], img.shape[:2])
        vis = JerseyNumberVisualizer(img, metadata)
        vis_pred = vis.draw_instance_predictions(predictions).get_image()

        vis = JerseyNumberVisualizer(img, metadata)
        vis_gt = vis.draw_dataset_dict(dic).get_image()

        concat = np.concatenate((vis_pred, vis_gt), axis=1)
        cv2.imwrite(os.path.join(args.output, basename), concat[:, :, ::-1])
