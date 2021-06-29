import io
import logging
import contextlib
import os
import datetime
import json
import numpy as np

from PIL import Image

from fvcore.common.timer import Timer
from detectron2.structures import BoxMode, PolygonMasks, Boxes
from fvcore.common.file_io import PathManager, file_lock


from detectron2.data import DatasetCatalog, MetadataCatalog

# fmt: off
CLASS_NAMES = [
    'person', '0', '1', '2', '3',
    '4', '5', '6', '7', '8', '9'
]
# follow the name in COCO, if only use 4 keypoints
# KEYPOINT_NAMES = ["left_shoulder", "right_shoulder", "right_hip", "left_hip"]
# KEYPOINT_CONNECTION_RULES = [
#     ("left_shoulder", "right_shoulder", (255, 32, 0)),
#     ("right_shoulder", "right_hip", (255, 32, 0)),
#     ("right_hip", "left_hip", (255, 32, 0)),
#     ("left_hip", "left_shoulder", (255, 32, 0)),
# ]
KEYPOINT_NAMES = (
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)

COCO_PERSON_KEYPOINT_FLIP_MAP = (
    ("left_eye", "right_eye"),
    ("left_ear", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
)

KEYPOINT_CONNECTION_RULES = [
    # face
    ("left_ear", "left_eye", (102, 204, 255)),
    ("right_ear", "right_eye", (51, 153, 255)),
    ("left_eye", "nose", (102, 0, 204)),
    ("nose", "right_eye", (51, 102, 255)),
    # upper-body
    ("left_shoulder", "right_shoulder", (255, 128, 0)),
    ("left_shoulder", "left_elbow", (153, 255, 204)),
    ("right_shoulder", "right_elbow", (128, 229, 255)),
    ("left_elbow", "left_wrist", (153, 255, 153)),
    ("right_elbow", "right_wrist", (102, 255, 224)),
    # lower-body
    ("left_hip", "right_hip", (255, 102, 0)),
    ("left_hip", "left_knee", (255, 255, 77)),
    ("right_hip", "right_knee", (153, 255, 204)),
    ("left_knee", "left_ankle", (191, 255, 128)),
    ("right_knee", "right_ankle", (255, 195, 77)),
]


# fmt: on

DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/jnw'))
assert os.path.exists(DATASET_ROOT), "Dataset jnw not found in {}".format(DATASET_ROOT)

# The annotation format:
# self.dataset:
# [{'filename': filename, 'width': width, 'height': height, 'polygons': polygons, \
#   'keypoints': output_kpts, 'persons': persons, 'digits': output_digits,
#   'digits_bboxes': output_digit_boxes, 'numbers': numbers, 'video_id': anno['file_attributes']['video_id']
#   }]
# the keypoints order is left_sholder, right_shoulder, right_hip, left_hip
# bbox annotation older is XYXY_ABS, keypoints is x, y, v

def get_dicts(data_dir, anno_dir, split=None, digit_only=False, num_images=-1):
    """
    Get annotations for specific split/video.
    Note these fields could be an empty list:
        "digit_bboxes": [],
        "digit_labels": [],
        "keypoints": [],

    data_dir: datasets/jnw/total
    anno_dir: datasets/jnw/annotations/jnw_annotations.json
    split:    list of video ids. Eg. [0, 1, 2, 3]
    """
    annotations = json.load(open(anno_dir, 'r'))
    split = [split] if isinstance(split, int) else split
    # get only annotations in specific videos
    annotations = [annotation for annotation in annotations if annotation['video_id'] in split] if split else annotations
    # add actual dataset path prefix, and extra fields
    for i in range(len(annotations)): # file level
        # construct full path for each image
        annotations[i]['file_name'] = os.path.join(data_dir, annotations[i]['file_name'])
        # rename 'instances' to 'annotations'
        annotations[i]['annotations'] = annotations[i].pop('instances')
        for j in range(len(annotations[i]['annotations'])): # instance level
            if not digit_only:
                annotations[i]['annotations'][j]['category_id'] = CLASS_NAMES.index('person')
            # broadcast the bbox mode to each instance
            annotations[i]['annotations'][j]['bbox_mode'] = BoxMode.XYXY_ABS
            annotations[i]['annotations'][j]['digit_ids'] = \
                [CLASS_NAMES.index(str(digit)) for digit in annotations[i]['annotations'][j]['digit_labels']]
    if num_images > 0:
        # fixed order
        annotations = annotations[:num_images]

    return annotations






def register_jerseynumbers(cfg):
    """
    The jersey number dataset needs config file s.t. it has multiple settings.
    """
    if cfg.DATASETS.DIGIT_ONLY:
        CLASS_NAMES.pop(0)
    train_video_ids, test_video_ids = cfg.DATASETS.TRAIN_VIDEO_IDS, cfg.DATASETS.TEST_VIDEO_IDS
    dataset_dir =  os.path.join(DATASET_ROOT, 'total/')
    annotation_dir = os.path.join(DATASET_ROOT, 'annotations/jnw_annotations.json')
    for name, d in zip(['train', 'val'], [train_video_ids, test_video_ids]):
        DatasetCatalog.register("jerseynumbers_" + name,
                                lambda d=d: get_dicts(dataset_dir, annotation_dir, d,
                                                      digit_only=cfg.DATASETS.DIGIT_ONLY,
                                                      num_images=cfg.DATASETS.NUM_IMAGES))
        metadataCat = MetadataCatalog.get("jerseynumbers_" + name)
        metadataCat.set(thing_classes=CLASS_NAMES)
        metadataCat.set(keypoint_names=KEYPOINT_NAMES)
        metadataCat.set(keypoint_connection_rules=KEYPOINT_CONNECTION_RULES)
        metadataCat.set(keypoint_flip_map=COCO_PERSON_KEYPOINT_FLIP_MAP) # no flip map will be used though

# register_jerseynumbers()
