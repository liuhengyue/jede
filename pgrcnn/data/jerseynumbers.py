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
logger = logging.getLogger(__name__)
# fmt: off
CHAR_NAMES = [
     '0', '1', '2', '3',
    '4', '5', '6', '7', '8', '9'
]

CLASS_NAMES = ['person'] + CHAR_NAMES + [str(i) + str(j) for i in range(10) for j in range(10)]
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

    Args:
        data_dir: datasets/jnw/total
        anno_dir: datasets/jnw/annotations/jnw_annotations.json
        split:    list of video ids. Eg. [0, 1, 2, 3]

    Returns:
        annotations: List[Dict]
        for each dict, it has fields of:
            image_id: int
            file_name: str
            width: int
            height: int
            video_id: int
            annotations: list of dicts:
                each annotation is an instance of a person, with fields of:
                    person_bbox: list[float] of 4 coordinates
                    category_id: 0 # person class id is 0
                    keypoints: list[int], shape of (num_keypoints x 3)
                    digit_bboxes: list[list[float]]
                    digit_ids: list[int]
                    bbox_mode: BoxMode.XYXY_ABS

            Note these fields could be an empty list:
                "digit_bboxes": [],
                "digit_ids": [],
                "keypoints": [],

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
            # add jersey number box and ids
            digit_bboxes = annotations[i]['annotations'][j]['digit_bboxes']
            # list of one list
            annotations[i]['annotations'][j]['number_bbox'] = get_union_box(digit_bboxes)
            annotations[i]['annotations'][j]['number_sequence'] = annotations[i]['annotations'][j]['digit_ids']
            # for eval only
            number_id = ''.join([str(digit) for digit in annotations[i]['annotations'][j]['digit_labels']])
            number_id = CLASS_NAMES.index(number_id) if number_id else -1
            annotations[i]['annotations'][j]['number_id'] = number_id
    if num_images > 0:
        # fixed order
        annotations = annotations[:num_images]

    get_statistics(annotations)

    return annotations

def get_union_box(digit_bboxes):
    """

    Args:
        digit_bboxes: List[List[4 int]]

    Returns:
        number_bbox: List[4 int]
    """
    if len(digit_bboxes) == 0:
        return digit_bboxes
    if len(digit_bboxes) == 1:
        return digit_bboxes[0]
    digit_bboxes_np = np.asarray(digit_bboxes) # (N, 4)
    x1 = digit_bboxes_np[:, 0].min().item()
    y1 = digit_bboxes_np[:, 1].min().item()
    x2 = digit_bboxes_np[:, 2].max().item()
    y2 = digit_bboxes_np[:, 3].max().item()
    return [x1, y1, x2, y2]


def get_statistics(annotations):
    img_ws, img_hs = [], []
    area_digit_to_person_ratios = []
    h_digit_to_person_ratios = []
    w_digit_to_person_ratios = []
    p_ws, p_hs, d_ws, d_hs = [], [], [], []
    num_annotated_kpts = 0
    for anno_dict in annotations:
        img_ws.append(anno_dict["width"])
        img_hs.append(anno_dict["height"])
        annos = anno_dict["annotations"]
        for anno in annos:
            num_annotated_kpts += len(anno["keypoints"]) > 0
            p_x1, p_y1, p_x2, p_y2 = anno["person_bbox"]
            p_w = p_x2 - p_x1
            p_h = p_y2 - p_y1
            p_ws.append(p_w)
            p_hs.append(p_h)
            for digit_bbox in anno["digit_bboxes"]:
                d_x1, d_y1, d_x2, d_y2 = digit_bbox
                d_w = d_x2 - d_x1
                d_h = d_y2 - d_y1
                d_ws.append(d_w)
                d_hs.append(d_h)
                w_digit_to_person_ratios.append(d_w / p_w)
                h_digit_to_person_ratios.append(d_h / p_h)
                area_digit_to_person_ratios.append((d_w * d_h) / (p_w * p_h))
    h_mean_ratio = np.mean(h_digit_to_person_ratios) if len(h_digit_to_person_ratios) else 0.
    w_mean_ratio = np.mean(w_digit_to_person_ratios) if len(w_digit_to_person_ratios) else 0.
    h_std_ratio = np.std(h_digit_to_person_ratios) if len(h_digit_to_person_ratios) else 0.
    w_std_ratio = np.std(w_digit_to_person_ratios) if len(w_digit_to_person_ratios) else 0.
    p_w_mean = np.mean(p_ws) if len(p_ws) else 0.
    p_w_std = np.std(p_ws) if len(p_ws) else 0.
    p_h_mean = np.mean(p_hs) if len(p_hs) else 0.
    p_h_std = np.std(p_hs) if len(p_hs) else 0.
    d_w_mean = np.mean(d_ws) if len(d_ws) else 0.
    d_w_std = np.std(d_ws) if len(d_ws) else 0.
    d_h_mean = np.mean(d_hs) if len(d_hs) else 0.
    d_h_std = np.std(d_hs) if len(d_hs) else 0.
    area_mean = np.mean(area_digit_to_person_ratios) if len(area_digit_to_person_ratios) else 0.
    area_std = np.std(area_digit_to_person_ratios) if len(area_digit_to_person_ratios) else 0.
    img_w_mean = np.mean(img_ws) if len(img_ws) else 0.
    img_w_std = np.std(img_ws) if len(img_ws) else 0.
    img_h_mean = np.mean(img_hs) if len(img_hs) else 0.
    img_h_std = np.std(img_hs) if len(img_hs) else 0.
    # logger.info("Person box height: {}".format(str("{:.2f}".format(p_h_mean)) + "+-" + str("{:.2f}".format(p_h_std))))
    # logger.info("Person box width: {}".format(str("{:.2f}".format(p_w_mean)) + "+-" + str("{:.2f}".format(p_w_std))))
    # logger.info("Digit box height: {}".format(str("{:.2f}".format(d_h_mean)) + "+-" + str("{:.2f}".format(d_h_std))))
    # logger.info("Digit box width: {}".format(str("{:.2f}".format(d_w_mean)) + "+-" + str("{:.2f}".format(d_w_std))))
    # logger.info("Height digit / person bbox ratio: {}".format(str("{:.2f}".format(h_mean_ratio)) + "+-" + str("{:.2f}".format(h_std_ratio))))
    # logger.info("Width digit / person bbox ratio: {}".format(str("{:.2f}".format(w_mean_ratio)) + "+-" + str("{:.2f}".format(w_std_ratio))))
    # logger.info("Area digit / person bbox ratio: {}".format(str("{:.2f}".format(area_mean)) + "+-" + str("{:.2f}".format(area_std))))
    # logger.info("Image width: {}".format(
    #     str("{:.2f}".format(img_w_mean)) + "+-" + str("{:.2f}".format(img_w_std))))
    # logger.info("Image height: {}".format(
    #     str("{:.2f}".format(img_h_mean)) + "+-" + str("{:.2f}".format(img_h_std))))
    # logger.info("Number of annotated keypoints: {}".format(num_annotated_kpts))

    data = [len(annotations), num_annotated_kpts, img_w_mean, img_w_std, img_h_mean, img_h_std, p_w_mean, p_w_std,
            p_h_mean, p_h_std, d_w_mean, d_w_std, d_h_mean, d_h_std, area_mean, area_std,
            w_mean_ratio, w_std_ratio, h_mean_ratio, h_std_ratio]
    data = ",".join(["%.2f" % d for d in data])
    headers = "num_imgs, num_annotated_kpts, img_w_mean, img_w_std, img_h_mean, img_h_std, p_w_mean, p_w_std," \
              "p_h_mean, p_h_std, d_w_mean, d_w_std, d_h_mean, d_h_std, area_mean, area_std, " \
              "w_mean_ratio, w_std_ratio, h_mean_ratio, h_std_ratio"
    logger.info(headers)
    logger.info(data)



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
        metadataCat.set(char_names=CHAR_NAMES)
        metadataCat.set(keypoint_names=KEYPOINT_NAMES)
        metadataCat.set(keypoint_connection_rules=KEYPOINT_CONNECTION_RULES)
        metadataCat.set(keypoint_flip_map=COCO_PERSON_KEYPOINT_FLIP_MAP) # no flip map will be used though

# register_jerseynumbers()
