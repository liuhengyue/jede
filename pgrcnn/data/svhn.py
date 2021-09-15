import os
import json
import logging
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

from .jerseynumbers import CLASS_NAMES, KEYPOINT_NAMES, COCO_PERSON_KEYPOINT_FLIP_MAP, KEYPOINT_CONNECTION_RULES, get_union_box
# DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/svhn'))
DATASET_ROOT = 'datasets/svhn'

bbox_prop = ['height', 'left', 'top', 'width', 'label']

def get_img_name(f, idx=0):
    img_name = ''.join(map(chr, f[f['digitStruct/name'][idx][0]][()].flatten()))
    return(img_name)


def get_img_boxes(f, idx=0):
    """
    get the 'height', 'left', 'top', 'width', 'label' of bounding boxes of an image
    :param f: h5py.File
    :param idx: index of the image
    :return: dictionary
    """
    def parse_box(box, idx):
        x1 = int(box['left'][idx][0])
        y1 = int(box['top'][idx][0])
        h = int(box['height'][idx][0])
        w = int(box['width'][idx][0])
        x2 = x1 + w
        y2 = y1 + h
        # the svhn dataset assigns the class label "10" to the digit 0
        # then we have person class used for jerseynumber dataset, keep it consistent
        label = int(box['label'][idx][0]) % 10 + 1
        return [x1, y1, x2, y2], label

    def parse_box_multiple(f, box, idx):
        x1 = int(f[box['left'][idx][0]][()])
        y1 = int(f[box['top'][idx][0]][()])
        h = int(f[box['height'][idx][0]][()])
        w = int(f[box['width'][idx][0]][()])
        x2 = x1 + w
        y2 = y1 + h
        # the svhn dataset assigns the class label "10" to the digit 0
        # then we have person class used for jerseynumber dataset, keep it consistent
        label = int(f[box['label'][idx][0]][()]) % 10 + 1
        return [x1, y1, x2, y2], label

    digit_bboxes = []
    digit_ids = []
    bbox_mode = BoxMode.XYXY_ABS
    annotations = []
    box = f[f['digitStruct/bbox'][idx][0]]
    num_instances = box['label'].shape[0]
    if num_instances == 1:
        box_xyxy, label = parse_box(box, 0)
        digit_bboxes.append(box_xyxy)
        digit_ids.append(label)
    else:
        for i in range(num_instances):
            box_xyxy, label = parse_box_multiple(f, box, i)
            digit_bboxes.append(box_xyxy)
            digit_ids.append(label)
    for box, digit_id in zip(digit_bboxes, digit_ids):
        annotations.append({
            "digit_bboxes": [box],
            "digit_ids": [digit_id],
            "bbox_mode": bbox_mode,
            "number_bbox": [box],
            "number_sequence": [digit_id],
            "number_id": digit_id
        })
    return annotations

def create_sequences(annos):
    """

    Args:
        anno: list of one dict.

    Returns:

    """
    # copy the annos of single digit
    results = [anno for anno in annos]
    num_boxes = len(annos)
    for i in range(1, num_boxes):
        digit_bboxes = annos[i-1]['digit_bboxes'] + annos[i]['digit_bboxes']
        digit_ids = annos[i-1]['digit_ids'] + annos[i]['digit_ids']
        union_box = get_union_box(digit_bboxes)
        number_id = ''.join([CLASS_NAMES[digit_id] for digit_id in digit_ids])
        number_id = CLASS_NAMES.index(number_id) if number_id in CLASS_NAMES else -1
        results.append({
            "digit_bboxes": digit_bboxes,
            "digit_ids": digit_ids,
            "bbox_mode": annos[i]['bbox_mode'],
            "number_bbox": union_box,
            "number_sequence": digit_ids,
            "number_id": number_id
        })
    return results

def save_dicts(dataset_dir: str, save_path: str, num_images=-1):
    import h5py
    from PIL import Image
    loaded_mat = h5py.File(os.path.join(dataset_dir, "digitStruct.mat"))
    annotations = []
    N = num_images if num_images > 0 else loaded_mat['digitStruct/name'].size
    for i in range(N):
        img_name = get_img_name(loaded_mat, i)
        img_name = os.path.join(dataset_dir, img_name)
        # width, height = 0, 0
        width, height = Image.open(img_name).size
        anno = get_img_boxes(loaded_mat, i)
        anno = create_sequences(anno)
        annotations.append({
            "file_name": img_name,
            "height": height,
            "width": width,
            "image_id": i,
            "annotations": anno
        })
    with open(save_path, 'w') as outfile:
        json.dump(annotations, outfile)


def get_dicts(root: str,
              split: str = "train",
              num_images=-1,
              reset_cache=False):
    """
    Get annotations for SVHN dataset.
    Some codes are from https://pytorch.org/vision/stable/_modules/torchvision/datasets/svhn.html.
    """
    dataset_dir = os.path.join(root, split)
    annotation_dir = os.path.join(root, "annotations")
    annotation_path = os.path.join(annotation_dir, split + ".json")
    if not os.path.exists(annotation_dir):
        os.makedirs(annotation_dir)
    if not os.path.exists(annotation_path) or reset_cache:
        save_dicts(dataset_dir, annotation_path, num_images=num_images)
    with open(annotation_path) as annotation_json:
        annotations = json.load(annotation_json)

        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        # np.place(labels, labels == 10, 0)
        # data = np.transpose(data, (3, 2, 0, 1))
        if num_images > 0:
            # fixed order
            annotations = annotations[:num_images]

        return annotations

if os.path.exists(DATASET_ROOT):
    DatasetCatalog.register("svhn_train", lambda: get_dicts(DATASET_ROOT, "train", num_images=-1, reset_cache=False))
    metadataCat = MetadataCatalog.get("svhn_train")
    metadataCat.set(dataset_root=DATASET_ROOT)
    metadataCat.set(thing_classes=CLASS_NAMES)
    metadataCat.set(keypoint_names=KEYPOINT_NAMES)
    metadataCat.set(keypoint_connection_rules=KEYPOINT_CONNECTION_RULES)
    metadataCat.set(keypoint_flip_map=COCO_PERSON_KEYPOINT_FLIP_MAP)