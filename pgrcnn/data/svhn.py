import os
import numpy as np
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

DATASET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/svhn'))
# assert os.path.exists(DATASET_ROOT), "Dataset svhn not found in {}".format(DATASET_ROOT)

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
        label = int(box['label'][idx][0]) % 10
        return [x1, y1, x2, y2], label

    def parse_box_multiple(f, box, idx):
        x1 = int(f[box['left'][idx][0]][()])
        y1 = int(f[box['top'][idx][0]][()])
        h = int(f[box['height'][idx][0]][()])
        w = int(f[box['width'][idx][0]][()])
        x2 = x1 + w
        y2 = y1 + h
        # the svhn dataset assigns the class label "10" to the digit 0
        label = int(f[box['label'][idx][0]][()]) % 10
        return [x1, y1, x2, y2], label

    annotations = []
    box = f[f['digitStruct/bbox'][idx][0]]
    num_instances = box['label'].shape[0]
    if num_instances == 1:
        box_xyxy, label = parse_box(box, 0)
        annotations.append({
            "bbox": box_xyxy,
            "category_id": label,
            "bbox_mode": BoxMode.XYXY_ABS
        })
    else:
        for i in range(num_instances):
            box_xyxy, label = parse_box_multiple(f, box, i)
            annotations.append({
                "bbox": box_xyxy,
                "category_id": label,
                "bbox_mode": BoxMode.XYXY_ABS
            })
    return annotations

def get_dicts(root: str,
              split: str = "train",
              num_images=-1):
    """
    Get annotations for SVHN dataset.
    Some codes are from https://pytorch.org/vision/stable/_modules/torchvision/datasets/svhn.html.
    """
    import h5py
    from PIL import Image
    dataset_dir = os.path.join(root, split)
    loaded_mat  = h5py.File(os.path.join(dataset_dir, "digitStruct.mat"))
    annotations = []
    N = loaded_mat['digitStruct/name'].size
    for i in range(N):
        img_name = get_img_name(loaded_mat, i)
        img_name = os.path.join(dataset_dir, img_name)
        width, height = 0, 0
        # width, height = Image.open(img_name).size
        anno = get_img_boxes(loaded_mat, i)
        annotations.append({
            "file_name": img_name,
            "height": height,
            "width": width,
            "image_id": i,
            "annotations": anno
        })

    # this makes it inconsistent with several loss functions
    # which expect the class labels to be in the range [0, C-1]
    # np.place(labels, labels == 10, 0)
    # data = np.transpose(data, (3, 2, 0, 1))
    if num_images > 0:
        # fixed order
        annotations = annotations[:num_images]

    return annotations

# DatasetCatalog.register("svhn_train", get_dicts(DATASET_ROOT, "train"))