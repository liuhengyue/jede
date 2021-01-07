import colorsys
import logging
import os
import math
import numpy as np
from enum import Enum, unique
import cv2
from PIL import Image
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import pycocotools.mask as mask_util
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg

from detectron2.structures import BitMasks, Boxes, BoxMode, Keypoints, PolygonMasks, RotatedBoxes

from detectron2.utils.colormap import random_color, colormap

from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data import DatasetCatalog, MetadataCatalog

from pgrcnn.data.custom_mapper import DatasetMapper
from pgrcnn.data.build import build_sequential_dataloader
logger = logging.getLogger(__name__)

_SMALL_OBJECT_AREA_THRESH = 1000
_LARGE_MASK_AREA_THRESH = 120000
_OFF_WHITE = (1.0, 1.0, 240.0 / 255)
_BLACK = (0, 0, 0)
_RED = (1.0, 0, 0)

_KEYPOINT_THRESHOLD = 0.05


class JerseyNumberVisualizer(Visualizer):
    def __init__(self, img_rgb, metadata, scale=1.0, instance_mode=ColorMode.IMAGE):
        super().__init__(img_rgb, metadata, scale=scale, instance_mode=instance_mode)


    def instances_to_dict(self, data):
        """
        data: acutual data returned by dataloader
        data['instances'] -> detectron2.structure.Instances

        """
        instances_list = []
        for i in range(len(data['instances'])):
            instance = data['instances'][i]
            instances_list.append({
            "person_bbox": instance.gt_boxes.tensor.numpy(),
            "keypoints": instance.gt_keypoints.tensor.numpy() if hasattr(instance, 'gt_keypoints') else None,
            "digit_bboxes": instance.gt_digit_boxes.tensor.squeeze_().numpy() if hasattr(instance, 'gt_digit_boxes') else None,
            "digit_ids": instance.gt_digit_classes.squeeze_().numpy() if hasattr(instance, 'gt_digit_classes') else None,
            "category_id": instance.gt_classes.squeeze_().numpy()
            })
        return {"annotations": instances_list}

    def draw_dataloader_instances(self, instances):
        instances_list = self.instances_to_dict(instances)
        return self.draw_dataset_dict(instances_list)

    def draw_single_instance(self, instance):
        """
        Given a instance dict, draw the corresponding
        person bbox, keypoints(if available), and digit bounding boxes (if available)

        instance has four fields: digit_bboxes, digit_labels, person_bbox, keypoints
        """
        if instance is not None:
            person_bbox = np.array(instance.get("person_bbox", np.empty((0, 4))))
            keypoints   = np.array(instance.get("keypoints", np.empty((0, 4, 3))))
            digit_bboxes = np.array(instance.get("digit_bboxes", np.empty((0, 4)))).reshape((-1, 4))
            digit_ids = np.array(instance.get("digit_ids", np.empty((0,))))
            category_id = instance.get("category_id", None)
            bbox_mode = instance.get("bbox_mode", None)


            # filter empty boxes
            if isinstance(digit_bboxes, list) and len(digit_bboxes) == 0:
                digit_bboxes = np.empty((0, 4))
            digit_bboxes = digit_bboxes[np.where(~np.all(digit_bboxes == 0, axis=1))]
            digit_ids = digit_ids[np.where(digit_ids > -1)]

            keypts = keypoints.reshape(1, -1, 3) if keypoints is not None else None


            # here, we have two different bbox (person and digit)
            if bbox_mode is not None:
                person_bbox = [BoxMode.convert(person_bbox, bbox_mode, BoxMode.XYXY_ABS)]
                digit_bboxes = [BoxMode.convert(each_digit_bbox, bbox_mode, BoxMode.XYXY_ABS) for
                           each_digit_bbox in digit_bboxes]
            # person labels, digit labels
            labels = [category_id] if category_id is not None else None
            if self.metadata:
                names = self.metadata.get("thing_classes", None)
                # not too necessary
                if names:
                    labels = [names[i] for i in labels] if labels is not None else None
                    digit_labels = [names[i] for i in digit_ids] if digit_ids is not None else None
            if labels is not None:
                self.overlay_instances(labels=labels, boxes=person_bbox, masks=None, keypoints=keypts)
            if digit_labels is not None:
                self.overlay_instances(labels=digit_labels, boxes=digit_bboxes, masks=None, keypoints=None)
            return self.output





    def draw_dataset_dict(self, dic):
        """
        Draw annotations/segmentaions in Detectron2 Dataset format. With my customization for jersey number dataset.

        Args:
            dic (dict): annotation/segmentation data of one image, in Detectron2 Dataset format.

        Returns:
            output (VisImage): image object with visualizations.
        """
        annos = dic.get("annotations", None)
        if annos:
            for anno in annos:
                self.draw_single_instance(anno)
        return self.output

    def overlay_instances(
        self,
        *,
        boxes=None,
        labels=None,
        digit_labels=None,
        digit_boxes=None,
        masks=None,
        keypoints=None,
        assigned_colors=None,
        alpha=0.5
    ):
        """
        Args:
            boxes (Boxes, RotatedBoxes or ndarray): either a :class:`Boxes`,
                or an Nx4 numpy array of XYXY_ABS format for the N objects in a single image,
                or a :class:`RotatedBoxes`,
                or an Nx5 numpy array of (x_center, y_center, width, height, angle_degrees) format
                for the N objects in a single image,
            labels (list[str]): the text to be displayed for each instance.
            digit boxes should be Nxkx4
            masks (masks-like object): Supported types are:

                * `structures.masks.PolygonMasks`, `structures.masks.BitMasks`.
                * list[list[ndarray]]: contains the segmentation masks for all objects in one image.
                    The first level of the list corresponds to individual instances. The second
                    level to all the polygon that compose the instance, and the third level
                    to the polygon coordinates. The third level should have the format of
                    [x0, y0, x1, y1, ..., xn, yn] (n >= 3).
                * list[ndarray]: each ndarray is a binary mask of shape (H, W).
                * list[dict]: each dict is a COCO-style RLE.
            keypoints (Keypoint or array like): an array-like object of shape (N, K, 3),
                where the N is the number of instances and K is the number of keypoints.
                The last dimension corresponds to (x, y, visibility or score).
            assigned_colors (list[matplotlib.colors]): a list of colors, where each color
                corresponds to each mask or box in the image. Refer to 'matplotlib.colors'
                for full list of formats that the colors are accepted in.

        Returns:
            output (VisImage): image object with visualizations.
        """
        num_instances = None
        if boxes is not None:
            boxes = self._convert_boxes(boxes)
            num_instances = len(boxes)
        if digit_boxes is not None:
            digit_boxes = self._convert_boxes(digit_boxes)
        if masks is not None:
            masks = self._convert_masks(masks)
            if num_instances:
                assert len(masks) == num_instances
            else:
                num_instances = len(masks)
        if keypoints is not None:
            if num_instances:
                assert len(keypoints) == num_instances
            else:
                num_instances = len(keypoints)
            keypoints = self._convert_keypoints(keypoints)
        if labels is not None:
            assert len(labels) == num_instances
        if assigned_colors is None:
            # assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]
            assigned_colors = [colormap(rgb=True, maximum=1)[i] for i in range(num_instances)]

        if num_instances == 0:
            return self.output
        if boxes is not None and boxes.shape[1] == 5:
            return self.overlay_rotated_instances(
                boxes=boxes, labels=labels, assigned_colors=assigned_colors
            )

        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        elif masks is not None:
            areas = np.asarray([x.area() for x in masks])

        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs] if boxes is not None else None
            labels = [labels[k] for k in sorted_idxs] if labels is not None else None
            masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]
            keypoints = keypoints[sorted_idxs] if keypoints is not None else None
            digit_boxes = digit_boxes[sorted_idxs] if digit_boxes is not None else None
            digit_labels = [digit_labels[k] for k in sorted_idxs] if digit_labels is not None else None

        for i in range(num_instances):
            color = assigned_colors[i]
            if boxes is not None:
                self.draw_box(boxes[i], edge_color=color)

            if digit_boxes is not None and digit_boxes[i] is not None:
                for digit_box in digit_boxes[i]:
                    self.draw_box(digit_box, edge_color=color)

            if masks is not None:
                for segment in masks[i].polygons:
                    self.draw_polygon(segment.reshape(-1, 2), color, alpha=alpha)

            if labels is not None:
                # first get a box
                if boxes is not None:
                    x0, y0, x1, y1 = boxes[i]
                    text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
                    horiz_align = "left"
                elif masks is not None:
                    x0, y0, x1, y1 = masks[i].bbox()

                    # draw text in the center (defined by median) when box is not drawn
                    # median is less sensitive to outliers.
                    text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
                    horiz_align = "center"
                else:
                    continue  # drawing the box confidence for keypoints isn't very useful.
                # for small objects, draw text at the side to avoid occlusion
                instance_area = (y1 - y0) * (x1 - x0)
                if (
                    instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale
                    or y1 - y0 < 40 * self.output.scale
                ):
                    if y1 >= self.output.height - 5:
                        text_pos = (x1, y0)
                    else:
                        text_pos = (x0, y1)

                height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
                lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
                font_size = (
                    np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                    * 0.5
                    * self._default_font_size
                )
                self.draw_text(
                    labels[i],
                    text_pos,
                    color=lighter_color,
                    horizontal_alignment=horiz_align,
                    font_size=font_size,
                )
            # draw digit labels
            if digit_labels is not None:
                # first get a box
                if digit_boxes is not None:
                    # loop over each digit box
                    for j, digit_box in enumerate(digit_boxes[i]):
                        x0, y0, x1, y1 = digit_box
                        text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
                        horiz_align = "left"

                        # for small objects, draw text at the side to avoid occlusion
                        instance_area = (y1 - y0) * (x1 - x0)
                        if (
                                instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale
                                or y1 - y0 < 40 * self.output.scale
                        ):
                            if y1 >= self.output.height - 5:
                                text_pos = (x1, y0)
                            else:
                                text_pos = (x0, y1)

                        height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
                        lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
                        font_size = (
                                np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                                * 0.5
                                * self._default_font_size
                        )
                        self.draw_text(
                            digit_labels[i][j],
                            text_pos,
                            color=lighter_color,
                            horizontal_alignment=horiz_align,
                            font_size=font_size,
                        )
                else:
                    continue  # drawing the box confidence for keypoints isn't very useful.


        # draw keypoints
        if keypoints is not None:
            for i, keypoints_per_instance in enumerate(keypoints):
                color = assigned_colors[i]
                self.draw_and_connect_keypoints(keypoints_per_instance, color=color)

        return self.output

    def draw_and_connect_keypoints(self, keypoints, color=_RED, draw_mid=False):
        """
        Draws keypoints of an instance and follows the rules for keypoint connections
        to draw lines between appropriate keypoints. This follows color heuristics for
        line color.

        Args:
            keypoints (Tensor): a tensor of shape (K, 3), where K is the number of keypoints
                and the last dimension corresponds to (x, y, probability).

        Returns:
            output (VisImage): image object with visualizations.
        """
        visible = {}
        keypoint_names = self.metadata.get("keypoint_names")
        for idx, keypoint in enumerate(keypoints):
            # draw keypoint
            x, y, prob = keypoint
            if prob > _KEYPOINT_THRESHOLD:
                self.draw_circle((x, y), color=color)
                if keypoint_names:
                    keypoint_name = keypoint_names[idx]
                    visible[keypoint_name] = (x, y)


        if self.metadata.get("keypoint_connection_rules"):
            for kp0, kp1, color in self.metadata.keypoint_connection_rules:
                if kp0 in visible and kp1 in visible:
                    x0, y0 = visible[kp0]
                    x1, y1 = visible[kp1]
                    color = tuple(x / 255.0 for x in color)
                    self.draw_line([x0, x1], [y0, y1], color=color)

        # draw lines from nose to mid-shoulder and mid-shoulder to mid-hip
        # Note that this strategy is specific to person keypoints.
        # For other keypoints, it should just do nothing
        if draw_mid:
            try:
                ls_x, ls_y = visible["left_shoulder"]
                rs_x, rs_y = visible["right_shoulder"]
                mid_shoulder_x, mid_shoulder_y = (ls_x + rs_x) / 2, (ls_y + rs_y) / 2
            except KeyError:
                pass
            else:
                # draw line from nose to mid-shoulder
                nose_x, nose_y = visible.get("nose", (None, None))
                if nose_x is not None:
                    self.draw_line([nose_x, mid_shoulder_x], [nose_y, mid_shoulder_y], color=_RED)

                try:
                    # draw line from mid-shoulder to mid-hip
                    lh_x, lh_y = visible["left_hip"]
                    rh_x, rh_y = visible["right_hip"]
                except KeyError:
                    pass
                else:
                    mid_hip_x, mid_hip_y = (lh_x + rh_x) / 2, (lh_y + rh_y) / 2
                    self.draw_line([mid_hip_x, mid_shoulder_x], [mid_hip_y, mid_shoulder_y], color=_RED)
        return self.output

def visualize_data(cfg, scale=1.0, only_show_multi_instances=True, set='train'):
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    # set to batch 1
    cfg.SOLVER.IMS_PER_BATCH = 1
    train_data_loader = build_sequential_dataloader(cfg, mapper=DatasetMapper(cfg, True), set=set)
    for batch in train_data_loader:
        for per_image in batch:
            if only_show_multi_instances and len(per_image["instances"]) == 1:
                continue
            # print(per_image['file_name'])
            # Pytorch tensor is in (C, H, W) format
            img = per_image["image"].permute(1, 2, 0)
            if cfg.INPUT.FORMAT == "BGR":
                img = img[:, :, [2, 1, 0]]
            else:
                img = np.asarray(Image.fromarray(img, mode=cfg.INPUT.FORMAT).convert("RGB"))

            visualizer = JerseyNumberVisualizer(img, metadata=metadata, scale=scale)
            target_fields = per_image["instances"].get_fields()
            labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
            # shape is Nx2, label has -1, mark as '' in label
            digit_labels = [[metadata.thing_classes[i] if i != -1 else '' for i in digits_pp] for digits_pp in target_fields["gt_digit_classes"]]
            boxes = target_fields.get("gt_boxes", None)
            digit_boxes = target_fields.get("gt_digit_boxes", None)
            # print(digit_boxes.tensor.size())
            # print(digit_labels)
            # print(digit_boxes)
            vis = visualizer.overlay_instances(
                labels=labels,
                boxes=boxes,
                digit_labels=digit_labels,
                digit_boxes=digit_boxes,
                masks=target_fields.get("gt_masks", None),
                keypoints=target_fields.get("gt_keypoints", None),
            )
            output(vis, "id " + str(per_image["image_id"]) + " " + per_image['file_name'])

def output(vis, fname, show=True, dirname='/../output/'):
    if show:
        print(fname)
        cv2.imshow("window", vis.get_image()[:, :, ::-1])
        cv2.waitKey()
    else:
        filepath = os.path.join(dirname, fname)
        print("Saving to {} ...".format(filepath))
        vis.save(filepath)