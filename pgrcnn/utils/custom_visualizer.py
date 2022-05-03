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
import matplotlib.pyplot as plt
import matplotlib.figure as mplfigure
import pycocotools.mask as mask_util
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg

from detectron2.structures import BitMasks, BoxMode, Keypoints, PolygonMasks, RotatedBoxes

from detectron2.utils.colormap import random_color, colormap

from detectron2.utils.visualizer import VisImage, Visualizer, ColorMode
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data import DatasetCatalog, MetadataCatalog

from pgrcnn.structures import Boxes
from pgrcnn.data.dataset_mapper import DatasetMapper
from pgrcnn.data.build import build_sequential_dataloader
from pgrcnn.data.det_utils import pad_full_keypoints

logger = logging.getLogger(__name__)

_SMALL_OBJECT_AREA_THRESH = 1000
_LARGE_MASK_AREA_THRESH = 120000
_OFF_WHITE = (1.0, 1.0, 240.0 / 255)
_BLACK = (0, 0, 0)
_DARK_GRAY = (0.2, 0.2, 0.2)
_MID_GRAY = (0.5, 0.5, 0.5)
_RED = (1.0, 0, 0)

_KEYPOINT_THRESHOLD = 0.05

def _create_text_labels(classes, scores, class_names, is_crowd=None):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):
        is_crowd (list[bool] or None):

    Returns:
        list[str] or None
    """
    labels = None
    if classes is not None:
        if class_names is not None and len(class_names) > 0:
            labels = [class_names[i] for i in classes]
        else:
            labels = [str(i) for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{}\n{:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    if labels is not None and is_crowd is not None:
        labels = [l + ("|crowd" if crowd else "") for l, crowd in zip(labels, is_crowd)]
    return labels

def _create_person_labels(classes, scores, jersey_numbers, jersey_scores, class_names):
    """
    Args:
        classes (list[int] or None):
        scores (list[float] or None):
        class_names (list[str] or None):
        is_crowd (list[bool] or None):

    Returns:
        list[str] or None
    """
    # labels = None
    # if classes is not None:
    #     if class_names is not None and len(class_names) > 0:
    #         labels = [class_names[i] for i in classes]
    #     else:
    #         labels = [str(i) for i in classes]
    # if scores is not None:
    #     if labels is None:
    #         labels = ["{:.0f}%".format(s * 100) for s in scores]
    #     else:
    #         labels = ["{}\n{:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    jersey_labels = []
    if jersey_numbers is not None:
        if class_names is not None and len(class_names) > 0:
            jersey_labels = [''.join([class_names[i] for i in jersey_numbers])]
        else:
            jersey_labels = [''.join([str(i) for i in jersey_numbers])]
        # labels = [l + jl for l, jl in zip(labels, jersey_labels)]
    if jersey_scores is not None:
        jersey_labels = ["{}\n{:.0f}%".format(l, s * 100) for l, s in zip(jersey_labels, jersey_scores)]
    return jersey_labels

class MontageImage:
    def __init__(self, img, nrows=1, ncols=2, scale=1.0):
        """
        Args:
            img (ndarray): an RGB image of shape (H, W, 3).
            scale (float): scale the input image
        """
        self.img = img
        self.scale = scale
        self.nrows = nrows
        self.ncols = ncols
        self.width, self.height = img.shape[1] * ncols, img.shape[0] * nrows
        self._setup_figure(img)

    def _setup_figure(self, img):
        """
        Args:
            Same as in :meth:`__init__()`.

        Returns:
            fig (matplotlib.pyplot.figure): top level container for all the image plot elements.
            ax (matplotlib.pyplot.Axes): contains figure elements and sets the coordinate system.
        """
        fig, axs = plt.subplots(self.nrows, self.ncols)
        self.dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / self.dpi,
            (self.height * self.scale + 1e-2) / self.dpi,
        )
        self.canvas = FigureCanvasAgg(fig)
        for ax in axs.flat:
            ax.axis("off")
            # Need to imshow this first so that other patches can be drawn on top
            ax.imshow(img, interpolation="nearest")
            # ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation="nearest")

        fig.tight_layout(pad=0.0, w_pad=0.5, h_pad=0.0)
        self.fig = fig
        self.axs = axs[None, ...] if (self.ncols == 1) or (self.nrows == 1) else axs
        self.ax = None # will be set during drawing

    def get_ax(self, row, col):
        return self.ax[row, col]

    def set_ax(self, row, col):
        self.ax = self.axs[row, col]

    def save(self, filepath):
        """
        Args:
            filepath (str): a string that contains the absolute path, including the file name, where
                the visualized image will be saved.
        """
        self.fig.savefig(filepath)
        plt.close(self.fig)

    def get_image(self):
        """
        Returns:
            ndarray:
                the visualized image of shape (H, W, 3) (RGB) in uint8 type.
                The shape is scaled w.r.t the input image using the given `scale` argument.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        # buf = io.BytesIO()  # works for cairo backend
        # canvas.print_rgba(buf)
        # width, height = self.width, self.height
        # s = buf.getvalue()

        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")

class JerseyNumberVisualizer(Visualizer):
    def __init__(self,
                 img_rgb,
                 metadata=None,
                 scale=2.0,
                 instance_mode=ColorMode.IMAGE,
                 digit_only=False,
                 montage=False,
                 nrows=1,
                 ncols=1):
        """
        Args:
            img_rgb: a numpy array of shape (H, W, C), where H and W correspond to
                the height and width of the image respectively. C is the number of
                color channels. The image is required to be in RGB format since that
                is a requirement of the Matplotlib library. The image is also expected
                to be in the range [0, 255].
            metadata (Metadata): dataset metadata (e.g. class names and colors)
            instance_mode (ColorMode): defines one of the pre-defined style for drawing
                instances on an image.
            digit_only: if digit_only, draw as normal
            montage: gt and pred side by side output
            nrows: number of rows for montage
            ncols: number of cols for montage
        """
        self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
        if metadata is None:
            metadata = MetadataCatalog.get("__nonexist__")
        self.metadata = metadata
        self.montage = montage
        if montage:
            self.output = MontageImage(self.img, scale=scale, nrows=nrows, ncols=ncols)
        else:
            self.output = VisImage(self.img, scale=scale)
        self.cpu_device = torch.device("cpu")

        # too small texts are useless, therefore clamp to 9
        # self._default_font_size = max(
        #     np.sqrt(self.output.height * self.output.width) / (nrows * ncols) // 90, 10 // scale
        # )
        self._default_font_size = 5
        self._instance_mode = instance_mode
        self.digit_only = digit_only


    def gts_to_dict(self, data):
        """
        data: acutual data returned by dataloader
        data['instances'] -> detectron2.structure.Instances

        """
        instances_list = []
        for i in range(len(data['instances'])):
            instance = data['instances'][i]
            digit_ids = torch.cat(instance.gt_digit_classes).numpy() if instance.has('gt_digit_classes') else np.empty((0,))
            number_id = ''.join([self.metadata.thing_classes[digit_id] for digit_id in digit_ids])
            number_id = self.metadata.thing_classes.index(number_id) if number_id in self.metadata.thing_classes else -1
            # tensor to numpy
            instances_list.append({
            "person_bbox": instance.gt_boxes.tensor.numpy() if instance.has('gt_boxes') else np.empty((0, 4)),
            "keypoints": instance.gt_keypoints.tensor.numpy() if instance.has('gt_keypoints') else np.empty((0, 17, 3)),
            "digit_bboxes": Boxes.cat(instance.gt_digit_boxes).tensor.numpy() if instance.has('gt_digit_boxes') else np.empty((0, 4)),
            "digit_ids": torch.cat(instance.gt_digit_classes).numpy() if instance.has('gt_digit_classes') else np.empty((0,)),
            "category_id": instance.gt_classes.numpy() if instance.has('gt_classes') else np.empty((0,)),
            "number_bbox": Boxes.cat(instance.gt_number_boxes).tensor.numpy() if instance.has('gt_number_boxes') else np.empty((0, 4)),
            "number_id": np.asarray([number_id])
            # "number_id": instance.gt_number_ids.numpy() if instance.has('gt_number_ids') else np.empty((0,)),
            })
        return {"annotations": instances_list}

    def predictions_to_list(self, instances):
        """
        data (Players): The instances of Players for an image.
        """
        instances_list = []
        for i in range(len(instances)):
            data = instances[i]
            # tensor to numpy
            instances_list.append({
            "person_bbox": data.pred_boxes.tensor.numpy() if data.has('pred_boxes') else np.empty((0, 4)),
            "scores": data.scores.numpy() if data.has('scores') else np.empty((0,)),
            "category_id": data.pred_classes.numpy() if data.has('pred_classes') else np.empty((0,)),
            # pred keypoints is tensor not Keypoints()
            "keypoints": data.pred_keypoints.numpy() if data.has('pred_keypoints') else np.empty((0, 17, 3)),
            "digit_bboxes": Boxes.cat(data.pred_digit_boxes).tensor.numpy() if data.has('pred_digit_boxes') else np.empty((0, 4)),
            "digit_ids": torch.cat(data.pred_digit_classes).numpy() if data.has('pred_digit_classes') else np.empty((0,)),
            "digit_scores": torch.cat(data.digit_scores).numpy() if data.has('digit_scores') else np.empty((0,)),
            "number_bbox": Boxes.cat(data.pred_number_boxes).tensor.numpy() if data.has('pred_number_boxes') else np.empty((0, 4)),
            "number_id": torch.cat(data.pred_number_classes).numpy() if data.has('pred_number_classes') else np.empty((0,)),
            "number_score": torch.cat(data.pred_number_scores).numpy() if data.has('pred_number_scores') else np.empty((0,)),
            })
        return instances_list

    def draw_dataloader_instances(self, instances):
        instances_list = self.gts_to_dict(instances)
        return self.draw_dataset_dict(instances_list)

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

    def draw_instance_predictions(self, instances):
        if self.digit_only:
            return super(JerseyNumberVisualizer, self).draw_instance_predictions(instances)
        predictions = self.predictions_to_list(instances)
        for p in predictions:
            self.draw_single_instance(p)
        return self.output

    def draw_montage(self, instances, dic):
        assert self.montage
        # draw gt
        self.output.set_ax(0, 0)
        annos = dic.get("annotations", None)
        if annos:
            for anno in annos:
                self.draw_single_instance(anno)
        # draw predictions
        self.output.set_ax(0, 1)
        predictions = self.predictions_to_list(instances)
        for p in predictions:
            self.draw_single_instance(p)

        return self.output

    def draw_single_instance(self, instance, draw_digit=True):
        """
        instance can be either from detection or gt
        Given a instance dict, draw the corresponding
        person bbox, keypoints(if available), and digit bounding boxes (if available)

        instance has four fields: digit_bboxes, digit_labels, person_bbox, keypoints
        """
        if instance is not None:
            person_bbox = np.array(instance.get("person_bbox", np.empty((0, 4))))
            keypoints   = np.array(instance.get("keypoints", np.empty((0, 4, 3))))
            digit_bboxes = np.array(instance.get("digit_bboxes", np.empty((0, 4)))).reshape((-1, 4))
            digit_ids = np.array(instance.get("digit_ids", np.empty((0,))))
            digit_ids = digit_ids.reshape(-1).tolist()
            category_id = np.array([instance.get("category_id")]) if "category_id" in instance else np.empty((0,))
            category_id = category_id.reshape(-1).tolist()
            bbox_mode = instance.get("bbox_mode", None)
            # we will have score if coming from predictions
            scores = instance.get("scores", None)
            digit_scores = instance.get("digit_scores", None)
            # get jersey number predictions
            number_bbox = np.array(instance.get("number_bbox", np.empty((0, 4)))).reshape(-1, 4)
            number_id = np.array(instance.get("number_id")).reshape(-1) if "number_id" in instance else np.empty((0,))
            # remove -1
            valid = number_id > -1
            number_id = number_id[valid]
            number_id = number_id.tolist()
            if len(number_id):
                number_bbox = number_bbox[valid]
            # todo: -1 in the number id
            number_score = instance.get("number_score", None)
            if number_score is not None:
                number_score = number_score[valid]
            # labels = _create_person_labels(category_id, scores, jersey_numbers, jersey_scores, self.metadata.get("thing_classes", None))
            labels = _create_text_labels(category_id, scores, self.metadata.get("thing_classes", None))
            digit_labels = _create_text_labels(digit_ids, digit_scores, self.metadata.get("thing_classes", None))
            number_labels = _create_text_labels(number_id, number_score, self.metadata.get("thing_classes", None))

            keypoints = pad_full_keypoints(keypoints)
            # no corresponding keypoint annotation
            if person_bbox.size and (not keypoints.size):
                keypoints = np.zeros((1, keypoints.shape[1], keypoints.shape[2]), dtype=np.float32)



            # here, we have two different bbox (person and digit)
            if bbox_mode is not None:
                person_bbox = [BoxMode.convert(person_bbox, bbox_mode, BoxMode.XYXY_ABS)]
                digit_bboxes = [BoxMode.convert(each_digit_bbox, bbox_mode, BoxMode.XYXY_ABS) for
                           each_digit_bbox in digit_bboxes]
                number_bbox = BoxMode.convert(number_bbox, bbox_mode, BoxMode.XYXY_ABS)

            if len(labels):
                person_colors = [_RED for _ in range(len(labels))]
                self.overlay_instances(labels=labels, boxes=person_bbox, masks=None, keypoints=keypoints, assigned_colors=person_colors)
            if number_labels and len(number_labels):
                number_colors = [_MID_GRAY for _ in range(len(number_labels))]
                self.overlay_instances(labels=number_labels, boxes=number_bbox, masks=None, keypoints=None,
                                       assigned_colors=number_colors, pos="up_mid")
            if draw_digit and len(digit_labels):
                self.overlay_instances(labels=digit_labels, boxes=digit_bboxes, masks=None, keypoints=None)
            return self.output





    def overlay_instances(
        self,
        *,
        boxes=None,
        labels=None,
        masks=None,
        keypoints=None,
        assigned_colors=None,
        alpha=1.0,
        pos="default"
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

        for i in range(num_instances):
            color = assigned_colors[i]
            if boxes is not None:
                self.draw_box(boxes[i], edge_color=color)


            if masks is not None:
                for segment in masks[i].polygons:
                    self.draw_polygon(segment.reshape(-1, 2), color, alpha=alpha)

            if labels is not None:
                # first get a box
                if boxes is not None:
                    x0, y0, x1, y1 = boxes[i]
                    text_pos = (x0, y0) # if drawing boxes, put text on the box corner.
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
                if pos == "up_mid":
                    horiz_align = "center"
                    shift = max(self._default_font_size / 4, 1) * self.output.scale * 1.2
                    num_rows = int('\n' in labels[i]) + 1
                    text_pos = ((x0 + x1) / 2, y0 - font_size * num_rows - shift)
                self.draw_text(
                    labels[i],
                    text_pos,
                    color=lighter_color,
                    horizontal_alignment=horiz_align,
                    font_size=font_size,
                )


        # draw keypoints
        if keypoints is not None:
            for i, keypoints_per_instance in enumerate(keypoints):
                color = assigned_colors[i]
                self.draw_and_connect_keypoints(keypoints_per_instance, color=color, draw_mid=False)

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
                self.draw_circle((x, y), radius=self._default_font_size // 2 + 1, color=color)
                if keypoint_names:
                    keypoint_name = keypoint_names[idx]
                    visible[keypoint_name] = (x, y)


        if self.metadata.get("keypoint_connection_rules"):
            for kp0, kp1, color in self.metadata.keypoint_connection_rules:
                if kp0 in visible and kp1 in visible:
                    x0, y0 = visible[kp0]
                    x1, y1 = visible[kp1]
                    color = tuple(x / 255.0 for x in color)
                    self.draw_line([x0, x1], [y0, y1], color=color, alpha=0.6)

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

    def draw_line(self, x_data, y_data, color, linestyle="-", linewidth=None, alpha=0.6):
        """
        Args:
            x_data (list[int]): a list containing x values of all the points being drawn.
                Length of list should match the length of y_data.
            y_data (list[int]): a list containing y values of all the points being drawn.
                Length of list should match the length of x_data.
            color: color of the line. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            linestyle: style of the line. Refer to `matplotlib.lines.Line2D`
                for a full list of formats that are accepted.
            linewidth (float or None): width of the line. When it's None,
                a default value will be computed and used.

        Returns:
            output (VisImage): image object with line drawn.
        """
        if linewidth is None:
            linewidth = self._default_font_size / 3
        linewidth = max(linewidth, 1)
        self.output.ax.add_line(
            mpl.lines.Line2D(
                x_data,
                y_data,
                linewidth=linewidth * self.output.scale,
                color=color,
                linestyle=linestyle,
                alpha=alpha
            )
        )
        return self.output

    def draw_circle(self, circle_coord, color, radius=3, alpha=0.6):
        """
        Args:
            circle_coord (list(int) or tuple(int)): contains the x and y coordinates
                of the center of the circle.
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            radius (int): radius of the circle.

        Returns:
            output (VisImage): image object with box drawn.
        """
        x, y = circle_coord
        self.output.ax.add_patch(
            mpl.patches.Circle(circle_coord, radius=radius, fill=True, color=color, alpha=alpha)
        )
        return self.output

    def draw_text(
        self,
        text,
        position,
        *,
        font_size=None,
        color="g",
        horizontal_alignment="center",
        rotation=0,
    ):
        """
        Args:
            text (str): class label
            position (tuple): a tuple of the x and y coordinates to place text on image.
            font_size (int, optional): font of the text. If not provided, a font size
                proportional to the image width is calculated and used.
            color: color of the text. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            horizontal_alignment (str): see `matplotlib.text.Text`
            rotation: rotation angle in degrees CCW

        Returns:
            output (VisImage): image object with text drawn.
        """
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))

        x, y = position
        self.output.ax.text(
            x,
            y,
            text,
            size=font_size * self.output.scale,
            family="sans-serif",
            bbox={"facecolor": "black", "alpha": 0.6, "pad": 0.7, "edgecolor": "none"},
            verticalalignment="top",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
        )
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