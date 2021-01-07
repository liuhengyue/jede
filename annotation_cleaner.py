#!/usr/bin/env python3
# I am too tired of dealing with the annotations, so this file is aiming to clean the VIA project file directly
# by changing the order of regions to make sure the annotations can be parsed without any problem
# @Author Henry

import os
import math
from collections import defaultdict
import json
import shutil
import skimage.io
import numpy as np
from shapely import geometry, ops
from functools import reduce
import operator


# ================== Macro JSON structure ==================
#     '_via_settings': dict
#     '_via_img_metadata'': dict
#         {{}, {}, {'filename':str, 'size': int, 'regions': list, 'file_attributes': {}}}
#     '_via_attributes'

# 'file_attributes' : {'video_id': int}
# regions: list []
# regions[0]: dict {'shape_attributes': {}, 'region_attributes': {}}

# ================== digit with segmentation ==========================================
# 'shape_attributes': {'name': 'polygon', 'all_points_x': [], 'all_points_y': []}
# 'region_attributes': {'digit': str, 'label': 'digit'}

# ================== keypoints ========================================================
# 'shape_attributes': {'name': 'polygon', 'all_points_x': [], 'all_points_y': []}
# 'region_attributes': {'digit': str, 'label': 'keypoints'}

# ================== person bounding box  =============================================
# 'shape_attributes': {'name': 'rect', 'x': int, 'y': int, 'width': int, 'height': int}
# 'region_attributes': {'digit': str, 'label': 'keypoints'}

# the problem here: the list of regions are not ordered properly
# it would be nice if they are arranged such as:
#   person_1 bb - digit1 (left) - digit2 (right) - keypoints_1 - person_2 bb ...

class AnnotationCleaner():
    def __init__(self):
        self.VIA_PROJECT_FILE_PATH  = '../../datasets/jnw/annotations/reordered_via_project.json'
        self.DATASET_PATH           = '../../datasets/jnw/total/'
        self.OUTPUT_FILE_PATH       = '../../datasets/jnw/reordered_via_project.json'
        self.annotations            = self.load_via_project_json()
        self.CURRENT_FILE_KEY       = '' # for recording current processing name
        self.OUTPUT_ANNOTATION_PATH = '../../datasets/jnw/annotations/jnw_annotations.json'


    def load_via_project_json(self):
        """
        Load the json dictionary from the json file.
        :return: None
        """
        return json.load(open(self.VIA_PROJECT_FILE_PATH))

    def _sort_regions(self, data):
        """
        data is the annotation for single file
        """
        person_bboxes, keypoints, digit_bboxes = [], [], []
        person_ids, keypoints_ids, digit_ids = [], [], []
        image_path = os.path.join(self.DATASET_PATH, data['filename'])
        image = skimage.io.imread(image_path)
        height, width = image.shape[:2]
        num_regions = len(data['regions'])
        for i, region in enumerate(data['regions']):
            label = region['region_attributes']['label']
            shape = region['shape_attributes']
            if label == 'person':
                polygon = [shape['x'], shape['y'], shape['width'], shape['height']]
                polygon = xywh2points(polygon)
                person_bboxes.append(polygon)
                person_ids.append(i)
            elif label == 'keypoints':
                polygon = [(shape['all_points_x'][i], shape['all_points_y'][i]) for i in range(0, len(shape['all_points_x']))]
                polygon = self.verify_keypoints_order(polygon)
                # return a sorted keypoints
                shape_dict = self.polygons2annotation(polygon)
                # modify in-place
                self.annotations['_via_img_metadata'][self.CURRENT_FILE_KEY]['regions'][i]['shape_attributes'] = shape_dict
                keypoints.append(polygon)
                keypoints_ids.append(i)
            elif label == 'digit':
                # has segmentation annotation
                if shape['name'] == 'rect':
                    polygon = [shape['x'], shape['y'], shape['width'], shape['height']]
                    polygon = xywh2points(polygon)
                else:
                    polygon = polygon2points(shape["all_points_x"], \
                                      shape["all_points_y"], \
                                      height, width)
                digit_bboxes.append(polygon)
                digit_ids.append(i)
            else:
                raise NameError("label not found.")
        # print(person_bboxes)
        # print(keypoints)
        # print(digit_bboxes)
        return {"person_bboxes": person_bboxes, "keypoints": keypoints, "digit_bboxes": digit_bboxes, \
                "person_ids": person_ids, "keypoints_ids": keypoints_ids, "digit_ids": digit_ids, "num_regions": num_regions}

    def match_instance(self, regions_dict):
        """
        A dict {"person_bboxes": person_bboxes, "keypoints": keypoints, "digit_bboxes": digit_bboxes, \
                "person_ids": person_ids, "keypoints_ids": keypoints_ids, "digit_ids": digit_ids, "num_regions": num_regions}
                for matching

        """
        # a dict of list

        # ======================= start with person vs keypoints ===============
        instances = defaultdict(list)
        def get_instance(target):
            for k, v in instances.items():
                if target in v:
                    return k
            return None
        # init by number of persons
        for i in range(len(regions_dict["person_ids"])):
            instances[i].append(regions_dict["person_ids"][i])
        # create a dict for matching status
        match = {id: None for id in regions_dict["person_ids"] + regions_dict["keypoints_ids"]}
        # iter over each person polygon
        for person_idx, person_bbox in enumerate(regions_dict["person_bboxes"]):
            keypoints_idx = self.match_person_keypoints(person_bbox, regions_dict["keypoints"])
            if keypoints_idx > -1 and (not match[regions_dict["person_ids"][person_idx]]) \
                    and (not match[regions_dict["keypoints_ids"][keypoints_idx]]):
                match[regions_dict["person_ids"][person_idx]] = regions_dict["keypoints_ids"][keypoints_idx]
                match[regions_dict["keypoints_ids"][keypoints_idx]] = regions_dict["person_ids"][person_idx]
                # find which person_id is associated with the instances
                instance_key = get_instance(regions_dict["person_ids"][person_idx])
                instances[instance_key].append(regions_dict["keypoints_ids"][keypoints_idx])

        # print(match)
        # print(instances)
        # if anything is not matched
        # assert None not in match.values()

        # ======================= keypoints vs digit_bboxes ===============
        # Note: one keypoints can be associated with two digits
        match = {id: None for id in regions_dict["digit_ids"]}
        # iter over each person polygon
        for digit_idx, digit_bbox in enumerate(regions_dict["digit_bboxes"]):
            keypoints_idx = self.match_digit_keypoints(digit_bbox, regions_dict["keypoints"])
            if keypoints_idx > -1 and (not match[regions_dict["digit_ids"][digit_idx]]):
                match[regions_dict["digit_ids"][digit_idx]] = regions_dict["keypoints_ids"][keypoints_idx]
                # find which keypoints is associated with the instances
                instance_key = get_instance(regions_dict["keypoints_ids"][keypoints_idx])
                instances[instance_key].append(regions_dict["digit_ids"][digit_idx])
            # no match
            # ======================= person_bboxes vs digit_bboxes ===============
            else:
                person_idx = self.match_digit_person(digit_bbox, regions_dict["person_bboxes"])
                if person_idx > -1:
                    match[regions_dict["digit_ids"][digit_idx]] = regions_dict["person_ids"][person_idx]
                    instance_key = get_instance(regions_dict["person_ids"][person_idx])
                    instances[instance_key].append(regions_dict["digit_ids"][digit_idx])
                else:
                    raise Exception("No matching for digit bbox!")

        # here, not every digit is needed to match any keypoints
        # print(match)
        # print(instances)

        # return the re-ordered region indices
        reordered_idx = [var for _, l in instances.items() for var in l]
        # print(reordered_idx)
        if len(reordered_idx) != regions_dict['num_regions']:
            print('please mannally change the order on this image {}.'.format(self.CURRENT_FILE_KEY))
            return [i for i in range(regions_dict["num_regions"])]
        return reordered_idx


    def match_person_keypoints(self, person_bbox, keypoints):
        """
        rule: if all keypoints are inside the person_bbox, it is a match
        Note: there may be multiple matches since players can be closed
        May change the order manually
        """
        p_min, _, p_max, _ = person_bbox
        for idx, keypoint_ins in enumerate(keypoints):
            score = all([p_min[0] <= keypoint[0] <= p_max[0] and p_min[1] <= keypoint[1] <= p_max[1] for keypoint in keypoint_ins])
            if score == 1:
                return idx
        return -1

    def match_digit_keypoints(self, digit_bbox, keypoints):
        iou_s = []
        for kpts in keypoints:
            iou = compute_iou(digit_bbox, kpts)
            iou_s.append(iou)
        # get largest iou
        if np.amax(iou_s) > 0:
            return np.argmax(iou_s)
        else:
            return -1

    def match_digit_person(self, digit_bbox, person_bboxes):
        """
        rule: if all digit boundingbox vertex are inside the person_bbox, it is a match
        Note: there may be multiple matches since players can be closed
        May change the order manually
        """

        for idx, person_bbox in enumerate(person_bboxes):
            p_min, _, p_max, _ = person_bbox
            score = all([p_min[0] <= p[0] <= p_max[0] and p_min[1] <= p[1] <= p_max[1] for p in digit_bbox])
            if score == 1:
                return idx
        return -1

    def verify_keypoints_order(self, polygon):
        """
        polygon [(x, y) * 4] a list of tuples should be in the order
        left_shoulder, right_shoulder, right_hip, left_hip
        """
        # first point should be left shoulder
        center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), polygon), [len(polygon)] * 2))
        polygon = sorted(polygon, key=lambda coord:
        math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))

        # if (not polygons[1][0] > polygons[0][0]) or (not polygons[2][1] > polygons[1][1]) \
        #         or (not polygons[3][0] < polygons[2][0]) or (not polygons[0][1] < polygons[3][1]):
        #     print("Sort does not work, order wrong in image {}".format(self.CURRENT_FILE_NAME))

        assert polygon[1][0] > polygon[0][0], "left right shoulder order wrong in image {}".format(self.CURRENT_FILE_KEY)
        assert polygon[2][1] > polygon[1][1], "right shoulder / hip order wrong in image {}".format(self.CURRENT_FILE_KEY)
        assert polygon[3][0] < polygon[2][0], "left right hip order wrong in image {}".format(self.CURRENT_FILE_KEY)
        assert polygon[0][1] < polygon[3][1], "left shoulder / hip order wrong in image {}".format(self.CURRENT_FILE_KEY)

        return polygon

    def polygons2annotation(self, polygon):
        shape_annotation = {"name": "polygon"}
        shape_annotation['all_points_x'] = [p[0] for p in polygon]
        shape_annotation['all_points_y'] = [p[1] for p in polygon]
        return shape_annotation

    def get_one_example(self, idx=0, key=None):
        if key:
            return self.annotations['_via_img_metadata'][key]
        example = list(self.annotations['_via_img_metadata'])
        key = example[idx]
        return self.annotations['_via_img_metadata'][key]


    def gather_single_file_annotation(self, data, image_id):
        """
        data is the annotation for single file.
        Return the instance dict:
        {
          'image_id': int,
          'filename': str,
          'width': int,
          'height': int,
          'video_id': int,
          'instances': list
        }

        in 'instances' field, each is a dict:
            {'person_bbox'   : list size 1x4},
            {'keypoints'     : list size 1x12 [x_ls, y_ls, v _ls, x_rs, ...]},
            {'digit_bboxes'  : list (max size 2) of lists size 1x4},
            {'digit_labels'     : list (max size 2)}

        """

        annotation = {
          'image_id': None,
          'file_name': None,
          'width': None,
          'height': None,
          'video_id': None,
          'instances': []
        }
        # general info
        annotation['image_id'] = int(image_id)
        annotation['file_name'] = data['filename']
        annotation['video_id'] = int(data['file_attributes']['video_id'])

        image_path = os.path.join(self.DATASET_PATH, data['filename'])
        image = skimage.io.imread(image_path)
        height, width = image.shape[:2]

        annotation['height'] = height
        annotation['width']  = width


        instance = {'digit_bboxes': [], 'digit_labels': [], 'keypoints': []}


        for i, region in enumerate(data['regions']):
            label = region['region_attributes']['label']
            shape = region['shape_attributes']
            if label == 'person':
                if 'person_bbox' in instance:
                    annotation['instances'].append(instance)
                    instance = {'digit_bboxes': [], 'digit_labels': [], 'keypoints': []}
                bbox = [shape['x'], shape['y'], shape['width'], shape['height']]
                bbox = xywh2xyxy(bbox)
                instance['person_bbox'] = bbox

            elif label == 'keypoints':
                kpts = keypoints2list(shape['all_points_x'], shape['all_points_y'])
                instance['keypoints'] = kpts
            elif label == 'digit':
                digit_label = int(region['region_attributes']['digit'])
                # has segmentation annotation
                if shape['name'] == 'rect':
                    bbox = [shape['x'], shape['y'], shape['width'], shape['height']]
                    bbox = xywh2xyxy(bbox)
                else:
                    bbox = polygon2xyxy(shape["all_points_x"], \
                                             shape["all_points_y"], \
                                             height, width)
                instance['digit_bboxes'].append(bbox)
                instance['digit_labels'].append(digit_label)
            else:
                raise NameError("label not found.")
        if 'person_bbox' in instance:
            annotation['instances'].append(instance)

        for instance in annotation['instances']:
            if len(instance['digit_bboxes']) > 2:
                print("error on image {}".format(annotation['filename']))

        # print(annotation)
        return annotation

    def export(self):
        """

        the output annotations:
        [{'image_id'   : int,
          'filename'   : str,
          'width'      : int,
          'height'     : int,
          'video_id'   : int,
          'instances'  : list
          }, {}]

        in 'instances' field, each is a dict:
            {'person_bbox'   : list size 1x4},
            {'keypoints'     : list size 1x12 [x_ls, y_ls, v _ls, x_rs, ...]},
            {'digit_bboxes'  : list (max size 2) of lists size 1x4},
            {'digit_ids'     : list (max size 2)}

        All bounding boxes follow the order of XYXY_ABS

        """

        output_annotations = []
        image_id = 0
        for _, anno in self.annotations['_via_img_metadata'].items():
            print(anno['filename'])
            res = self.gather_single_file_annotation(anno, image_id)
            output_annotations.append(res)
            image_id += 1
        with open(self.OUTPUT_ANNOTATION_PATH, "w") as write_file:
            json.dump(output_annotations, write_file)

    def test(self):
        example = self.get_one_example(key='new_3218_1.png15805')
        print(example['filename'])
        # print(example)
        regions = example['regions']
        print([region['region_attributes']['label'] for region in regions])
        region_tuples = self._sort_regions(example)
        print(region_tuples)
        sorted_idx = self.match_instance(region_tuples)
        example['regions'] = [example['regions'][i] for i in sorted_idx]
        print([region['region_attributes']['label'] for region in example['regions']])


    def reorder_region_attributes(self):
        for key, annotation in self.annotations['_via_img_metadata'].items():
            self.CURRENT_FILE_KEY = key
            region_dict = self._sort_regions(annotation)
            if len(region_dict["keypoints"]) > 0:
                sorted_idx = self.match_instance(region_dict)
                self.annotations['_via_img_metadata'][key]['regions'] = [annotation['regions'][i] for i in sorted_idx]
            else:
                print('please mannally change the order on this image {}.'.format(self.CURRENT_FILE_KEY))

    def save(self):
        # basename = os.path.basename(self.json_path) # with .json ext
        with open(self.OUTPUT_FILE_PATH, "w") as write_file:
            json.dump(self.annotations, write_file)

def polygon2xyxy(points_x, points_y, h, w):
    # should exclude the boundary points
    x_max = min(max(points_x) + 1, w)
    x_min = max(min(points_x) - 1, 0)
    y_max = min(max(points_y) + 1, h)
    y_min = max(min(points_y) - 1, 0)
    return [x_min, y_min, x_max, y_max]

def polygon2points(points_x, points_y, h, w):
    # should exclude the boundary points
    points_x = np.array(points_x)
    points_y = np.array(points_y)
    x2 = np.minimum(np.amax(points_x) + 1, w)
    x = np.maximum(np.amin(points_x) - 1, 0)
    y2 = np.minimum(np.amax(points_y) + 1, h)
    y = np.maximum(np.amin(points_y) - 1, 0)
    return [(x, y), (x2, y), (x2, y2), (x, y2)]

def keypoints2list(points_x, points_y):
    # a list of all points in the order of ls, rs, rh, lh
    # x, y, v
    return [v for x, y in zip(points_x, points_y) for v in [x, y, 2]]

def xywh2xyxy(xywh_list):
    x, y, width, height = xywh_list
    x2 = x + width
    y2 = y + height
    return [x, y, x2, y2]

def xywh2points(xywh_list):
    x, y, width, height = xywh_list
    x2 = x + width + 1
    y2 = y + height + 1
    return [(x, y), (x2, y), (x2, y2), (x, y2)]

def mesh_grid_indices(list_lengths):
    # generate all combinations of indices
    grid = [list(range(length)) for length in list_lengths]
    return np.array(np.meshgrid(*grid)).T.reshape(-1, len(list_lengths))


def compute_iou(region1, region2):
    """
    Given two regions, compute its iou

    Each region is defined with four 2D points [tuple, tuple, tuple, tuple]
    """
    polygon1     = geometry.Polygon(tuple(region1))
    polygon2     = geometry.Polygon(tuple(region2))
    intersection = polygon1.intersection(polygon2)
    union        = polygon1.union(polygon2)
    iou = intersection.area / union.area
    return iou

def compute_iou_regions(regions):
    """
    Given a list of regions, compute the iou considering all regions
    In this case, regions should be only a list with length 3

    Each region is defined with four 2D points [tuple, tuple, tuple, tuple]
    """

    intersection = geometry.Polygon(tuple(regions[0]))
    union = geometry.Polygon(tuple(regions[0]))

    for region in regions[1:]:
        polygon      = geometry.Polygon(tuple(region))
        intersection = intersection.intersection(polygon)
        union        = union.union(polygon)

    iou = intersection.area / union.area
    return iou



if __name__ == '__main__':
    annotation_cleaner = AnnotationCleaner()
    # annotation_cleaner.test()
    # annotation_cleaner.reorder_region_attributes()
    # annotation_cleaner.save()

    annotation_cleaner.export()




