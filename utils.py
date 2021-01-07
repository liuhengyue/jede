import os
import json
from functools import reduce  # forward compatibility for Python 3
import operator
import pprint
import shutil
import skimage.io
import numpy as np
import datetime
from collections import defaultdict
def compute_center_distance(box, boxes):
    y_c = (box[2] + box[0]) / 2
    x_c = (box[3] + box[1]) / 2
    ys_c = (boxes[:, 2] + boxes[:, 0]) / 2
    xs_c = (boxes[:, 3] + boxes[:, 1]) / 2
    distances = np.square(ys_c - y_c) + np.square(xs_c - x_c)
    return distances

def compute_digit2upperbody_distance(box, boxes):
    y_c = (box[2] + box[0]) / 2
    x_c = (box[3] + box[1]) / 2
    ys_c = (boxes[:, 2] + boxes[:, 0]) / 3
    xs_c = (boxes[:, 3] + boxes[:, 1]) / 2
    distances = np.square(ys_c - y_c) + np.square(xs_c - x_c)
    return distances

def compute_digit2person_distance(box, boxes):
    y_c = (box[2] + box[0]) / 2
    x_c = (box[3] + box[1]) / 2
    # person centers (uper body center)
    y_person_c, x_person_c = (boxes[:, 2] + 3 * boxes[:, 0]) / 4, (boxes[:, 3] + boxes[:, 1]) / 2
    # y_person_c, x_person_c = boxes[:, 0], boxes[:, 1]
    # vector from digit center to digit person center
    x1, y1 = x_person_c - x_c, y_person_c - y_c
    x2, y2 = 0, 1
    dot = x1 * x2 + y1 * y2  # dot product
    det = x1 * y2 - y1 * x2  # determinant
    angle = np.abs(np.arctan2(det, dot))
    magnitude = np.square(x1**2 + y1**2)
    # check if box inside person box
    x1 = box[1] - boxes[:, 1]
    y1 = box[0] - boxes[:, 0]
    x2 = boxes[:, 3] - box[3]
    y2 = boxes[:, 2] - box[2]
    inside = np.logical_and(np.logical_and(np.logical_and(x1 >= 0, y1 >= 0), x2 >= 0), y2 >= 0)
    penalty = np.where(inside, 1, float("inf"))
    # print(angle * magnitude * penalty, "penalty ", penalty)
    # distances = box[0] - y1
    # distances = np.where(distances > 0, distances, float('inf'))
    # distances = np.square(y2 - y_c) + np.square(x2 - x_c) + np.square(y1 - y_c) + np.square(x1 - x_c)
    return magnitude * penalty

def compute_distances(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    distance_mat = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(distance_mat.shape[1]):
        box2 = boxes2[i]
        distance_mat[:, i] = compute_digit2person_distance(box2, boxes1)
    return distance_mat

def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps

def extract_bbox_from_polygon(points_x, points_y, h, w):
        # should exclude the boundary points
        points_x = np.array(points_x)
        points_y = np.array(points_y)
        x_max = np.minimum(np.amax(points_x) + 1, w)
        x_min = np.maximum(np.amin(points_x) - 1, 0)
        y_max = np.minimum(np.amax(points_y) + 1, h)
        y_min = np.maximum(np.amin(points_y) - 1, 0)
        return np.array([y_min, x_min, y_max, x_max])

def compute_keypoints_boxes_distances(boxes, keypoints):
    # boxes: [N1, 4]
    # keypoints: [N2, 4, (x, y, v)]
    # return [N1, N2]
    def compute_points_box_distance(keypoints, box):
        # check keypoints
        # box: (y1, x1, y2, x2).
        # convert points as [[[x1, y1], [x2, y2], ...]]
        points = keypoints[:,:,:2]
        points_x = np.mean(points[:, :, 0], axis=1)
        points_y = np.mean(points[:, :, 1], axis=1)
        y_c = (box[2] + box[0]) / 2
        x_c = (box[3] + box[1]) / 2
        distances = np.sqrt(np.square(points_y - y_c) + np.square(points_x - x_c))
        return distances

    distance_mat = np.zeros((boxes.shape[0], keypoints.shape[0]))
    for i in range(distance_mat.shape[0]):
        box = boxes[i]
        distance_mat[i, :] = compute_points_box_distance(keypoints, box)
    return distance_mat

def compute_points_inside_boxes(boxes, keypoints):
    # boxes: [N1, 4]
    # keypoints: [N2, 4, (x, y, v)]
    # return [N1, N2]
    def compute_points_inside(keypoints, box):
        # check keypoints
        # box: (y1, x1, y2, x2).
        # convert points as [[[x1, y1], [x2, y2], ...]]
        points = keypoints[:,:,:2]
        check = np.logical_and(points >= box[[1, 0]], points <= box[[3, 2]])
        if_inside = np.all(np.logical_and(points >= box[[1, 0]], points <= box[[3, 2]]), axis=(1,2))
        return if_inside

    inside_mat = np.zeros((boxes.shape[0], keypoints.shape[0]))
    for i in range(inside_mat.shape[0]):
        box = boxes[i]
        inside_mat[i, :] = compute_points_inside(keypoints, box)
    return inside_mat

class VIAConverter:
    def __init__(self, json_path, dataset_path=None):
        self.json_path = json_path
        self.dataset_path = dataset_path


    def load_json(self):
        """
        Load the json dictionary from the json file.
        :return: None
        """
        self.annotations = json.load(open(self.json_path))

    def combine_multi_anno_files(self, list_files):
        self.annotations = {}
        for json_file in list_files:
            self.annotations.update(json.load(open(json_file)))


    def remove_key_from_annotations(self, key_to_remove_list):
        """
        Remove a certain key from the annotations
        :param key_to_remove: A list of keys from the root of the dictionary
        An example: ['file_attributes', 'number'] will remove the number key
        :return:
        """

        def getFromDict(dataDict, mapList):
            return reduce(operator.getitem, mapList, dataDict)
        # in case we need to set the key value
        def setInDict(dataDict, mapList, value):
            getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value
        # loop over each annotation
        for k, v in self.annotations.items():
            # pop the key (the last element in the key list)
            getFromDict(v, key_to_remove_list[:-1]).pop(key_to_remove_list[-1], None)



    def sort_clockwise(self, points):
        def angle_with_start(coord, start):
            vec = coord - start
            return np.angle(np.complex(vec[0], vec[1]))
        # convert into a coordinate system
        # (1, 1, 1, 2) -> (1, 1), (1, 2)
        coords = points.tolist()
        coords = [np.array(coord) for coord in coords]
        # make sure the first coord is the left shoulder
        start = coords[0]
        # sort the remaining coordinates by angle
        # with reverse=True because we want to sort by clockwise angle
        rest = sorted(coords[1:], key=lambda coord: angle_with_start(coord, start), reverse=False)

        # our first coordinate should be our starting point
        rest.insert(0, start)

        points = np.stack(rest)
        # convert into the proper coordinate format
        # (1, 1), (1, 2) -> (1, 1, 1, 2)
        return points

    def match_digit_box(self, dis_mat):
        # person id dict
        person_digit_map = defaultdict(list)
        num_persons, num_digits = dis_mat.shape
        un_matched_ids = [i for i in range(num_digits)]
        while len(un_matched_ids) > 0:
            digit_id = un_matched_ids.pop(0)
            num_matches = np.sum(dis_mat[:, digit_id] < float("inf"))
            print(dis_mat[:, digit_id])
            if num_matches == 1:
                person_id = np.where(dis_mat[:, digit_id] < float("inf"))[0][0]
                person_digit_map[person_id].append(digit_id)
            elif num_matches > 1:
                got_match = True
                for id, dis in enumerate(dis_mat[:, digit_id]):
                    if dis < float("inf") and len(person_digit_map[id]) < 2:
                        person_digit_map[id].append(digit_id)
                        break
        print(person_digit_map)
        person_ids = []
        digit_ids =[]
        for k, v_s in person_digit_map.items():
            for v in v_s:
                person_ids.append(k)
                digit_ids.append(v)
        print(person_ids)
        print(digit_ids)
        list1, list2 = zip(*sorted(zip(digit_ids, person_ids)))
        return np.array(list2)




    def match_boxes(self, persons, keypoints, digits, digits_bboxes):
        """
        Match the bounding boxes between digit, person and keypoints
        :return:
        """
        numbers = []
        # [N_persons, N_digitboxes]
        distances = compute_distances(np.array(persons), np.array(digits_bboxes))
        # assert each digit is matched
        all_inf = np.all(distances == float("inf"), axis=0)
        assert np.any(all_inf) == False, "not all digits are matched"
        # print(distances)
        # ids_associated_by_center_distance = self.match_digit_box(distances)
        # for each digit, get the person id
        ids_associated_by_center_distance = np.argmin(distances, axis=0)
        # print(ids_associated_by_center_distance.shape)
        #        ids_associated_by_overlaps = np.argmax(overlaps, axis=0)
        associated_person = ids_associated_by_center_distance
        # shape: [num person boxes, num keypoint annotations]
        output_kpts = []  # same number of persons
        output_digits = []  # same number of persons
        output_digit_boxes = []  # same number of persons
        for i in range(len(persons)):
            numbers.append("")
            output_kpts.append(np.zeros((4, 3), dtype=np.int32).tolist())
            output_digits.append([])
            output_digit_boxes.append([])
        if len(keypoints) > 0:
            # shape: [N_boxes, N_kpt_map]
            kpts_distances = compute_keypoints_boxes_distances(np.array(persons), np.array(keypoints))
            kpts_inside_mask = compute_points_inside_boxes(np.array(persons), np.array(keypoints))
            kpts_distances = np.where(kpts_inside_mask, kpts_distances, float('inf'))
            # print(kpts_distances)
            sorted_kpts_ids = np.argsort(np.sum(kpts_distances != float('inf'), axis=0))
            # print(sorted_kpts_ids)
            # match one by one for each keypoint map
            person_matches = [False] * len(persons)
            matched_person_ids = []
            for id in sorted_kpts_ids:
                # sord distances for each
                # sorted_distances = np.sort(kpts_distances[:, id])
                sorted_person_ids = np.argsort(kpts_distances[:, id])
                # print(sorted_person_ids)
                # print(sorted_distances)
                for p_id in sorted_person_ids:
                    if not person_matches[p_id]:
                        matched_person_ids.append(p_id)
                        person_matches[p_id] = True
                        break
            matched_person_ids = np.array(matched_person_ids)
            assert matched_person_ids.shape == np.unique(
                matched_person_ids).shape, "Wrong keypoint match on image."
            for idx, person_id in enumerate(matched_person_ids):
                output_kpts[person_id] = keypoints[sorted_kpts_ids[idx]]
        # generate numbers from associations, for each person roi (even no association)
        for idx, person_id in enumerate(associated_person):
            numbers[person_id] = numbers[person_id] + digits[idx]
            output_digits[person_id].append(digits[idx])
            output_digit_boxes[person_id].append(digits_bboxes[idx])
        # assert
        # total_num_digits = len([number for numbers in output_digits for number in numbers])
        # assert total_num_digits == len(digits), "not all numbers are matched, check boundary"
        for number in numbers:
            if len(number) > 2:
                print("wrong number")
            # assert len(number) < 3, "wrong numbers associated with person"

        return numbers, output_kpts, output_digits, output_digit_boxes

    def process_regions(self, regions_anno, height, width, filename=None):
        persons = []
        keypoints = []
        polygons = []
        digits = []
        numbers = []
        digits_bboxes = []

        for region in regions_anno:
            # first check the label type
            label = region['region_attributes']['label']
            if label == 'digit':
                # class label
                digits.append(region['region_attributes']['digit'])
                # digit bounding box
                try:  # original mask annotation
                    polygons.append(region["shape_attributes"])
                    digit_bbox = extract_bbox_from_polygon(region["shape_attributes"]["all_points_x"], \
                                                           region["shape_attributes"]["all_points_y"], \
                                                           height, width).tolist()
                except:  # bbox annotation
                    x1, x2, y1, y2 = region["shape_attributes"]["x"], \
                                     region["shape_attributes"]["x"] + region["shape_attributes"][
                                         "width"], \
                                     region["shape_attributes"]["y"], \
                                     region["shape_attributes"]["y"] + region["shape_attributes"][
                                         "height"]
                    digit_bbox = [y1, x1, y2, x2]
                digits_bboxes.append(digit_bbox)
            elif label == 'person':
                x1, x2, y1, y2 = region["shape_attributes"]["x"], \
                                 region["shape_attributes"]["x"] + region["shape_attributes"][
                                     "width"], \
                                 region["shape_attributes"]["y"], \
                                 region["shape_attributes"]["y"] + region["shape_attributes"][
                                     "height"]
                persons.append([y1, x1, y2, x2])
            else: # label is keypoint
                # print(region)
                p = region["shape_attributes"]
                # shape: (4, 3)
                kpts = np.stack((p['all_points_x'], p['all_points_y']), axis=-1)
                kpts = self.sort_clockwise(kpts)
                assert kpts.shape == (4, 2), "Wrong shape of keypoints on image {}.".format(filename)
                kpts = np.concatenate((kpts, np.ones((kpts.shape[0], 1), dtype=np.int8) * 2), axis=1)
                keypoints.append(kpts.tolist())

        # loop end
        numbers, output_kpts, output_digits, output_digit_boxes = self.match_boxes(persons, keypoints, digits,
                                                                                   digits_bboxes)
        return persons, polygons, numbers, output_kpts, output_digits, output_digit_boxes
        # numbers, output_kpts, output_digits, output_digit_boxes = self.match_boxes(persons, keypoints, digits, digits_bboxes)
        # output_anno = {'filename': a['filename'], \
        #                               'width': width, 'height': height, 'polygons': polygons, \
        #                               'keypoints': output_kpts, 'persons': persons, 'digits': digits, 'associated_person': associated_person.tolist(), 'numbers': numbers,
        #                               'digits_bboxes': digits_bboxes, 'video_id': a['file_attributes']['video_id']}
        # self.output_annotations.append()

    def convert_via_annotations(self):
        """
        Convert the via annotations to a better format, with verifications of annotations.
        :return:
        """
        output_annotations = []
        for _, anno in self.annotations.items():
            filename = anno['filename']
            self.filename = filename
            print(filename)
            # only process annotation with regions label
            if anno['regions']:
                image_path = os.path.join(self.dataset_path, anno['filename'])
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]
                persons, polygons, numbers, output_kpts, output_digits, output_digit_boxes = self.process_regions(anno['regions'], height, width, filename=filename)
                res = {'filename': filename, 'width': width, 'height': height, 'polygons': polygons, \
                       'keypoints': output_kpts, 'persons': persons, 'digits': output_digits,
                       'digits_bboxes': output_digit_boxes, 'numbers': numbers, 'video_id': anno['file_attributes']['video_id']
                       }
                output_annotations.append(res)
                # copy image to the SJNM folder
                self.copy_image_to_path(image_path, self.dataset_path)
        return output_annotations

    def copy_image_to_path(self, image_old_path, image_folder):
        """
        We have different image folders for different batches, one easy way is to move all the images into one folder.
        :param image_old_path:
        :param image_folder:
        :return:
        """
        if not os.path.exists(image_folder):
            raise Exception("No such folder")
        # create new image path
        basename = os.path.basename(image_old_path)
        image_new_path = os.path.join(image_folder, basename)
        # for those images are not in the target folder, do the copy
        if not os.path.exists(image_new_path):
            shutil.copyfile(image_old_path, image_new_path)


    def via_old2new(self, json_file):
        """
        Convert old VIA project json to the new format with label attribute
        :param json_file:
        :return:
        """
        old_annotations = json.load(open(json_file))
        import copy
        new_annotations = copy.deepcopy(old_annotations)
        for k, v in old_annotations['_via_img_metadata'].items():
            # remove file attributes
            for target_key in ['Number', 'number', 'single']:
                new_annotations['_via_img_metadata'][k]['file_attributes'].pop(target_key, None)
            if v['regions']:
                for i, region in enumerate(v['regions']):
                    for key, val in region['region_attributes'].items():
                        region_attrs = new_annotations['_via_img_metadata'][k]['regions'][i]['region_attributes']
                        # check the entry properties
                        if (key == "digit" or key == "digits") and val != None:
                            region_attrs.pop("keypoints", None)
                            region_attrs.pop("person", None)
                            region_attrs['digit'] = region_attrs.pop("digits", None)
                            region_attrs['label'] = 'digit'
                        elif key == "keypoints" and val == "true":
                            region_attrs['label'] = 'keypoints'
                            region_attrs.pop("keypoints", None)
                            region_attrs.pop("person", None)
                        elif key == "person" and val == "true":
                            region_attrs['label'] = 'person'
                            region_attrs.pop("keypoints", None)
                            region_attrs.pop("person", None)
                        else:
                            # actually there are several files with wrong annotation type, did it manually
                            pass
                            # raise Exception("Annotation format incorrect on image {}".format(v['filename']))
        return new_annotations



    def test_print(self):
        pp = pprint.PrettyPrinter(indent=2)
        # pp.pprint(self.annotations["nba01_35_0.png257617"])
        pp.pprint(self.annotations["nba01_35_0.png257617"])

def save(ds_to_dump, save_dir="./", file_name='processed_via_total.json'):
    # basename = os.path.basename(self.json_path) # with .json ext
    with open(os.path.join(save_dir, "{}".format(file_name)), "w") as write_file:
        json.dump(ds_to_dump, write_file)

def convert_single_via_project(json_path, dataset_path):
    """
    The json path is the VIA exported json file with annotations
    """
    # json_path = r"D:\research\playground-mask-rcnn\json\batch5.json"
    # dataset_path = r"D:\research\batch_nba_01"
    via_converter = VIAConverter(json_path, dataset_path)
    via_converter.load_json()
    # via_converter.remove_key_from_annotations(['file_attributes', 'number'])
    # via_converter.remove_key_from_annotations(['file_attributes', 'single'])
    output_annotations = via_converter.convert_via_annotations()
    save(output_annotations, save_dir='./datasets/jnw/annotations', file_name="processed_annotations.json")

def process_multi_batch_files(list_batch_files, list_data_paths):
    output_annotations = {}
    i = 0
    for batch_file, data_path in zip(list_batch_files, list_data_paths):
        cur_converter = VIAConverter(batch_file, data_path)
        cur_converter.load_json()
        cur_annotations = cur_converter.convert_via_annotations()
        for annotation in cur_annotations:
            output_annotations[i] = annotation
            i += 1
    save(output_annotations, file_name="batch_all.json")


def remove_via_empty_annotations(via_project_dict):
    new_via_project_dict = {'_via_settings': via_project_dict['_via_settings'],
                            '_via_img_metadata': {},
                            '_via_attributes': via_project_dict['_via_attributes']}

    for key, val in via_project_dict['_via_img_metadata'].items():
        if val['regions']:
            new_via_project_dict['_via_img_metadata'][key] = val
    return new_via_project_dict


def merge_via_projects(list_project_files, save_project=True):
    """
    The via project json format:
    {
        "_via_settings" : {},
        "_via_img_metadata" : {},
        "_via_attributes" : {}
    }
    """
    if len(list_project_files) == 0:
        return None
    if len(list_project_files) == 1:
        return list_project_files[0]
    # keep the first dict, and update its keys
    merged_via_project_dict = json.load(open(list_project_files[0]))
    for i in range(1, len(list_project_files)):
        cur_via_project_dict = json.load(open(list_project_files[i]))
        # remove empty annotations
        cur_via_project_dict = remove_via_empty_annotations(cur_via_project_dict)
        merged_via_project_dict['_via_img_metadata'].update(cur_via_project_dict['_via_img_metadata'])
    if save_project:
        save(merged_via_project_dict, save_dir='./data/', file_name='merged_via_project.json')
    return merged_via_project_dict

class VIA2Coco:

    """
    NOT IMPLEMENTED!!!
    The annotation format:
    [{'filename': filename, 'width': width, 'height': height, 'polygons': polygons, \
                       'keypoints': output_kpts, 'persons': persons, 'digits': output_digits,
                       'digits_bboxes': output_digit_boxes, 'numbers': numbers, 'video_id': anno['file_attributes']['video_id']
                       }]

    COCO format:
    {
     "info": info,
     "licenses": [license],
     "categories": [category],
     "images": [
            {"file_name": "0.jpg", "height": 600, "width": 800, "id": 0},...
          ],
     "annotations": [{
        "segmentation": [[510.66,423.01,511.72,420.03,...,510.45,423.01]],
        "area": 702.1057499999998,
        "iscrowd": 0,
        "image_id": 289343,
        "bbox": [473.07,395.93,38.65,28.67],
        "category_id": 18,
        "id": 1768}, ...]
    }

    """

    def __init__(self, annotation_file):
        self.annotation_file = annotation_file
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.__coco_base()
        self.__load_annotations()

    def __coco_base(self):
        INFO = {
            "description": "Example Dataset",
            "url": "https://github.com/waspinator/pycococreator",
            "version": "0.1.0",
            "year": 2018,
            "contributor": "waspinator",
            "date_created": datetime.datetime.utcnow().isoformat(' ')
        }

        LICENSES = [
            {
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
            }
        ]

        CATEGORIES, IAMGES, ANNOTATIONS = [], [], []
        self.coco = {
         "info": INFO,
         "licenses": LICENSES,
         "categories": CATEGORIES,
         "images": IAMGES,
         "annotations": ANNOTATIONS
        }

    def __load_annotations(self):
        # list of annotations grouped by image [{}, {}]
        self.via_project_dict = json.load(open(self.annotation_file))

    def _image(self, i):
        ann = self.via_project_dict[i]
        image = {}
        image['height'] = ann['height']
        image['width'] = ann['width']
        image['id'] = self.img_id
        image['file_name'] = ann['filename']
        return image





if __name__ == '__main__':
    # batch_files = [r'D:\research\playground-mask-rcnn\json\batch_all.json', r'D:\research\playground-mask-rcnn\json\batch5.json']
    # data_paths = [r'D:\research\SJNM', r'D:\research\batch_nba_01']
    # process_multi_batch_files(batch_files, data_paths)

    # merged_via_project_dict = merge_via_projects(['./data/processed_via_total.json', './data/via_project_batch5.json'])

    convert_single_via_project('/home/henry/Research/da-det/datasets/jnw/annotations/via_export_json.json', '/home/henry/Research/da-det/datasets/jnw/total/')

    # new_annotations = via_converter.via_old2new(r"D:\research\playground-mask-rcnn\json\via_total.json")
    # via_converter.save(new_annotations)
