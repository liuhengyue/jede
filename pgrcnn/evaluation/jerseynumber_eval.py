from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import datetime
import time
from collections import defaultdict
from pycocotools import mask as maskUtils
import copy

class JerseyNumberEval(COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType='segm'):
        self.task_type = iouType
        iou_type = iouType if iouType == "keypoints" else "bbox"
        super().__init__(cocoGt, cocoDt, iouType=iou_type)

    def evaluate(self):
        '''
        Run per image evaluation on given images and store results (a list of dict) in self.evalImgs
        :return: None
        '''
        tic = time.time()
        print('Running per image evaluation...')
        p = self.params
        # add backward compatibility if useSegm is specified in params
        if not p.useSegm is None:
            p.iouType = 'segm' if p.useSegm == 1 else 'bbox'
            print('useSegm (deprecated) is not None. Running {} evaluation'.format(p.iouType))
        print('Evaluate annotation type *{}*'.format(p.iouType))
        p.imgIds = list(np.unique(p.imgIds))
        if p.useCats:
            p.catIds = list(np.unique(p.catIds))
        p.maxDets = sorted(p.maxDets)
        self.params=p

        self._prepare()
        # loop through images, area range, max detection number
        catIds = p.catIds if p.useCats else [-1]

        if p.iouType == 'segm' or p.iouType == 'bbox':
            computeIoU = self.computeIoU
        elif p.iouType == 'keypoints':
            computeIoU = self.computeOks
        if self.task_type in {"jersey_number", "jersey_number_box", "jersey_number_class"}:
            computeIoU = self.computeIoU
        if self.task_type == "digit_bbox_class_agnostic":
            computeIoU = self.computeIoU
        self.ious = {(imgId, catId): computeIoU(imgId, catId) \
                        for imgId in p.imgIds
                        for catId in catIds}

        evaluateImg = self.evaluateImg # if self.task_type == "jersey_number" else super().evaluateImg
        maxDet = p.maxDets[-1]
        self.evalImgs = [evaluateImg(imgId, catId, areaRng, maxDet)
                 for catId in catIds
                 for areaRng in p.areaRng
                 for imgId in p.imgIds
             ]
        self._paramsEval = copy.deepcopy(self.params)
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc-tic))

    # def computeMatches(self, imgId, catId):
    #     p = self.params
    #     if p.useCats:
    #         gt = self._gts[imgId,catId]
    #         dt = self._dts[imgId,catId]
    #     else:
    #         gt = [_ for cId in p.catIds for _ in self._gts[imgId,cId]]
    #         dt = [_ for cId in p.catIds for _ in self._dts[imgId,cId]]
    #     if len(gt) == 0 and len(dt) ==0:
    #         return []
    #     inds = np.argsort([-d['score'] for d in dt], kind='mergesort')
    #     dt = [dt[i] for i in inds]
    #     if len(dt) > p.maxDets[-1]:
    #         dt=dt[0:p.maxDets[-1]]
    #
    #     if p.iouType == 'segm':
    #         g = [g['segmentation'] for g in gt]
    #         d = [d['segmentation'] for d in dt]
    #     elif p.iouType == 'bbox':
    #         g = [g['bbox'] for g in gt]
    #         d = [d['bbox'] for d in dt]
    #     else:
    #         raise Exception('unknown iouType for iou computation')
    #
    #     # compute iou between each dt and gt region
    #     iscrowd = [int(o['iscrowd']) for o in gt]
    #     ious = maskUtils.iou(d,g,iscrowd)
    #     # get the jersey matching matrix
    #     g_jersey = [self.jersey_number_mapping[tuple(g["jersey_number"])] for g in gt]
    #     d_jersey = [self.jersey_number_mapping[tuple(d["jersey_number"])]
    #                 if tuple(d["jersey_number"]) in self.jersey_number_mapping else -1 for d in dt]
    #     m = len(d_jersey)
    #     n = len(g_jersey)
    #     matches = np.zeros((m, n), dtype=np.double)
    #     for i in range(m):
    #         for j in range(n):
    #             matches[i, j] = d_jersey[i] == g_jersey[j]
    #     ious = ious * matches
    #     return ious
