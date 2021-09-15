# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
import torch

import detectron2.export.torchscript  # apply patch # noqa
from detectron2 import model_zoo
from detectron2.layers import ShapeSpec
from detectron2.utils.env import TORCH_VERSION

import pgrcnn
from pgrcnn.config import get_cfg
from pgrcnn.modeling import build_digit_neck_branch, build_digit_neck_output


class TestDigitNeckBranch(unittest.TestCase):
    @unittest.skipIf(TORCH_VERSION < (1, 8), "Insufficient pytorch version")
    def test_KptsAttentionBranch(self):
        cfg = get_cfg()
        name = cfg.MODEL.ROI_DIGIT_NECK_BRANCHES.KEYPOINTS_BRANCH.NAME = "KptsAttentionBranch"
        cfg.MODEL.ROI_DIGIT_NECK_BRANCHES.KEYPOINTS_BRANCH.UP_SCALE = 2
        neck_branch = build_digit_neck_branch(name, cfg, ShapeSpec(channels=17, height=56, width=56))
        inp = torch.rand(2, 17, 56, 56)
        out = neck_branch(inp)
        print(neck_branch)
        self.assertTrue(out.shape == (2, 64, 56, 56))
    def test_neck(self):
        cfg = get_cfg()
        name = cfg.MODEL.ROI_DIGIT_NECK_BRANCHES.KEYPOINTS_BRANCH.NAME = "KptsAttentionBranch"
        cfg.MODEL.ROI_DIGIT_NECK_BRANCHES.KEYPOINTS_BRANCH.UP_SCALE = 2
        cfg.MODEL.ROI_DIGIT_NECK.FUSION_TYPE = "cat"
        input_shapes = {
            "keypoint_heatmap_shape": ShapeSpec(channels=17, height=56, width=56),
            "person_box_features_shape": ShapeSpec(channels=256, height=14, width=14)
        }
        neck = build_digit_neck_output(cfg, input_shapes)
        inp1 = torch.rand(2, 17, 56, 56)
        inp2 = torch.rand(2, 256, 14, 14)
        outputs = neck(inp1, inp2)
        print(neck)
        list_head_names = cfg.MODEL.ROI_DIGIT_NECK.OUTPUT_HEAD_NAMES
        list_out_channels = cfg.MODEL.ROI_DIGIT_NECK.OUTPUT_HEAD_CHANNELS
        for name, out_channels, output in zip(list_head_names, list_out_channels, outputs):
            print(name + "--OK")
            self.assertTrue(output.shape == (2, out_channels, 56, 56))


if __name__ == "__main__":
    unittest.main()