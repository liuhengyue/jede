# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
import torch

import detectron2.export.torchscript  # apply patch # noqa
from detectron2 import model_zoo
from detectron2.layers import ShapeSpec
from detectron2.utils.env import TORCH_VERSION

from pgrcnn.config import get_cfg
from pgrcnn.modeling import build_digit_neck_branch


class TestDigitNeckBranch(unittest.TestCase):
    @unittest.skipIf(TORCH_VERSION < (1, 8), "Insufficient pytorch version")
    def test_KptsAttentionBranch(self):
        cfg = get_cfg()
        cfg.MODEL.ROI_DIGIT_NECK_BRANCHES.KEYPOINTS_BRANCH.NAME = "KptsAttentionBranch"
        neck_branch = build_digit_neck_branch(cfg, ShapeSpec(channels=17, height=56, width=56))
        inp = torch.rand(2, 17, 56, 56)
        out = neck_branch(inp)
        self.assertTrue(isinstance(out, torch.Tensor))


if __name__ == "__main__":
    unittest.main()