from detectron2.config.config import CfgNode
from detectron2.config.defaults import _C
def get_cfg() -> CfgNode:
    """
    Get a copy of the default config.
    Then add extra fields.

    We

    Returns:
        a detectron2 CfgNode instance.
    """
    add_poseguide_config(_C)

    return _C.clone()

def add_poseguide_config(cfg):
    """
    # Add custom config for pose-guided heads.
    """

    # 10 digit recognition

    cfg.DATASETS.DIGIT_ONLY = True
    cfg.DATASETS.TRAIN_VIDEO_IDS = [0,1,3]
    cfg.DATASETS.TEST_VIDEO_IDS =[2]

    _C.MODEL.ROI_DIGIT_HEAD = CfgNode(new_allowed=True)
    cfg.MODEL.ROI_DIGIT_HEAD.NUM_DIGITS = 10

    # augmentation
    _C.INPUT.AUG = CfgNode(new_allowed=True)
    cfg.INPUT.AUG.COLOR = False
    cfg.INPUT.AUG.GRAYSCALE = False
    cfg.INPUT.AUG.EXTEND = False