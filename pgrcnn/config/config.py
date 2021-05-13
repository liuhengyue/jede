from detectron2.config import CfgNode as CN
# def get_cfg() -> CfgNode:
#     """
#     Get a copy of the default config.
#     Then add extra fields.
#
#     We
#
#     Returns:
#         a detectron2 CfgNode instance.
#     """
#     add_poseguide_config(_C)
#
#     return _C.clone()

def add_tridentnet_config(cfg):
    """
    Add config for tridentnet.
    """
    _C = cfg

    _C.MODEL.TRIDENT = CN()

    # Number of branches for TridentNet.
    _C.MODEL.TRIDENT.NUM_BRANCH = 3
    # Specify the dilations for each branch.
    _C.MODEL.TRIDENT.BRANCH_DILATIONS = [1, 2, 3]
    # Specify the stage for applying trident blocks. Default stage is Res4 according to the
    # TridentNet paper.
    _C.MODEL.TRIDENT.TRIDENT_STAGE = "res4"
    # Specify the test branch index TridentNet Fast inference:
    #   - use -1 to aggregate results of all branches during inference.
    #   - otherwise, only using specified branch for fast inference. Recommended setting is
    #     to use the middle branch.
    _C.MODEL.TRIDENT.TEST_BRANCH_IDX = 1

def add_poseguide_config(cfg):
    """
    # Add custom config for pose-guided heads.
    """

    # 10 digit recognition

    _C = cfg

    # dataset configurations
    _C.DATASETS.DIGIT_ONLY = True
    _C.DATASETS.TRAIN_VIDEO_IDS = [0,1,3]
    _C.DATASETS.TEST_VIDEO_IDS =[2]
    _C.DATASETS.NUM_INTERESTS = 3 # we have 3 potential digit locations (L, C, R), also we can have more
    _C.DATASETS.NUM_KEYPOINTS = 4 # we only have annotations of 4 keypoints

    _C.MODEL.ROI_DIGIT_HEAD = CN()
    _C.MODEL.ROI_DIGIT_HEAD.NAME = "Kpts2DigitHead"
    _C.MODEL.ROI_DIGIT_HEAD.NUM_DIGITS = 10
    _C.MODEL.ROI_DIGIT_HEAD.NUM_DIGIT_CLASSES = 10
    _C.MODEL.ROI_DIGIT_HEAD.DEFORMABLE = False
    _C.MODEL.ROI_DIGIT_HEAD.TRANSFORM_DIM = 9 # legacy
    _C.MODEL.ROI_DIGIT_HEAD.NUM_PROPOSAL = 3
    _C.MODEL.ROI_DIGIT_HEAD.NUM_CONV = 5
    _C.MODEL.ROI_DIGIT_HEAD.CONV_DIM = 64
    _C.MODEL.ROI_DIGIT_HEAD.NUM_FC = 0
    _C.MODEL.ROI_DIGIT_HEAD.FC_DIM = 256
    _C.MODEL.ROI_DIGIT_HEAD.NORM = ""
    # input
    _C.INPUT.RANDOM_FLIP = "none" # we do not flip since it does not make sense to flip a digit
    # augmentation
    _C.INPUT.AUG = CN()
    _C.INPUT.AUG.COLOR = False
    _C.INPUT.AUG.GRAYSCALE = False
    _C.INPUT.AUG.EXTEND = False