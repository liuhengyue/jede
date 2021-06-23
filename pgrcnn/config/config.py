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

    # some customizations of RPN
    _C.MODEL.PROPOSAL_GENERATOR.NAME = "PlayerRPN"
    # disable shuffle for debugging
    _C.DATALOADER.SHUFFLE = True
    # dataset configurations
    _C.DATASETS.DIGIT_ONLY = True
    _C.DATASETS.NUM_IMAGES = -1 # -1 means all
    _C.DATASETS.TRAIN_VIDEO_IDS = [0,1,3]
    _C.DATASETS.TEST_VIDEO_IDS =[2]
    _C.DATASETS.NUM_INTERESTS = 1 # we have 1 map, or 3 potential digit locations (L, C, R), also we can have more
    _C.DATASETS.NUM_KEYPOINTS = 4 # we only have annotations of 4 keypoints (LS, RS, RH, LH)
    _C.DATASETS.PAD_TO_FULL = True # if true, we use all 17 keypoints, o/w use 4
    _C.DATASETS.KEYPOINTS_INDS = [5, 6, 12, 11] # the indices of keypoints we have in the order wrt COCO

    _C.MODEL.ROI_HEADS.NAME = "PGROIHeads"

    _C.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE = 0 # we can train without keypoints

    _C.MODEL.ROI_DIGIT_HEAD = CN()
    _C.MODEL.ROI_DIGIT_HEAD.NAME = "Kpts2DigitHead"
    _C.MODEL.ROI_DIGIT_HEAD.USE_PERSON_BOX_FEATURES = True
    _C.MODEL.ROI_DIGIT_HEAD.NUM_DIGIT_CLASSES = 10
    _C.MODEL.ROI_DIGIT_HEAD.DEFORMABLE = False
    _C.MODEL.ROI_DIGIT_HEAD.TRANSFORM_DIM = 9 # legacy
    _C.MODEL.ROI_DIGIT_HEAD.BATCH_DIGIT_SIZE_PER_IMAGE = 256 # number of digit rois to train per image
    _C.MODEL.ROI_DIGIT_HEAD.NUM_PROPOSAL = 3
    _C.MODEL.ROI_DIGIT_HEAD.NUM_CONV = 5
    _C.MODEL.ROI_DIGIT_HEAD.CONV_DIM = 64
    _C.MODEL.ROI_DIGIT_HEAD.NUM_FC = 0
    _C.MODEL.ROI_DIGIT_HEAD.FC_DIM = 256
    _C.MODEL.ROI_DIGIT_HEAD.NORM = ""
    _C.MODEL.ROI_DIGIT_HEAD.FOCAL_BIAS = -2.19 # −log((1−pi)/pi), pi=0.1 -> -2.19
    # input
    _C.INPUT.RANDOM_FLIP = "none" # we do not flip since it does not make sense to flip a digit
    # augmentation
    _C.INPUT.AUG = CN()
    _C.INPUT.AUG.COLOR = False
    _C.INPUT.AUG.GRAYSCALE = False
    _C.INPUT.AUG.EXTEND = False

    _C.INPUT.AUG.COPY_PASTE_MIX = 0 # 0 means do not apply, otherwise the number of images to add