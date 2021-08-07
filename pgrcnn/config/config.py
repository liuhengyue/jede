from detectron2.config import CfgNode as CN
from detectron2.config.defaults import _C

__all__ = ["get_cfg", "add_poseguide_config", "add_tridentnet_config"]

def get_cfg() -> CN:
    """
    Get a copy of the default config.
    Then add extra fields.

    We

    Returns:
        a detectron2 CfgNode instance.
    """
    add_poseguide_config(_C)

    return _C.clone()

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
    _C.DATASETS.DIGIT_ONLY = False
    _C.DATASETS.NUM_IMAGES = -1 # -1 means all
    _C.DATASETS.TRAIN_VIDEO_IDS = [0,1,3]
    _C.DATASETS.TEST_VIDEO_IDS =[2]
    _C.DATASETS.NUM_INTERESTS = 1 # we have 1 map, or 3 potential digit locations (L, C, R), also we can have more
    _C.DATASETS.NUM_KEYPOINTS = 4 # we only have annotations of 4 keypoints (LS, RS, RH, LH)
    _C.DATASETS.PAD_TO_FULL = True # if true, we use all 17 keypoints, o/w use 4
    _C.DATASETS.KEYPOINTS_INDS = [5, 6, 12, 11] # the indices of keypoints we have in the order wrt COCO

    # _C.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 6000
    # _C.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000
    _C.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05

    _C.MODEL.ROI_HEADS.NAME = "PGROIHeads"
    _C.MODEL.ROI_HEADS.ENABLE_POSE_GUIDE = True
    _C.MODEL.ROI_JERSEY_NUMBER_DET = CN()
    _C.MODEL.ROI_JERSEY_NUMBER_DET.NAME  = None # "NumDigitClassification"
    _C.MODEL.ROI_HEADS.OFFSET_TEST = 0.0 # add a small number (feature level wrt 56x56) to get a larger bounding box

    _C.MODEL.ROI_BOX_HEAD.DIGIT_POOLER_RESOLUTION = 7

    _C.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE = 1 # we can train without keypoints

    _C.MODEL.ROI_DIGIT_NECK = CN()
    _C.MODEL.ROI_DIGIT_NECK.NAME = "DigitNeck"
    _C.MODEL.ROI_DIGIT_NECK.OUTPUT_HEAD_NAMES = ("center", "size", "offset")
    _C.MODEL.ROI_DIGIT_NECK.OUTPUT_HEAD_CHANNELS = (1, 2, 2)
    _C.MODEL.ROI_DIGIT_NECK.OUTPUT_HEAD_WEIGHTS = (1.0, 0.1, 1.0)
    _C.MODEL.ROI_DIGIT_NECK.USE_PERSON_BOX_FEATURES = True
    _C.MODEL.ROI_DIGIT_NECK.USE_KEYPOINTS_FEATURES = True
    _C.MODEL.ROI_DIGIT_NECK.BBOX_REG_LOSS_TYPE = "smooth_l1"
    _C.MODEL.ROI_DIGIT_NECK.NUM_DIGIT_CLASSES = 10
    _C.MODEL.ROI_DIGIT_NECK.DEFORMABLE = False
    _C.MODEL.ROI_DIGIT_NECK.BATCH_DIGIT_SIZE_PER_IMAGE = 256 # number of digit rois to train per image
    # these two are the number of digit proposals per person roi
    _C.MODEL.ROI_DIGIT_NECK.SIZE_TARGET_SCALE = "feature"
    _C.MODEL.ROI_DIGIT_NECK.SIZE_TARGET_TYPE = "wh"
    # per person roi
    _C.MODEL.ROI_DIGIT_NECK.NUM_PROPOSAL_TRAIN = 100
    _C.MODEL.ROI_DIGIT_NECK.NUM_PROPOSAL_TEST = 20
    _C.MODEL.ROI_DIGIT_NECK.NUM_CONV = 4
    _C.MODEL.ROI_DIGIT_NECK.CONV_DIM = 64
    _C.MODEL.ROI_DIGIT_NECK.NORM = ""
    _C.MODEL.ROI_DIGIT_NECK.FOCAL_BIAS = -2.19 # −log((1−pi)/pi), pi=0.1 -> -2.19
    _C.MODEL.ROI_DIGIT_NECK.SCORE_THRESH_TEST = 0.01
    _C.MODEL.ROI_DIGIT_NECK.NMS_THRESH_TEST = 0.5
    _C.MODEL.ROI_DIGIT_NECK.OFFSET_REG = True
    _C.MODEL.ROI_DIGIT_NECK.FUSION_TYPE = "cat" # cat, sum, ""

    # digit neck branches
    _C.MODEL.ROI_DIGIT_NECK_BRANCHES = CN()
    _C.MODEL.ROI_DIGIT_NECK_BRANCHES.NORM = ""

    _C.MODEL.ROI_DIGIT_NECK_BRANCHES.PERSON_BRANCH = CN()
    _C.MODEL.ROI_DIGIT_NECK_BRANCHES.PERSON_BRANCH.NAME =  "PersonROIBranch"
    _C.MODEL.ROI_DIGIT_NECK_BRANCHES.PERSON_BRANCH.UP_SCALE = 2
    _C.MODEL.ROI_DIGIT_NECK_BRANCHES.PERSON_BRANCH.DECONV_KERNEL = 4
    _C.MODEL.ROI_DIGIT_NECK_BRANCHES.PERSON_BRANCH.CONV_DIMS = [64, 64]

    _C.MODEL.ROI_DIGIT_NECK_BRANCHES.KEYPOINTS_BRANCH = CN()
    _C.MODEL.ROI_DIGIT_NECK_BRANCHES.KEYPOINTS_BRANCH.NAME = "KptsROIBranch"
    _C.MODEL.ROI_DIGIT_NECK_BRANCHES.KEYPOINTS_BRANCH.UP_SCALE = 1
    _C.MODEL.ROI_DIGIT_NECK_BRANCHES.KEYPOINTS_BRANCH.DECONV_KERNEL = 0
    # vailid for KptsROIBranch
    _C.MODEL.ROI_DIGIT_NECK_BRANCHES.KEYPOINTS_BRANCH.CONV_SPECS = (3, 2, 1) # kernel_size, stride, padding
    _C.MODEL.ROI_DIGIT_NECK_BRANCHES.KEYPOINTS_BRANCH.CONV_DIMS = [64, 64]

    # attention based
    # _C.MODEL.ROI_DIGIT_NECK_BRANCHES.KEYPOINTS_BRANCH.NAME = "KptsAttentionBranch"
    # _C.MODEL.ROI_DIGIT_NECK_BRANCHES.KEYPOINTS_BRANCH.UP_SCALE = 2
    # _C.MODEL.ROI_DIGIT_NECK_BRANCHES.KEYPOINTS_BRANCH.CONV_SPECS = (3, 1, 1)  # kernel_size, stride, padding
    # _C.MODEL.ROI_DIGIT_NECK_BRANCHES.KEYPOINTS_BRANCH.CONV_DIMS = [64, 64]

    # input
    _C.INPUT.RANDOM_FLIP = "none" # we do not flip since it does not make sense to flip a digit
    # augmentation
    _C.INPUT.AUG = CN()
    _C.INPUT.AUG.COLOR = False
    _C.INPUT.AUG.GRAYSCALE = False
    _C.INPUT.AUG.EXTEND = False

    _C.INPUT.AUG.COPY_PASTE_MIX = 0 # 0 means do not apply, otherwise the number of images to add