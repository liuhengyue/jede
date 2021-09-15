from .roi_heads import (
    BaseROIHeads,
    BasePGROIHeads
)

from .keypoint_head import KPGRCNNHead

from .necks.digit_neck import (
    build_digit_neck_output,
    DigitNeck
)

from .necks.number_neck import build_number_neck_output

from .jersey_number_head import build_jersey_number_head, JerseyNumberOutputLayers

from .layers import coord_attention, dual_attention

from .meta_arch import pgrcnn

from .utils import ctdet_decode, compute_targets, compute_number_targets

_EXCLUDE = {"ShapeSpec"}
__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]

from detectron2.utils.env import fixup_module_metadata

fixup_module_metadata(__name__, globals(), __all__)
del fixup_module_metadata