from .roi_heads import (
    BaseROIHeads
)
from .pg_head import (
    PGROIHeads
)
from .keypoint_head import KPGRCNNHead

from .digit_neck import (
    build_digit_neck,
    DigitNeck
)

from .digit_neck_branches import (
    build_digit_neck_branch,
    PersonROIBranch,
    KptsROIBranch,
    KptsAttentionBranch,
)

from .meta_arch import pgrcnn
from .utils import compute_targets

_EXCLUDE = {"ShapeSpec"}
__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]

from detectron2.utils.env import fixup_module_metadata

fixup_module_metadata(__name__, globals(), __all__)
del fixup_module_metadata