from .roi_heads import (
    PersonROIHeads
)
from .pg_head import (
    PGROIHeads
)
from .keypoint_head import KPGRCNNHead

from .meta_arch import (
    META_ARCH_REGISTRY,
    pgrcnn
)

_EXCLUDE = {"ShapeSpec"}
__all__ = [k for k in globals().keys() if k not in _EXCLUDE and not k.startswith("_")]

from detectron2.utils.env import fixup_module_metadata

fixup_module_metadata(__name__, globals(), __all__)
del fixup_module_metadata