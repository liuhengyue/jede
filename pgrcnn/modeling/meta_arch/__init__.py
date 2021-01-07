# -*- coding: utf-8 -*-

from detectron2.modeling import META_ARCH_REGISTRY, build_model  # isort:skip

# import all the meta_arch, so they will be registered
from .pgrcnn import PGRCNN
