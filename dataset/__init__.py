""" Init dataset """

from .transforms import *
from . import ModelNet40v1
from .ModelNet40v1 import *
from . import aflw2000
from .aflw2000 import *

__all__ = []
__all__.extend(ModelNet40v1.__all__)
__all__.extend(aflw2000.__all__)
