"""Point cloud transforms functions."""

import numpy as np
import mindspore.dataset.transforms.py_transforms as trans
from engine.class_factory import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.PIPELINE)
class PcToTensor(trans.PyTensorOperation):
    """
    Convert the input point cloud in type numpy.ndarray of shape (N, C)
    to numpy.ndarray of shape (C, N).

    Args:
       new_order(tuple), new_order of output.

    Examples:
      >>> #  Convert the input video frames in type numpy
      >>> trans = [transform.PcToTensor()]
   """

    def __init__(self, order=(1, 0)):
        self.order = tuple(order)

    def __call__(self, x):
        """
        Args:
           Video(list): Video to be tensor.

        Returns:
           seq video: Tensor of seq video.
        """
        if isinstance(x, np.ndarray):
            return np.transpose(x, self.order).astype(np.float32)
        raise AssertionError(
            "Type of input should be numpy but got {}.".format(
                type(x).__name__))
