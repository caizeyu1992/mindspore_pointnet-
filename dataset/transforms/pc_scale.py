"""Point cloud transforms functions."""

import numpy as np
import mindspore.dataset.transforms.py_transforms as trans
from engine.class_factory import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.PIPELINE)
class PcScale(trans.PyTensorOperation):
    """
    Random scale the input point cloud with: output = input * scale

    Args:
       scale_low(float): low bound for the scale size.
       scale_high(float): upper bound for the scale size.

    Examples:
      >>> #  Random scale the input point cloud in type numpy.
      >>> trans = [PcScale(scale_low=0.8, scale_high=1.2)]
   """

    def __init__(self, scale_low=0.8, scale_high=1.2):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, x):
        """
        Args:
           Point cloud(numpy array): point cloud data.

        Returns:
           transformed Point cloud: point cloud data.
        """
        if isinstance(x, np.ndarray):
            scale = np.random.uniform(self.scale_low, self.scale_high)
            output = x * scale
            return output
        raise AssertionError(
            "Type of input should be numpy but got {}.".format(
                type(x).__name__))
